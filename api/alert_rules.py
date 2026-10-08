"""HSF alert rules: user-owned conditions on canonical HSF intelligence, evaluated
server-side by the real-time alert worker that already runs in hsf-api.

Rule types (all evaluated from the latest saved full-market scan, the same data the
Scanner, Stock Intelligence and Watchlist Intelligence show; see
api.watchlist_intel.market_observation):

    HSF_SCORE_ABOVE / HSF_SCORE_BELOW            level: score >= / < threshold
    HSF_SCORE_CROSS_ABOVE / HSF_SCORE_CROSS_BELOW transition: last seen < t <= now / last seen >= t > now
    PREBREAKOUT_ACTIVE                            transition: PreBreakout signal off -> on (Premium)
    SETUP_APPEARED                                transition: not a ranked setup -> ranked setup
                                                  (optionally only a given setup, e.g. "Breakout")
    RVOL_ABOVE                                    level: scan relative volume >= threshold
    RANK_IMPROVED                                 transition: HSF rank moved up >= threshold places
                                                  since the last scan this rule saw

"Last seen" is persisted per (rule, ticker) in hsf_alert_rule_state, so a crossing
fires once (77 -> 82 crosses 80; 83 -> 84 doesn't) and re-evaluating an observation
the rule already saw does nothing. Level rules fire on each new scan while true, at
most once per cooldown. The first scan a rule sees only sets its baseline for the
transition types.

Lifecycle of one pass (evaluate_once, called from the worker loop):
    latest scan -> skip if none / stale (a scheduled scan was missed) -> enabled rules
    (+ owners' plans) -> plan limits -> per-ticker state -> pure rule check -> cooldown
    -> one transaction: insert events (UNIQUE rule/ticker/observation), upsert state
    -> deliver (in-app = the event row; email via the worker's SMTP sender) -> record
    per-channel delivery status; failed emails are retried on later passes.

Unsupported on purpose (no canonical server-side data to evaluate reliably):
Day Trader / Stair-Stepper setups (built from live 1-minute bars on request, not
persisted per symbol), RSI (not in the scans), web push (no sender yet).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("hsf_api.alert_rules")

# ---- catalog ----------------------------------------------------------------------------------------
LEVEL, EDGE = "level", "transition"
RULE_TYPES: Dict[str, Dict[str, Any]] = {
    "HSF_SCORE_ABOVE": {"operator": ">=", "kind": LEVEL, "threshold": (0.0, 100.0), "default": 80.0,
                        "label": "HSF Score at or above",
                        "description": "Fires on each new market scan where the HSF Score is at or above your "
                                       "value (at most once per cooldown)."},
    "HSF_SCORE_BELOW": {"operator": "<", "kind": LEVEL, "threshold": (0.0, 100.0), "default": 60.0,
                        "label": "HSF Score below",
                        "description": "Fires on each new market scan where the ticker is ranked and its HSF "
                                       "Score is below your value (at most once per cooldown)."},
    "HSF_SCORE_CROSS_ABOVE": {"operator": "crosses_above", "kind": EDGE, "threshold": (0.0, 100.0),
                              "default": 80.0, "label": "HSF Score crosses above",
                              "description": "Fires once when the HSF Score moves from below your value to at "
                                             "or above it between scans."},
    "HSF_SCORE_CROSS_BELOW": {"operator": "crosses_below", "kind": EDGE, "threshold": (0.0, 100.0),
                              "default": 60.0, "label": "HSF Score drops below",
                              "description": "Fires once when the HSF Score moves from at or above your value "
                                             "to below it between scans."},
    "PREBREAKOUT_ACTIVE": {"operator": "becomes_true", "kind": EDGE, "threshold": None,
                           "label": "Becomes PreBreakout", "entitlement": "can_early_breakout",
                           "description": "Fires when the PreBreakout model signal turns on for the ticker."},
    "SETUP_APPEARED": {"operator": "appears", "kind": EDGE, "threshold": None, "value": True,
                       "label": "New HSF setup",
                       "description": "Fires when the ticker becomes a ranked HSF setup (optionally only a "
                                      "given setup, e.g. Breakout or Golden Cross)."},
    "RVOL_ABOVE": {"operator": ">=", "kind": LEVEL, "threshold": (1.0, 1000.0), "default": 2.0,
                   "label": "Relative volume at or above",
                   "description": "Fires on each new market scan where the ticker trades at your multiple of "
                                  "its 20-day average volume or more (at most once per cooldown)."},
    "RANK_IMPROVED": {"operator": "improves_by", "kind": EDGE, "threshold": (1.0, 10000.0), "default": 10.0,
                      "label": "HSF rank improves by",
                      "description": "Fires when the ticker moves up at least your number of places in the HSF "
                                     "ranking since the last scan."},
}
CHANNELS = ("in_app", "email")
CHANNEL_ENTITLEMENT = {"email": "can_email_alerts"}
DEFAULT_COOLDOWN = {LEVEL: 86400, EDGE: 3600}
MAX_COOLDOWN_S = 7 * 86400
EMAIL_RETRY_ATTEMPTS = 3
EMAIL_RETRY_WINDOW_S = 6 * 3600


class RuleError(ValueError):
    """Invalid rule input (422)."""


class RuleForbidden(PermissionError):
    """The plan doesn't include this rule type or channel (403)."""


def rule_types(entitlements: Dict[str, bool]) -> List[Dict[str, Any]]:
    out = []
    for t, spec in RULE_TYPES.items():
        thr = spec.get("threshold")
        out.append({"type": t, "label": spec["label"], "description": spec["description"],
                    "operator": spec["operator"], "kind": spec["kind"],
                    "threshold": ({"min": thr[0], "max": thr[1], "default": spec.get("default")} if thr else None),
                    "takes_value": bool(spec.get("value")),
                    "default_cooldown_seconds": DEFAULT_COOLDOWN[spec["kind"]],
                    "available": _allowed(spec, entitlements)})
    return out


def _allowed(spec: Dict[str, Any], entitlements: Dict[str, bool]) -> bool:
    need = spec.get("entitlement")
    return bool(entitlements.get(need)) if need else True


def capabilities(ent: Dict[str, Any], *, watchlist_max: int, tickers_per_request: int) -> Dict[str, Any]:
    """The server-side limits for watchlists and alerts on this plan, in one place.
    Only limits that already exist are numbers; null means no plan limit is defined."""
    e = ent["entitlements"]
    return {
        "tier": ent["tier"],
        "max_watchlists": watchlist_max,
        "max_symbols_per_watchlist": None,
        "max_symbols_per_request": tickers_per_request,
        "max_active_alerts": int(ent["alert_limit"]),
        "alert_rule_types": [t for t, s in RULE_TYPES.items() if _allowed(s, e)],
        "delivery_channels": [c for c in CHANNELS if not CHANNEL_ENTITLEMENT.get(c) or e.get(CHANNEL_ENTITLEMENT[c])],
    }


# ---- validation -------------------------------------------------------------------------------------
def validate(entitlements: Dict[str, bool], *, rule_type: str, threshold: Optional[float], value: Optional[str],
             ticker: Optional[str], watchlist_id: Optional[int], delivery_channels: Optional[Sequence[str]],
             cooldown_seconds: Optional[int], partial: bool = False) -> Dict[str, Any]:
    from db.watchlists import normalize_watchlist_ticker

    spec = RULE_TYPES.get(str(rule_type or "").upper())
    if spec is None:
        raise RuleError(f"Unknown rule type. Use one of: {', '.join(RULE_TYPES)}.")
    if not _allowed(spec, entitlements):
        raise RuleForbidden(f"{spec['label']} alerts are a Premium feature.")
    out: Dict[str, Any] = {}
    if not partial:
        t = normalize_watchlist_ticker(ticker) if ticker else ""
        if ticker and not t:
            raise RuleError("Enter a valid ticker symbol.")
        if bool(t) == (watchlist_id is not None):
            raise RuleError("Set exactly one of ticker or watchlist_id.")
        out.update(ticker=t or None, watchlist_id=int(watchlist_id) if watchlist_id is not None else None,
                   rule_type=str(rule_type).upper(), operator=spec["operator"])
    rng = spec.get("threshold")
    if rng:
        if threshold is None:
            if not partial:
                raise RuleError("Enter a threshold.")
        else:
            if not (rng[0] <= float(threshold) <= rng[1]):
                raise RuleError(f"Threshold must be between {rng[0]:g} and {rng[1]:g}.")
            out["threshold"] = float(threshold)
    elif threshold is not None:
        raise RuleError("This rule type takes no threshold.")
    if value is not None:
        if not spec.get("value"):
            raise RuleError("This rule type takes no value.")
        out["value"] = str(value).strip()[:40] or None
    elif not partial and spec.get("value"):
        out["value"] = None
    if delivery_channels is not None or not partial:
        chans = list(dict.fromkeys(str(c).strip().lower() for c in (delivery_channels or ["in_app"])))
        bad = [c for c in chans if c not in CHANNELS]
        if bad or not chans:
            raise RuleError(f"Delivery channels must be from: {', '.join(CHANNELS)}.")
        for c in chans:
            need = CHANNEL_ENTITLEMENT.get(c)
            if need and not entitlements.get(need):
                raise RuleForbidden("Email alerts are included with Pro and Premium.")
        out["delivery_channels"] = chans
    if cooldown_seconds is not None:
        if not (0 <= int(cooldown_seconds) <= MAX_COOLDOWN_S):
            raise RuleError(f"cooldown_seconds must be between 0 and {MAX_COOLDOWN_S}.")
        out["cooldown_seconds"] = int(cooldown_seconds)
    elif not partial:
        out["cooldown_seconds"] = DEFAULT_COOLDOWN[spec["kind"]]
    if not partial:
        out.setdefault("threshold", None)
        out.setdefault("value", None)
    return out


# ---- pure rule check --------------------------------------------------------------------------------
def check(rule: Dict[str, Any], view: Dict[str, Any], state: Optional[Dict[str, Any]]
          ) -> Tuple[bool, Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    """(fires, trigger value, previous value, new state) for one rule on one ticker's
    canonical view. new state None = this observation says nothing about the ticker
    (absent from the scan), so the stored state is left as it was."""
    if not view.get("present"):
        return False, None, None, None
    rt, thr = rule["rule_type"], rule.get("threshold")
    state = state or {}
    prev_value, prev_flag = state.get("last_value"), state.get("last_flag")
    has_baseline = bool(state.get("observation_id"))
    score, rank = view.get("hsf_score"), view.get("rank")

    if rt in ("HSF_SCORE_ABOVE", "HSF_SCORE_BELOW", "HSF_SCORE_CROSS_ABOVE", "HSF_SCORE_CROSS_BELOW"):
        # Unranked = no HSF score: keep the last known score as the comparison point.
        new = {"last_value": score if score is not None else prev_value, "last_flag": None}
        if score is None:
            return False, None, prev_value, new
        if rt == "HSF_SCORE_ABOVE":
            return score >= thr, score, prev_value, new
        if rt == "HSF_SCORE_BELOW":
            return score < thr, score, prev_value, new
        if prev_value is None:
            return False, score, None, new
        if rt == "HSF_SCORE_CROSS_ABOVE":
            return prev_value < thr <= score, score, prev_value, new
        return prev_value >= thr > score, score, prev_value, new
    if rt == "RVOL_ABOVE":
        rvol = view.get("rvol")
        new = {"last_value": rvol, "last_flag": None}
        return (rvol is not None and rvol >= thr), rvol, prev_value, new
    if rt == "PREBREAKOUT_ACTIVE":
        flag = bool(view.get("prebreakout"))
        new = {"last_value": view.get("prebreakout_score"), "last_flag": flag}
        return (has_baseline and prev_flag is False and flag), view.get("prebreakout_score"), prev_value, new
    if rt == "SETUP_APPEARED":
        want = str(rule.get("value") or "").strip().lower()
        setup = view.get("setup")
        flag = bool(view.get("ranked")) and (not want or str(setup or "").strip().lower() == want)
        new = {"last_value": score, "last_flag": flag, "last_text": setup}
        return (has_baseline and prev_flag is False and flag), score, prev_value, new
    if rt == "RANK_IMPROVED":
        new = {"last_value": float(rank) if rank is not None else None, "last_flag": None}
        if rank is None or prev_value is None:
            return False, (float(rank) if rank is not None else None), prev_value, new
        return (prev_value - rank) >= thr, float(rank), prev_value, new
    return False, None, None, None


def message_for(rule: Dict[str, Any], ticker: str, value: Optional[float], prev: Optional[float],
                view: Dict[str, Any]) -> str:
    rt, thr = rule["rule_type"], rule.get("threshold")
    if rt == "HSF_SCORE_ABOVE":
        return f"{ticker} HSF Score {value:.0f} is at or above {thr:g}"
    if rt == "HSF_SCORE_BELOW":
        return f"{ticker} HSF Score {value:.0f} is below {thr:g}"
    if rt == "HSF_SCORE_CROSS_ABOVE":
        return f"{ticker} HSF Score crossed above {thr:g} ({prev:.0f} -> {value:.0f})"
    if rt == "HSF_SCORE_CROSS_BELOW":
        return f"{ticker} HSF Score dropped below {thr:g} ({prev:.0f} -> {value:.0f})"
    if rt == "RVOL_ABOVE":
        return f"{ticker} is trading {value:.1f}x its 20-day average volume (your level {thr:g}x)"
    if rt == "PREBREAKOUT_ACTIVE":
        return f"{ticker} is now a PreBreakout setup"
    if rt == "SETUP_APPEARED":
        return f"{ticker} is a new HSF setup: {view.get('setup') or 'Signal'}"
    if rt == "RANK_IMPROVED":
        return f"{ticker} moved up {prev - value:.0f} places to #{value:.0f} in the HSF ranking"
    return f"{ticker} alert"


# ---- metrics ----------------------------------------------------------------------------------------
_metrics_lock = threading.Lock()
_metrics: Dict[str, Any] = {"passes": 0, "rules_evaluated": 0, "rules_triggered": 0, "events_deduplicated": 0,
                            "delivery_successes": 0, "delivery_failures": 0, "last_pass_ms": None,
                            "last_observation_id": None, "last_skip_reason": None}


def _bump(**kw: Any) -> None:
    with _metrics_lock:
        for k, v in kw.items():
            if isinstance(v, int) and not isinstance(v, bool) and isinstance(_metrics.get(k), int):
                _metrics[k] += v
            else:
                _metrics[k] = v


def metrics() -> Dict[str, Any]:
    with _metrics_lock:
        return dict(_metrics)


# ---- evaluator --------------------------------------------------------------------------------------
def _plan(rule: Dict[str, Any]) -> Tuple[str, Dict[str, bool], int]:
    from api.main import entitlements_for

    ent = entitlements_for({"tier": rule.get("owner_tier"), "is_admin": rule.get("owner_is_admin")})
    return ent["tier"], ent["entitlements"], int(ent["alert_limit"])


def _apply_limits(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Newest rules first, up to each owner's plan limit minus their enabled ticker
    alerts (one limit covers both), and only the types and channels the plan has
    now, so a downgrade takes effect at the next pass."""
    kept, used = [], {}
    for r in rules:
        if not r.get("owner_is_active", True):
            continue
        tier, ent, limit = _plan(r)
        spec = RULE_TYPES.get(r["rule_type"])
        if spec is None or not _allowed(spec, ent):
            continue
        u = r["user_id"]
        if u not in used:
            used[u] = int(r.get("owner_legacy_alerts") or 0)
        if used[u] >= limit:
            continue
        used[u] += 1
        chans = [c for c in r["delivery_channels"] if not CHANNEL_ENTITLEMENT.get(c) or ent.get(CHANNEL_ENTITLEMENT[c])]
        kept.append({**r, "delivery_channels": chans or ["in_app"], "_premium": bool(ent.get("can_early_breakout")),
                     "_tier": tier})
    return kept


def _cooling(state: Optional[Dict[str, Any]], cooldown_s: int, now: dt.datetime) -> bool:
    last = (state or {}).get("last_triggered_at")
    if not last or cooldown_s <= 0:
        return False
    try:
        t = dt.datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return False
    t = t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    return (now - t).total_seconds() < cooldown_s


def evaluate_once(now: Optional[dt.datetime] = None, *, observation: Optional[Dict[str, Any]] = None,
                  deliver: bool = True) -> Dict[str, Any]:
    """One evaluation pass. Idempotent: a rule that already saw this observation is skipped,
    and events are unique per (rule, ticker, observation)."""
    from api import watchlist_intel
    from db import alert_rules as store

    t0 = time.perf_counter()
    now = now or dt.datetime.now(dt.timezone.utc)
    result: Dict[str, Any] = {"observation_id": None, "rules": 0, "evaluated": 0, "triggered": 0,
                              "deduplicated": 0, "skipped": None, "delivered": 0, "delivery_failed": 0}
    rules = _apply_limits(store.load_enabled_rules())
    result["rules"] = len(rules)
    if not rules:  # don't load scan data for nothing
        _bump(passes=1, last_skip_reason="no_rules")
        result["skipped"] = "no_rules"
        return result
    obs = observation if observation is not None else watchlist_intel.market_observation()
    if obs is None:
        result["skipped"] = "no_scan"
    elif watchlist_intel.observation_stale(obs, now):
        result["skipped"] = "stale_scan"  # wait for a fresh scan; state is untouched, nothing is lost
    if result["skipped"]:
        _bump(passes=1, last_skip_reason=result["skipped"])
        return result
    obs_id = obs["observation_id"]
    result["observation_id"] = obs_id
    wl = store.watchlist_tickers({(r["user_id"], r["watchlist_id"]) for r in rules if r.get("watchlist_id")})
    states = store.load_states([r["id"] for r in rules])
    now_s = now.isoformat(timespec="microseconds")
    views: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    new_states, events, evaluated, triggered = [], [], set(), set()
    for r in rules:
        if r.get("watchlist_id"):
            tickers = wl.get(int(r["watchlist_id"]))
            if tickers is None:  # watchlist deleted or not the owner's: nothing to watch
                continue
        else:
            tickers = [r["ticker"]]
        for t in tickers:
            st = states.get((r["id"], t))
            if st and st.get("observation_id") == obs_id:
                continue  # this rule already saw this scan for this ticker
            key = (t, r["_premium"])
            if key not in views:
                views[key] = watchlist_intel.ticker_view(obs, t, premium=r["_premium"])
            view = views[key]
            fires, value, prev, new = check(r, view, st)
            evaluated.add(r["id"])
            result["evaluated"] += 1
            if new is None:
                continue
            last_trig = (st or {}).get("last_triggered_at")
            if fires and _cooling(st, int(r["cooldown_seconds"]), now):
                fires = False
                result["deduplicated"] += 1
            if fires:
                last_trig = now_s
                triggered.add(r["id"])
                events.append({
                    "user_id": r["user_id"], "rule_id": r["id"], "watchlist_id": r.get("watchlist_id"),
                    "ticker": t, "rule_type": r["rule_type"], "operator": r["operator"],
                    "threshold": r.get("threshold"), "trigger_value": value, "previous_value": prev,
                    "hsf_score": view.get("hsf_score"), "setup": view.get("setup"),
                    "message": message_for(r, t, value, prev, view), "observation_id": obs_id,
                    "market_data_as_of": watchlist_intel._iso(obs["scan_at"]), "triggered_at": now_s,
                    "delivery": {c: "pending" if c != "in_app" else "delivered" for c in r["delivery_channels"]},
                    "_channels": r["delivery_channels"], "_tier": r["_tier"],
                    "_email_verified": r.get("owner_email_verified"),
                })
            new_states.append({"rule_id": r["id"], "ticker": t, **new, "observation_id": obs_id,
                               "last_triggered_at": last_trig})
    inserted = store.save_evaluation(states=new_states, events=events, evaluated_rule_ids=sorted(evaluated),
                                     triggered_rule_ids=sorted(triggered), now=now_s)
    result["triggered"] = len(inserted)
    result["deduplicated"] += len(events) - len(inserted)  # another pass/process already recorded them
    if deliver:
        for ev in inserted:
            ok, failed = _deliver(ev)
            result["delivered"] += ok
            result["delivery_failed"] += failed
        retried_ok, retried_failed = retry_failed_deliveries(now)
        result["delivered"] += retried_ok
        result["delivery_failed"] += retried_failed
    ms = round((time.perf_counter() - t0) * 1000, 1)
    result["ms"] = ms
    _bump(passes=1, rules_evaluated=result["evaluated"], rules_triggered=result["triggered"],
          events_deduplicated=result["deduplicated"], delivery_successes=result["delivered"],
          delivery_failures=result["delivery_failed"], last_pass_ms=ms, last_observation_id=obs_id,
          last_skip_reason=None)
    if result["evaluated"] or result["triggered"]:
        log.info(json.dumps({"event": "alert_rules_pass", **result}))
    return result


# ---- delivery ---------------------------------------------------------------------------------------
def _send_email(user_id: str, message: str) -> Tuple[bool, Optional[str]]:
    """Email through the real-time worker's SMTP sender, with the alerts unsubscribe
    link. (sent, error); 'skipped' cases return (False, reason) with reason prefixed
    'skip:' so they aren't retried."""
    from billing_service import realtime_alerts as rt

    if "@" not in str(user_id):
        return False, "skip:no_email_address"
    try:
        from db.email_prefs import unsubscribe_url, wants_email

        if not wants_email(user_id, "alerts"):
            return False, "skip:alert_emails_off"
        unsub = unsubscribe_url(user_id, "alerts")
    except Exception:
        unsub = None
    try:
        sent = rt._send_email(user_id, "HSF alert rule triggered", message, unsubscribe_url=unsub)
    except Exception as e:
        return False, type(e).__name__
    return (True, None) if sent else (False, "smtp_failed")


def _deliver(ev: Dict[str, Any], *, first: bool = True) -> Tuple[int, int]:
    """Deliver one stored event. The event row already exists, so a failed channel only
    changes its delivery status (kept for retry), never the event. In-app delivery is
    the event row itself."""
    from db import alert_rules as store

    delivery = dict(ev.get("delivery") or {})
    ok = failed = 0
    error = None
    attempts = 0
    for ch in list(delivery):
        if ch == "in_app":
            ok += 1 if first else 0
            continue
        if delivery[ch] not in ("pending", "failed"):
            continue
        if ch == "email":
            if not ev.get("_email_verified", True):
                delivery[ch], error = "skipped", "email_not_verified"
                continue
            attempts = 1
            sent, why = _send_email(ev["user_id"], ev["message"])
            if sent:
                delivery[ch] = "sent"
                ok += 1
            elif why and why.startswith("skip:"):
                delivery[ch], error = "skipped", why[5:]
            else:
                delivery[ch], error = "failed", why
                failed += 1
    try:
        store.set_delivery(int(ev["id"]), delivery, attempts_inc=attempts, error=error)
    except Exception as e:  # the event stays; its status may read 'pending' until the next retry
        log.warning(json.dumps({"event": "alert_rule_delivery_status_failed", "error": type(e).__name__}))
    return ok, failed


def retry_failed_deliveries(now: dt.datetime) -> Tuple[int, int]:
    from db import alert_rules as store

    since = (now - dt.timedelta(seconds=EMAIL_RETRY_WINDOW_S)).isoformat(timespec="microseconds")
    try:
        pending = store.events_to_retry(since_iso=since, before_iso=now.isoformat(timespec="microseconds"),
                                        max_attempts=EMAIL_RETRY_ATTEMPTS)
    except Exception:
        return 0, 0
    ok = failed = 0
    for ev in pending:
        a, b = _deliver(ev, first=False)
        ok += a
        failed += b
    return ok, failed


# ---- worker hook ------------------------------------------------------------------------------------
def enabled() -> bool:
    """Kill switch: HSF_ALERT_RULES_ENABLED=0 stops evaluation (rules stay saved)."""
    return os.environ.get("HSF_ALERT_RULES_ENABLED", "1").strip() != "0"


def worker_pass() -> None:
    """Called by the real-time alert worker each loop (billing_service.realtime_alerts).
    Only during extended hours (4:00-20:00 ET weekdays), when scans land and the
    price-alert worker is querying anyway, so the database can still sleep overnight
    and at weekends. Cheap when nothing changed: market_runs() is cached and a rule
    that saw the latest scan is skipped. Never raises into the worker."""
    from billing_service.realtime_alerts import market_session_open

    if not enabled() or not market_session_open():
        return
    try:
        evaluate_once()
    except Exception as e:
        # Type only: driver error text can carry connection details.
        log.warning(json.dumps({"event": "alert_rules_pass_failed", "error": type(e).__name__}))
    finally:
        try:
            from db.engine import release_thread_connection

            release_thread_connection()
        except Exception:
            pass


# ---- API service layer ------------------------------------------------------------------------------
def _db(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from api.user_data import _db as user_db

    return user_db(fn, *args, **kwargs)


def _public(rule: Dict[str, Any]) -> Dict[str, Any]:
    from db.alert_rules import RULE_COLUMNS

    return {k: rule.get(k) for k in RULE_COLUMNS if k != "user_id"}


def list_rules(user: str) -> List[Dict[str, Any]]:
    from db import alert_rules as store

    return [_public(r) for r in _db(store.list_rules, user)]


def get_rule(user: str, rule_id: int) -> Dict[str, Any]:
    from api.user_data import NotFound
    from db import alert_rules as store

    rule = _db(store.get_rule, user, int(rule_id))
    if rule is None:
        raise NotFound("alert rule")
    return _public(rule)


def create_rule(user: str, ent: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    from api import user_data
    from db import alert_rules as store

    clean = validate(ent["entitlements"], rule_type=body.get("rule_type"), threshold=body.get("threshold"),
                     value=body.get("value"), ticker=body.get("ticker"), watchlist_id=body.get("watchlist_id"),
                     delivery_channels=body.get("delivery_channels"), cooldown_seconds=body.get("cooldown_seconds"))
    if clean.get("watchlist_id") is not None:
        user_data._owned(user, clean["watchlist_id"])  # 404 when missing or someone else's
    try:
        rule = _db(store.create_rule, user, enabled=bool(body.get("enabled", True)),
                   max_active=int(ent["alert_limit"]), **clean)
    except store.RuleLimitReached as e:
        raise user_data.LimitReached(str(e)) from e
    except ValueError as e:
        raise user_data.Conflict(str(e)) from e
    log.info(json.dumps({"event": "alert_rule_created", "rule_id": rule["id"], "rule_type": rule["rule_type"]}))
    return _public(rule)


def update_rule(user: str, ent: Dict[str, Any], rule_id: int, body: Dict[str, Any]) -> Dict[str, Any]:
    from api import user_data
    from db import alert_rules as store

    current = get_rule(user, rule_id)
    clean = validate(ent["entitlements"], rule_type=current["rule_type"], threshold=body.get("threshold"),
                     value=body.get("value"), ticker=None, watchlist_id=None,
                     delivery_channels=body.get("delivery_channels"), cooldown_seconds=body.get("cooldown_seconds"),
                     partial=True)
    if body.get("enabled") is not None:
        clean["enabled"] = bool(body["enabled"])
    try:
        rule = _db(store.update_rule, user, int(rule_id), clean, max_active=int(ent["alert_limit"]))
    except store.RuleLimitReached as e:
        raise user_data.LimitReached(str(e)) from e
    if rule is None:
        raise user_data.NotFound("alert rule")
    return _public(rule)


def delete_rule(user: str, rule_id: int) -> None:
    from api.user_data import NotFound
    from db import alert_rules as store

    if not _db(store.delete_rule, user, int(rule_id)):
        raise NotFound("alert rule")


def count_active(user: str) -> int:
    from db import alert_rules as store

    return int(_db(store.count_active, user))


# ---- events: ticker alerts and rule alerts in one feed ----------------------------------------------
def _parse_ts(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    try:
        v = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)


_SRC_RANK = {"alert": 0, "rule": 1}


def encode_cursor(ts: dt.datetime, source: str, row_id: int) -> str:
    import base64

    raw = json.dumps([ts.isoformat(timespec="microseconds"), source, int(row_id)])
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> Tuple[dt.datetime, str, int]:
    import base64

    try:
        pad = "=" * (-len(cursor) % 4)
        ts, src, rid = json.loads(base64.urlsafe_b64decode(cursor + pad).decode())
        t = _parse_ts(ts)
        if t is None or src not in _SRC_RANK:
            raise ValueError
        return t, src, int(rid)
    except Exception as e:
        raise RuleError("Invalid cursor.") from e


def list_events(user: str, *, limit: int, ticker: Optional[str] = None, rule_id: Optional[int] = None,
                watchlist_id: Optional[int] = None, triggered_after: Optional[dt.datetime] = None,
                triggered_before: Optional[dt.datetime] = None, cursor: Optional[str] = None,
                source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Newest first across both kinds of alert. rule_id / watchlist_id select rule events only."""
    from db import alert_rules as store
    from db import alerts as legacy

    cur_key = decode_cursor(cursor) if cursor else None
    before = triggered_before
    if cur_key and (before is None or cur_key[0] < before):
        before = cur_key[0]
    after = triggered_after
    rows: List[Tuple[Tuple[dt.datetime, int, int], Dict[str, Any]]] = []
    fetch = limit + 1
    if source in (None, "alert") and rule_id is None and watchlist_id is None:
        for e in _db(legacy.list_events_filtered, user, fetch, ticker=ticker, after=after, before=before):
            ts = _parse_ts(e.get("fired_at"))
            if ts is None:
                continue
            rows.append(((ts, 0, int(e["id"])), {
                "id": int(e["id"]), "source": "alert", "alert_id": e.get("alert_id"), "ticker": e.get("ticker"),
                "message": str(e.get("message") or ""), "delivery": {"in_app": "delivered"}}))
    if source in (None, "rule"):
        iso = (lambda d: d.astimezone(dt.timezone.utc).isoformat(timespec="microseconds") if d else None)
        for e in _db(store.list_events, user, limit=fetch, ticker=ticker, rule_id=rule_id,
                     watchlist_id=watchlist_id, after=iso(after), before=iso(before)):
            ts = _parse_ts(e.get("triggered_at"))
            if ts is None:
                continue
            rows.append(((ts, 1, int(e["id"])), {
                "id": int(e["id"]), "source": "rule", "ticker": e.get("ticker"), "message": e.get("message") or "",
                **{k: e.get(k) for k in ("rule_id", "watchlist_id", "rule_type", "operator", "threshold",
                                         "trigger_value", "previous_value", "hsf_score", "setup",
                                         "market_data_as_of", "delivery")}}))
    if cur_key:
        ck = (cur_key[0], _SRC_RANK[cur_key[1]], cur_key[2])
        rows = [r for r in rows if r[0] < ck]
    rows.sort(key=lambda r: r[0], reverse=True)
    out = []
    for key, ev in rows[:limit]:
        when = key[0].isoformat()
        out.append({**ev, "event_id": f"{ev['source']}:{ev['id']}", "fired_at": when, "triggered_at": when,
                    "cursor": encode_cursor(key[0], ev["source"], ev["id"])})
    return out
