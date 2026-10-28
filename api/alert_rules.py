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
    EARNINGS_SOON                                 heads-up: the ticker reports earnings within
                                                  threshold days (once per earnings date; Pro).
                                                  Read from earnings_calendar, so it works for
                                                  tickers that aren't in the scan
    INSIDER_BUY                                   heads-up: a company insider reported an open-market
                                                  buy (SEC Form 4) worth at least threshold dollars
                                                  (once per newest buy date; Pro). Read from EDGAR
                                                  (api.filings), so no scan row is needed
    RANK_IMPROVED                                 transition: HSF rank moved up >= threshold places
                                                  since the last scan this rule saw
    SCAN_MATCH                                    scan alert (Pro): saved Scanner filters on the whole
                                                  market (HSF Score >= threshold, plus optional setup,
                                                  price range and relative volume in `value` as JSON).
                                                  Fires once per scan that brings new matching
                                                  stocks, listing them in one alert

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
    "EARNINGS_SOON": {"operator": "within_days", "kind": EDGE, "threshold": (0.0, 14.0), "default": 1.0,
                      "label": "Reports earnings within (days)", "entitlement": "can_earnings", "plan": "Pro",
                      "description": "Fires once per earnings date when the ticker reports within your number of "
                                     "days (1 = the day before). Uses the HSF earnings calendar."},
    "INSIDER_BUY": {"operator": "insider_buy", "kind": EDGE, "threshold": (0.0, 1_000_000_000.0), "default": 0.0,
                    "label": "Insider buys at least ($)", "entitlement": "can_insider_alerts", "plan": "Pro",
                    "description": "Fires when a company insider (officer, director or 10% owner) reports buying "
                                   "shares on the open market in an SEC Form 4 worth at least your amount "
                                   "(0 = any buy). Checked after each market scan."},
    "RANK_IMPROVED": {"operator": "improves_by", "kind": EDGE, "threshold": (1.0, 10000.0), "default": 10.0,
                      "label": "HSF rank improves by",
                      "description": "Fires when the ticker moves up at least your number of places in the HSF "
                                     "ranking since the last scan."},
    "SCAN_MATCH": {"operator": "new_matches", "kind": EDGE, "threshold": (0.0, 100.0), "default": 75.0,
                   "value": True, "scope": "market", "entitlement": "can_scan_alerts", "plan": "Pro",
                   "cooldown": 0, "label": "New scanner matches",
                   "description": "Saves your Scanner filters and checks them after every market scan. Fires when "
                                  "stocks that weren't matching on the previous scan now match, listing them in "
                                  "one alert."},
}
MARKET = "*"  # state/target key for market-wide rules (SCAN_MATCH)
SCAN_SIGNALS = ("breakout", "gapper", "golden_cross", "gainer", "prebreakout")
SCAN_FILTER_KEYS = ("name", "setup", "min_price", "max_price", "min_rvol", "earnings")
SCAN_EARNINGS = ("today", "tomorrow", "week")  # only names that report then (ET days; week = next 7)
EARNINGS_CACHE_S = 300
INSIDER_MAX_TICKERS = 150  # EDGAR lookups per pass (each cached for hours)
_earnings_cache: Dict[Tuple[str, str], Tuple[float, frozenset]] = {}
SCAN_MATCH_LIST_MAX = 1000  # matches remembered per rule between scans
SCAN_MATCH_NAMES_IN_MESSAGE = 8
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
                    "scope": spec.get("scope", "symbol"),
                    "default_cooldown_seconds": _default_cooldown(spec),
                    "available": _allowed(spec, entitlements)})
    return out


def _default_cooldown(spec: Dict[str, Any]) -> int:
    return int(spec["cooldown"]) if "cooldown" in spec else DEFAULT_COOLDOWN[spec["kind"]]


def _market(spec_or_rule: Dict[str, Any]) -> bool:
    spec = RULE_TYPES.get(str(spec_or_rule.get("rule_type") or "")) if "rule_type" in spec_or_rule else spec_or_rule
    return bool(spec and spec.get("scope") == "market")


def parse_scan_filters(value: Any) -> Dict[str, Any]:
    """SCAN_MATCH filters from the rule's `value` (JSON object). Unknown keys are
    rejected; every key is optional. Raises RuleError on bad input."""
    if value is None or not str(value).strip():
        return {}
    try:
        raw = json.loads(str(value))
    except ValueError as e:
        raise RuleError("Scan filters must be a JSON object.") from e
    if not isinstance(raw, dict):
        raise RuleError("Scan filters must be a JSON object.")
    bad = [k for k in raw if k not in SCAN_FILTER_KEYS]
    if bad:
        raise RuleError(f"Unknown scan filter: {', '.join(sorted(bad))}. Use: {', '.join(SCAN_FILTER_KEYS)}.")
    out: Dict[str, Any] = {}
    name = str(raw.get("name") or "").strip()[:60]
    if name:
        out["name"] = name
    setup = str(raw.get("setup") or "").strip().lower()
    if setup:
        if setup not in SCAN_SIGNALS:
            raise RuleError(f"Setup must be one of: {', '.join(SCAN_SIGNALS)}.")
        out["setup"] = setup
    earnings = str(raw.get("earnings") or "").strip().lower()
    if earnings:
        if earnings not in SCAN_EARNINGS:
            raise RuleError(f"earnings must be one of: {', '.join(SCAN_EARNINGS)}.")
        out["earnings"] = earnings
    for k, lo, hi in (("min_price", 0.0, 1e6), ("max_price", 0.0, 1e6), ("min_rvol", 0.0, 1000.0)):
        v = raw.get(k)
        if v is None or v == "":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError) as e:
            raise RuleError(f"{k} must be a number.") from e
        if not (lo <= f <= hi) or f != f:
            raise RuleError(f"{k} must be between {lo:g} and {hi:g}.")
        out[k] = f
    if "min_price" in out and "max_price" in out and out["min_price"] > out["max_price"]:
        raise RuleError("min_price must not exceed max_price.")
    return out


def _scan_value(filters: Dict[str, Any]) -> Optional[str]:
    return json.dumps(filters, sort_keys=True, separators=(",", ":")) if filters else None


def _reporting(window: str) -> frozenset:
    """Ticker keys (letters and digits only) that report earnings in `window`, cached for a
    few minutes so one alert pass reads the calendar once."""
    from api.scans import EARNINGS_WINDOWS, _et_date, _ticker_key
    from db.earnings import fetch_earnings_this_week

    lo, hi = EARNINGS_WINDOWS[window]
    start = _et_date() + dt.timedelta(days=lo)
    key = (start.isoformat(), window)
    hit = _earnings_cache.get(key)
    if hit and time.monotonic() - hit[0] < EARNINGS_CACHE_S:
        return hit[1]
    try:
        rows = fetch_earnings_this_week(days_ahead=hi - lo, start=start) or []
    except Exception as e:  # skip earnings rules this pass, never the whole pass
        raise RuleError("The earnings calendar is unavailable.") from e
    keys = frozenset(_ticker_key(r.get("symbol")) for r in rows if r.get("earnings_date") is not None)
    _earnings_cache.clear()
    _earnings_cache[key] = (time.monotonic(), keys)
    return keys


def scan_matches(obs: Dict[str, Any], rule: Dict[str, Any], *, premium: bool) -> List[str]:
    """Tickers in the latest scan's ranked setups that pass a SCAN_MATCH rule, best
    rank first. Uses the same canonical per-ticker view as every other rule."""
    from api import watchlist_intel

    f = parse_scan_filters(rule.get("value"))
    min_score = float(rule.get("threshold") or 0)
    hits: List[Tuple[int, str]] = []
    reporting = _reporting(f["earnings"]) if f.get("earnings") else None
    picks = None
    if f.get("setup") == "prebreakout":
        # Same names as the Scanner's PreBreakout tab: the model's strongest this scan.
        from ui.results_intelligence import prebreakout_picks

        ranked = obs.get("ranked") or {}
        top = {id(o) for o in prebreakout_picks(list(ranked.values()))} if premium else set()
        picks = {str(t).upper() for t, o in ranked.items() if id(o) in top}
    for t in obs.get("ranked") or {}:
        v = watchlist_intel.ticker_view(obs, t, premium=premium)
        score = v.get("hsf_score")
        if not v.get("ranked") or score is None or score < min_score:
            continue
        if picks is not None:
            if t.upper() not in picks:
                continue
        elif f.get("setup"):
            setup = str(v.get("setup") or "").lower()
            if f["setup"] != setup and f["setup"] not in [s.lower() for s in v.get("signals") or []]:
                continue
        p = v.get("price")
        if "min_price" in f and (p is None or p < f["min_price"]):
            continue
        if "max_price" in f and (p is None or p > f["max_price"]):
            continue
        if "min_rvol" in f and (v.get("rvol") is None or v["rvol"] < f["min_rvol"]):
            continue
        if reporting is not None:
            from api.scans import _ticker_key

            if _ticker_key(t) not in reporting:
                continue
        hits.append((int(v.get("rank") or 10**9), t.upper()))
    return [t for _, t in sorted(hits)]


def check_scan(matches: Sequence[str], state: Optional[Dict[str, Any]]
               ) -> Tuple[bool, List[str], Dict[str, Any]]:
    """(fires, new tickers best rank first, new state) for a SCAN_MATCH rule. The first
    scan a rule sees only records its baseline."""
    state = state or {}
    keep = list(matches)[:SCAN_MATCH_LIST_MAX]
    new_state = {"last_value": float(len(matches)), "last_flag": True, "last_text": json.dumps(keep)}
    if not state.get("observation_id"):
        return False, [], new_state
    try:
        before = set(json.loads(state.get("last_text") or "[]"))
    except ValueError:
        before = set()
    fresh = [t for t in matches if t not in before]
    return bool(fresh), fresh, new_state


def scan_message(rule: Dict[str, Any], fresh: Sequence[str]) -> str:
    f = parse_scan_filters(rule.get("value"))
    name = f.get("name") or "your scan alert"
    shown = ", ".join(fresh[:SCAN_MATCH_NAMES_IN_MESSAGE])
    more = f" and {len(fresh) - SCAN_MATCH_NAMES_IN_MESSAGE} more" if len(fresh) > SCAN_MATCH_NAMES_IN_MESSAGE else ""
    noun = "stock matches" if len(fresh) == 1 else "stocks match"
    return f"{len(fresh)} new {noun} {name}: {shown}{more}"


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
        plan = spec.get("plan", "Premium")
        raise RuleForbidden(f"{spec['label']} alerts are {'part of Pro' if plan == 'Pro' else 'a Premium feature'}.")
    out: Dict[str, Any] = {}
    market = _market(spec)
    if not partial and market:
        if ticker or watchlist_id is not None:
            raise RuleError(f"{spec['label']} alerts watch the whole market: leave ticker and watchlist_id empty.")
        out.update(ticker=None, watchlist_id=None, rule_type=str(rule_type).upper(), operator=spec["operator"])
    elif not partial:
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
        if market:
            filters = parse_scan_filters(value)
            if filters.get("setup") == "prebreakout" and not entitlements.get("can_early_breakout"):
                raise RuleForbidden("PreBreakout scan alerts are a Premium feature.")
            out["value"] = _scan_value(filters)
        else:
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
        out["cooldown_seconds"] = _default_cooldown(spec)
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
    if rule["rule_type"] == "EARNINGS_SOON":
        return _check_earnings(rule, view, state or {})
    if rule["rule_type"] == "INSIDER_BUY":
        return _check_insider(rule, view, state or {})
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


def _check_earnings(rule: Dict[str, Any], view: Dict[str, Any], state: Dict[str, Any]
                    ) -> Tuple[bool, Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    """EARNINGS_SOON: fires the first time a known earnings date is within the window.
    last_text holds the earnings date it already fired for, so it fires once per report
    (a moved date fires again). Needs no scan row: the date comes from the calendar."""
    days, date = view.get("earnings_days"), view.get("earnings_date")
    fired_for = state.get("last_text")
    if days is None or date is None:
        return False, None, None, {"last_value": None, "last_flag": False, "last_text": fired_for}
    inside = 0 <= float(days) <= float(rule.get("threshold") or 0)
    fires = inside and fired_for != date
    return fires, float(days), None, {"last_value": float(days), "last_flag": inside,
                                      "last_text": date if inside else fired_for}


def _check_insider(rule: Dict[str, Any], view: Dict[str, Any], state: Dict[str, Any]
                   ) -> Tuple[bool, Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    """INSIDER_BUY: fires once for the newest insider buy date at or above the rule's dollar
    amount. last_text holds the buy date it last fired for (or the baseline date the first
    pass saw, so a rule never fires for buys filed before it was created)."""
    date, value = view.get("insider_buy_date"), view.get("insider_buy_value")
    seen = state.get("last_text")
    if not view.get("insider_known"):
        return False, None, None, None  # EDGAR unreadable this pass: keep the stored state
    if date is None:
        return False, None, None, {"last_value": None, "last_flag": False, "last_text": seen or "-"}
    big_enough = float(value or 0) >= float(rule.get("threshold") or 0)
    fires = bool(seen) and big_enough and str(date) > str(seen)
    return fires, float(value or 0), None, {"last_value": float(value or 0), "last_flag": big_enough,
                                            "last_text": max(str(date), str(seen or ""))}


def insider_lookup(tickers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """{ticker: insider_known, insider_buy_date, insider_buy_value, insider_buy_who} from the
    newest open-market buy in recent Form 4s (api.filings, cached for hours). A ticker EDGAR
    couldn't answer for is left out, so its rules keep their state."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from api import filings

        if not filings.enabled():
            return {}
        for t in sorted({str(x).strip().upper() for x in tickers if str(x).strip()})[:INSIDER_MAX_TICKERS]:
            f = filings.stock_filings(t)
            if not f.get("available"):
                continue
            buys = [x for x in f.get("insider") or [] if x.get("type") == "Buy" and x.get("date")]
            if not buys:
                out[t] = {"insider_known": True, "insider_buy_date": None, "insider_buy_value": None}
                continue
            newest = max(x["date"] for x in buys)
            same_day = [x for x in buys if x["date"] == newest]
            out[t] = {"insider_known": True, "insider_buy_date": newest,
                      "insider_buy_value": round(sum(x.get("value") or 0 for x in same_day), 2),
                      "insider_buy_who": f"{same_day[0].get('insider')} ({same_day[0].get('role')})"}
    except Exception as e:
        log.warning(json.dumps({"event": "alert_rules_insider_failed", "error": type(e).__name__}))
    return out


def _earnings_when(view: Dict[str, Any]) -> str:
    days = int(view.get("earnings_days") or 0)
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days ({view.get('earnings_date')})"
    t = str(view.get("earnings_time") or "").strip().lower()
    timing = {"bmo": " before the open", "amc": " after the close"}.get(t, "")
    return f"{when}{timing}"


def earnings_lookup(tickers: Sequence[str], now: dt.datetime) -> Dict[str, Dict[str, Any]]:
    """{ticker: earnings_days, earnings_date, earnings_time} for upcoming reports, from
    the earnings calendar (one query). Days count from today in New York. Best effort:
    an unreadable calendar means no earnings alerts this pass, never a failed pass."""
    syms = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not syms:
        return {}
    try:
        from zoneinfo import ZoneInfo

        from db.earnings import load_earnings_details_map

        details = load_earnings_details_map(syms)
        today = now.astimezone(ZoneInfo("America/New_York")).date()
    except Exception as e:
        log.warning(json.dumps({"event": "alert_rules_earnings_failed", "error": type(e).__name__}))
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for t in syms:
        d, when = details.get(t) or details.get(t.replace(".", "-")) or (None, None)
        if isinstance(d, dt.datetime):
            d = d.date()
        if not isinstance(d, dt.date) or d < today:
            continue
        out[t] = {"earnings_days": (d - today).days, "earnings_date": d.isoformat(),
                  "earnings_time": (str(when).strip().lower() or None) if when else None}
    return out


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
    if rt == "EARNINGS_SOON":
        return f"{ticker} reports earnings {_earnings_when(view)}"
    if rt == "INSIDER_BUY":
        amount = f"${value:,.0f}" if value else "shares"
        return f"{ticker}: insider buy of {amount} reported {view.get('insider_buy_date')} by {view.get('insider_buy_who') or 'an insider'}"
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
    from api import plan_access
    from api.main import entitlements_for

    tier = plan_access.effective_tier(str(rule.get("user_id") or ""), rule.get("owner_tier"),
                                      bool(rule.get("owner_is_admin")))["tier"]
    ent = entitlements_for({"tier": tier, "is_admin": rule.get("owner_is_admin")})
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


def _evaluate_market_rule(r: Dict[str, Any], obs: Dict[str, Any], st: Optional[Dict[str, Any]], now_s: str,
                          now: dt.datetime) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """SCAN_MATCH: (new state, event or None); None when this rule already saw the scan
    or its saved filters can't be read (the rule is skipped, never the pass)."""
    from api import watchlist_intel

    obs_id = obs["observation_id"]
    if st and st.get("observation_id") == obs_id:
        return None
    try:
        matches = scan_matches(obs, r, premium=r["_premium"])
    except RuleError:
        return None
    fires, fresh, new = check_scan(matches, st)
    last_trig = (st or {}).get("last_triggered_at")
    if fires and _cooling(st, int(r["cooldown_seconds"]), now):
        fires = False
    state = {"rule_id": r["id"], "ticker": MARKET, **new, "observation_id": obs_id,
             "last_triggered_at": now_s if fires else last_trig}
    if not fires:
        return state, None
    top = watchlist_intel.ticker_view(obs, fresh[0], premium=r["_premium"])
    event = {
        "user_id": r["user_id"], "rule_id": r["id"], "watchlist_id": None,
        "ticker": fresh[0], "rule_type": r["rule_type"], "operator": r["operator"],
        "threshold": r.get("threshold"), "trigger_value": float(len(fresh)), "previous_value": None,
        "hsf_score": top.get("hsf_score"), "setup": top.get("setup"),
        "message": scan_message(r, fresh), "observation_id": obs_id,
        "market_data_as_of": watchlist_intel._iso(obs["scan_at"]), "triggered_at": now_s,
        "delivery": {c: "pending" if c != "in_app" else "delivered" for c in r["delivery_channels"]},
        "_channels": r["delivery_channels"], "_tier": r["_tier"],
        "_email_verified": r.get("owner_email_verified"), "_view": top,
    }
    return state, event


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
    earn_tickers = set()
    for r in rules:
        if r["rule_type"] == "EARNINGS_SOON":
            earn_tickers.update((wl.get(int(r["watchlist_id"])) or []) if r.get("watchlist_id") else [r["ticker"]])
    earnings = earnings_lookup(sorted(earn_tickers), now) if earn_tickers else {}
    insider_tickers = set()
    for r in rules:
        if r["rule_type"] == "INSIDER_BUY":
            insider_tickers.update((wl.get(int(r["watchlist_id"])) or []) if r.get("watchlist_id") else [r["ticker"]])
    for t, info in (insider_lookup(sorted(insider_tickers)) if insider_tickers else {}).items():
        earnings.setdefault(t, {}).update(info)
    now_s = now.isoformat(timespec="microseconds")
    views: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    new_states, events, evaluated, triggered = [], [], set(), set()
    for r in rules:
        if _market(r):
            ev = _evaluate_market_rule(r, obs, states.get((r["id"], MARKET)), now_s, now)
            if ev is None:
                continue
            evaluated.add(r["id"])
            result["evaluated"] += 1
            state, event = ev
            new_states.append(state)
            if event is not None:
                triggered.add(r["id"])
                events.append(event)
            continue
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
                views[key] = {**watchlist_intel.ticker_view(obs, t, premium=r["_premium"]),
                              **earnings.get(t.upper(), {})}
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
                    "_email_verified": r.get("owner_email_verified"), "_view": view,
                })
            new_states.append({"rule_id": r["id"], "ticker": t, **new, "observation_id": obs_id,
                               "last_triggered_at": last_trig})
    if deliver and events:
        try:  # Premium: one AI "Why:" line on the message (own daily cap; never blocks the pass)
            from api.ai_premium import annotate_alerts

            result["ai_why"] = annotate_alerts(events, obs)
        except Exception as e:
            log.warning(json.dumps({"event": "alert_ai_why_failed", "error": type(e).__name__}))
    inserted = store.save_evaluation(states=new_states, events=events, evaluated_rule_ids=sorted(evaluated),
                                     triggered_rule_ids=sorted(triggered), now=now_s)
    result["triggered"] = len(inserted)
    result["deduplicated"] += len(events) - len(inserted)  # another pass/process already recorded them
    if deliver:
        for ev in inserted:
            ok, failed = _deliver(ev)
            result["delivered"] += ok
            result["delivery_failed"] += failed
            _push(ev)
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


def _push(ev: Dict[str, Any]) -> None:
    """Browser notifications (api.webpush): every browser the owner turned them on in.
    Best effort and not retried; the event row and the other channels don't depend on it."""
    try:
        from api import webpush

        webpush.alert_fired(str(ev["user_id"]), str(ev["message"]))
    except Exception as e:
        log.warning(json.dumps({"event": "alert_rule_push_failed", "error": type(e).__name__}))


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
