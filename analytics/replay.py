"""Run 43 — Historical Replay & Signal Timeline (pure, deterministic, no I/O).

Reconstructs how HSF's view of a symbol DEVELOPED over a session from the
canonical observation history (Run 36), reusing the Run 40 OpportunityView engine
for state/reasons/lifecycle. It extracts timeline events, collapses stable
periods, and keeps matured outcomes strictly SEPARATE from the point-in-time
historical state.

THE cardinal rule (Task 3): the reconstructed view at time T uses ONLY
observations at or before T. Future observations and any matured outcome must
never influence primary setup / scanner agreement / priority / lifecycle /
reasons / feature snapshot / market context. `state_at` and the leakage tests
enforce this.

Not a backtest, simulator, or predictive model. Alert Priority stays an attention
signal. No scanner/model/DT change.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence

from analytics import opportunity_view as ov

SCHEMA_VERSION = "hsf-replay-1.0"

# change-string (Run 40) -> (event_type, importance)
_CHANGE_EVENTS = [
    ("Moved above VWAP", ("VWAP_CROSS_UP", "NOTABLE")),
    ("Lost VWAP", ("VWAP_CROSS_DOWN", "NOTABLE")),
    ("New scanner", ("SCANNER_ADDED", "NOTABLE")),
    ("Direction:", ("DIRECTION_CHANGED", "MAJOR")),
    ("PreBreakout probability", ("PREBREAKOUT_CHANGE", "INFO")),
    ("RVOL:", ("RVOL_CHANGE", "NOTABLE")),
    ("Scanner agreement", ("SCANNER_AGREEMENT_CHANGE", "INFO")),
]
_IMPORTANCE_RANK = {"MAJOR": 3, "NOTABLE": 2, "INFO": 1}
_PRIORITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _parse(ts: Any) -> Optional[_dt.datetime]:
    try:
        d = ts if isinstance(ts, _dt.datetime) else _dt.datetime.fromisoformat(
            str(ts).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _quality_ok(obs: Dict[str, Any], *, min_completeness: float,
                allow_stale: bool, allow_fallback: bool) -> bool:
    dq = obs.get("data_quality") or {}
    if not allow_stale and dq.get("stale"):
        return False
    if not allow_fallback and dq.get("fallback_used"):
        return False
    fc = dq.get("feature_completeness")
    if fc is not None and fc < min_completeness:
        return False
    return True


def _source_of(obs: Dict[str, Any]) -> str:
    ctx = obs.get("market_context") or {}
    return str(ctx.get("source") or str(obs.get("context") or "").split(":")[0] or "unknown")


def _prepare_observations(
    observations: Sequence[Dict[str, Any]], *, symbol: str, date: Optional[str],
    source: Optional[str], min_completeness: float, allow_stale: bool,
    allow_fallback: bool,
) -> List[Dict[str, Any]]:
    sym = str(symbol or "").upper()
    prepared: List[Dict[str, Any]] = []
    for o in observations or []:
        try:
            if str(o.get("symbol") or "").upper() != sym:
                continue
            ts = _parse(o.get("scan_timestamp") or o.get("timestamp"))
            if ts is None:  # malformed timestamp → skip (Task 24)
                continue
            if date is not None and ts.date().isoformat() != date:
                continue
            if source is not None and _source_of(o) != source:
                continue
            if not _quality_ok(o, min_completeness=min_completeness,
                               allow_stale=allow_stale, allow_fallback=allow_fallback):
                continue
            prepared.append({"_obs": o, "_ts": ts})
        except Exception:
            continue  # one malformed observation never breaks the replay
    prepared.sort(key=lambda x: x["_ts"])
    return prepared


def _view_at_index(prepared: List[Dict[str, Any]], i: int) -> Dict[str, Any]:
    """OpportunityView for observation i, using ONLY the immediately prior
    observation (point-in-time: nothing after i)."""
    obs = prepared[i]["_obs"]
    prior = prepared[i - 1]["_obs"] if i > 0 else None
    v = ov.build_opportunity_view(obs, prior_obs=prior)
    v["timestamp"] = prepared[i]["_ts"].isoformat()
    v["_feature_snapshot"] = dict(obs.get("indicators") or {})
    v["market_regime"] = (obs.get("market_context") or {}).get("market_regime")
    v["no_active_setup"] = (v.get("scanner_count", 0) == 0)
    return v


def _events_between(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    ts = cur.get("timestamp")

    def add(etype, imp, title, details=""):
        events.append({"timestamp": ts, "event_type": etype, "importance": imp,
                       "title": title, "details": details})

    prev_setup = (not prev) or prev.get("no_active_setup")
    if prev_setup and not cur.get("no_active_setup"):
        add("SETUP_APPEARED", "MAJOR",
            f"NEW — {cur.get('primary_setup') or 'Setup'}")
    if prev and not prev.get("no_active_setup") and cur.get("no_active_setup"):
        add("SETUP_RESOLVED", "MAJOR", "Setup resolved")
    if prev and prev.get("primary_setup") and cur.get("primary_setup") \
            and prev["primary_setup"] != cur["primary_setup"] \
            and not cur.get("no_active_setup"):
        add("PRIMARY_SETUP_CHANGED", "NOTABLE",
            f"Primary setup: {prev['primary_setup']} → {cur['primary_setup']}")
    if prev:
        pr, cr = _PRIORITY_RANK.get(prev.get("alert_priority"), 0), _PRIORITY_RANK.get(cur.get("alert_priority"), 0)
        if cr > pr:
            add("PRIORITY_INCREASED", "MAJOR" if cur.get("alert_priority") == "HIGH" else "NOTABLE",
                f"Priority {prev.get('alert_priority')} → {cur.get('alert_priority')}")
        elif cr < pr:
            add("PRIORITY_DECREASED", "NOTABLE",
                f"Priority {prev.get('alert_priority')} → {cur.get('alert_priority')}")
        if prev.get("lifecycle_state") != cur.get("lifecycle_state") and \
                cur.get("lifecycle_state") in ("STRENGTHENING", "WEAKENING"):
            add(f"LIFECYCLE_{cur['lifecycle_state']}",
                "NOTABLE", cur["lifecycle_state"].title())
    for change in (cur.get("changes_since_prior") or []):
        for needle, (etype, imp) in _CHANGE_EVENTS:
            if needle.lower() in change.lower():
                add(etype, imp, change)
                break
    return events


def build_replay_session(
    symbol: str, date: Optional[str], observations: Sequence[Dict[str, Any]],
    *, outcomes: Optional[Dict[str, Dict[str, Any]]] = None,
    source: Optional[str] = "scheduled", min_completeness: float = 0.0,
    allow_stale: bool = False, allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Deterministic ReplaySession (Task 2/7). Outcomes are kept SEPARATE in
    `outcomes_by_timestamp` and never influence the views/events."""
    prepared = _prepare_observations(
        observations, symbol=symbol, date=date, source=source,
        min_completeness=min_completeness, allow_stale=allow_stale,
        allow_fallback=allow_fallback)
    views: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for i in range(len(prepared)):
        v = _view_at_index(prepared, i)
        events.extend(_events_between(views[-1] if views else None, v))
        views.append(v)

    # Outcomes are looked up by observation timestamp — read-only, separate.
    outcomes = outcomes or {}
    outcomes_by_ts = {v["timestamp"]: outcomes.get(v["timestamp"])
                      for v in views if outcomes.get(v["timestamp"])}

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": str(symbol or "").upper(),
        "date": date,
        "session": views[-1].get("session") if views else None,
        "observations": views,
        "events": events,
        "timeline": collapse_stable(views, events),
        "outcomes_by_timestamp": outcomes_by_ts,
        "outcomes_available": bool(outcomes_by_ts),
        "coverage": _coverage(prepared, views),
        "summary": _session_summary(views, events),
    }


def collapse_stable(views: Sequence[Dict[str, Any]],
                    events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse consecutive stable observations (no events) into a span (Task 6)."""
    event_ts = {e["timestamp"] for e in events}
    timeline: List[Dict[str, Any]] = []
    run_start: Optional[Dict[str, Any]] = None
    prev: Optional[Dict[str, Any]] = None
    for v in views:
        if v["timestamp"] in event_ts:
            if run_start is not None:
                timeline.append(_span(run_start, prev))
                run_start = None
            timeline.append({"kind": "event", "timestamp": v["timestamp"],
                             "state": v.get("lifecycle_state"),
                             "priority": v.get("alert_priority"),
                             "primary_setup": v.get("primary_setup"),
                             "no_active_setup": v.get("no_active_setup"),
                             "view": v})
        else:
            if run_start is None:
                run_start = v
        prev = v
    if run_start is not None:
        timeline.append(_span(run_start, prev))
    return timeline


def _span(start: Dict[str, Any], end: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "stable", "from": start["timestamp"], "to": end["timestamp"],
            "state": end.get("lifecycle_state"), "priority": end.get("alert_priority"),
            "primary_setup": end.get("primary_setup"),
            "no_active_setup": end.get("no_active_setup"),
            "note": "No meaningful setup change"}


def _coverage(prepared, views) -> Dict[str, Any]:
    n = len(views)
    complete = sum(1 for v in views
                   if (v.get("data_quality") or {}).get("feature_completeness", 0) >= 0.999)
    return {"observations": n,
            "complete": complete,
            "first": views[0]["timestamp"] if views else None,
            "last": views[-1]["timestamp"] if views else None}


def _session_summary(views, events) -> Dict[str, Any]:
    """Descriptive facts only (Task 18) — never 'best entry'/'profit'."""
    first_setup = next((v["timestamp"] for v in views if not v.get("no_active_setup")), None)
    resolved = next((e["timestamp"] for e in events if e["event_type"] == "SETUP_RESOLVED"), None)
    peak_agree = max((v.get("scanner_count") or 0 for v in views), default=0)
    priority_changes = sum(1 for e in events
                           if e["event_type"] in ("PRIORITY_INCREASED", "PRIORITY_DECREASED"))
    return {
        "observations": len(views),
        "first_setup": first_setup,
        "peak_scanner_agreement": peak_agree,
        "priority_changes": priority_changes,
        "setup_resolved": resolved,
        "major_events": sum(1 for e in events if e["importance"] == "MAJOR"),
    }


def state_at(session: Dict[str, Any], timestamp: Any) -> Optional[Dict[str, Any]]:
    """Strict point-in-time view: the reconstructed state as of `timestamp`,
    using ONLY observations at or before it (Task 3). Returns the last view whose
    timestamp <= the requested time — which was itself built with no future data.
    Never consults outcomes."""
    target = _parse(timestamp)
    if target is None:
        return None
    eligible = [v for v in session.get("observations", []) if _parse(v["timestamp"]) and _parse(v["timestamp"]) <= target]
    return eligible[-1] if eligible else None


def outcomes_at(session: Dict[str, Any], timestamp: Any) -> Dict[str, Any]:
    """Matured outcomes for an observation, clearly SEPARATE from its state (Task
    15/16). Pending horizons are reported as pending, never estimated."""
    v = state_at(session, timestamp)
    if not v:
        return {"available": False, "horizons": {}}
    oc = (session.get("outcomes_by_timestamp") or {}).get(v["timestamp"])
    horizons = {}
    for h in ("+5m", "+15m", "+30m", "+60m"):
        rec = (oc or {}).get(h) if isinstance(oc, dict) else None
        horizons[h] = ({"raw_return": rec.get("raw_return"), "status": rec.get("data_status", "MATURED")}
                       if rec else {"status": "PENDING"})
    return {"available": bool(oc), "at": v["timestamp"],
            "mfe": (oc or {}).get("mfe") if isinstance(oc, dict) else None,
            "mae": (oc or {}).get("mae") if isinstance(oc, dict) else None,
            "horizons": horizons}


def available_dates(observations: Sequence[Dict[str, Any]], symbol: str,
                    *, source: Optional[str] = "scheduled") -> List[str]:
    """Distinct dates with replayable observations for a symbol (Task 9/23)."""
    sym = str(symbol or "").upper()
    dates = set()
    for o in observations or []:
        if str(o.get("symbol") or "").upper() != sym:
            continue
        if source is not None and _source_of(o) != source:
            continue
        ts = _parse(o.get("scan_timestamp") or o.get("timestamp"))
        if ts is not None:
            dates.add(ts.date().isoformat())
    return sorted(dates)
