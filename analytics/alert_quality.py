"""Run 24 — HSF Intelligence Alert QUALITY measurement (pure + background).

The question this answers: *did an intelligence alert identify a meaningful
subsequent HSF state change, or was it noise?* It is measurement only — it never
modifies HSF Score v1.0, movement thresholds, event definitions, preferences, or
delivery, and it never calls Claude.

Pipeline shape reuses the canonical pattern already proven by signal_outcomes:
frozen signal-time record -> wait for a horizon -> find the first comparable
later observation -> classify -> persist (idempotent). But unlike
signal_outcomes (which measures *price* return), quality is measured purely from
canonical HSF state — the frozen alert payload vs a subsequent opportunity
snapshot, compared with ui.opportunities.compare_opportunities. Price is never
used here (a FADING alert can be useful even if price rises; an upgrade can fail
to persist even if price rises).

Stages:
  1. signal-time observation = the frozen `hsf_intelligence_alerts` row
     (payload already holds score/status/version/signals/snapshot_time).
  2. subsequent state        = first comparable 'market_brief' opportunity
     snapshot at or after (alert_time + horizon offset).
  3. classification          = deterministic, event-aware (Step 5).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

# Canonical movement threshold (NEVER redefined here — imported meaning).
_MIN_DELTA = 3
_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}

# Centralized minimum sample before a quality rate is presented as meaningful.
MIN_QUALITY_SAMPLE = 10

# Deterministic horizons. Snapshot cadence in this repo is intraday + ~4
# scheduled runs per weekday, so exact trading-day boundaries do not map cleanly
# onto the data. We therefore use calendar-hour offsets with a "first comparable
# observation AT OR AFTER the offset" rule (honest approximation, reported in the
# final notes). NEXT = the first snapshot strictly after the alert.
HORIZONS: List[tuple] = [
    ("NEXT", 0),
    ("D1", 24),
    ("D3", 72),
    ("D5", 120),
]
_HORIZON_OFFSETS = {name: hours for name, hours in HORIZONS}

# Normalized quality classifications (Step 5).
CONFIRMED = "CONFIRMED"
PERSISTED = "PERSISTED"
REVERSED = "REVERSED"
DETERIORATED = "DETERIORATED"
RECOVERED = "RECOVERED"
NEUTRAL = "NEUTRAL"
VERSION_CHANGED = "VERSION_CHANGED"
# Data statuses.
MATURED = "MATURED"
PENDING = "PENDING"
UNAVAILABLE = "UNAVAILABLE"


def _rank(status: Optional[str]) -> int:
    return _STATUS_RANK.get(str(status or "").upper(), 0)


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _versions_incompatible(alert_ver: Any, subseq_ver: Any) -> bool:
    """Same rule as compare_opportunities: a missing version is compatible (only
    v1.0 has ever existed); two present-and-different versions block comparison."""
    a, s = alert_ver, subseq_ver
    if a is None or s is None:
        return False
    return str(a) != str(s)


def classify_outcome(
    event_type: str,
    alert: Dict[str, Any],
    subsequent: Dict[str, Any],
) -> str:
    """Deterministic, event-aware quality classification (Step 5).

    `alert`      = {status, score, previous_status, score_version}
    `subsequent` = {status, score, present (bool), fading, score_version}

    Measures HSF state follow-through only — never price, never independence of
    confirming signals. Returns one of the normalized classifications.
    """
    if _versions_incompatible(alert.get("score_version"), subsequent.get("score_version")):
        return VERSION_CHANGED

    present = bool(subsequent.get("present"))
    a_rank = _rank(alert.get("status"))
    s_rank = _rank(subsequent.get("status"))
    prev_rank = _rank(alert.get("previous_status"))
    a_score = _num(alert.get("score"))
    s_score = _num(subsequent.get("score"))
    delta = (s_score - a_score) if (a_score is not None and s_score is not None) else None
    strengthened = delta is not None and delta >= _MIN_DELTA
    reversed_down = delta is not None and delta <= -_MIN_DELTA
    follows_through = delta is None or delta > -_MIN_DELTA  # not reversed below threshold

    et = str(event_type or "").upper()

    if et == "STATUS_UPGRADE":
        if not present:
            return REVERSED
        return CONFIRMED if s_rank >= a_rank else REVERSED

    if et == "STATUS_DOWNGRADE":
        # Recovered if it returns to or exceeds the pre-downgrade state.
        if prev_rank and s_rank >= prev_rank:
            return RECOVERED
        if s_rank <= a_rank:
            return CONFIRMED  # weaker state persisted or deteriorated further
        return NEUTRAL

    if et == "FADING":
        if (not present) or reversed_down or s_rank < a_rank or bool(subsequent.get("fading")):
            return CONFIRMED  # continued weakening / downgrade / drop / still fading
        if strengthened or s_rank > a_rank:
            return RECOVERED
        return NEUTRAL

    if et == "RISING":
        if not present:
            return REVERSED
        if reversed_down or s_rank < a_rank:
            return REVERSED
        return CONFIRMED if follows_through else NEUTRAL

    if et == "FALLING":
        if strengthened or s_rank > a_rank:
            return RECOVERED
        return CONFIRMED  # stayed weak / fell further / dropped

    if et == "DROPPED":
        # DROPPED = left the HSF ranking (not a price claim). Confirmed while it
        # remains absent or re-enters only weakly; recovered on meaningful re-entry.
        if not present:
            return CONFIRMED
        return RECOVERED if s_rank >= _STATUS_RANK["WATCH"] else CONFIRMED

    if et == "NEW_OPPORTUNITY":
        if not present:
            return REVERSED
        if s_rank >= a_rank or follows_through:
            return PERSISTED
        return NEUTRAL

    if et == "SIGNAL_ADDED":
        if not present:
            return REVERSED
        if s_rank >= a_rank or follows_through:
            return CONFIRMED
        return NEUTRAL

    if et == "SIGNAL_REMOVED":
        if (not present) or reversed_down or s_rank < a_rank:
            return DETERIORATED
        return NEUTRAL

    # Unknown / VERSION_CHANGED event types are not quality-classified.
    return NEUTRAL


def _find_subsequent_opp(
    snapshots: List[Dict[str, Any]], ticker: str,
) -> Dict[str, Any]:
    """Locate the ticker in the chosen subsequent snapshot. Returns {present,
    status, score, score_version, fading} (present=False when not ranked)."""
    t = str(ticker or "").upper()
    for o in (snapshots or []):
        if str(o.get("ticker") or "").upper() == t:
            return {
                "present": True,
                "status": o.get("status"),
                "score": o.get("score"),
                "score_version": o.get("score_version"),
                "fading": bool(o.get("fading")),
            }
    return {"present": False, "status": None, "score": None,
            "score_version": None, "fading": False}


def evaluate_alert_at_horizon(
    alert: Dict[str, Any],
    horizon: str,
    candidate_snapshots: List[Dict[str, Any]],
    *,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Pure horizon evaluation (no DB). Deterministic.

    `alert` = {id, ticker, event_type, alert_time (datetime), payload (note)}.
    `candidate_snapshots` = 'market_brief' snapshots strictly usable as the
    subsequent observation, newest-first or any order, each {snapshot_time,
    opportunities}. We pick the FIRST comparable observation AT OR AFTER the
    horizon cutoff.

    Returns a result dict carrying data_status (PENDING/UNAVAILABLE/MATURED),
    quality_classification, and the frozen+subsequent fields for persistence.
    Never uses data before the cutoff (no leakage); never mutates the alert.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    offset_h = _HORIZON_OFFSETS.get(horizon, 0)
    alert_time = alert.get("alert_time")
    note = alert.get("payload") or {}
    ticker = alert.get("ticker") or note.get("ticker")
    event_type = alert.get("event_type") or note.get("event_type")

    alert_state = {
        "status": note.get("current_status"),
        "score": note.get("current_score"),
        "previous_status": note.get("previous_status"),
        "score_version": note.get("score_version"),
    }
    base = {
        "alert_id": alert.get("id"),
        "ticker": (str(ticker).upper() if ticker else None),
        "event_type": event_type,
        "alert_time": alert_time,
        "evaluation_horizon": horizon,
        "alert_score": note.get("current_score"),
        "alert_status": note.get("current_status"),
        "score_version": note.get("score_version"),
    }

    if alert_time is None:
        return {**base, "data_status": UNAVAILABLE, "quality_classification": UNAVAILABLE,
                "evaluation_time": None, "subsequent_score": None, "subsequent_status": None,
                "score_delta": None, "still_present": None, "fading": None}

    cutoff = alert_time + _dt.timedelta(hours=offset_h)
    # Not enough wall-clock time has elapsed yet -> genuinely PENDING.
    if now < cutoff:
        return {**base, "data_status": PENDING, "quality_classification": PENDING,
                "evaluation_time": None, "subsequent_score": None, "subsequent_status": None,
                "score_delta": None, "still_present": None, "fading": None}

    # First comparable observation AT OR AFTER the cutoff (NEXT: strictly after
    # alert_time). Deterministic: earliest qualifying snapshot wins.
    qualifying = []
    for s in (candidate_snapshots or []):
        st = s.get("snapshot_time")
        if st is None:
            continue
        if offset_h == 0:
            ok = st > alert_time
        else:
            ok = st >= cutoff
        if ok:
            qualifying.append(s)
    if not qualifying:
        # Time elapsed but no later snapshot exists. Not a negative outcome; it
        # is transient (a later snapshot may appear) so it is not persisted.
        return {**base, "data_status": UNAVAILABLE, "quality_classification": UNAVAILABLE,
                "evaluation_time": None, "subsequent_score": None, "subsequent_status": None,
                "score_delta": None, "still_present": None, "fading": None}

    chosen = min(qualifying, key=lambda s: s.get("snapshot_time"))
    subseq = _find_subsequent_opp(chosen.get("opportunities") or [], base["ticker"])
    classification = classify_outcome(event_type, alert_state, subseq)

    a_score = _num(alert_state.get("score"))
    s_score = _num(subseq.get("score"))
    score_delta = None
    if classification != VERSION_CHANGED and a_score is not None and s_score is not None:
        score_delta = int(round(s_score - a_score))

    return {
        **base,
        "data_status": MATURED,
        "quality_classification": classification,
        "evaluation_time": chosen.get("snapshot_time"),
        "subsequent_score": subseq.get("score"),
        "subsequent_status": subseq.get("status"),
        "score_delta": score_delta,
        "still_present": subseq.get("present"),
        "fading": subseq.get("fading"),
    }


def mature_alert_outcomes(*, lookback_days: int = 30, limit: int = 2000) -> int:
    """Background maturation (cron-owned). For each persisted intelligence alert
    and each horizon whose time has elapsed and that has no terminal outcome yet,
    find the first comparable subsequent snapshot, classify, and persist once
    (idempotent per alert_id + horizon). Returns the number of outcomes written.

    Only terminal (MATURED / VERSION_CHANGED) results are persisted; PENDING and
    UNAVAILABLE are transient and retried on a later run. Never raises.
    """
    try:
        from db.intelligence_alerts import (
            fetch_alerts_for_maturation,
            persist_alert_outcome,
        )
        from db.opportunity_snapshots import load_recent_snapshots
    except Exception:
        return 0
    try:
        alerts = fetch_alerts_for_maturation(days_back=lookback_days, limit=limit)
    except Exception:
        return 0
    if not alerts:
        return 0
    # One read of the market_brief snapshot history powers every horizon lookup.
    try:
        snapshots = load_recent_snapshots(context="market_brief", limit=400)
    except Exception:
        snapshots = []
    now = _dt.datetime.now(_dt.timezone.utc)
    written = 0
    for alert in alerts:
        # Skip version-change internal records — never user alerts.
        if str(alert.get("event_type") or "").upper() == "VERSION_CHANGED":
            continue
        for horizon, _ in HORIZONS:
            try:
                res = evaluate_alert_at_horizon(alert, horizon, snapshots, now=now)
            except Exception:
                continue
            if res.get("data_status") in (PENDING, UNAVAILABLE):
                continue  # transient — re-evaluate on a future run
            try:
                if persist_alert_outcome(res):
                    written += 1
            except Exception:
                continue
    return written
