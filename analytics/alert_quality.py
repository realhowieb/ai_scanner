"""Run 24 + 24A — HSF Intelligence Alert QUALITY measurement (pure + background).

The question this answers: *did an intelligence alert identify a meaningful
subsequent HSF state change, or was it noise?* Measurement only — it never
modifies HSF Score v1.0, movement thresholds, event definitions, preferences, or
delivery, and it never calls Claude.

Pipeline shape reuses the canonical pattern proven by signal_outcomes: frozen
signal-time record -> wait for a horizon -> find the FIRST valid comparable later
observation -> classify -> persist (idempotent, first-observation immutable).
Unlike signal_outcomes (which measures *price* return), quality is measured
purely from canonical HSF state — the frozen alert payload vs a subsequent
opportunity snapshot. Price is never used (Step 6): a FADING alert can be useful
even if price rises; an upgrade can fail to persist even if price rises.

Canonical reuse (24A): the meaningful score-change threshold and the
score-version rule are imported from ui.opportunities — there is exactly ONE
definition of each, never a private fork here.

Horizons (24A, Preferred B): the repo has no pure, deterministic trading-session
calendar (only a network-gated Alpaca lookup), so horizons are HONEST elapsed
wall-clock offsets from the source snapshot, named and labelled as elapsed time
(NEXT / H24 / H72 / H120) — never "1 day / 3 day", which would misrepresent
weekends and holidays.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from ui.opportunities import MIN_SCORE_DELTA, versions_incompatible

_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}

# Centralized minimum sample before a quality rate is presented as meaningful.
MIN_QUALITY_SAMPLE = 10

# --- Canonical horizon definitions (24A Step 3): ONE structure, consumed by
# maturation, persistence, aggregation, UI, and tests. No duplicated hardcoded
# horizon logic anywhere. offset_hours = elapsed wall-clock from the SOURCE
# snapshot; NEXT (offset 0) = the first comparable snapshot strictly after it.
QUALITY_HORIZONS: List[Dict[str, Any]] = [
    {"key": "NEXT", "offset_hours": 0, "order": 0, "label": "Next HSF observation"},
    {"key": "H24", "offset_hours": 24, "order": 1, "label": "24h+ follow-through"},
    {"key": "H72", "offset_hours": 72, "order": 2, "label": "72h+ follow-through"},
    {"key": "H120", "offset_hours": 120, "order": 3, "label": "120h+ follow-through"},
]
_HORIZON_OFFSETS = {h["key"]: h["offset_hours"] for h in QUALITY_HORIZONS}


def get_quality_horizons() -> List[Dict[str, Any]]:
    """The canonical horizon list (copy), ordered. Single source of truth."""
    return [dict(h) for h in sorted(QUALITY_HORIZONS, key=lambda h: h["order"])]


# Normalized quality classifications (Step 5).
CONFIRMED = "CONFIRMED"      # directional / event thesis held
PERSISTED = "PERSISTED"      # non-directional existence held (NEW_OPPORTUNITY)
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


def _to_dt(v: Any) -> Optional[_dt.datetime]:
    """Coerce a datetime or ISO string to a datetime; None when impossible."""
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, str) and v and v != "None":
        try:
            return _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def is_valid_quality_snapshot(snap: Dict[str, Any]) -> bool:
    """Lightweight snapshot validity gate (24A Step 20). A snapshot is usable as
    a subsequent observation only when it has a real timestamp and a structurally
    valid opportunities list. A VALID EMPTY list is allowed — zero opportunities
    still carries meaningful absence information. Rejects missing/invalid
    timestamps and non-list / malformed payloads (so ticker absence from a broken
    snapshot is never mistaken for a meaningful DROPPED/FADING/FALLING outcome)."""
    if not isinstance(snap, dict):
        return False
    if _to_dt(snap.get("snapshot_time")) is None:
        return False
    return isinstance(snap.get("opportunities"), list)


def select_first_valid_observation(
    snapshots: List[Dict[str, Any]],
    *,
    source_time: Optional[_dt.datetime],
    cutoff: _dt.datetime,
) -> Optional[Dict[str, Any]]:
    """Deterministic, leakage-safe observation selection (24A Step 5).

    Returns the EARLIEST snapshot that is (a) structurally valid, (b) strictly
    after the source snapshot (never the alert's own market state), and (c) at or
    after the horizon cutoff. Robust to out-of-order input, duplicate/ malformed
    timestamps, and duplicate rows. None when nothing qualifies.
    """
    qualifying = []
    for s in (snapshots or []):
        if not is_valid_quality_snapshot(s):
            continue
        st = _to_dt(s.get("snapshot_time"))
        if source_time is not None and not (st > source_time):
            continue  # strictly after the source snapshot — no same-state reuse
        if st < cutoff:
            continue  # horizon not yet reached by this observation
        qualifying.append((st, s))
    if not qualifying:
        return None
    # Earliest qualifying observation wins (stable, deterministic).
    return min(qualifying, key=lambda pair: pair[0])[1]


def classify_outcome(
    event_type: str,
    alert: Dict[str, Any],
    subsequent: Dict[str, Any],
) -> str:
    """Deterministic, event-aware quality classification (Step 5).

    `alert`      = {status, score, previous_status, score_version}
    `subsequent` = {status, score, present (bool), fading, score_version}

    Measures HSF state follow-through only — never price, never independence of
    confirming signals. Uses the CANONICAL score threshold (MIN_SCORE_DELTA) and
    the CANONICAL version rule. Absence (present=False) means only "not in that
    HSF ranking" — callers gate it behind snapshot validity before reaching here.
    """
    if versions_incompatible(alert.get("score_version"), subsequent.get("score_version")):
        return VERSION_CHANGED

    present = bool(subsequent.get("present"))
    a_rank = _rank(alert.get("status"))
    s_rank = _rank(subsequent.get("status"))
    prev_rank = _rank(alert.get("previous_status"))
    a_score = _num(alert.get("score"))
    s_score = _num(subsequent.get("score"))
    delta = (s_score - a_score) if (a_score is not None and s_score is not None) else None
    strengthened = delta is not None and delta >= MIN_SCORE_DELTA
    reversed_down = delta is not None and delta <= -MIN_SCORE_DELTA
    follows_through = delta is None or delta > -MIN_SCORE_DELTA  # not reversed below threshold

    et = str(event_type or "").upper()

    if et == "STATUS_UPGRADE":
        if not present:
            return REVERSED  # fell out of the ranked set entirely
        return CONFIRMED if s_rank >= a_rank else REVERSED

    if et == "STATUS_DOWNGRADE":
        # Recovered if it returns to or exceeds the pre-downgrade state.
        if prev_rank and s_rank >= prev_rank:
            return RECOVERED
        # Absent (rank 0) or weaker/equal to the downgraded state = thesis held.
        if s_rank <= a_rank:
            return CONFIRMED
        return NEUTRAL

    if et == "FADING":
        if (not present) or reversed_down or s_rank < a_rank or bool(subsequent.get("fading")):
            return CONFIRMED  # continued weakening / downgrade / drop / still fading
        if strengthened or s_rank > a_rank:
            return RECOVERED  # meaningful strengthening only (canonical threshold)
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
        return CONFIRMED  # stayed weak / fell further / dropped from a valid set

    if et == "DROPPED":
        # DROPPED = left the HSF ranking (not a price claim). Confirmed while it
        # remains absent or re-enters only weakly; recovered on meaningful
        # re-entry (>= WATCH). Absence only reaches here from a VALID snapshot.
        if not present:
            return CONFIRMED
        return RECOVERED if s_rank >= _STATUS_RANK["WATCH"] else CONFIRMED

    if et == "NEW_OPPORTUNITY":
        if not present:
            return REVERSED
        if s_rank >= a_rank or follows_through:
            return PERSISTED  # existence/presence held (non-directional)
        return NEUTRAL

    if et == "SIGNAL_ADDED":
        if not present:
            return REVERSED
        if s_rank >= a_rank or follows_through:
            return CONFIRMED  # HSF state persisted/strengthened (confirming signals)
        return NEUTRAL

    if et == "SIGNAL_REMOVED":
        if (not present) or reversed_down or s_rank < a_rank:
            return DETERIORATED
        return NEUTRAL  # stable state after a confirming-signal removal

    return NEUTRAL  # unknown types are not quality-classified


def _empty_result(base: Dict[str, Any], data_status: str, classification: str) -> Dict[str, Any]:
    return {**base, "data_status": data_status, "quality_classification": classification,
            "evaluation_time": None, "subsequent_score": None, "subsequent_status": None,
            "score_delta": None, "still_present": None, "fading": None}


def evaluate_alert_at_horizon(
    alert: Dict[str, Any],
    horizon: str,
    candidate_snapshots: List[Dict[str, Any]],
    *,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Pure horizon evaluation (no DB). Deterministic and leakage-safe.

    `alert` = {id, ticker, event_type, alert_time, payload(note)}. The payload's
    `current_snapshot_time` is the SOURCE market state — the observation must be
    strictly after it (never the same state). Horizon offsets are elapsed hours
    from the source snapshot; `now` gates PENDING. Returns a result carrying
    data_status (PENDING/UNAVAILABLE/MATURED) + classification + fields to persist.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    offset_h = _HORIZON_OFFSETS.get(horizon, 0)
    note = alert.get("payload") or {}
    ticker = alert.get("ticker") or note.get("ticker")
    event_type = alert.get("event_type") or note.get("event_type")

    alert_time = _to_dt(alert.get("alert_time"))
    # Reference = the SOURCE snapshot that produced the alert (fallback: the
    # alert record time, which is >= the source snapshot, hence safe/stricter).
    source_time = _to_dt(note.get("current_snapshot_time")) or alert_time

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
        "alert_time": alert.get("alert_time"),
        "evaluation_horizon": horizon,
        "alert_score": note.get("current_score"),
        "alert_status": note.get("current_status"),
        "score_version": note.get("score_version"),
    }

    if source_time is None:
        return _empty_result(base, UNAVAILABLE, UNAVAILABLE)

    cutoff = source_time + _dt.timedelta(hours=offset_h)
    if now < cutoff:  # not enough wall-clock time has elapsed yet
        return _empty_result(base, PENDING, PENDING)

    chosen = select_first_valid_observation(
        candidate_snapshots, source_time=source_time, cutoff=cutoff)
    if chosen is None:
        # Elapsed but no valid later observation yet. Transient (a later snapshot
        # may arrive); not a negative outcome and not persisted.
        return _empty_result(base, UNAVAILABLE, UNAVAILABLE)

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
        "evaluation_time": _to_dt(chosen.get("snapshot_time")),
        "subsequent_score": subseq.get("score"),
        "subsequent_status": subseq.get("status"),
        "score_delta": score_delta,
        "still_present": subseq.get("present"),
        "fading": subseq.get("fading"),
    }


def _find_subsequent_opp(opps: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
    """Locate the ticker in the chosen subsequent snapshot. Returns {present,
    status, score, score_version, fading}. present=False means only "not ranked
    in this HSF opportunity set" — never a price/tradability claim."""
    t = str(ticker or "").upper()
    for o in (opps or []):
        if isinstance(o, dict) and str(o.get("ticker") or "").upper() == t:
            return {
                "present": True,
                "status": o.get("status"),
                "score": o.get("score"),
                "score_version": o.get("score_version"),
                "fading": bool(o.get("fading")),
            }
    return {"present": False, "status": None, "score": None,
            "score_version": None, "fading": False}


def mature_alert_outcomes(*, lookback_days: int = 30, limit: int = 2000,
                          now: Optional[_dt.datetime] = None) -> int:
    """Background maturation (cron-owned). For each persisted alert and each
    horizon whose time has elapsed and that has no terminal outcome yet, find the
    first valid comparable subsequent snapshot, classify, and persist once
    (idempotent per alert_id + horizon; first-observation immutable via ON
    CONFLICT DO NOTHING). Only terminal (MATURED / VERSION_CHANGED classification)
    results are persisted; PENDING and UNAVAILABLE are transient and retried.
    Returns the number of outcomes written. Never raises."""
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
    try:
        snapshots = load_recent_snapshots(context="market_brief", limit=400)
    except Exception:
        snapshots = []
    now = now or _dt.datetime.now(_dt.timezone.utc)
    written = 0
    for alert in alerts:
        if str(alert.get("event_type") or "").upper() == "VERSION_CHANGED":
            continue  # internal record — never a user alert
        for horizon in _HORIZON_OFFSETS:
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
