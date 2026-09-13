"""Run 25 — HSF Opportunity Outcome Intelligence (pure + background).

The broader question (vs Run 24's alert quality): *what happens to EVERY frozen
HSF opportunity after it is identified* — not only the ones that produced alerts.
Each frozen opportunity observation becomes a measurable historical datum; its
subsequent canonical HSF state is classified at the Run 24A horizons.

This is HSF STATE outcome intelligence, never trading performance. It never
modifies HSF Score v1.0, components, ranking, thresholds, movement, version
semantics, or alerts; it never uses price; it never calls Claude.

Canonical reuse (no parallel architectures):
  * Horizons + validity + leakage-safe selection  -> analytics.alert_quality
  * Score threshold + version rule + status rank    -> ui.opportunities
  * Frozen signal-time observations                 -> db.signal_outcomes
    (source='opportunity'), grouped by snapshot_time into snapshot-shaped sets.
  * Subsequent observations come from the SAME frozen store, so subsequent
    signals/fading are available (the market_brief snapshot rows are minimal).

Kept separate (Step 29/30): Run 24 alert-quality tables, and the price/return
columns of signal_outcomes. Those are different questions and stay different.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from analytics.alert_quality import (  # canonical reuse
    MATURED,
    PENDING,
    UNAVAILABLE,
    _to_dt,
    get_quality_horizons,
    is_valid_quality_snapshot,
    select_first_valid_observation,
)
from ui.opportunities import (  # canonical reuse
    HSF_SCORE_VERSION,
    MIN_SCORE_DELTA,
    versions_incompatible,
)

_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}

# Canonical opportunity-evolution vocabulary (Step 7).
STRENGTHENED = "STRENGTHENED"
PERSISTED = "PERSISTED"
WEAKENED = "WEAKENED"
FADED = "FADED"
DROPPED = "DROPPED"
RECOVERED = "RECOVERED"
VERSION_CHANGED = "VERSION_CHANGED"
NEUTRAL = "NEUTRAL"

# Explicit precedence (Step 15) — highest wins, so the same inputs always map to
# one class: VERSION_CHANGED > DROPPED > RECOVERED > FADED > STRENGTHENED >
# WEAKENED > PERSISTED > NEUTRAL.


def _rank(status: Optional[str]) -> int:
    return _STATUS_RANK.get(str(status or "").upper(), 0)


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def is_eligible_opportunity_observation(obs: Dict[str, Any]) -> bool:
    """Canonical eligibility (Step 3). An observation is measurable only when it
    is structurally valid, has a real timestamp + ticker + HSF score + status, and
    uses a KNOWN score version (the current one, or a missing version which is
    canonically v1.0). Legacy rows with an unknown present version are skipped
    honestly rather than reconstructed."""
    if not isinstance(obs, dict):
        return False
    if _to_dt(obs.get("snapshot_time")) is None:
        return False
    if not str(obs.get("ticker") or "").strip():
        return False
    if _num(obs.get("score")) is None or not obs.get("status"):
        return False
    ver = obs.get("score_version")
    if ver is not None and str(ver) != str(HSF_SCORE_VERSION):
        return False  # unknown/legacy version — ineligible, never reconstructed
    return True


def classify_opportunity_outcome(
    initial: Dict[str, Any], subsequent: Dict[str, Any],
) -> str:
    """Deterministic opportunity-evolution classification with explicit
    precedence (Step 15). Reuses the canonical score threshold, version rule, and
    fading flag — never a new definition.

    `initial`    = {status, score, score_version, fading}
    `subsequent` = {present, status, score, score_version, fading}
    """
    if versions_incompatible(initial.get("score_version"), subsequent.get("score_version")):
        return VERSION_CHANGED
    if not bool(subsequent.get("present")):
        return DROPPED  # absent from a VALID later ranking (callers gate validity)

    i_rank, s_rank = _rank(initial.get("status")), _rank(subsequent.get("status"))
    i_score, s_score = _num(initial.get("score")), _num(subsequent.get("score"))
    delta = (s_score - i_score) if (i_score is not None and s_score is not None) else None
    up = (s_rank > i_rank) or (delta is not None and delta >= MIN_SCORE_DELTA)
    down = (s_rank < i_rank) or (delta is not None and delta <= -MIN_SCORE_DELTA)

    initial_weak = bool(initial.get("fading")) or i_rank <= _STATUS_RANK["CAUTION"]
    # RECOVERED only from a weak/fading source that meaningfully improves and is
    # no longer fading (a healthy source that improves is STRENGTHENED instead).
    if initial_weak and up and not bool(subsequent.get("fading")):
        return RECOVERED
    if bool(subsequent.get("fading")):
        return FADED  # canonical fading explicitly true takes precedence
    if up:
        return STRENGTHENED
    if down:
        return WEAKENED
    return PERSISTED  # present and within canonical movement tolerance


def _reconstruct_snapshots(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group frozen signal-time observations by snapshot_time into snapshot-shaped
    sets {snapshot_time, opportunities:[...]}. This is what the Run 24A validity +
    selection helpers consume. Deterministic; malformed rows are dropped."""
    by_time: Dict[Any, Dict[str, Any]] = {}
    for o in (observations or []):
        st = _to_dt(o.get("snapshot_time"))
        if st is None or not str(o.get("ticker") or "").strip():
            continue
        bucket = by_time.setdefault(st, {"snapshot_time": st, "opportunities": []})
        bucket["opportunities"].append({
            "ticker": str(o.get("ticker")).upper(),
            "score": o.get("score"), "status": o.get("status"),
            "score_version": o.get("score_version"),
            "signals": list(o.get("signals") or []), "fading": bool(o.get("fading")),
        })
    return list(by_time.values())


def _find(opps: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
    t = str(ticker or "").upper()
    for o in (opps or []):
        if isinstance(o, dict) and str(o.get("ticker") or "").upper() == t:
            return {"present": True, "status": o.get("status"), "score": o.get("score"),
                    "score_version": o.get("score_version"), "fading": bool(o.get("fading"))}
    return {"present": False, "status": None, "score": None, "score_version": None, "fading": False}


def _base(obs: Dict[str, Any], horizon: str) -> Dict[str, Any]:
    return {
        "opportunity_observation_id": obs.get("observation_id"),
        "ticker": str(obs.get("ticker") or "").upper(),
        "source_snapshot_time": obs.get("snapshot_time"),
        "evaluation_horizon": horizon,
        "initial_score": obs.get("score"), "initial_status": obs.get("status"),
        "initial_score_version": obs.get("score_version"),
        "initial_fading": bool(obs.get("fading")),
        "initial_signal_count": int(obs.get("n_signals") or len(obs.get("signals") or []) or 0),
        "initial_regime": obs.get("regime"),  # None unless frozen with the obs
    }


def _empty(base: Dict[str, Any], data_status: str) -> Dict[str, Any]:
    return {**base, "data_status": data_status, "outcome_classification": data_status,
            "evaluation_time": None, "subsequent_score": None, "subsequent_status": None,
            "subsequent_score_version": None, "subsequent_fading": None, "still_present": None,
            "score_delta": None, "status_transition": None}


def evaluate_observation_at_horizon(
    obs: Dict[str, Any], horizon: str, snapshots: List[Dict[str, Any]],
    *, now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Pure, deterministic, leakage-safe horizon evaluation for one observation.
    Reuses select_first_valid_observation (strictly-after-source + cutoff, earliest
    valid wins). Returns a result with data_status + outcome_classification."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    offsets = {h["key"]: h["offset_hours"] for h in get_quality_horizons()}
    offset_h = offsets.get(horizon, 0)
    source_time = _to_dt(obs.get("snapshot_time"))
    base = _base(obs, horizon)
    if source_time is None:
        return _empty(base, UNAVAILABLE)
    cutoff = source_time + _dt.timedelta(hours=offset_h)
    if now < cutoff:
        return _empty(base, PENDING)
    chosen = select_first_valid_observation(snapshots, source_time=source_time, cutoff=cutoff)
    if chosen is None:
        return _empty(base, UNAVAILABLE)  # transient; not persisted

    subseq = _find(chosen.get("opportunities") or [], base["ticker"])
    initial = {"status": obs.get("status"), "score": obs.get("score"),
               "score_version": obs.get("score_version"), "fading": bool(obs.get("fading"))}
    classification = classify_opportunity_outcome(initial, subseq)

    i_score, s_score = _num(obs.get("score")), _num(subseq.get("score"))
    score_delta = None
    if classification != VERSION_CHANGED and i_score is not None and s_score is not None:
        score_delta = int(round(s_score - i_score))
    transition = None
    if subseq.get("present") and subseq.get("status") and obs.get("status") \
            and str(subseq["status"]) != str(obs["status"]):
        transition = f"{obs['status']}→{subseq['status']}"

    return {
        **base, "data_status": MATURED, "outcome_classification": classification,
        "evaluation_time": _to_dt(chosen.get("snapshot_time")),
        "subsequent_score": subseq.get("score"), "subsequent_status": subseq.get("status"),
        "subsequent_score_version": subseq.get("score_version"),
        "subsequent_fading": subseq.get("fading"), "still_present": subseq.get("present"),
        "score_delta": score_delta, "status_transition": transition,
    }


def mature_opportunity_outcomes(
    *, lookback_days: int = 30, limit: int = 20000, now: Optional[_dt.datetime] = None,
) -> Dict[str, int]:
    """Background maturation (cron-owned). For each eligible frozen observation and
    each elapsed horizon without a persisted outcome, classify the first valid
    subsequent HSF observation and persist once (idempotent; first-observation
    immutable). Only terminal outcomes are persisted — PENDING/UNAVAILABLE retry.
    Never raises; returns restrained operational counts."""
    metrics = {"observations_checked": 0, "horizons_checked": 0, "matured": 0,
               "pending": 0, "unavailable": 0, "skipped": 0, "failed": 0}
    try:
        from db.opportunity_outcomes import (
            persist_opportunity_outcome,
            persisted_outcome_keys,
        )
        from db.signal_outcomes import fetch_opportunity_observations
    except Exception:
        return metrics
    try:
        observations = fetch_opportunity_observations(days_back=lookback_days, limit=limit)
    except Exception:
        return metrics
    if not observations:
        return metrics
    snapshots = _reconstruct_snapshots(observations)
    try:
        existing = persisted_outcome_keys(days_back=lookback_days)
    except Exception:
        existing = set()
    now = now or _dt.datetime.now(_dt.timezone.utc)
    horizons = [h["key"] for h in get_quality_horizons()]
    for obs in observations:
        if not is_eligible_opportunity_observation(obs):
            metrics["skipped"] += 1
            continue
        metrics["observations_checked"] += 1
        oid = obs.get("observation_id")
        for horizon in horizons:
            metrics["horizons_checked"] += 1
            if (oid, horizon) in existing:
                continue  # already matured — never recomputed (immutability)
            try:
                res = evaluate_observation_at_horizon(obs, horizon, snapshots, now=now)
            except Exception:
                metrics["failed"] += 1
                continue
            ds = res.get("data_status")
            if ds == PENDING:
                metrics["pending"] += 1
                continue
            if ds == UNAVAILABLE:
                metrics["unavailable"] += 1
                continue
            try:
                if persist_opportunity_outcome(res):
                    metrics["matured"] += 1
            except Exception:
                metrics["failed"] += 1
    return metrics


# Centralized score bands (Step 22). Aligned with the canonical status tiers
# (STRONG >= 75). Kept deliberately few; never tuned to flatter results.
SCORE_BANDS = [(80, 101, "80+"), (70, 80, "70-79"), (60, 70, "60-69"), (0, 60, "<60")]


def score_band(score: Any) -> Optional[str]:
    s = _num(score)
    if s is None:
        return None
    for lo, hi, label in SCORE_BANDS:
        if lo <= s < hi:
            return label
    return None


def signal_count_bucket(n: Any) -> str:
    try:
        c = int(n or 0)
    except (TypeError, ValueError):
        c = 0
    if c >= 3:
        return "3+"
    if c == 2:
        return "2"
    return "1"
