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
from ui.opportunities import HSF_SCORE_VERSION, MIN_SCORE_DELTA, versions_incompatible  # canonical reuse
from ui.opportunities import HSF_STATUS_RANK as _STATUS_RANK
from ui.opportunities import status_rank as _rank

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


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def normalized_signals(signals: Any) -> List[str]:
    """Confirming signals as a deterministic set (Step 14): lower-cased, trimmed,
    de-duplicated, sorted. Reuses no external definition — only normalizes form,
    never the signal meanings. Duplicates never inflate the count."""
    out = set()
    for s in (signals or []):
        t = str(s or "").strip().lower()
        if t:
            out.add(t)
    return sorted(out)


def _is_degraded(status: Any, fading: Any) -> bool:
    """Canonical degraded/weak HSF condition (recovery-eligible): explicit canonical
    fading, or status CAUTION (which canonically means fading OR score < 50 — a
    genuinely degraded tier, not merely lower-ranked)."""
    return bool(fading) or _rank(status) <= _STATUS_RANK["CAUTION"]


def _is_healthy(status: Any, fading: Any) -> bool:
    """Meaningfully healthy HSF state: WATCH or STRONG and not fading."""
    return (not bool(fading)) and _rank(status) >= _STATUS_RANK["WATCH"]


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

    # RECOVERED only when the SOURCE was in a genuinely degraded/fading canonical
    # state AND the later state returns to a meaningfully HEALTHIER tier (>= WATCH,
    # not fading) with real improvement. A healthy source that improves is
    # STRENGTHENED instead; a degraded source that merely ticks up but stays
    # degraded is STRENGTHENED, not RECOVERED.
    if _is_degraded(initial.get("status"), initial.get("fading")) and up \
            and _is_healthy(subsequent.get("status"), subsequent.get("fading")):
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
    sets {snapshot_time, opportunities:[...]}. Consumed by the Run 24A validity +
    selection helpers.

    All source rows come from ONE canonical context (signal_outcomes
    source='opportunity', i.e. market_brief), so timestamp grouping never merges
    different scans/contexts. A ticker is de-duplicated WITHIN a snapshot by the
    lowest observation_id (deterministic — never "first DB row returned"), so
    reversed input order yields an identical reconstruction."""
    by_time: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for o in (observations or []):
        st = _to_dt(o.get("snapshot_time"))
        t = str(o.get("ticker") or "").strip().upper()
        if st is None or not t:
            continue
        row = {
            "ticker": t, "score": o.get("score"), "status": o.get("status"),
            "score_version": o.get("score_version"),
            "signals": list(o.get("signals") or []), "fading": bool(o.get("fading")),
            "_oid": o.get("observation_id"),
        }
        bucket = by_time.setdefault(st, {})
        prev = bucket.get(t)
        if prev is None or _oid_key(row) < _oid_key(prev):
            bucket[t] = row  # deterministic: lowest observation_id wins
    return [{"snapshot_time": st, "opportunities": sorted(rows.values(), key=lambda r: r["ticker"])}
            for st, rows in by_time.items()]


def _oid_key(row: Dict[str, Any]):
    oid = row.get("_oid")
    return (0, int(oid)) if isinstance(oid, int) else (1, str(oid))


def _snapshot_version(snapshot: Dict[str, Any]) -> Any:
    """Canonical score version of a reconstructed snapshot = the version shared by
    its present opportunities (the comparison universe). Used to decide, for an
    ABSENT ticker, whether the ranking universe itself changed incompatibly —
    never fabricated from current data. None when unknown/mixed."""
    vers = {str(o.get("score_version")) for o in (snapshot.get("opportunities") or [])
            if o.get("score_version") is not None}
    return next(iter(vers)) if len(vers) == 1 else None


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
        # Count from the FROZEN signal-time signals, de-duplicated (never from
        # current/future state). Falls back to frozen n_signals only if no list.
        "initial_signal_count": (len(normalized_signals(obs.get("signals")))
                                 if obs.get("signals") is not None
                                 else int(obs.get("n_signals") or 0)),
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
    if not subseq.get("present"):
        # For an absent ticker there is no per-ticker version; infer the ranking
        # universe's version so an incompatible-version universe classifies as
        # VERSION_CHANGED (precedence) rather than a misleading DROPPED (Step 7).
        subseq["score_version"] = _snapshot_version(chosen)
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


def _dedupe_source_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse to ONE logical observation per (ticker, snapshot_time,
    score_version), keeping the lowest observation_id deterministically. The
    frozen store already enforces uniqueness per (snapshot, ticker); this is a
    defensive guard so duplicate rows can never inflate opportunity counts or
    depend on DB row ordering. Historical rows are never deleted."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for o in (observations or []):
        st = _to_dt(o.get("snapshot_time"))
        t = str(o.get("ticker") or "").strip().upper()
        if st is None or not t:
            continue
        key = (t, st, str(o.get("score_version")))
        prev = best.get(key)
        if prev is None or _oid_key({"_oid": o.get("observation_id")}) < _oid_key({"_oid": prev.get("observation_id")}):
            best[key] = o
    return list(best.values())


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
    observations = _dedupe_source_observations(observations)
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


# Centralized score bands (Step 11). Chosen BEFORE looking at any results and
# aligned to the CANONICAL status thresholds so no band straddles a tier edge:
# STRONG >= 75, CAUTION < 50 (WATCH in between). Deliberately few; never tuned to
# flatter results. A malformed/out-of-range score returns None (never a band).
SCORE_BANDS = [(75, 101, "75+"), (60, 75, "60-74"), (50, 60, "50-59"), (0, 50, "<50")]


def score_band(score: Any) -> Optional[str]:
    s = _num(score)
    if s is None or s < 0 or s > 100:
        return None  # malformed scores never enter a misleading band
    for lo, hi, label in SCORE_BANDS:
        if lo <= s < hi:
            return label
    return None


def signal_count_bucket(n: Any) -> str:
    """Confirming-signal bucket. Zero is kept DISTINCT (a ranked opportunity
    normally carries >= 1 confirming signal; zero signals a malformed/incomplete
    observation and must not hide inside '1')."""
    try:
        c = int(n or 0)
    except (TypeError, ValueError):
        c = 0
    if c <= 0:
        return "0"
    if c >= 3:
        return "3+"
    return str(c)
