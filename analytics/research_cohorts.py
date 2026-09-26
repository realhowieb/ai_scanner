"""Run 47 — tiered research cohorts (candidate / near-miss / control), pure.

Builds a bounded, deterministic, point-in-time-safe research dataset so HSF can
compare what it SELECTED (candidates) against what it REJECTED (near-misses =
close-but-below-cut, controls = deterministic sample of the broad evaluated
universe) — without persisting all ~11.6K securities per scan.

No scoring/scanner/ranking change: cohorts are labels over existing scan outputs.
Controls are selected by a seeded hash of (scan_run_id, symbol) — never by future
returns or outcomes. Everything here is point-in-time (features known at T only);
outcomes stay in the separate outcomes table.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from analytics.observation_capture import build_scan_observations

# Cohort labels (stable, unambiguous).
CANDIDATE, NEAR_MISS, CONTROL = "CANDIDATE", "NEAR_MISS", "CONTROL"
LEGACY_CANDIDATE = "CANDIDATE"  # legacy candidate observations map here

# Conservative defaults + hard caps so config can never persist the whole market.
DEFAULT_NEAR_MISS_N = 50
DEFAULT_CONTROL_N = 100
HARD_CAP = 500


def near_miss_n() -> int:
    return _bounded(os.getenv("RESEARCH_NEAR_MISS_N"), DEFAULT_NEAR_MISS_N)


def control_n() -> int:
    return _bounded(os.getenv("RESEARCH_CONTROL_N"), DEFAULT_CONTROL_N)


def _bounded(raw: Any, default: int) -> int:
    try:
        v = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        v = default
    return max(0, min(v, HARD_CAP))


def _seed(scan_run_id: str, symbol: str) -> int:
    """Deterministic, reproducible ranking key for control sampling. Stable across
    runs and machines; depends ONLY on point-in-time identity."""
    h = hashlib.sha256(f"{scan_run_id}|{str(symbol).upper()}".encode()).hexdigest()
    return int(h[:16], 16)


def select_control_symbols(
    evaluated_symbols: Sequence[str], *, scan_run_id: str,
    exclude: Optional[Sequence[str]] = None, n: Optional[int] = None,
) -> List[str]:
    """Deterministic seeded control sample of evaluated NON-candidate symbols
    (Task 4). Reproducible; never uses future information; bounded by HARD_CAP."""
    n = _bounded(n, DEFAULT_CONTROL_N) if n is not None else control_n()
    excl = {str(s).upper() for s in (exclude or [])}
    pool = sorted({str(s).upper() for s in (evaluated_symbols or [])} - excl)
    # Lowest-hash n symbols → a stable pseudo-random sample.
    pool.sort(key=lambda s: _seed(scan_run_id, s))
    return sorted(pool[:n])


def _tag(obs: Dict[str, Any], cohort: str, reason: str) -> Dict[str, Any]:
    obs["research_cohort"] = cohort
    obs["selection_reason"] = reason
    obs.setdefault("market_context", {})["research_cohort"] = cohort
    return obs


def tag_candidates(candidate_observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Label existing candidate observations (Task 1/2) — unchanged otherwise."""
    return [_tag(o, CANDIDATE, "top_n_candidate") for o in candidate_observations]


def build_near_miss_observations(
    near_miss_rows: Sequence[Dict[str, Any]], *, universe: str, scan_timestamp: Any,
    session: Optional[str] = None, scan_id: Optional[str] = None,
    coverage_health: Optional[str] = None, n: Optional[int] = None,
    research_run_context: Optional[Dict[str, Any]] = None, top_n: Optional[int] = None,
    price_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Full-feature NEAR_MISS observations from ranked rows just below the cut.
    Same builder/features as candidates → fair comparison (Task 3/7).

    Run 57: optional point-in-time `research_metadata` (rank = top_n + position
    below the cut), same block as candidates. Additive, non-fatal."""
    limit = _bounded(n, DEFAULT_NEAR_MISS_N) if n is not None else near_miss_n()
    rows = list(near_miss_rows or [])[:limit]
    obs = build_scan_observations(
        rows, universe=universe, scan_timestamp=scan_timestamp, session=session,
        scan_id=scan_id, coverage_health=coverage_health)
    obs = [_tag(o, NEAR_MISS, "rank_below_cutoff") for o in obs]
    if research_run_context:
        from analytics.research_metadata import attach
        attach(obs, research_run_context, rows=rows,
               rank_offset=int(top_n) if top_n is not None else None, price_meta=price_meta)
    return obs


def build_control_observations(
    control_symbols: Sequence[str], price_snapshot: Dict[str, Dict[str, Any]], *,
    universe: str, scan_timestamp: Any, session: Optional[str] = None,
    scan_id: Optional[str] = None, coverage_health: Optional[str] = None,
    research_run_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Compact CONTROL observations (identity + price/volume) for a deterministic
    sample of the broad evaluated universe (Task 4). These securities were
    filtered before scoring, so richer technicals are honestly absent.

    Run 57: optional point-in-time `research_metadata`: the same run-level
    provenance/regime/tier block as the scored cohorts plus the symbol's provider
    tag. Controls are unranked, so rank and row features stay None/empty."""
    from analytics.hsf_observation import build_observation
    price_snapshot = price_snapshot or {}
    out: List[Dict[str, Any]] = []
    for sym in control_symbols:
        snap = price_snapshot.get(str(sym).upper()) or {}
        o = build_observation(
            symbol=sym, timestamp=scan_timestamp, context=f"scheduled:{str(universe).lower()}",
            session=session, universe_version=universe,
            market={k: snap.get(k) for k in ("price", "volume") if snap.get(k) is not None},
            indicators={}, scanners=[],
            market_context={"source": "scheduled", "scan_id": scan_id,
                            "coverage_health": coverage_health},
            scan_timestamp=scan_timestamp, data_source="scheduled_control_sample")
        out.append(_tag(o, CONTROL, "deterministic_sample"))
    if research_run_context:
        from analytics.research_metadata import attach
        attach(out, research_run_context, rows=None, rank_offset=None, price_meta=price_snapshot)
    return out


def cohort_of(obs: Dict[str, Any]) -> str:
    """Read an observation's cohort, treating legacy (untagged candidate) rows as
    CANDIDATE (Task 21 backward compatibility)."""
    c = obs.get("research_cohort") or (obs.get("market_context") or {}).get("research_cohort")
    return str(c) if c else LEGACY_CANDIDATE


def cohort_balance(observations: Sequence[Dict[str, Any]],
                   *, evaluated: Optional[int] = None) -> Dict[str, Any]:
    """Per-scan cohort counts (Task 16). Makes a missing cohort obvious."""
    counts = Counter(cohort_of(o) for o in observations)
    return {
        "evaluated": evaluated,
        "candidate": counts.get(CANDIDATE, 0),
        "near_miss": counts.get(NEAR_MISS, 0),
        "control": counts.get(CONTROL, 0),
        "research_rows": len(observations),
    }
