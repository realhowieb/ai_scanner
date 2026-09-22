"""Run 34 — DT historical↔live parity diagnostics (pure, no I/O, no ML).

The scoring stack (direction votes, agreement, confirmation, conflicts, DT Score,
quality tier) is SHARED code: both the live path (ui.day_trader →
day_trade_intelligence on build_day_trader_metrics rows) and the historical path
(reconstruct_observations → day_trade_intelligence) call the SAME
analytics.day_trade_intel functions. Parity therefore reduces entirely to the
seven feature inputs. These helpers make the per-observation feature fidelity —
and any silent fallback to an incomplete input set — explicit and quantifiable.

Read-only: no scoring is changed here; we only re-describe an already-scored
observation from its recorded ``diagnostic_inputs``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from analytics import day_trade_intel as di

# The 7 inputs the classifier consumes (kept in sync with build_observation's
# diagnostic_inputs and reconstruct_observations' feat).
FEATURE_FIELDS = ("chg_pct", "gap_pct", "rvol", "vs_vwap_pct",
                  "adx", "supertrend_direction", "ewo")
# Daily-derived inputs. When ALL are absent, the observation came from the
# intraday-only fallback (no daily-indicator reconstruction): it cannot fire the
# Strong tier (no ADX/RVOL confirmation) and must not be treated as full fidelity.
DAILY_DERIVED_FIELDS = ("adx", "supertrend_direction", "ewo", "gap_pct", "rvol")


def _present(value: Any) -> bool:
    """A feature is present when it is a real, usable value (not None/NaN/blank)."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def classify_fallback(inputs: Dict[str, Any]) -> tuple[str, str]:
    """Classify an observation's feature completeness.

    Returns (status, reason):
      * ``full_feature`` — all 7 inputs present (true live fidelity).
      * ``fallback``     — every daily-derived input absent (intraday-only path).
      * ``partial``      — some (but not all) inputs present.
    """
    present = {f: _present(inputs.get(f)) for f in FEATURE_FIELDS}
    if all(present.values()):
        return "full_feature", ""
    if not any(present.get(f) for f in DAILY_DERIVED_FIELDS):
        return ("fallback",
                "all daily-derived inputs (adx/supertrend/ewo/gap/rvol) missing "
                "— intraday-only reconstruction")
    missing = [f for f in FEATURE_FIELDS if not present[f]]
    return "partial", "missing: " + ",".join(missing)


def _confirmation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Replicate the intel engine's confirmation read (ADX>=20 or RVOL>=1.5)."""
    adx = di._num(inputs.get("adx"))
    rvol = di._num(inputs.get("rvol"))
    adx_ok = adx is not None and adx >= 20.0
    rvol_ok = rvol is not None and rvol >= 1.5
    return {
        "confirmation": adx_ok or rvol_ok,
        "confirmation_count": int(adx_ok) + int(rvol_ok),
        "confirmation_status": ("both" if adx_ok and rvol_ok else "adx_only" if adx_ok
                                else "rvol_only" if rvol_ok else "neither"),
    }


def parity_record(obs: Dict[str, Any]) -> Dict[str, Any]:
    """One per-observation parity/diagnostic record.

    Uses the observation's recorded ``diagnostic_inputs`` (the exact inputs the
    score was computed from) and re-describes votes/agreement/confirmation/
    conflicts/score/tier via the shared intel engine, plus an explicit
    fallback_status/reason so incomplete-data rows are obvious.
    """
    inputs = obs.get("diagnostic_inputs") or {}
    votes = di._direction_votes(inputs)
    dir_votes = [v for v in votes.values() if v != 0]
    conflicts = di.day_trade_conflicts(inputs)
    status, reason = classify_fallback(inputs)
    rec: Dict[str, Any] = {
        "symbol": obs.get("ticker"),
        "timestamp": obs.get("timestamp"),
        "feature_source": obs.get("feature_source"),
        "fallback_status": status,
        "fallback_reason": reason,
        "feature_coverage": sum(_present(inputs.get(f)) for f in FEATURE_FIELDS),
    }
    for f in FEATURE_FIELDS:
        rec[f] = inputs.get(f)
    rec.update(_confirmation(inputs))
    rec.update({
        "directional_vote_count": len(dir_votes),
        "agreement": di._agreement(inputs),
        "conflict_count": len(conflicts),
        "conflict_reasons": conflicts,
        # Stored outputs (computed by the shared engine at build time).
        "dt_score": obs.get("score"),
        "direction": obs.get("direction"),
        "quality_tier": obs.get("setup_quality"),
    })
    return rec


def _bucket(value: Optional[float], edges: Sequence[float], labels: Sequence[str]) -> str:
    if value is None:
        return "missing"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def parity_summary(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate parity diagnostics for a set of observations (pure).

    Reports coverage, fallback rates, and the distributions Run 34 asks for:
    directional-vote count, agreement, confirmation, conflict count, and
    individual conflict-reason frequencies — plus tier and direction counts.
    """
    records = [parity_record(o) for o in observations]
    n = len(records)

    def _pct(x: int) -> Optional[float]:
        return (x / n) if n else None

    status_counts = Counter(r["fallback_status"] for r in records)
    source_counts = Counter(str(r.get("feature_source") or "unknown") for r in records)
    vote_counts = Counter(r["directional_vote_count"] for r in records)
    confirm_counts = Counter(r["confirmation_status"] for r in records)
    conflict_n = Counter(min(r["conflict_count"], 4) for r in records)
    tier_counts = Counter(str(r["quality_tier"]) for r in records)
    dir_counts = Counter(str(r["direction"]) for r in records)
    agree_counts: Counter = Counter(
        _bucket(r["agreement"], (0.55, 0.70, 0.80, 1.0),
                ("<0.55", "0.55-0.69", "0.70-0.79", "0.80-0.99", "1.00")) for r in records)

    conflict_reason_freq: Counter = Counter()
    for r in records:
        conflict_reason_freq.update(r["conflict_reasons"])

    per_field = {f: sum(_present(r[f]) for r in records) for f in FEATURE_FIELDS}

    return {
        "n": n,
        "coverage": {
            "full_feature": {"count": status_counts.get("full_feature", 0),
                             "pct": _pct(status_counts.get("full_feature", 0))},
            "partial": {"count": status_counts.get("partial", 0),
                        "pct": _pct(status_counts.get("partial", 0))},
            "fallback": {"count": status_counts.get("fallback", 0),
                         "pct": _pct(status_counts.get("fallback", 0))},
            "per_field": {f: {"present": c, "pct": _pct(c)} for f, c in per_field.items()},
            "by_source": dict(source_counts),
        },
        "directional_vote_count": dict(sorted(vote_counts.items())),
        "agreement": dict(agree_counts),
        "confirmation": dict(confirm_counts),
        "conflict_count": {(str(k) if k < 4 else "4+"): conflict_n[k] for k in range(5)},
        "conflict_reasons": dict(conflict_reason_freq),
        "quality_tier": dict(tier_counts),
        "direction": dict(dir_counts),
    }
