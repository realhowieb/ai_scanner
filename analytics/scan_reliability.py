"""Run 45 — full-market scan reliability: telemetry, health-gated snapshot
safety, failure taxonomy, and repeatability comparison (pure, no I/O).

Operational hardening only. It reuses the existing Run 37 coverage/health system
(analytics.coverage) rather than inventing a second one, and NEVER touches
scoring, signals, ranking, ML, or intelligence. It only decides whether a scan is
trustworthy enough to (a) be promoted as the canonical daily snapshot and (b) be
considered normal versus recent runs.
"""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from analytics.coverage import classify_failure

# Operational health states (folds coverage STALE into DEGRADED).
OPERATIONAL_STATES = ("HEALTHY", "DEGRADED", "FAILED")
# Only a HEALTHY scan may become the canonical daily snapshot.
SNAPSHOT_MIN_COVERAGE = 0.90
# Operational-level failure reasons (superset of the per-symbol coverage taxonomy).
OPERATIONAL_FAILURES = (
    "RATE_LIMIT", "TIMEOUT", "PROVIDER_ERROR", "NO_PRICE_DATA",
    "FILTERED_BY_POLICY", "MALFORMED_RESPONSE", "UNIVERSE_PROVIDER_FAILURE",
    "OVERLAPPING_RUN", "UNKNOWN",
)


def derive_event_counts(skipped: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """Classify per-symbol skips into operational event counts (Run 37 taxonomy).

    Distinguishes intentional policy skips from real provider trouble — e.g.
    "195 FILTERED_BY_POLICY" (deliberate) vs "195 RATE_LIMIT" (throttled)."""
    counts: Counter = Counter()
    for item in (skipped or []):
        reason = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item
        counts[classify_failure(reason)] += 1
    return {
        "by_reason": dict(counts),
        "rate_limit_events": counts.get("RATE_LIMIT", 0),
        "timeout_events": counts.get("TIMEOUT", 0),
        "provider_error_events": counts.get("API_ERROR", 0),
        "no_price_data_events": counts.get("NO_PRICE_DATA", 0),
        "policy_filtered_events": counts.get("FILTERED_BY_POLICY", 0),
        # Provider-trouble = failures that are NOT intentional policy skips.
        "provider_trouble_events": sum(
            v for k, v in counts.items() if k != "FILTERED_BY_POLICY"),
    }


def to_operational_state(coverage_health_state: Optional[str]) -> str:
    """Map a coverage health state to the operational triad."""
    s = str(coverage_health_state or "").upper()
    if s == "HEALTHY":
        return "HEALTHY"
    if s == "FAILED":
        return "FAILED"
    return "DEGRADED"  # DEGRADED / STALE / unknown → DEGRADED


def snapshot_decision(
    coverage_health_state: Optional[str], coverage_pct: Optional[float],
    *, min_coverage: float = SNAPSHOT_MIN_COVERAGE,
) -> Dict[str, Any]:
    """Decide whether a scan may be promoted as the canonical daily snapshot
    (Task 4 — the critical safety gate). Only a HEALTHY scan with sufficient
    coverage promotes; DEGRADED/FAILED are suppressed (artifact/diagnostics are
    still retained by the caller — failures are never made invisible)."""
    state = to_operational_state(coverage_health_state)
    if state != "HEALTHY":
        return {"promote": False,
                "reason": f"{state.lower()} scan — not promoted as canonical snapshot"}
    if coverage_pct is not None and coverage_pct < min_coverage:
        return {"promote": False,
                "reason": f"coverage {coverage_pct:.1%} < snapshot floor {min_coverage:.0%}"}
    return {"promote": True, "reason": None}


def build_performance_record(
    *, run_id: Optional[str], started_at: Any, completed_at: Any,
    market_session: Optional[str], universe: Optional[str],
    universe_source: Optional[str], provider_asset_count: Optional[int],
    eligible_symbol_count: Optional[int], attempted_symbol_count: Optional[int],
    priced_symbol_count: Optional[int], skipped_symbol_count: Optional[int],
    candidate_count: Optional[int], coverage_percentage: Optional[float],
    coverage_health: Optional[str], timings: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None, skipped: Optional[Sequence[Any]] = None,
    snapshot_promoted: Optional[bool] = None,
    snapshot_suppression_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble one structured performance record (Task 1). Unmeasurable stage
    timings are honestly null, not faked."""
    timings = timings or {}
    total = timings.get("total_runtime_seconds")
    symbols_per_second = None
    if total and attempted_symbol_count:
        symbols_per_second = round(attempted_symbol_count / total, 1)
    events = derive_event_counts(skipped)
    return {
        "schema": "hsf-scan-perf-1.0",
        "run_id": run_id,
        "started_at": str(started_at) if started_at is not None else None,
        "completed_at": str(completed_at) if completed_at is not None else None,
        "market_session": market_session,
        "universe": universe,
        "universe_source": universe_source,
        "provider_asset_count": provider_asset_count,
        "eligible_symbol_count": eligible_symbol_count,
        "attempted_symbol_count": attempted_symbol_count,
        "priced_symbol_count": priced_symbol_count,
        "skipped_symbol_count": skipped_symbol_count,
        "candidate_count": candidate_count,
        "coverage_percentage": coverage_percentage,
        "coverage_health": coverage_health,
        "operational_state": to_operational_state(coverage_health),
        "timings": {
            "universe_load_seconds": timings.get("universe_load_seconds"),
            "price_fetch_seconds": timings.get("price_fetch_seconds"),
            "scanner_processing_seconds": timings.get("scanner_processing_seconds"),
            "persistence_seconds": timings.get("persistence_seconds"),
            "automation_export_seconds": timings.get("automation_export_seconds"),
            "total_runtime_seconds": total,
        },
        "symbols_per_second": symbols_per_second,
        "batch_size": batch_size,
        # The engine does not currently expose batch-level counters; honestly null.
        "batches_attempted": timings.get("batches_attempted"),
        "batches_succeeded": timings.get("batches_succeeded"),
        "batches_failed": timings.get("batches_failed"),
        "rate_limit_events": events["rate_limit_events"],
        "timeout_events": events["timeout_events"],
        "provider_error_events": events["provider_error_events"],
        "policy_filtered_events": events["policy_filtered_events"],
        "provider_trouble_events": events["provider_trouble_events"],
        "retry_count": timings.get("retry_count"),
        "failure_by_reason": events["by_reason"],
        "snapshot_promoted": snapshot_promoted,
        "snapshot_suppression_reason": snapshot_suppression_reason,
    }


def render_run_summary(rec: Dict[str, Any]) -> str:
    """Human-readable scheduled-scan health summary (Task 11)."""
    def _n(x):
        return "—" if x is None else (f"{x:,}" if isinstance(x, int) else x)

    cov = rec.get("coverage_percentage")
    return "\n".join([
        f"HSF {rec.get('universe')} SCAN",
        f"Universe source: {rec.get('universe_source')}",
        f"Eligible:   {_n(rec.get('eligible_symbol_count'))}",
        f"Attempted:  {_n(rec.get('attempted_symbol_count'))}",
        f"Priced:     {_n(rec.get('priced_symbol_count'))}",
        f"Skipped:    {_n(rec.get('skipped_symbol_count'))}",
        f"Coverage:   {f'{cov:.1%}' if isinstance(cov, (int, float)) else '—'}",
        f"Candidates: {_n(rec.get('candidate_count'))}",
        f"Runtime:    {rec.get('timings', {}).get('total_runtime_seconds')}s",
        f"Throughput: {rec.get('symbols_per_second')} symbols/sec",
        f"Rate limits: {rec.get('rate_limit_events')}  Timeouts: {rec.get('timeout_events')}  "
        f"Provider errors: {rec.get('provider_error_events')}",
        f"Policy-filtered: {rec.get('policy_filtered_events')}",
        f"Health: {rec.get('operational_state')}",
        f"Snapshot promoted: {'YES' if rec.get('snapshot_promoted') else 'NO'}"
        + (f" ({rec.get('snapshot_suppression_reason')})"
           if rec.get('snapshot_promoted') is False and rec.get('snapshot_suppression_reason') else ""),
    ])


def compare_to_recent(
    current: Dict[str, Any], recent: Sequence[Dict[str, Any]],
    *, min_history: int = 3,
) -> Dict[str, Any]:
    """Flag (not fail) meaningful operational deviations vs recent HEALTHY runs
    (Tasks 2 & 13). Tolerant of normal listing/delisting/provider variation."""
    healthy = [r for r in recent if r.get("operational_state") == "HEALTHY"]
    if len(healthy) < min_history:
        return {"comparable": False, "reason": "insufficient healthy history",
                "flags": []}

    def _med(key, path=None):
        vals = []
        for r in healthy:
            v = (r.get(path, {}) or {}).get(key) if path else r.get(key)
            if isinstance(v, (int, float)):
                vals.append(v)
        return median(vals) if vals else None

    flags: List[str] = []
    us = current.get("eligible_symbol_count")
    us_med = _med("eligible_symbol_count")
    if us and us_med and abs(us - us_med) / us_med > 0.20:
        flags.append(f"universe size {us} deviates >20% from median {us_med:.0f}")
    cov, cov_med = current.get("coverage_percentage"), _med("coverage_percentage")
    if cov is not None and cov_med is not None and (cov_med - cov) > 0.05:
        flags.append(f"coverage {cov:.1%} down >5pts from median {cov_med:.1%}")
    rt = current.get("timings", {}).get("total_runtime_seconds")
    rt_med = _med("total_runtime_seconds", path="timings")
    if rt and rt_med and rt > 2 * rt_med:
        flags.append(f"runtime {rt}s >2x median {rt_med}s")
    tp, tp_med = current.get("symbols_per_second"), _med("symbols_per_second")
    if tp and tp_med and tp < 0.5 * tp_med:
        flags.append(f"throughput {tp}/s <0.5x median {tp_med}/s")
    pt = current.get("provider_trouble_events", 0)
    pt_med = _med("provider_trouble_events") or 0
    if pt > max(50, 3 * pt_med):
        flags.append(f"provider-trouble events {pt} >3x median {pt_med:.0f}")
    return {"comparable": True, "flags": flags, "is_anomalous": bool(flags),
            "healthy_samples": len(healthy)}
