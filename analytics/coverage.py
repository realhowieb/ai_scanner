"""Run 37 — whole-market scan coverage funnel + data-health (pure, no I/O).

Answers "when HSF says it scanned the market, how much did it actually scan?"
with measurable evidence. Consumes what a scan already produces (universe size,
eligible/tradable count, fetched price symbols, provider skip reasons, timing)
and builds a coverage funnel, a per-reason failure breakdown, and a health
classification. Reuses data.provider_diagnostics for the low-level skip taxonomy.

Dependency-free so headless jobs, CI, and tests can import it anywhere. No
scanning, no scoring, no thresholds baked into production scan logic — all health
thresholds are parameters with documented defaults.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.provider_diagnostics import classify_skip_reason

# Run 37 failure taxonomy — every symbol that leaves the funnel gets one of these.
FAILURE_REASONS = (
    "NO_PRICE_DATA", "STALE_DATA", "INSUFFICIENT_HISTORY", "INVALID_SYMBOL",
    "DELISTED", "UNSUPPORTED_SECURITY", "INDICATOR_FAILURE", "API_ERROR",
    "RATE_LIMIT", "TIMEOUT", "FILTERED_BY_POLICY", "UNKNOWN",
)

# Map the provider-diagnostics categories to the Run 37 taxonomy.
_PROVIDER_TO_REASON = {
    "provider_missing": "API_ERROR",
    "rate_limited": "RATE_LIMIT",
    "timeout": "TIMEOUT",
    "auth": "API_ERROR",
    "empty_response": "NO_PRICE_DATA",
    "duplicate_data": "NO_PRICE_DATA",
    "download_error": "API_ERROR",
    "invalid_data": "INDICATOR_FAILURE",
    "other": "UNKNOWN",
    "unknown": "UNKNOWN",
}

HEALTH_STATES = ("HEALTHY", "DEGRADED", "STALE", "FAILED")


def classify_failure(reason: object) -> str:
    """Map a raw provider/scan skip reason to the Run 37 failure taxonomy."""
    text = str(reason or "").strip().lower()
    # Direct taxonomy hits (scan-side reasons that aren't provider skips).
    for r in FAILURE_REASONS:
        if r.lower() in text:
            return r
    if "insufficient" in text or "history" in text or "not enough" in text:
        return "INSUFFICIENT_HISTORY"
    if "delist" in text:
        return "DELISTED"
    if "untradable" in text or "not tradable" in text or "policy" in text:
        return "FILTERED_BY_POLICY"
    return _PROVIDER_TO_REASON.get(classify_skip_reason(reason), "UNKNOWN")


def _pct(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def build_coverage_funnel(
    *,
    universe_version: Optional[str] = None,
    expected: int,
    eligible: int,
    attempted: int,
    price_success: int,
    skipped: Optional[Sequence[Any]] = None,
    indicator_complete: Optional[int] = None,
    results: Optional[int] = None,
    scan_started_at: Any = None,
    scan_completed_at: Any = None,
    market_session: Optional[str] = None,
    duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the coverage funnel with counts, percentages, and failure reasons.

    `skipped` is an iterable of (symbol, reason) pairs (or bare reasons). Nothing
    is hardcoded: `expected`/`eligible` are supplied by the caller from the live
    universe. `indicator_complete` is optional (None = not measured this run).
    """
    skipped = list(skipped or [])
    reason_counts: Counter = Counter()
    failed_symbols: List[Tuple[str, str]] = []
    for item in skipped:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            sym, raw = str(item[0]), item[1]
        else:
            sym, raw = "?", item
        reason = classify_failure(raw)
        reason_counts[reason] += 1
        failed_symbols.append((sym, reason))

    price_failure = max(0, attempted - price_success)
    excluded = max(0, expected - eligible)

    funnel = {
        "universe_version": universe_version,
        "market_session": market_session,
        "counts": {
            "expected": int(expected),
            "eligible": int(eligible),
            "excluded": int(excluded),
            "attempted": int(attempted),
            "price_success": int(price_success),
            "price_failure": int(price_failure),
            "indicator_complete": (int(indicator_complete)
                                   if indicator_complete is not None else None),
            "results": (int(results) if results is not None else None),
            "failures": len(skipped),
        },
        "percentages": {
            "eligible_pct": _pct(eligible, expected),
            "price_coverage_pct": _pct(price_success, eligible),
            "price_coverage_vs_expected_pct": _pct(price_success, expected),
            "indicator_coverage_pct": (_pct(indicator_complete, eligible)
                                       if indicator_complete is not None else None),
        },
        "coverage_pct": _pct(price_success, eligible),
        "failure_reasons": dict(reason_counts),
        "top_failure_categories": [
            {"reason": r, "count": c} for r, c in reason_counts.most_common(5)
        ],
        "timing": {
            "scan_started_at": str(scan_started_at) if scan_started_at is not None else None,
            "scan_completed_at": str(scan_completed_at) if scan_completed_at is not None else None,
            "duration_sec": duration_sec,
        },
    }
    return funnel


def classify_health(
    funnel: Dict[str, Any],
    *,
    universe_age_hours: Optional[float] = None,
    price_ts_age_min: Optional[float] = None,
    healthy_floor: float = 0.95,
    degraded_floor: float = 0.80,
    universe_max_age_hours: float = 216.0,   # weekly refresh (Sun) + 1-day grace
    price_max_age_min: float = 60.0,
    model_version_age_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify overall data health. Precedence: FAILED > STALE > DEGRADED > HEALTHY.

    Thresholds are documented defaults grounded in current behavior (weekly
    universe refresh; existing cron guard treats a near-empty large scan as a
    failure) and are all overridable. None-valued ages are simply not checked.
    """
    counts = funnel.get("counts", {})
    coverage = funnel.get("coverage_pct")
    reasons: List[str] = []

    # FAILED — no eligible universe, or zero successful price coverage.
    if counts.get("eligible", 0) <= 0:
        return {"state": "FAILED", "reasons": ["no eligible universe"],
                "coverage_pct": coverage}
    if counts.get("price_success", 0) <= 0:
        return {"state": "FAILED", "reasons": ["zero price coverage"],
                "coverage_pct": coverage}

    # STALE — inputs too old to trust even if coverage looks fine.
    stale = False
    if universe_age_hours is not None and universe_age_hours > universe_max_age_hours:
        stale = True
        reasons.append(f"universe age {universe_age_hours:.0f}h > {universe_max_age_hours:.0f}h")
    if price_ts_age_min is not None and price_ts_age_min > price_max_age_min:
        stale = True
        reasons.append(f"price age {price_ts_age_min:.0f}m > {price_max_age_min:.0f}m")
    if stale:
        return {"state": "STALE", "reasons": reasons, "coverage_pct": coverage}

    # DEGRADED / HEALTHY on coverage.
    if coverage is None or coverage < degraded_floor:
        reasons.append(f"coverage {coverage} < degraded_floor {degraded_floor}")
        return {"state": "DEGRADED", "reasons": reasons, "coverage_pct": coverage}
    if coverage < healthy_floor:
        reasons.append(f"coverage {coverage} < healthy_floor {healthy_floor}")
        return {"state": "DEGRADED", "reasons": reasons, "coverage_pct": coverage}
    if model_version_age_days is not None and model_version_age_days > 400:
        reasons.append(f"model version age {model_version_age_days:.0f}d")
        return {"state": "DEGRADED", "reasons": reasons, "coverage_pct": coverage}
    return {"state": "HEALTHY", "reasons": ["coverage ok"], "coverage_pct": coverage}


def health_summary(
    funnel: Dict[str, Any], health: Dict[str, Any], *, stale_symbols: int = 0,
) -> Dict[str, Any]:
    """Compact structure for an eventual admin/status surface (Task 6)."""
    c = funnel.get("counts", {})
    p = funnel.get("percentages", {})
    return {
        "state": health.get("state"),
        "universe_version": funnel.get("universe_version"),
        "price_coverage_pct": p.get("price_coverage_pct"),
        "indicator_coverage_pct": p.get("indicator_coverage_pct"),
        "eligible": c.get("eligible"),
        "price_success": c.get("price_success"),
        "stale_symbols": int(stale_symbols),
        "errors": c.get("failures"),
        "top_failure_categories": funnel.get("top_failure_categories"),
        "duration_sec": funnel.get("timing", {}).get("duration_sec"),
        "last_scan_completed_at": funnel.get("timing", {}).get("scan_completed_at"),
    }


def render_coverage_text(funnel: Dict[str, Any], health: Dict[str, Any]) -> str:
    """Human-readable coverage report (Task 5)."""
    c = funnel.get("counts", {})
    p = funnel.get("percentages", {})

    def _p(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lines = [
        f"Coverage report — universe_version={funnel.get('universe_version')} "
        f"session={funnel.get('market_session')} health={health.get('state')}",
        f"  Expected            {c.get('expected'):>8}",
        f"  Eligible            {c.get('eligible'):>8}  ({_p(p.get('eligible_pct'))} of expected)",
        f"  Attempted           {c.get('attempted'):>8}",
        f"  Price success       {c.get('price_success'):>8}  ({_p(p.get('price_coverage_pct'))} of eligible)",
        f"  Price failure       {c.get('price_failure'):>8}",
    ]
    if c.get("indicator_complete") is not None:
        lines.append(f"  Indicator complete  {c.get('indicator_complete'):>8}  "
                     f"({_p(p.get('indicator_coverage_pct'))} of eligible)")
    if c.get("results") is not None:
        lines.append(f"  Results produced    {c.get('results'):>8}")
    lines.append(f"  Coverage            {_p(funnel.get('coverage_pct'))}")
    if funnel.get("top_failure_categories"):
        cats = ", ".join(f"{x['reason']}={x['count']}" for x in funnel["top_failure_categories"])
        lines.append(f"  Top failures        {cats}")
    if health.get("reasons"):
        lines.append(f"  Health notes        {'; '.join(health['reasons'])}")
    return "\n".join(lines)


def coverage_report(
    funnel: Dict[str, Any], health: Dict[str, Any], *, stale_symbols: int = 0,
) -> Dict[str, Any]:
    """Machine-readable + human-readable coverage report bundle (Task 5)."""
    return {
        "schema": "hsf-coverage-1.0",
        "funnel": funnel,
        "health": health,
        "summary": health_summary(funnel, health, stale_symbols=stale_symbols),
        "text": render_coverage_text(funnel, health),
    }
