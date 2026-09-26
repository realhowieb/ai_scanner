"""Run 38A — production scanner-observation capture (side-effect only).

Turns a completed scheduled scan's result rows into canonical Run 36
observations and persists them, so HSF accumulates durable, versioned,
per-scanner market observations for future performance analysis.

Strictly observe-only: it reads the scan's already-computed results and NEVER
changes scanner logic, thresholds, ranking, or output. If persistence fails or is
removed entirely, the scan produces identical results. Pure builders here; the
one I/O call (batch save) is isolated and best-effort.

Capture boundary: the scheduled breakout scan's result rows (the scanner
triggers), captured in `scheduler.cron_runner.run_and_save` AFTER the scan
completes and BEFORE results are serialized/discarded. Non-trigger control is a
compact scan-level denominator (Run 37 coverage), not per-symbol rows — see
docs/PRODUCTION_OBSERVATION_CAPTURE.md for the storage estimate behind that.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence

from analytics.hsf_observation import build_observation

# Result-df column -> canonical field. Only columns the scheduled breakout scan
# actually produces (scan/breakout.py). Absent fields are flagged missing, never
# invented (Task 3).
_MARKET_MAP = {"Last": "price", "Volume": "volume"}
_INDICATOR_MAP = {
    "GapPct": "gap_pct", "PctChange": "chg_pct", "VolRel20": "rvol",
    "Volatility20D%": "atr_pct",
}


def bucket_timestamp(ts: Any, *, bucket: str = "hour") -> str:
    """Floor a scan timestamp to a bucket so workflow retries within the same
    scheduled slot produce the SAME observation id (idempotency). Scheduled cron
    slots are >= 1h apart, so hour-flooring dedupes retries without collapsing
    distinct scans."""
    try:
        d = ts if isinstance(ts, _dt.datetime) else _dt.datetime.fromisoformat(
            str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        d = d.astimezone(_dt.timezone.utc)
        if bucket == "hour":
            d = d.replace(minute=0, second=0, microsecond=0)
        elif bucket == "day":
            d = d.replace(hour=0, minute=0, second=0, microsecond=0)
        return d.isoformat()
    except Exception:
        return str(ts)


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def derive_scanner_triggers(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Which scanners a breakout candidate row satisfies (multi-scanner capture).

    Every result row is already a breakout candidate, so `breakout` always
    triggers. The strategy sub-scanners mirror scan.strategies' predicates WITHOUT
    calling or changing them (measurement only, no shared mutable state)."""
    gap = _num(row.get("GapPct"))
    vrel = _num(row.get("VolRel20"))
    t20, t10 = _num(row.get("Trend20D%")), _num(row.get("Trend10D%"))
    dvol = _num(row.get("DollarVol20"))
    score = _num(row.get("BreakoutScore"))
    is_bo = bool(row.get("IsBreakout"))

    scanners: List[Dict[str, Any]] = [{
        "name": "breakout", "version": "breakout-1", "triggered": True,
        "score": score, "direction": "long",
        "meta": {"is_breakout": is_bo, "pattern": row.get("PatternTag")},
    }]
    if gap is not None and gap > 0:
        scanners.append({"name": "gap_up", "version": "strategy-1", "triggered": True,
                         "score": gap, "direction": "long"})
    if gap is not None and gap < 0:
        scanners.append({"name": "gap_down", "version": "strategy-1", "triggered": True,
                         "score": gap, "direction": "short"})
    if vrel is not None and vrel >= 2:
        scanners.append({"name": "unusual_vol", "version": "strategy-1", "triggered": True,
                         "score": vrel, "direction": "long"})
    if t20 is not None and t10 is not None and t20 > 0 and t10 > 0:
        scanners.append({"name": "momentum", "version": "strategy-1", "triggered": True,
                         "score": t20, "direction": "long"})
    if is_bo:
        scanners.append({"name": "breakout_only", "version": "strategy-1", "triggered": True,
                         "score": score, "direction": "long"})
    if dvol is not None:
        scanners.append({"name": "most_active", "version": "strategy-1", "triggered": True,
                         "score": dvol, "direction": "long"})
    return scanners


def _row_get(row: Dict[str, Any], keys) -> Dict[str, Any]:
    out = {}
    for col, field in keys.items():
        if col in row and row.get(col) is not None:
            out[field] = row.get(col)
    return out


def build_scan_observations(
    result_rows: Sequence[Dict[str, Any]],
    *,
    universe: str,
    scan_timestamp: Any,
    session: Optional[str] = None,
    universe_version: Optional[str] = None,
    scan_id: Optional[str] = None,
    coverage_health: Optional[str] = None,
    market_regime: Optional[str] = None,
    ts_bucket: str = "hour",
    research_cohort: Optional[str] = None,
    selection_reason: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build one canonical observation per scanner-trigger result row. When
    `research_cohort` is given (Run 47), each observation is tagged with its
    cohort + selection_reason (point-in-time labels only)."""
    obs_ts = bucket_timestamp(scan_timestamp, bucket=ts_bucket)
    context = f"scheduled:{str(universe).lower()}"
    out: List[Dict[str, Any]] = []
    for row in result_rows or []:
        sym = row.get("Ticker") or row.get("Symbol")
        if not sym:
            continue
        o = build_observation(
            symbol=sym,
            timestamp=obs_ts,
            context=context,
            session=session,
            universe_version=universe_version or universe,
            market=_row_get(row, _MARKET_MAP),
            indicators=_row_get(row, _INDICATOR_MAP),
            scanners=derive_scanner_triggers(row),
            market_context={"source": "scheduled", "scan_id": scan_id,
                            "coverage_health": coverage_health,
                            "market_regime": market_regime},
            scan_timestamp=scan_timestamp,
            data_source="scheduled_breakout_scan",
        )
        if research_cohort:
            o["research_cohort"] = research_cohort
            o["selection_reason"] = selection_reason or ""
            o["market_context"]["research_cohort"] = research_cohort
        out.append(o)
    return out


def summarize_capture(observations: Sequence[Dict[str, Any]],
                      *, coverage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture statistics + quality distribution for a set of observations (Task
    14/15). `coverage` (Run 37 funnel counts) supplies the control denominator."""
    n = len(observations)
    quality = {"complete": 0, "partial": 0, "fallback": 0, "stale": 0}
    for o in observations:
        dq = o.get("data_quality", {})
        if dq.get("fallback_used"):
            quality["fallback"] += 1
        elif dq.get("feature_completeness", 0) >= 0.999:
            quality["complete"] += 1
        else:
            quality["partial"] += 1
        if dq.get("stale"):
            quality["stale"] += 1
    ccounts = (coverage or {}).get("counts", {}) if coverage else {}
    return {
        "trigger_observations": n,
        "eligible_symbols": ccounts.get("eligible"),
        "control_denominator": {  # compact non-trigger control (Task 4 option 3)
            "eligible": ccounts.get("eligible"),
            "price_success": ccounts.get("price_success"),
            "non_trigger_estimate": (
                max(0, ccounts.get("price_success", 0) - n)
                if ccounts.get("price_success") is not None else None),
        },
        "quality_distribution": quality,
    }


def _capture_health(stats: Dict[str, Any]) -> str:
    attempted = stats.get("attempted", 0) or 0
    failed = stats.get("write_failures", 0) or 0
    if attempted and failed == attempted:
        return "FAILED"
    if failed:
        return "DEGRADED"
    return "HEALTHY"


def capture_scan_observations(
    result_rows: Sequence[Dict[str, Any]],
    *,
    universe: str,
    scan_timestamp: Any,
    session: Optional[str] = None,
    universe_version: Optional[str] = None,
    scan_id: Optional[str] = None,
    coverage: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    conn=None,
    research_cohort: Optional[str] = None,
    selection_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and (unless dry_run) persist canonical observations for a scan.

    Best-effort and non-fatal: any failure is captured in the returned stats and
    never raised. Returns a capture-statistics dict (Task 14/17)."""
    coverage_health = None
    if coverage:
        coverage_health = (coverage.get("health") or {}).get("state") if "health" in coverage else None
    observations = build_scan_observations(
        result_rows, universe=universe, scan_timestamp=scan_timestamp,
        session=session, universe_version=universe_version, scan_id=scan_id,
        coverage_health=coverage_health, research_cohort=research_cohort,
        selection_reason=selection_reason,
    )
    cov_funnel = (coverage or {}).get("funnel") if coverage else None
    summary = summarize_capture(observations, coverage=cov_funnel)

    stats: Dict[str, Any] = {
        "universe": universe, "scan_id": scan_id, "session": session,
        "dry_run": bool(dry_run),
        "attempted": len(observations),
        "written": 0, "duplicates": 0, "write_failures": 0,
        **summary,
    }
    if dry_run:
        stats["would_write"] = len(observations)
        stats["capture_health"] = "DRY_RUN"
        return stats

    try:
        from db.hsf_observations import save_observations_batch
        res = save_observations_batch(observations, conn=conn)
        stats["written"] = res.get("written", 0)
        stats["duplicates"] = res.get("duplicates", 0)
        stats["write_failures"] = res.get("failed", 0)
    except Exception as e:  # never let capture break a scan
        stats["write_failures"] = len(observations)
        stats["error"] = f"{type(e).__name__}: {e}"
    stats["capture_health"] = _capture_health(stats)
    return stats


# Intraday maturation horizons (label -> forward minute bars).
HORIZON_BARS = {"+5m": 5, "+15m": 15, "+30m": 30, "+60m": 60}
# Maturation failure taxonomy (Task 8).
MATURATION_FAILURES = (
    "PRICE_DATA_UNAVAILABLE", "INSUFFICIENT_FUTURE_BARS", "INVALID_TIMESTAMP",
    "MARKET_CLOSED", "PROVIDER_ERROR", "DATABASE_ERROR", "UNKNOWN",
    "RATE_LIMITED",  # Run 51: persistent HTTP 429 — throttling, not missing data
)


def _parse_dt(v: Any):
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def horizon_eligibility(
    anchor: Any, now: Any, existing_horizons, *,
    horizon_bars: Optional[Dict[str, int]] = None, slack_min: int = 15,
) -> Dict[str, str]:
    """Per-horizon maturation status, independent of the others (Task 4/7).

    Returns {horizon -> "already" | "not_ready" | "ready"}. A horizon is `ready`
    only once `now >= anchor + horizon + slack` (slack lets the provider publish
    the bar). `already` when an outcome exists. Never marks ready before the
    horizon's wall-clock has elapsed — no lookahead.
    """
    horizon_bars = horizon_bars or HORIZON_BARS
    a, n = _parse_dt(anchor), _parse_dt(now)
    existing = set(existing_horizons or [])
    out: Dict[str, str] = {}
    for h, mins in horizon_bars.items():
        if h in existing:
            out[h] = "already"
        elif a is None or n is None:
            out[h] = "not_ready"
        elif n >= a + _dt.timedelta(minutes=mins + slack_min):
            out[h] = "ready"
        else:
            out[h] = "not_ready"
    return out


def compute_matured_outcomes(
    observation: Dict[str, Any],
    *,
    prices_after: Sequence[float],
    horizon_bars: Dict[str, int],
    evaluation_times: Optional[Dict[str, Any]] = None,
    direction: str = "long",
) -> List[Dict[str, Any]]:
    """Pure outcome builder for a matured observation (Task 10/11).

    `prices_after[0]` is the signal-time price; later entries are strictly-later
    bars (outcomes measured from the future only — no lookahead). Reuses
    day_trade_validation for the return/MFE-MAE math and hsf_observation.build_outcome
    for the schema + lookahead guard. Returns one outcome per horizon with data.
    """
    from analytics.day_trade_validation import (
        directional_return,
        forward_returns,
        mfe_mae,
    )
    from analytics.hsf_observation import build_outcome

    oid = observation.get("observation_id")
    symbol = observation.get("symbol")
    obs_ts = observation.get("scan_timestamp") or observation.get("timestamp")
    evaluation_times = evaluation_times or {}
    if not prices_after or len(prices_after) < 2:
        return []
    rets = forward_returns(prices_after, 0, horizon_bars)
    mm = mfe_mae(prices_after, 0, max(horizon_bars.values()), direction)
    out: List[Dict[str, Any]] = []
    for label, r in rets.items():
        if r is None:
            continue
        eval_time = evaluation_times.get(label)
        try:
            out.append(build_outcome(
                observation_id=oid, symbol=symbol, observation_timestamp=obs_ts,
                horizon=label if label.startswith("+") else f"+{label}",
                evaluation_time=eval_time,
                raw_return=round(r, 6),
                directional_return=directional_return(direction, r),
                mfe=mm.get("mfe"), mae=mm.get("mae"),
                data_status="MATURED" if eval_time is not None else "PENDING",
            ))
        except ValueError:
            # lookahead guard tripped — skip rather than record a bad label
            continue
    return out


def render_capture_text(stats: Dict[str, Any]) -> str:
    """Human-readable capture summary for logs (Task 14)."""
    q = stats.get("quality_distribution", {})
    lines = [
        "HSF Observation Capture",
        "-----------------------",
        f"Universe:            {stats.get('universe')} "
        f"(session={stats.get('session')}, dry_run={stats.get('dry_run')})",
        f"Eligible symbols:    {stats.get('eligible_symbols')}",
        f"Observations attempt {stats.get('attempted')}",
        f"Written:             {stats.get('written')}",
        f"Duplicates:          {stats.get('duplicates')}",
        f"Write failures:      {stats.get('write_failures')}",
        f"Trigger obs:         {stats.get('trigger_observations')}",
        f"Control denominator: {stats.get('control_denominator', {}).get('non_trigger_estimate')}",
        f"Quality:             complete={q.get('complete')} partial={q.get('partial')} "
        f"fallback={q.get('fallback')} stale={q.get('stale')}",
        f"Capture health:      {stats.get('capture_health')}",
    ]
    return "\n".join(lines)
