"""P2-26 — "stair-stepper" trend consistency on 1-minute bars.

A stair-stepper climbs (or falls) in a tight, straight line with shallow
pullbacks. ADX and raw momentum reward big moves, not smooth ones, so this
measures straightness directly: the R² of a linear regression of close against
time over the most recent 1-minute bars (0 = no line, 1 = a perfect line).

Descriptive only. It says how straight the recent path was; it is not a
prediction and never feeds HSF Score, ranking, cohorts or research data.
Pure functions (no Streamlit, no network) so it can be tested directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

DEFAULT_WINDOW = 45        # most recent 1-minute bars fitted
WINDOW_OPTIONS = (10, 15, 20, 30, 45, 60)
MIN_COVERAGE = 0.6         # bars / minutes spanned; below this the feed is too gappy


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def latest_session_bars(bars: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bars from the most recent US/Eastern trading date only, so a window never
    bridges the overnight gap. Bars without a usable time or close are dropped."""
    clean = []
    for b in bars or []:
        ts = _parse_ts(b.get("t"))
        try:
            close = float(b.get("c"))
        except (TypeError, ValueError):
            continue
        if ts is None or ts.tzinfo is None or close <= 0:
            continue
        clean.append((ts, close))
    if not clean:
        return []
    clean.sort(key=lambda x: x[0])
    last_day = clean[-1][0].astimezone(_ET).date()
    return [{"t": ts, "c": c} for ts, c in clean if ts.astimezone(_ET).date() == last_day]


def stair_step_metrics(
    bars: Sequence[Dict[str, Any]], window: int = DEFAULT_WINDOW
) -> Dict[str, Any]:
    """Fit the last `window` 1-minute bars of the latest session.

    Returns {status, bars, r2, trend_pct_per_hour, direction, max_pullback_pct,
    coverage, as_of}. `status` is "ok", "insufficient" (fewer than ~80% of the
    window) or "sparse" (the bars are spread over too many minutes, typical of a
    thin name on the IEX feed); metrics are None unless status is "ok".
    """
    session = latest_session_bars(bars)
    recent = session[-window:] if window > 0 else []
    out: Dict[str, Any] = {"status": "insufficient", "bars": len(recent), "r2": None,
                           "trend_pct_per_hour": None, "direction": None,
                           "max_pullback_pct": None, "coverage": None,
                           "slope_per_minute": None, "current_price": None,
                           "fitted_price": None,
                           "as_of": recent[-1]["t"] if recent else None}
    if len(recent) < max(3, int(window * 0.8)):
        return out

    t0 = recent[0]["t"]
    xs = [(b["t"] - t0).total_seconds() / 60.0 for b in recent]   # minutes, gaps respected
    ys = [b["c"] for b in recent]
    span = xs[-1] - xs[0]
    coverage = len(recent) / (span + 1) if span > 0 else 0.0
    out["coverage"] = round(min(coverage, 1.0), 2)
    if coverage < MIN_COVERAGE:
        out["status"] = "sparse"
        return out

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return out
    slope = sxy / sxx
    r2 = 0.0 if syy <= 0 else (sxy * sxy) / (sxx * syy)
    direction = "up" if slope > 0 else ("down" if slope < 0 else "flat")

    # Deepest retracement against the trend, from closes: drop from the running
    # high for an up-trend, bounce from the running low for a down-trend.
    worst = 0.0
    if direction == "down":
        low = ys[0]
        for y in ys:
            low = min(low, y)
            worst = max(worst, (y - low) / low)
    else:
        high = ys[0]
        for y in ys:
            high = max(high, y)
            worst = max(worst, (high - y) / high)

    fitted_price = my + slope * (xs[-1] - mx)
    out.update(status="ok", r2=round(r2, 3), direction=direction,
               trend_pct_per_hour=round(slope * 60.0 / my * 100.0, 2),
               max_pullback_pct=round(worst * 100.0, 2),
               slope_per_minute=round(slope, 8), current_price=ys[-1],
               fitted_price=round(fitted_price, 8))
    return out


def is_stair_stepper(
    m: Dict[str, Any], *, r2_min: float = 0.80, direction: str = "up",
    max_pullback_pct: float = 1.0, min_trend_pct_per_hour: float = 0.5,
) -> bool:
    """True when the fit is straight enough, in the chosen direction ("up",
    "down" or "either"), with shallow pullbacks and a real (not flat) slope."""
    if m.get("status") != "ok":
        return False
    if direction in ("up", "down") and m.get("direction") != direction:
        return False
    if m.get("direction") == "flat":
        return False
    return (m["r2"] >= r2_min
            and abs(m["trend_pct_per_hour"]) >= min_trend_pct_per_hour
            and m["max_pullback_pct"] <= max_pullback_pct)
