"""AI Notes generation module.
This module provides the `generate_ai_note` function used by the app.

The note adapts to the row schema: breakout-scan rows (BreakoutScore, GapPct,
Trend20D%, …) get a setup analysis; watchlist quote rows (Last, Change,
% Change, …) get a price snapshot.
"""

import pandas as pd


def _numopt(row, key):
    """Float from a row value, or None when missing/NaN (so callers can say 'n/a')."""
    try:
        v = row.get(key)
        if v is None:
            return None
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _num(row, key, default=0.0):
    """Best-effort float from a row value; NaN and missing both fall back to default."""
    v = _numopt(row, key)
    return float(default) if v is None else v


def _ticker(row) -> str:
    return str(row.get("Ticker") or row.get("Symbol") or "?")


def _breakout_note(row) -> str:
    ticker = _ticker(row)
    score = _num(row, "BreakoutScore")
    # Scan engine emits the gap column as "GapPct"; fall back to "Gap%".
    gap = _numopt(row, "GapPct")
    if gap is None:
        gap = _numopt(row, "Gap%")
    trend = _numopt(row, "Trend20D%")
    volrel = _numopt(row, "VolRel20")
    dvol = _numopt(row, "DollarVol20")
    vol20 = _numopt(row, "Volatility20D%")

    def pct(v):
        return "n/a" if v is None else f"{v:.2f}%"

    gap_s = pct(gap)
    if trend is None:
        direction = "flat"
    else:
        direction = "bullish" if trend > 0 else "bearish" if trend < 0 else "neutral"
    trend_s = "20‑day trend of n/a" if trend is None else f"{direction} 20‑day trend of {trend:.2f}%"
    volrel_s = "n/a" if volrel is None else f"{volrel:.2f}×"
    dvol_s = "n/a" if dvol is None else f"${dvol:,.0f}"
    return (
        f"{ticker} shows a breakout score of {score:.1f}. "
        f"Gap is {gap_s}, with a {trend_s}. "
        f"Volume relative is {volrel_s} and dollar volume is {dvol_s}. "
        f"Volatility (20D) sits at {pct(vol20)}."
    )


def _quote_note(row) -> str:
    ticker = _ticker(row)
    name = str(row.get("Name") or "").strip()
    last = _num(row, "Last")
    change = _num(row, "Change")
    pct = _num(row, "% Change", _num(row, "%Change"))
    high = _num(row, "High")
    low = _num(row, "Low")
    direction = "up" if change > 0 else "down" if change < 0 else "flat"
    label = f"{ticker}{f' ({name})' if name else ''}"
    note = (
        f"{label} last traded at ${last:,.2f}, {direction} "
        f"{change:+.2f} ({pct:+.2f}%) on the day."
    )
    if high or low:
        note += f" Day range: ${low:,.2f}–${high:,.2f}."
    return note


def generate_ai_note(row: pd.Series) -> str:
    """Generate a short, human-readable note that adapts to the row schema."""
    try:
        if row is None:
            return "AI Note unavailable for this row."
        # Breakout-scan row?
        if row.get("BreakoutScore") is not None:
            return _breakout_note(row)
        # Watchlist / quote row?
        if row.get("Last") is not None:
            return _quote_note(row)
        return f"{_ticker(row)}: no scan metrics available for an AI note."
    except Exception:
        return "AI Note unavailable for this row."
