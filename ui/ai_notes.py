"""AI Notes generation module.
This module provides the `generate_ai_note` function used by the app.

The note adapts to the row schema: breakout-scan rows (BreakoutScore, GapPct,
Trend20D%, …) get a setup analysis; watchlist quote rows (Last, Change,
% Change, …) get a price snapshot.
"""

import pandas as pd


def _num(row, key, default=0.0):
    """Best-effort float from a row value."""
    try:
        v = row.get(key)
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _ticker(row) -> str:
    return str(row.get("Ticker") or row.get("Symbol") or "?")


def _breakout_note(row) -> str:
    ticker = _ticker(row)
    score = _num(row, "BreakoutScore")
    # Scan engine emits the gap column as "GapPct"; fall back to "Gap%".
    gap = _num(row, "GapPct", _num(row, "Gap%"))
    trend = _num(row, "Trend20D%")
    volrel = _num(row, "VolRel20")
    dvol = _num(row, "DollarVol20")
    vol20 = _num(row, "Volatility20D%")
    direction = "bullish" if trend > 0 else "bearish" if trend < 0 else "neutral"
    return (
        f"{ticker} shows a breakout score of {score:.1f}. "
        f"Gap is {gap:.2f}%, with a {direction} 20‑day trend of {trend:.2f}%. "
        f"Volume relative is {volrel:.2f}× and dollar volume is ${dvol:,.0f}. "
        f"Volatility (20D) sits at {vol20:.2f}%."
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
