"""AI Notes generation module.
This module provides the `generate_ai_note` function used by the app.
"""

import pandas as pd


def generate_ai_note(row: pd.Series) -> str:
    """Generate AI commentary for a single stock row.

    Expected columns in `row`:
        - Ticker
        - BreakoutScore
        - Gap%
        - Trend20D%
        - VolRel20
        - DollarVol20
        - Volatility20D%

    Returns a short, human-readable analysis string.
    """
    try:
        ticker = row.get("Ticker", "?")
        score = row.get("BreakoutScore", 0)
        gap = row.get("Gap%", 0)
        trend = row.get("Trend20D%", 0)
        volrel = row.get("VolRel20", 0)
        dvol = row.get("DollarVol20", 0)
        vol20 = row.get("Volatility20D%", 0)

        direction = "bullish" if trend > 0 else "bearish" if trend < 0 else "neutral"

        return (
            f"{ticker} shows a breakout score of {score:.1f}. "
            f"Gap is {gap:.2f}%, with a {direction} 20‑day trend of {trend:.2f}%. "
            f"Volume relative is {volrel:.2f}× and dollar volume is ${dvol:,.0f}. "
            f"Volatility (20D) sits at {vol20:.2f}%."
        )

    except Exception:
        return "AI Note unavailable for this row."
