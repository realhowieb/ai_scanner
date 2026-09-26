"""Dependency-free alert labels and explanatory copy."""
from __future__ import annotations

from typing import Any

BREAKOUT_ALERT_DEFAULT = 8.0
BREAKOUT_ALERT_SCALE_COPY = (
    "**Breakout Score threshold** uses the scanner's supporting technical score, "
    "not the 0-100 HSF Score. Lower thresholds fire more often; higher thresholds "
    "are more selective. The alert fires when Breakout Score is at or above your value."
)
BREAKOUT_ALERT_SCALE_LEGEND = "Lower / more frequent  ←  8.0 default  →  Higher / more selective"


def breakout_alert_label(threshold: Any, watchlist_only: bool = False) -> str:
    """Human-readable saved-alert label, safe for malformed legacy values."""
    try:
        value = float(threshold or 0)
    except (TypeError, ValueError):
        value = 0.0
    scope = "watchlist only" if watchlist_only else "all tickers"
    return f"🚀 Breakout Score ≥ {value:g} ({scope})"
