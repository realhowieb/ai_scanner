"""Run 62 — scanner results empty states and tab label (pure, testable)."""
from __future__ import annotations

from typing import Any, Optional

# Run 62 — empty states. df is None until the user runs a scan this session;
# an empty DataFrame means their scan ran and matched nothing.
NO_SESSION_SCAN_MESSAGE = (
    "You haven't run a scan in this session yet. Choose what to scan with the "
    "controls above, then run it. The market status line shows when HSF last "
    "scanned the full market."
)
NO_MATCHES_MESSAGE = (
    "Your last scan found no stocks matching the current settings. Try another "
    "strategy, or loosen the price, gap or volume filters."
)


def results_empty_message(df: Optional[Any]) -> str:
    """The empty-state sentence for the results tab."""
    return NO_SESSION_SCAN_MESSAGE if df is None else NO_MATCHES_MESSAGE


def results_tab_label(df: Optional[Any]) -> str:
    """Tab label: row count once a scan has results, plain wording before."""
    rows = 0 if df is None else len(df)
    if df is None:
        return "📊 Your scan results"
    return f"📊 Latest scan results ({rows} rows)" if rows else "📊 Your scan results (no matches)"

