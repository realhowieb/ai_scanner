"""Run 62 — scanner results empty states and tab label (pure, testable)."""
from __future__ import annotations

from typing import Any, Optional

# Run 62 — empty states. df is None until the user runs a scan this session;
# an empty DataFrame means their scan ran and matched nothing.
NO_SESSION_SCAN_MESSAGE = (
    "You haven't run a scan in this session yet. Open the Custom Scan page "
    "(link below or in the menu), choose what to scan, then run it. The market status line shows when "
    "HSF last scanned the full market."
)
MARKET_UNAVAILABLE_MESSAGE = (
    "HSF's latest full-market scan isn't available right now. Results will appear "
    "after the next scheduled scan completes, or you can run your own scan on the Custom Scan page."
)
NO_MATCHES_MESSAGE = (
    "Your last scan found no stocks matching the current settings. Try another "
    "strategy, or loosen the price, gap or volume filters."
)


def results_empty_message(df: Optional[Any], market_unavailable: bool = False) -> str:
    """The empty-state sentence for the results tab. With no session scan, a
    missing market scan is named as such (Run 70, P2-10)."""
    if df is not None:
        return NO_MATCHES_MESSAGE
    return MARKET_UNAVAILABLE_MESSAGE if market_unavailable else NO_SESSION_SCAN_MESSAGE


def results_tab_label(df: Optional[Any], market_view: Optional[Any] = None) -> str:
    """Tab label: row count once a scan has results, plain wording before.
    Run 63: the default full-market view is labelled as the market scan."""
    rows = 0 if df is None else len(df)
    if df is None:
        return "📊 Your scan results"
    if market_view and rows:
        return f"📊 Latest market scan ({rows} setups)"
    return f"📊 Latest scan results ({rows} rows)" if rows else "📊 Your scan results (no matches)"

