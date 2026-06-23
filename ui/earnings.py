# ui/earnings.py

from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import date

from db.earnings import (
    fetch_earnings_this_week,
    add_earnings_days_column,
)

# Column name used everywhere
EARN_COL_DAYS = "earnings_in_days"


def render_earnings_this_week_panel(*, can_earnings: bool) -> None:
    """
    Render the 'Earnings this week' panel.
    DB-only. Never fetches from Yahoo.
    """
    if not can_earnings:
        st.caption("🔒 Earnings calendar is a Pro feature.")
        return

    try:
        rows = fetch_earnings_this_week()
    except Exception as e:
        st.error(f"Failed to load earnings calendar: {e}")
        return

    if not rows:
        st.info("No upcoming earnings found for this week.")
        return

    # rows may be a list[dict] or list[tuple]/list[list]
    df = pd.DataFrame(rows)

    # Normalize column names defensively (handles SQLAlchemy Row/tuples too)
    df.columns = [str(c).lower() for c in df.columns]

    # Some parts of the app use `ticker` (preferred) while older code uses `symbol`
    if "symbol" in df.columns and "ticker" not in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    if "earnings_date" in df.columns:
        today_ts = pd.Timestamp(date.today())
        # Use timestamps (not .dt.date) so subtraction stays vectorized
        earn_ts = pd.to_datetime(df["earnings_date"], errors="coerce")
        df[EARN_COL_DAYS] = (earn_ts - today_ts).dt.days

    cols = []
    for c in ["ticker", "earnings_date", EARN_COL_DAYS]:
        if c in df.columns:
            cols.append(c)

    view = df[cols]
    if EARN_COL_DAYS in view.columns:
        view = view.sort_values(EARN_COL_DAYS, na_position="last")

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
    )


__all__ = [
    "EARN_COL_DAYS",
    "render_earnings_this_week_panel",
    "fetch_earnings_this_week",
    "add_earnings_days_column",
]
