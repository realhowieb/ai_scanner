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

    df = pd.DataFrame(rows)

    # Normalize column names defensively
    df.columns = [c.lower() for c in df.columns]

    if "earnings_date" in df.columns:
        today = date.today()
        df[EARN_COL_DAYS] = (
            pd.to_datetime(df["earnings_date"], errors="coerce").dt.date - today
        ).dt.days

    # Friendly ordering
    cols = []
    for c in ["symbol", "earnings_date", "earnings_time", EARN_COL_DAYS]:
        if c in df.columns:
            cols.append(c)

    st.dataframe(
        df[cols].sort_values(EARN_COL_DAYS, na_position="last"),
        width="stretch",
        hide_index=True,
    )


__all__ = [
    "EARN_COL_DAYS",
    "render_earnings_this_week_panel",
    "fetch_earnings_this_week",
    "add_earnings_days_column",
]