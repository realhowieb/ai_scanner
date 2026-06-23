"""Small Streamlit panel describing scheduled scan runtime settings."""
from __future__ import annotations

import os

import streamlit as st


def render_scheduler() -> None:
    """Render scheduler configuration used by GitHub Actions/cron jobs."""
    st.subheader("Scheduled scans")
    st.caption("GitHub Actions runs SP500, NASDAQ, and Combo scans through scheduler.cron_runner.")

    settings = {
        "CRON_NASDAQ_LIMIT": os.getenv("CRON_NASDAQ_LIMIT", "2000"),
        "CRON_MIN_GAP": os.getenv("CRON_MIN_GAP", "0"),
        "CRON_MIN_PRICE": os.getenv("CRON_MIN_PRICE", "1"),
        "CRON_MAX_PRICE": os.getenv("CRON_MAX_PRICE", "1000"),
        "CRON_TOP_N": os.getenv("CRON_TOP_N", "100"),
        "CRON_PROFILE": os.getenv("CRON_PROFILE", "regular"),
        "AI_SCANNER_SQLITE_FALLBACK": os.getenv("AI_SCANNER_SQLITE_FALLBACK", "true"),
    }
    st.table({"Setting": list(settings.keys()), "Value": list(settings.values())})
