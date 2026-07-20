"""⚡ Day Trader — live: dedicated full page for the intraday monitor.

Own page so it can stay open all session (auto-refresh polling only this view,
not the whole scanner) and get maximum width for the table.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Day Trader — live", page_icon="⚡", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to use the Day Trader monitor.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()


def _session_watch_tickers() -> list[str]:
    """Use tickers already loaded on the main page; avoid DB work before first paint."""
    tickers = st.session_state.get("active_watchlist_tickers") or []
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


try:
    from ui.day_trader import render_day_trader_panel

    render_day_trader_panel(watch_tickers=_session_watch_tickers())
except Exception as e:
    st.error("Day Trader monitor failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
