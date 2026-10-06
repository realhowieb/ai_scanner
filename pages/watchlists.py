"""📋 My Stocks — watchlists and alerts in one place (P1-6).

Previously "My Watchlist"; alerts (also still at pages/alerts.py for the
"Alert me on this" shortcuts) now sit in a second tab.

Run 27 promotes the persisted watchlist into a read-only intelligence surface
while preserving the existing watchlist management panel and scan handoff.
"""
from __future__ import annotations

import streamlit as st

from ui.showcase import initial_sidebar_state

st.set_page_config(page_title="My Stocks", page_icon="📋", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # Run 62/P2: before any sign-in gate

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage your watchlists.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

def _render_watchlist_tab() -> None:
    from ui.personal_watchlist import render_personal_watchlist
    from ui.watchlists import render_watchlists_panel

    render_personal_watchlist(_username)
    st.markdown("---")
    st.markdown("### Manage Watchlists")
    st.caption("Your active list feeds the scanner, Day Trader monitor, Market Brief, and alerts.")
    render_watchlists_panel(_username)

    # Add / Remove / Clear act here; "Run Watchlist Scan" / "View as table"
    # open the Custom Scan page, which runs them and shows results on the Scanner.
    from ui.custom_scan import handle_watchlist_tools

    handle_watchlist_tools(_username)


try:
    from ui.alerts_page import render_alerts_body, session_watch_tickers
    from ui.header import render_page_logo

    st.session_state["hsf_my_watchlist_viewed"] = True
    render_page_logo()
    from ui.design_system import render_page_header

    render_page_header("My Stocks", "Your watchlists and alerts in one place.")
    _tab_watch, _tab_alerts = st.tabs(["📋 Watchlist", "🔔 Alerts"])
    with _tab_watch:
        _render_watchlist_tab()
    with _tab_alerts:
        render_alerts_body(_username, watch_tickers=session_watch_tickers())
except Exception as e:
    from ui.safe_errors import show_error

    show_error("your watchlists", e)
