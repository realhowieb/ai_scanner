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

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage your watchlists.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

def _render_watchlist_tab() -> None:
    from ui.personal_watchlist import render_personal_watchlist
    from ui.watchlists import handle_active_watchlist_actions, render_watchlists_panel

    render_personal_watchlist(_username)
    st.markdown("---")
    st.markdown("### Manage Watchlists")
    st.caption("Your active list feeds the scanner, Day Trader monitor, Market Brief, and alerts.")
    render_watchlists_panel(_username)

    _view, _run, _clear, _add, _remove, _sym, _scan_all = st.session_state.get(
        "_wl_tools_state", (False, False, False, False, False, "", False)
    )

    # Add / Remove / Clear only touch the watchlist DB — handle them right here
    # so they work without the scanner. (do_scan is never reached for these.)
    if _add or _remove or _clear:
        def _noop_scan(*_a, **_k):
            return None

        handle_active_watchlist_actions(
            view_watchlist=False, run_watchlist=False, clear_watchlist=_clear,
            add_symbol=_add, remove_symbol=_remove, symbol=_sym, username=_username,
            do_scan=_noop_scan, banner=lambda msg, kind="info": st.toast(msg),
            scan_all=bool(_scan_all),
        )
        st.rerun()

    # "Run Watchlist Scan" / "View as table" need the scanner's results pipeline,
    # which lives on the main page. Hand off via a durable flag and switch there
    # so the click actually does something instead of silently no-op'ing.
    if _run or _view:
        st.session_state["_wl_pending_scan"] = "run" if _run else "view"
        st.session_state["_wl_pending_scan_all"] = bool(_scan_all)
        st.switch_page("app.py")


try:
    from ui.alerts_page import render_alerts_body, session_watch_tickers
    from ui.header import render_page_logo

    st.session_state["hsf_my_watchlist_viewed"] = True
    render_page_logo()
    st.title("My Stocks")
    _tab_watch, _tab_alerts = st.tabs(["📋 Watchlist", "🔔 Alerts"])
    with _tab_watch:
        _render_watchlist_tab()
    with _tab_alerts:
        render_alerts_body(_username, watch_tickers=session_watch_tickers())
except Exception as e:
    from ui.safe_errors import show_error

    show_error("your watchlists", e)
