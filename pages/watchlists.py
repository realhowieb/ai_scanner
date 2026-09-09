"""📋 Watchlists — create, organize, and quote your watchlists.

Promotes watchlist management to its own page. Reuses
ui.watchlists.render_watchlists_panel.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Watchlists", page_icon="📋", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage your watchlists.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

st.markdown("## 📋 Watchlists")
st.caption("Create and organize watchlists; your active list feeds the scanner, "
           "the Day Trader monitor, and the Market Brief.")

try:
    from ui.header import render_page_logo
    from ui.watchlists import render_watchlists_panel

    render_page_logo()
    render_watchlists_panel(_username)

    from ui.watchlists import handle_active_watchlist_actions

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
except Exception as e:
    st.error("Watchlists failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
