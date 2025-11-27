from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from db.watchlists import (
    list_watchlists,
    create_watchlist,
    delete_watchlist,
    get_watchlist_tickers,
    set_watchlist_tickers,
)


def render_watchlists_panel(user_id: str) -> Tuple[Optional[int], List[str]]:
    """Render the 'My Watchlists' block in the sidebar.

    Returns:
        (active_watchlist_id, active_watchlist_tickers)
    """
    st.sidebar.markdown("### 📋 My Watchlists")

    try:
        watchlists = list_watchlists(user_id)
    except Exception as e:
        st.sidebar.caption("Watchlists require Neon DB (cloud) and may be unavailable.")
        # Show the underlying error in dev so we can diagnose connection/config issues.
        with st.sidebar.expander("Watchlist error details", expanded=False):
            st.code(str(e) or repr(e))
        # For safety, don't crash the app; just return empty.
        return None, []

    active_id: Optional[int] = None
    active_tickers: List[str] = []

    if watchlists:
        options = {f"{wl['name']} (#{wl['id']})": wl for wl in watchlists}
        selected_label = st.sidebar.selectbox(
            "Active watchlist",
            list(options.keys()),
            index=0,
        )
        active = options[selected_label]
        active_id = active["id"]
        active_tickers = get_watchlist_tickers(active_id, user_id)
    else:
        st.sidebar.caption("No watchlists yet. Create one below.")

    with st.sidebar.expander("Manage watchlists", expanded=False):
        new_name = st.text_input("New watchlist name", key="wl_new_name")
        if st.button("Create watchlist", key="wl_create_btn"):
            if new_name.strip():
                create_watchlist(user_id, new_name.strip())
                st.rerun()
            else:
                st.warning("Please enter a name for your watchlist.")

        if active_id is not None:
            if st.button("Delete active watchlist", key="wl_delete_btn"):
                delete_watchlist(active_id, user_id)
                st.rerun()

            tickers_str = ",".join(active_tickers)
            edited = st.text_area(
                "Tickers (comma-separated)",
                value=tickers_str,
                key="wl_tickers_edit",
                help="Example: AAPL, TSLA, NVDA",
            )
            if st.button("Save tickers", key="wl_save_tickers"):
                tickers = [
                    t.strip().upper()
                    for t in edited.split(",")
                    if t.strip()
                ]
                set_watchlist_tickers(active_id, user_id, tickers)
                st.success("Watchlist updated.")
                st.rerun()
        else:
            st.caption("Create a watchlist to add tickers.")

    return active_id, active_tickers