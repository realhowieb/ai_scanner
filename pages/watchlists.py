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
except Exception as e:
    st.error("Watchlists failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
