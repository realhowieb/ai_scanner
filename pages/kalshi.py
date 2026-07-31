"""🪙 Kalshi BTC Scanner — dedicated page.

Read-only directional scanner for Kalshi's up/down BTC event contracts. Its own
page so it can auto-refresh independently of the stock scanner.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Kalshi BTC Scanner", page_icon="🪙", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to use the Kalshi BTC scanner.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.header import render_page_logo
    from ui.kalshi_scanner import render_kalshi_scanner

    render_page_logo()
    render_kalshi_scanner()
except Exception as e:
    st.error("Kalshi BTC scanner failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
