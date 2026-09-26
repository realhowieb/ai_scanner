"""📅 Today — the daily entry point (P1-5)."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Today · HSF AI Stock Scanner", page_icon="📅", layout="centered")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to see Today.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.header import render_page_logo
    from ui.today import render_today

    render_page_logo()
    render_today(_username)
except Exception as e:
    from ui.safe_errors import show_error

    show_error("Today", e)
