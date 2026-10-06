"""🧪 Custom Scan — run your own scan with your own filters.

The Scanner shows the latest automatic full-market scan; this page holds the
filters and scan buttons (owner, 2026-10-06). A finished scan opens the Scanner
with its results.
"""
from __future__ import annotations

import streamlit as st

from ui.showcase import initial_sidebar_state

st.set_page_config(page_title="Custom Scan", page_icon="🧪", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # before any sign-in gate

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to run a custom scan.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.nav import render_sidebar_nav

    render_sidebar_nav()
except Exception:
    pass

_flags = st.session_state.get("entitlements")
_tier = st.session_state.get("tier")
if not _flags or _tier is None:  # plan not resolved in this session yet (the Scanner does it)
    st.info("Open the Scanner once to load your plan, then come back to run a custom scan.")
    st.page_link("app.py", label="Open the Scanner", icon="🔎")
    st.stop()

try:
    from ui.design_system import render_page_header
    from ui.header import render_page_logo

    render_page_logo()
    render_page_header("Custom Scan", "Pick a market and your own filters. Results open on the Scanner.")
except Exception:
    st.title("Custom Scan")

st.page_link("app.py", label="← Latest market scan (Scanner)", icon="🔎")

try:
    from ui.custom_scan import render_custom_scan

    render_custom_scan(_username, _tier, dict(_flags))
except Exception as e:
    if any(c.__name__ == "ScriptControlException" for c in type(e).__mro__):
        raise  # Streamlit control flow (st.stop / st.rerun / st.switch_page)
    from ui.safe_errors import show_error

    show_error("Custom Scan", e)

try:
    from ui.footer import render_footer

    render_footer()
except Exception:
    pass
