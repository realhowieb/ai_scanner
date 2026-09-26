"""🔔 Alerts: dedicated page for alert management.

Create/manage alerts, outcome scorecards, and the triggered feed live here;
the main page keeps the lightweight notification bell and a link. In-context
creation survives via the "Alert me on this" quick action on result details,
which pre-fills the price-alert form before this page renders it.
"""
from __future__ import annotations

import streamlit as st

from ui.design_system import render_page_header
from ui.showcase import initial_sidebar_state

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage alerts.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()


from ui.alerts_page import render_alerts_body  # noqa: E402
from ui.alerts_page import session_watch_tickers as _session_watch_tickers  # noqa: E402

try:
    from ui.header import render_page_logo
    from ui.onboarding import render_alerts_orientation

    render_page_logo()
    render_page_header("Alerts", "Meaningful intelligence changes for watched stocks.")
    render_alerts_orientation(_username)
except Exception as e:
    from ui.safe_errors import show_error

    show_error("the alerts page", e)
render_alerts_body(_username, watch_tickers=_session_watch_tickers())
st.caption("Alerts also live on **My Stocks**, next to your watchlists.")
st.page_link("pages/watchlists.py", label="Open My Stocks", icon="📋")
st.page_link("app.py", label="← Back to scanner", icon="🏠")
