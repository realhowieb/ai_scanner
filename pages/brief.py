"""📬 Market Brief — the morning-digest content, in the app.

Same Top gappers / Today's setups / PreBreakout picks the morning email sends,
on its own page so it's available any time — not just at 8am in your inbox.
"""
from __future__ import annotations

import streamlit as st

from ui.showcase import initial_sidebar_state

st.set_page_config(page_title="Market Brief", page_icon="📬", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to see your market brief.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.design_system import render_page_header
    from ui.header import render_page_logo
    from ui.market_brief import render_market_brief
    from ui.onboarding import render_market_brief_orientation

    render_page_logo()
    render_page_header("Market Brief", "What matters in the market right now.")
    from ui.trust_banner import render_trust_banner

    render_trust_banner()
    render_market_brief_orientation(_username)
    render_market_brief()
except Exception as e:
    from ui.safe_errors import show_error

    show_error("the market brief", e)

st.page_link("app.py", label="← Back to scanner", icon="🏠")
