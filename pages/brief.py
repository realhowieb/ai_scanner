"""📬 Market Brief — the morning-digest content, in the app.

Same Top gappers / Today's setups / PreBreakout picks the morning email sends,
on its own page so it's available any time — not just at 8am in your inbox.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Market Brief", page_icon="📬", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to see your market brief.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

st.markdown("## 📬 Market Brief")
st.caption("The same brief we email you each morning — top gappers, fresh setups, "
           "and PreBreakout picks from the latest scan.")

try:
    from ui.header import render_page_logo
    from ui.market_brief import render_market_brief

    render_page_logo()
    render_market_brief()
except Exception as e:
    st.error("Market brief failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
