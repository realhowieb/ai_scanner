"""📓 Journal — your trades, live P&L, and closed-trade stats.

Promotes the trade journal (previously buried at the bottom of the scanner) to
its own page. Reuses ui.journal.render_journal_panel.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Journal", page_icon="📓", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to see your journal.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

st.markdown("## 📓 Trade Journal")
st.caption("Positions logged from trade plans, marked to live quotes, with "
           "closed-trade win-rate and return stats.")

try:
    from ui.header import render_page_logo
    from ui.journal import render_journal_panel

    render_page_logo()

    has_trades = False
    try:
        from db.trades import list_trades

        has_trades = bool(list_trades(_username))
    except Exception:
        has_trades = False

    if has_trades:
        render_journal_panel(_username)
    else:
        st.info(
            "Your journal is empty. Use **📓 Log this trade** on a scan result's "
            "trade plan (or **💹 Paper Trade This Setup**) to start tracking "
            "positions here."
        )
except Exception as e:
    st.error("Journal failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
