"""🔬 HSF Stock Intelligence — canonical single-ticker deep dive.

Reads the selected ticker from session state (set when a user opens a name from
Market Brief / Scanner / Watchlist) or a manual ticker box, then renders the
read-only canonical intelligence view. No scan, no brief rebuild, no writes.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Stock Intelligence", page_icon="🔬", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to view HSF Stock Intelligence.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.nav import render_sidebar_nav

    render_sidebar_nav()
except Exception:
    pass

from ui.design_system import render_page_header

render_page_header("Stock Intelligence", "Understand the current HSF state of one ticker.")
try:
    from ui.onboarding import render_stock_intelligence_orientation

    render_stock_intelligence_orientation(_username)
except Exception:
    pass

_default = (st.session_state.get("hsf_stock_ticker") or "").strip().upper()
_ticker = st.text_input("Ticker", value=_default, placeholder="e.g. NVDA",
                        key="hsf_stock_ticker_input").strip().upper()
if _ticker:
    st.session_state["hsf_stock_ticker"] = _ticker
    st.session_state["hsf_stock_intelligence_viewed"] = True

if not _ticker:
    st.caption("Enter a ticker, or open one from Market Brief, Scanner, or your Watchlist.")
else:
    # Use the live opportunity carried from Scanner/Brief ONLY when it matches
    # the selected ticker — hard guard against a stale row from another ticker.
    _opp = st.session_state.get("hsf_stock_opp")
    if not (_opp and str(_opp.get("ticker") or "").strip().upper() == _ticker):
        _opp = None
    try:
        from ui.charts import render_chart_for_ticker
        from ui.stock_intelligence import render_stock_intelligence

        render_stock_intelligence(
            _ticker, current_opp=_opp, source="page",
            render_chart_for_ticker=lambda t: render_chart_for_ticker(t, key=f"si_page_chart_{t}"),
        )
    except Exception as e:
        st.error("Stock Intelligence failed to load.")
        st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
