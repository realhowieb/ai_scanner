"""🔬 HSF Stock Intelligence — canonical single-ticker deep dive.

Reads the selected ticker from session state (set when a user opens a name from
Market Brief / Scanner / Watchlist) or a manual ticker box, then renders the
read-only canonical intelligence view. No scan, no brief rebuild, no writes.
"""
from __future__ import annotations

import streamlit as st

from ui.showcase import initial_sidebar_state

st.set_page_config(page_title="Stock Intelligence", page_icon="🔬", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # Run 62/P2: before any sign-in gate

# P1-2: deep links (?ticker=NVDA) pre-select a ticker.
_qp_ticker = str(st.query_params.get("ticker") or "").strip().upper()
if _qp_ticker:
    if _qp_ticker.replace(".", "").replace("-", "").isalnum() and len(_qp_ticker) <= 10:
        st.session_state["hsf_stock_ticker"] = _qp_ticker
        st.session_state.pop("hsf_stock_ticker_input", None)
    st.query_params.pop("ticker", None)   # consume once, so typing a new ticker isn't overridden
_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    # P2-5: a shared link survives sign-in — the main page returns here after login.
    if st.session_state.get("hsf_stock_ticker"):
        st.session_state["hsf_after_login_page"] = "pages/stock.py"
    _shared = st.session_state.get("hsf_stock_ticker")
    st.info(f"Sign in to view HSF Stock Intelligence for {_shared}." if _shared
            else "Please log in on the main page to view HSF Stock Intelligence.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.nav import render_sidebar_nav

    render_sidebar_nav()
except Exception:
    pass

from ui.design_system import render_page_header

render_page_header("Stock Intelligence", "Understand the current HSF state of one ticker.")

_default = (st.session_state.get("hsf_stock_ticker") or "").strip().upper()
_ticker = st.text_input("Ticker", value=_default, placeholder="e.g. NVDA",
                        key="hsf_stock_ticker_input").strip().upper()
if _ticker:
    st.session_state["hsf_stock_ticker"] = _ticker
    st.session_state["hsf_stock_intelligence_viewed"] = True

if not _ticker:
    st.caption("Enter a ticker, or open one from Market Brief, Scanner, or your Watchlist.")
else:
    # Use the opportunity/row carried from the screen the user came from ONLY when
    # it matches this ticker. With no carried context (typed ticker, shared link),
    # Run 70 looks the ticker up in the latest scheduled full-market scan, so
    # Stock Intelligence agrees with Today and the Scanner (P0-9).
    from ui.stock_handoff import latest_market_context, session_context

    _opp, _row = session_context(_ticker)
    if _opp is None and _row is None:
        _opp, _row = latest_market_context(_ticker)
    try:
        from ui.charts import render_chart_for_ticker
        from ui.stock_intelligence import render_stock_intelligence

        render_stock_intelligence(
            _ticker, current_opp=_opp, current_row=_row, source="page",
            render_chart_for_ticker=lambda t: render_chart_for_ticker(t, key=f"si_page_chart_{t}"),
        )
    except Exception as e:
        from ui.safe_errors import show_error

        show_error("Stock Intelligence", e)

st.page_link("app.py", label="← Back to scanner", icon="🏠")
