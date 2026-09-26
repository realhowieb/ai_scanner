"""Run 70 (P0-9) — hand a ticker to Stock Intelligence with the scan it came from.

Every "Open in Stock Intelligence" path passes the CANONICAL opportunity for
that ticker from the scan the user was looking at (the same
consolidate_scanner_results path as the Scanner's HSF Score column and Today's
top setups), plus the raw row. Stock Intelligence then shows the same HSF
Score, status and setup as the screen the user came from.

A ticker opened with no scan context (typed in, or a shared link) is looked up
in the latest scheduled full-market scan before falling back to history.
Read-only; no scoring change.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

OPP_KEY = "hsf_stock_opp"
ROW_KEY = "hsf_stock_row"
TICKER_KEY = "hsf_stock_ticker"


def _norm(t: Any) -> str:
    return str(t or "").strip().upper()


def row_for(ticker: str, scan_df: Any) -> Optional[Dict[str, Any]]:
    """The first row for `ticker` in a results frame, as a plain dict."""
    if scan_df is None or getattr(scan_df, "empty", True):
        return None
    col = "Ticker" if "Ticker" in scan_df.columns else ("Symbol" if "Symbol" in scan_df.columns else None)
    if col is None:
        return None
    t = _norm(ticker)
    for rec in scan_df.to_dict(orient="records"):
        if _norm(rec.get(col)) == t:
            return rec
    return None


def canonical_opportunity(ticker: str, scan_df: Any) -> Optional[Dict[str, Any]]:
    """The ticker's HSF opportunity in `scan_df` via the canonical consolidation
    (None when the name doesn't qualify, exactly as in the Scanner column)."""
    if scan_df is None or getattr(scan_df, "empty", True):
        return None
    from ui.results_intelligence import consolidate_scanner_results

    t = _norm(ticker)
    for o in consolidate_scanner_results(scan_df.to_dict(orient="records"), top_n=None):
        if o.get("ticker") == t:
            return o
    return None


def handoff_state(ticker: str, *, scan_df: Any = None,
                  opp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Session values for opening `ticker` (testable without Streamlit)."""
    t = _norm(ticker)
    return {
        TICKER_KEY: t,
        OPP_KEY: opp if opp is not None else canonical_opportunity(t, scan_df),
        ROW_KEY: row_for(t, scan_df),
    }


def open_in_stock_intelligence(ticker: str, *, scan_df: Any = None,
                               opp: Optional[Dict[str, Any]] = None) -> None:
    """Set the hand-off state and switch to Stock Intelligence."""
    if st is None:
        return
    for k, v in handoff_state(ticker, scan_df=scan_df, opp=opp).items():
        if v is None:
            st.session_state.pop(k, None)
        else:
            st.session_state[k] = v
    st.switch_page("pages/stock.py")


def latest_market_context(ticker: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(opportunity, row) for `ticker` from the latest scheduled full-market scan."""
    try:
        from ui.market_scans import safe_recent_runs, safe_run_df

        runs = safe_recent_runs()
        df = safe_run_df(runs[0]["id"]) if runs else None
        return canonical_opportunity(ticker, df), row_for(ticker, df)
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("latest market context", exc)
        return None, None


def session_context(ticker: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(opportunity, row) carried in session for `ticker`, only if they match it
    (hard guard against a stale value from another ticker)."""
    if st is None:
        return None, None
    t = _norm(ticker)
    opp = st.session_state.get(OPP_KEY)
    row = st.session_state.get(ROW_KEY)
    opp = opp if isinstance(opp, dict) and _norm(opp.get("ticker")) == t else None
    row_t = _norm((row or {}).get("Ticker") or (row or {}).get("Symbol")) if isinstance(row, dict) else ""
    row = row if row_t == t else None
    return opp, row
