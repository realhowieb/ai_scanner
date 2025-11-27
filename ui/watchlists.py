from __future__ import annotations

import streamlit as st
import pandas as pd  # type: ignore

# ... (other existing imports and code)

def render_scan_controls():
    # ... (existing code before watchlist buttons)

    # Watchlist actions (uses active_watchlist_tickers from session_state)
    watchlist_tickers = st.session_state.get("active_watchlist_tickers", []) or []
    has_watchlist = isinstance(watchlist_tickers, list) and len(watchlist_tickers) > 0

    cw1, cw2, _ = st.columns([1, 1, 2])
    with cw1:
        view_watchlist_btn = st.button(
            "View Watchlist",
            key="btn_view_watchlist",
            use_container_width=True,
            disabled=not has_watchlist,
        )
    with cw2:
        run_watchlist_btn = st.button(
            "Run Watchlist Scan",
            key="btn_scan_watchlist",
            use_container_width=True,
            disabled=not has_watchlist,
        )

    st.caption("Use your active watchlist for viewing or scanning.")

    # ... (other existing code)

    if view_watchlist_btn:
        # Normalize and validate tickers from the active watchlist
        tickers = [
            str(t).strip().upper()
            for t in (st.session_state.get("active_watchlist_tickers") or [])
            if str(t).strip()
        ]
        if not tickers:
            _banner("Active watchlist has no tickers to view.", "warning")
        else:
            rows = []
            try:
                import yfinance as yf  # type: ignore
                for sym in tickers:
                    price = None
                    prev_close = None
                    try:
                        ticker_obj = yf.Ticker(sym)
                        fast_info = getattr(ticker_obj, "fast_info", None)
                        # Try to get last price and previous close from fast_info
                        if fast_info is not None:
                            price = getattr(fast_info, "last_price", None)
                            prev_close = getattr(fast_info, "previous_close", None)
                        # Fallbacks if fast_info is missing or incomplete
                        if price is None or prev_close is None:
                            hist = ticker_obj.history(period="2d")
                            if not hist.empty and "Close" in hist.columns:
                                closes = hist["Close"].tolist()
                                if len(closes) >= 1 and price is None:
                                    price = float(closes[-1])
                                if len(closes) >= 2 and prev_close is None:
                                    prev_close = float(closes[-2])
                    except Exception:
                        price = None
                        prev_close = None

                    change = None
                    change_pct = None
                    if price is not None and prev_close not in (None, 0):
                        change = float(price) - float(prev_close)
                        change_pct = (change / float(prev_close)) * 100.0

                    rows.append(
                        {
                            "Ticker": sym,
                            "Price": price,
                            "$ Change": change,
                            "% Change": change_pct,
                        }
                    )
            except Exception:
                # If yfinance or network is unavailable, still show tickers without prices
                rows = [
                    {
                        "Ticker": sym,
                        "Price": None,
                        "$ Change": None,
                        "% Change": None,
                    }
                    for sym in tickers
                ]

            df_view = pd.DataFrame(rows)
            st.session_state.results_df = df_view
            _banner(
                f"Showing active watchlist with {len(tickers)} tickers (with prices & daily change).",
                "info",
            )

    if run_watchlist_btn:
        # Normalize and validate tickers from the active watchlist
        tickers = [
            str(t).strip().upper()
            for t in (st.session_state.get("active_watchlist_tickers") or [])
            if str(t).strip()
        ]
        if not tickers:
            _banner("Active watchlist has no tickers to scan.", "warning")
        else:
            label = f"Watchlist ({len(tickers)} tickers)"
            do_scan(tickers, label)

    # ... (rest of existing render_scan_controls code)