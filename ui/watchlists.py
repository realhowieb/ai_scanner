from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from db.watchlists import (
    list_watchlists,
    create_watchlist,
    delete_watchlist,
    get_watchlist_tickers,
    set_watchlist_tickers,
)


def render_watchlists_panel(user_id: str) -> Tuple[Optional[int], List[str]]:
    """Render the 'My Watchlists' block in the sidebar.

    Returns:
        (active_watchlist_id, active_watchlist_tickers)
    """
    st.sidebar.markdown("### 📋 My Watchlists")

    try:
        watchlists = list_watchlists(user_id)
    except Exception as e:
        st.sidebar.caption("Watchlists require Neon DB (cloud) and may be unavailable.")
        # Show the underlying error in dev so we can diagnose connection/config issues.
        with st.sidebar.expander("Watchlist error details", expanded=False):
            # Include exception type, message, and repr for better debugging
            st.code(f"{type(e)}\n{str(e)}\n{repr(e)}")
        # For safety, don't crash the app; just return empty.
        return None, []

    active_id: Optional[int] = None
    active_tickers: List[str] = []

    if watchlists:
        options = {f"{wl['name']} (#{wl['id']})": wl for wl in watchlists}
        selected_label = st.sidebar.selectbox(
            "Active watchlist",
            list(options.keys()),
            index=0,
        )
        active = options[selected_label]
        active_id = active["id"]
        active_tickers = get_watchlist_tickers(active_id, user_id)
    else:
        st.sidebar.caption("No watchlists yet. Create one below.")

    # If there is an active watchlist with tickers, show a simple table with prices & changes
    if active_id is not None and active_tickers:
        with st.sidebar.expander("View active watchlist (with prices)", expanded=False):
            rows = []
            try:
                import yfinance as yf  # type: ignore
                for t in active_tickers:
                    sym = str(t).strip().upper()
                    price = None
                    prev_close = None
                    try:
                        ticker_obj = yf.Ticker(sym)
                        fast_info = getattr(ticker_obj, "fast_info", None)

                        # Try to get last price
                        if fast_info is not None:
                            price = getattr(fast_info, "last_price", None)
                            prev_close = getattr(fast_info, "previous_close", None)

                        # Fallbacks if fast_info is missing or incomplete
                        if price is None or prev_close is None:
                            hist = ticker_obj.history(period="2d")
                            if not hist.empty and "Close" in hist.columns:
                                closes = hist["Close"].tolist()
                                # Last close is current, prior close is previous bar if it exists
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
                        "Ticker": str(t).strip().upper(),
                        "Price": None,
                        "$ Change": None,
                        "% Change": None,
                    }
                    for t in active_tickers
                ]

            if pd is not None:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                # Fallback: simple text listing
                for row in rows:
                    price = row.get("Price")
                    change = row.get("$ Change")
                    change_pct = row.get("% Change")
                    if price is not None:
                        txt = f"{row['Ticker']}: {price:.2f}"
                        if change is not None and change_pct is not None:
                            txt += f" ({change:+.2f}, {change_pct:+.2f}%)"
                    else:
                        txt = f"{row['Ticker']}: —"
                    st.write(txt)

    with st.sidebar.expander("Manage watchlists", expanded=False):
        new_name = st.text_input("New watchlist name", key="wl_new_name")
        if st.button("Create watchlist", key="wl_create_btn"):
            if new_name.strip():
                create_watchlist(user_id, new_name.strip())
                st.rerun()
            else:
                st.warning("Please enter a name for your watchlist.")

        if active_id is not None:
            if st.button("Delete active watchlist", key="wl_delete_btn"):
                delete_watchlist(active_id, user_id)
                st.rerun()

            tickers_str = ",".join(active_tickers)
            edited = st.text_area(
                "Tickers (comma-separated)",
                value=tickers_str,
                key="wl_tickers_edit",
                help="Example: AAPL, TSLA, NVDA",
            )
            if st.button("Save tickers", key="wl_save_tickers"):
                tickers = [
                    t.strip().upper()
                    for t in edited.split(",")
                    if t.strip()
                ]
                set_watchlist_tickers(active_id, user_id, tickers)
                st.success("Watchlist updated.")
                st.rerun()
        else:
            st.caption("Create a watchlist to add tickers.")

    return active_id, active_tickers