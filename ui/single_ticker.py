"""Single-ticker search, chart, and scan controls."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import streamlit as st

from db.watchlists import get_watchlist_tickers, set_watchlist_tickers

BannerFn = Callable[[str, str], None]
ScanFn = Callable[[list[str], str], None]


@st.cache_data(ttl=60)
def get_live_quote(ticker: str) -> Optional[float]:
    """Best-effort live quote for a single ticker.

    Alpaca-first (centralized batched snapshot), yfinance only as a fallback.
    """
    if not ticker:
        return None
    sym = str(ticker).strip().upper()

    # --- Alpaca (preferred) ---
    try:
        from market_data import get_latest_quotes

        q = (get_latest_quotes([sym]) or {}).get(sym) or {}
        last = q.get("last")
        if last is not None:
            return float(last)
    except Exception:
        pass

    # --- yfinance fallback ---
    try:
        import yfinance as yf  # type: ignore

        t = yf.Ticker(sym)
        fast_info = getattr(t, "fast_info", None)
        if fast_info is not None:
            last_price = getattr(fast_info, "last_price", None)
            if last_price is not None:
                return float(last_price)
        hist = t.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None

    return None


def render_single_symbol_chart(symbol: str, days: int = 90) -> None:
    """Render a small price chart for a single symbol."""
    if not symbol:
        return

    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        st.info("Charting library (plotly) is not available.")
        return

    sym = str(symbol).strip().upper()
    if not sym:
        return

    # Centralized OHLC fetch (Alpaca-first, yfinance fallback) — same source the
    # rest of the app's charts use.
    try:
        from ui.charts import _fetch_unadjusted_ohlc

        hist = _fetch_unadjusted_ohlc(sym, period="6mo", interval="1d")
    except Exception as e:
        st.error(f"Failed to load price history for {sym}: {e}")
        return

    if hist is None or hist.empty:
        st.warning(
            f"No price history returned for {sym} over the last 6 months. "
            "Market data may be unavailable for this symbol."
        )
        return

    price_series = None
    if "Close" in hist.columns:
        price_series = hist["Close"].dropna()
    elif "Adj Close" in hist.columns:
        price_series = hist["Adj Close"].dropna()

    if price_series is None or price_series.empty:
        st.warning(
            f"Downloaded history for {sym} has no usable Close/Adj Close prices; "
            "cannot render chart."
        )
        try:
            st.caption("Raw history (tail):")
            st.dataframe(hist.tail(), width="stretch")
        except Exception:
            pass
        return

    if days is not None and days > 0 and price_series.shape[0] > days:
        price_series = price_series.iloc[-days:]

    fig = go.Figure()
    added_candles = False
    try:
        required_ohlc = {"Open", "High", "Low", "Close"}
        if required_ohlc.issubset(set(hist.columns)):
            ohlc = hist.loc[price_series.index]
            ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"], how="any")
            if not ohlc.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=ohlc.index,
                        open=ohlc["Open"],
                        high=ohlc["High"],
                        low=ohlc["Low"],
                        close=ohlc["Close"],
                        name="Price",
                    )
                )
                added_candles = True
    except Exception:
        added_candles = False

    if not added_candles:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                name="Price",
            )
        )

    try:
        ma20 = price_series.rolling(window=20).mean()
        ma50 = price_series.rolling(window=50).mean()
        if ma20.dropna().shape[0] >= 5:
            fig.add_trace(go.Scatter(x=ma20.index, y=ma20.values, mode="lines", name="MA20"))
        if ma50.dropna().shape[0] >= 5:
            fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines", name="MA50"))
    except Exception:
        pass

    fig.update_layout(
        title=f"{sym} - last {min(days, price_series.shape[0])} trading days",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    try:
        st.plotly_chart(fig, width="stretch", key=f"single_ticker_chart_{_safe_chart_key(sym)}")
    except Exception as e:
        st.warning(f"Failed to render chart for {sym} due to an internal plotting error: {e}")
        try:
            st.caption("Raw Close-series data (tail):")
            st.dataframe(price_series.to_frame(name="Close").tail(), width="stretch")
        except Exception:
            pass


def _safe_chart_key(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().upper())
    return safe or "symbol"


def render_single_ticker_panel() -> tuple[str, bool, bool]:
    """Render single-ticker controls and return ticker/chart/scan state."""
    st.markdown("### 🔍 Search & Scan Single Ticker")
    st.caption("Enter a symbol, view its chart, and optionally run a focused breakout scan.")

    c1, c2 = st.columns([3, 1])
    with c1:
        search_ticker = st.text_input(
            "Ticker symbol",
            key="single_search_ticker",
            placeholder="AAPL",
            label_visibility="collapsed",
        )
    with c2:
        show_chart_btn = st.button(
            "Show Chart 📈",
            key="single_show_chart_btn",
            width="stretch",
        )

    c3, c4 = st.columns([2, 2])
    with c3:
        st.checkbox(
            "Add to active watchlist",
            value=False,
            key="single_search_add_to_watchlist",
            help="If enabled, the searched ticker is added to your active watchlist when you run a scan.",
        )
    with c4:
        run_single_scan_btn = st.button(
            "Run Single-Ticker Scan 💸",
            key="single_search_scan_btn",
            width="stretch",
        )

    normalized_ticker = (search_ticker or "").strip().upper()
    if normalized_ticker:
        quote = get_live_quote(normalized_ticker)
        if quote is not None:
            st.caption(f"{normalized_ticker} ~ ${quote:.2f}")
        else:
            st.caption(f"{normalized_ticker}: live quote unavailable.")

    return normalized_ticker, bool(show_chart_btn), bool(run_single_scan_btn)


def handle_single_ticker_actions(
    *,
    ticker: str,
    show_chart: bool,
    run_scan: bool,
    username: str,
    do_scan: ScanFn,
    banner: BannerFn,
) -> None:
    """Handle chart and scan actions for the single-ticker panel."""
    if show_chart:
        if not ticker:
            banner("Please enter a ticker symbol to chart.", "warning")
        else:
            st.markdown("### 📈 Price chart")
            render_single_symbol_chart(ticker)

    if not run_scan:
        return

    if not ticker:
        banner("Please enter a ticker symbol to scan.", "warning")
        return

    add_to_watchlist = st.session_state.get("single_search_add_to_watchlist", True)
    if add_to_watchlist:
        active_watchlist_id = st.session_state.get("active_watchlist_id")
        if active_watchlist_id is not None:
            try:
                existing = get_watchlist_tickers(active_watchlist_id, username) or []
                norm_existing = {str(t).strip().upper() for t in existing}
                if ticker not in norm_existing:
                    updated = sorted(norm_existing | {ticker})
                    set_watchlist_tickers(active_watchlist_id, username, list(updated))
                    st.caption(f"Added {ticker} to your active watchlist.")
            except Exception:
                pass

    do_scan([ticker], f"Search: {ticker}")
