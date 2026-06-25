from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None


# ---------- OHLC fetch helper ----------

def _fetch_unadjusted_ohlc(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch recent OHLC data for a single ticker.

    This is intentionally self-contained so charts.py does not depend on
    the rest of the app. If yfinance is unavailable, returns an empty
    DataFrame and callers should handle that gracefully.
    """
    if yf is None or not ticker:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalise columns in case of MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Assume (field,) style for single-ticker
        df.columns = [c[0] for c in df.columns]

    return df


# ---------- Built‑in candlestick renderer ----------

def _render_builtin_candlestick(
    df: pd.DataFrame,
    ticker: str,
    *,
    height: int = 420,
    show_ma: bool = True,
    key: str | None = None,
) -> None:
    """Render a simple candlestick chart using Plotly if available.

    Falls back to a basic line chart if Plotly is not installed.
    """
    # Defensive: ensure we have a proper DataFrame copy
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No chart data available for this symbol.")
        return

    df = df.copy()

    # Ensure column names are unique; drop duplicate labels and keep first
    try:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    except Exception:
        # If anything goes wrong, leave df as-is; Plotly will still raise a clear error
        pass

    # Ensure required OHLC columns exist
    required_ohlc = {"Open", "High", "Low", "Close"}
    if "Close" not in df.columns:
        st.info("Price data missing 'Close' column for this symbol.")
        return

    if go is None:
        # Fallback: simple line chart
        st.line_chart(df["Close"])
        return

    fig = go.Figure()

    if required_ohlc.issubset(df.columns):
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"),
        )

    if show_ma and len(df) >= 10:
        ma20 = df["Close"].rolling(window=20).mean()
        ma50 = df["Close"].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=ma20, mode="lines", name="MA20"),
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=ma50, mode="lines", name="MA50"),
        )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=20),
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        title=f"{ticker} – Daily Chart",
    )

    st.plotly_chart(fig, width="stretch", key=key)


# ---------- Public chart API ----------

def render_chart_for_ticker(
    ticker: str,
    *,
    period: str = "3mo",
    interval: str = "1d",
    height: int = 420,
    show_ma: bool = True,
    key: str | None = None,
) -> None:
    """High‑level chart function used by the main app.

    This keeps all charting concerns inside charts.py so that app.py only
    needs to call a single function.
    """
    if not ticker:
        st.info("No symbol selected for chart.")
        return

    df = _fetch_unadjusted_ohlc(ticker, period=period, interval=interval)
    if df is None or df.empty:
        st.info("No price history available for this symbol.")
        return

    _render_builtin_candlestick(df, ticker, height=height, show_ma=show_ma, key=key)
