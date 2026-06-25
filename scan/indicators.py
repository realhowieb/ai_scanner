# ai_scanner/scan/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# Helpers to normalize inputs
# -----------------------------
def _to_series_close(x: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Accept either a Series (already Close) or a DataFrame with a 'Close' column.
    Returns a float Series.
    """
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        if "Close" not in x.columns:
            raise KeyError("DataFrame must contain a 'Close' column")
        s = x["Close"]
    else:
        raise TypeError("Expected pandas Series or DataFrame")
    return pd.Series(s, copy=False).astype(float)


# -----------------------------
# Public indicator functions
# -----------------------------
def ema(data: pd.Series | pd.DataFrame, span: int = 14) -> pd.Series:
    """
    Exponential Moving Average of Close.
    Accepts a Series of prices or a DataFrame with 'Close'.
    """
    s = _to_series_close(data)
    return s.ewm(span=span, adjust=False).mean()


def rsi(data: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's RSI on Close.
    Accepts a Series of prices or a DataFrame with 'Close'.
    """
    s = _to_series_close(data)
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing via ewm with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd(data: pd.Series | pd.DataFrame):
    """
    MACD line, Signal line, and Histogram on Close.
    Returns tuple (macd_line, signal, hist).
    """
    s = _to_series_close(data)
    macd_line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range with Wilder smoothing.
    Requires columns: High, Low, Close.
    """
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise KeyError(f"ATR requires '{col}' column")

    high = pd.Series(df["High"], copy=False).astype(float)
    low = pd.Series(df["Low"], copy=False).astype(float)
    close = pd.Series(df["Close"], copy=False).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_wilder = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr_wilder


def rs_20d_vs_spy(df_ticker: pd.DataFrame, df_spy: pd.DataFrame) -> float:
    """
    Relative strength vs SPY over last ~20 trading days, in percent.
    Aligns on index, forward-fills, and returns the most recent (rt - rs)*100.
    Returns np.nan if insufficient data.
    """
    try:
        ct = pd.Series(df_ticker["Close"]).astype(float)
        cs = pd.Series(df_spy["Close"]).astype(float)
        idx = ct.index.union(cs.index)
        ct = ct.reindex(idx).ffill()
        cs = cs.reindex(idx).ffill()

        if ct.dropna().empty or cs.dropna().empty:
            return np.nan

        # try 20, then 10, then 5 to be more tolerant on short histories
        for lb in (20, 10, 5):
            if ct.notna().sum() >= lb + 1 and cs.notna().sum() >= lb + 1:
                rt = ct.pct_change(lb)
                rs = cs.pct_change(lb)
                # Use the last date common to the ticker dataframe
                last_date = df_ticker.index[-1]
                rel = (rt - rs).loc[:last_date].dropna()
                if not rel.empty:
                    return round(float(rel.iloc[-1] * 100), 2)
        return np.nan
    except Exception:
        return np.nan


# -----------------------------
# Inserted helper function for downcasting OHLCV DataFrames
def _downcast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast common OHLCV columns to smaller dtypes to save memory.
    - Float columns (Open, High, Low, Close, Adj Close) -> float32
    - Volume -> int32
    Returns a *copy* to avoid SettingWithCopy warnings.
    Safe to import as a utility from other modules.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    out = df.copy()

    float_cols = ["Open", "High", "Low", "Close", "Adj Close", "AdjClose", "Adj_Close"]
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").astype("int32")

    return out

# -----------------------------
# Back-compat wrapper names
# -----------------------------
def calc_ema(data: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series:
    return ema(data, span=period)


def calc_rsi(data: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series:
    return rsi(data, period=period)


def calc_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr(data, period=period)
