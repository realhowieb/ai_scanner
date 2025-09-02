import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
from pathlib import Path

st.set_page_config(page_title="AI Stock Scanner", layout="wide")
st.title("AI Stock Scanner — Penny & Breakout Scanner")

# ---------------------- Load Universe ---------------------- #
@st.cache_data(ttl=3600)
def load_universe(file_path="universe.txt"):
    p = Path(file_path)
    if not p.exists():
        st.error(f"Missing {file_path}. Please create it with tickers.")
        return []
    return [line.strip().upper() for line in p.read_text().splitlines() if line.strip()]

tickers = load_universe()
st.sidebar.write(f"Loaded {len(tickers)} tickers from universe.txt")

# ---------------------- Helper Functions ---------------------- #
def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def sma(series, window):
    return series.rolling(window).mean()

def atr14(high, low, close):
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1)
    tr_max = tr.max(axis=1)
    return tr_max.rolling(14).mean().iloc[-1]

@st.cache_data(ttl=3600)
def fetch_history(tickers, period_days=60):
    hist = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(" ".join(batch), period=f"{period_days}d", interval="1d", auto_adjust=False, threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    sub = data.xs(t, axis=1, level=1, drop_level=False).droplevel(1, axis=1)
                    if not sub.empty: hist[t] = sub.dropna()
            else:
                t = batch[0]
                if not data.empty: hist[t] = data.dropna()
        except Exception:
            continue
    return hist

def latest_close_pct_change(df):
    if len(df) < 2: return np.nan
    prev_close = df["Close"].iloc[-2]
    last_close = df["Close"].iloc[-1]
    if prev_close == 0 or math.isnan(prev_close) or math.isnan(last_close): return np.nan
    return (last_close / prev_close - 1.0) * 100.0

# ---------------------- Scanners ---------------------- #
def scan_penny_runners(hist, min_price=0.5, max_price=5.0, min_change_pct=5.0):
    rows = []
    for t, df in hist.items():
        try:
            price = df["Close"].iloc[-1]
            if not (min_price <= price <= max_price): continue
            chg = latest_close_pct_change(df)
            if pd.isna(chg) or chg < min_change_pct: continue
            rvol = df["Volume"].iloc[-1]/df["Volume"].iloc[-10:].mean()
            rows.append({"Ticker": t, "Price": round(price,2), "%Chg_d": round(chg,2), "RVOL10": round(rvol,2)})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    return out.sort_values(by=["%Chg_d","RVOL10"], ascending=[False,False]).reset_index(drop=True) if not out.empty else out

def scan_breakout(hist, min_price=5.0, max_price=50.0, rsi_min=55, rsi_max=70):
    rows = []
    for t, df in hist.items():
        if len(df) < 25: continue
        price = df["Close"].iloc[-1]
        if not (min_price <= price <= max_price): continue
        rsi_val = rsi(df["Close"]).iloc[-1]
        if not (rsi_min <= rsi_val <= rsi_max): continue
        ma9 = sma(df["Close"],9).iloc[-1]
        ma21 = sma(df["Close"],21).iloc[-1]
        cond_ma9 = price > ma9
        cond_ma21 = price > ma21
        if not cond_ma9 or not cond_ma21: continue
        atr = atr14(df["High"], df["Low"], df["Close"])
        # Trend label
        trend = "Strong Up" if rsi_val>=67 else "Up" if rsi_val>=60 else "Neutral" if rsi_val>=50 else "Down"
        rows.append({"Ticker": t, "Price": round(price,2), "RSI14": round(rsi_val,2), "ATR14": round(atr,2), "Trend": trend})
    out = pd.DataFrame(rows)
    return out

# ---------------------- Streamlit UI ---------------------- #
st.sidebar.header("Scanner Settings")
scanner_type = st.sidebar.radio("Select scanner type:", ["Penny Runners","Breakout Momentum"])
min_price = st.sidebar.number_input("Min Price", 0.1, 100.0, 0.5)
max_price = st.sidebar.number_input("Max Price", 0.5, 100.0, 50.0)
days_history = st.sidebar.slider("History Days", 20, 120, 60)

if st.button("Run Scanner"):
    st.info(f"Fetching {days_history}d historical data for {len(tickers)} tickers...")
    hist = fetch_history(tickers, days_history)
    st.success("Data fetched!")

    if scanner_type == "Penny Runners":
        df = scan_penny_runners(hist, min_price, max_price)
    else:
        df = scan_breakout(hist, min_price, max_price)

    st.dataframe(df)
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "scanner_results.csv", "text/csv")

        # Display line charts for top 5 tickers
        st.subheader("Top 5 ticker charts")
        for t in df["Ticker"].head(5):
            st.write(t)
            st.line_chart(hist[t]["Close"])