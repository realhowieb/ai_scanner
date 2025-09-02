import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
from pathlib import Path
from io import StringIO
import requests

st.set_page_config(page_title="AI Stock Scanner", layout="wide")
st.title("AI Stock Scanner — Breakout Scanner")

# ---------------------- Load Universe ---------------------- #
@st.cache_data(ttl=3600)
def load_universe(file_path="universe.txt", top_n_nasdaq=100, top_n_sp500=500):
    from pathlib import Path
    import pandas as pd
    import requests

    p = Path(file_path)
    tickers = []

    try:
        # --- S&P 500 ---
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url_sp500, headers=headers)
        tables = pd.read_html(response.text)
        sp500_df = tables[0]
        sp500_tickers = sp500_df['Symbol'].tolist()[:top_n_sp500]

        # --- NASDAQ 100 ---
        url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
        response = requests.get(url_nasdaq, headers=headers)
        tables = pd.read_html(response.text)
        nasdaq_df = tables[3]  # usually the 4th table
        nasdaq_tickers = nasdaq_df['Ticker'].tolist()[:top_n_nasdaq]

        # Combine and remove duplicates
        tickers = list(dict.fromkeys([t.upper().replace('.', '-') for t in nasdaq_tickers + sp500_tickers]))
        p.write_text("\n".join(tickers))
        st.info(f"Auto-populated universe.txt with {len(tickers)} tickers (NASDAQ 100 + S&P 500)")
        return tickers

    except Exception as e:
        st.warning(f"Failed to auto-populate tickers: {e}")
        if p.exists():
            return [line.strip().upper() for line in p.read_text().splitlines() if line.strip()]
        return []

tickers = load_universe()
tickers = [t.replace('.', '-') for t in tickers]  # Yahoo Finance format
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
def fetch_history_safe(tickers, period_days=60):
    """
    Fetch historical data safely, skipping tickers with missing/delisted data.
    """
    hist = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=f"{period_days}d", interval="1d",
                               auto_adjust=False, threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t not in data['Close']:
                        st.warning(f"Skipped {t}: no data")
                        continue
                    sub = pd.DataFrame({
                        'Open': data['Open'][t],
                        'High': data['High'][t],
                        'Low': data['Low'][t],
                        'Close': data['Close'][t],
                        'Volume': data['Volume'][t]
                    }).dropna()
                    if not sub.empty:
                        hist[t] = sub
            else:
                for t in batch:
                    sub = data.dropna()
                    if not sub.empty:
                        hist[t] = sub
        except Exception as e:
            st.warning(f"Error fetching batch {batch}: {e}")
            continue
    st.info(f"Fetched data for {len(hist)} / {len(tickers)} tickers")
    return hist

def latest_close_pct_change(df):
    if len(df) < 2: return np.nan
    prev_close = df["Close"].iloc[-2]
    last_close = df["Close"].iloc[-1]
    if prev_close == 0 or math.isnan(prev_close) or math.isnan(last_close): return np.nan
    return (last_close / prev_close - 1.0) * 100.0

# ---------------------- Scanners ---------------------- #
def scan_breakout(hist, min_price=5.0, max_price=1500.0, rsi_min=55, rsi_max=70):
    rows = []
    for t, df in hist.items():
        if len(df) < 25: continue
        price = float(df["Close"].iloc[-1])
        if not (min_price <= price <= max_price): continue
        rsi_val = rsi(df["Close"]).iloc[-1]
        if not (rsi_min <= rsi_val <= rsi_max): continue
        ma9 = sma(df["Close"],9).iloc[-1]
        ma21 = sma(df["Close"],21).iloc[-1]
        cond_ma9 = price > ma9
        cond_ma21 = price > ma21
        if not cond_ma9 or not cond_ma21: continue
        atr = atr14(df["High"], df["Low"], df["Close"])
        trend = "Strong Up" if rsi_val>=67 else "Up" if rsi_val>=60 else "Neutral"
        rows.append({"Ticker": t, "Price": round(price,2), "RSI14": round(rsi_val,2), "ATR14": round(atr,2), "Trend": trend})
    return pd.DataFrame(rows)

# ---------------------- Streamlit UI ---------------------- #
st.sidebar.header("Scanner Settings")
scanner_type = st.sidebar.radio("Select scanner type:", ["Breakout Momentum"])
min_price = st.sidebar.number_input("Min Price", 0.1, 100.0, 0.5)
max_price = st.sidebar.number_input("Max Price", 0.5, 5000.0, 1500.0)
days_history = st.sidebar.slider("History Days", 20, 120, 60)

if st.button("Run Scanner"):
    st.info(f"Fetching {days_history}d historical data for {len(tickers)} tickers...")
    hist = fetch_history_safe(tickers, days_history)
    st.success("Data fetched!")

    df = scan_breakout(hist, min_price, max_price)

    st.dataframe(df)
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "scanner_results.csv", "text/csv")

        st.subheader("Top 5 ticker charts")
        for t in df["Ticker"].head(5):
            st.write(t)
            st.line_chart(hist[t]["Close"])