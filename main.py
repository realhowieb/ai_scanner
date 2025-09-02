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
def load_universe(file_path="universe.txt", top_n_nasdaq=100, top_n_sp500=500, top_n_russell=2000, min_price=1.0, max_price=15.0):
    p = Path(file_path)
    tickers = []

    def get_tickers_from_wikipedia(url, top_n):
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            for table in tables:
                # Find a column with 'Symbol' or 'Ticker'
                ticker_col = None
                for col in table.columns:
                    if "Symbol" in col or "Ticker" in col:
                        ticker_col = col
                        break
                if ticker_col is not None:
                    tickers_list = table[ticker_col].dropna().astype(str).str.upper().str.replace('.', '-').tolist()
                    return tickers_list[:top_n]
            raise ValueError("Ticker column not found in tables")
        except Exception as e:
            return None

    def get_tickers_from_file(file_name, top_n):
        f = Path(file_name)
        if f.exists():
            lines = f.read_text().splitlines()
            tickers_list = [line.strip().upper().replace('.', '-') for line in lines if line.strip()]
            return tickers_list[:top_n]
        return []

    try:
        # --- S&P 500 ---
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_tickers = get_tickers_from_wikipedia(url_sp500, top_n_sp500)
        if sp500_tickers is None:
            sp500_tickers = get_tickers_from_file("sp500.txt", top_n_sp500)

        # --- NASDAQ 100 ---
        url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
        nasdaq_tickers = get_tickers_from_wikipedia(url_nasdaq, top_n_nasdaq)
        if nasdaq_tickers is None:
            nasdaq_tickers = get_tickers_from_file("nasdaq100.txt", top_n_nasdaq)

        # --- Russell 2000 ---
        url_russell = "https://en.wikipedia.org/wiki/Russell_2000_Index"
        russell_tickers = get_tickers_from_wikipedia(url_russell, top_n_russell)
        if russell_tickers is None:
            russell_tickers = get_tickers_from_file("russell2000.txt", top_n_russell)

        # Combine tickers from all indices
        combined_tickers = list(dict.fromkeys(nasdaq_tickers + sp500_tickers + russell_tickers))

        # Filter tickers by price using yfinance
        filtered_tickers = []
        batch_size = 50
        for i in range(0, len(combined_tickers), batch_size):
            batch = combined_tickers[i:i+batch_size]
            data = yf.download(batch, period="1d", interval="1d", progress=False, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close'].iloc[-1]
            else:
                close_prices = data['Close'].iloc[-1:]
                close_prices = close_prices.squeeze()
            for ticker in batch:
                price = close_prices.get(ticker, None)
                if price is not None and not pd.isna(price):
                    if min_price <= price <= max_price:
                        filtered_tickers.append(ticker)

        tickers = filtered_tickers
        p.write_text("\n".join(tickers))
        st.info(f"Auto-populated universe.txt with {len(tickers)} tickers (price between ${min_price} and ${max_price})")
        return tickers

    except Exception as e:
        st.warning(f"Failed to auto-populate tickers: {e}")
        if p.exists():
            return [line.strip().upper().replace('.', '-') for line in p.read_text().splitlines() if line.strip()]
        return []

# Add sidebar selection for index
scanner_index = st.sidebar.selectbox("Select Index to Scan:", ["NASDAQ 100", "S&P 500", "Russell 2000"])

# Load all tickers from universe.txt or fetch new
all_tickers = load_universe()

# Filter tickers by selected index
def filter_tickers_by_index(tickers, index_name):
    index_name = index_name.lower()
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if index_name == "s&p 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            df = tables[0]
            index_tickers = set(df['Symbol'].str.upper().str.replace('.', '-').tolist())
        elif index_name == "nasdaq 100":
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            nasdaq_df = tables[3]
            ticker_col = None
            for col in nasdaq_df.columns:
                if "Symbol" in col or "Ticker" in col:
                    ticker_col = col
                    break
            if ticker_col is None:
                return []
            index_tickers = set(nasdaq_df[ticker_col].str.upper().str.replace('.', '-').tolist())
        elif index_name == "russell 2000":
            url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            index_tickers = set()
            for table in tables:
                for col in table.columns:
                    if "Ticker" in col or "Symbol" in col:
                        index_tickers = set(table[col].str.upper().str.replace('.', '-').tolist())
                        break
                if index_tickers:
                    break
        else:
            return tickers
        filtered = [t for t in tickers if t in index_tickers]
        return filtered
    except Exception as e:
        st.warning(f"Failed to filter tickers by index {index_name}: {e}")
        return tickers

tickers = filter_tickers_by_index(all_tickers, scanner_index)
tickers = [t.replace('.', '-') for t in tickers]  # Yahoo Finance format
st.sidebar.write(f"Loaded {len(tickers)} tickers from universe.txt for {scanner_index}")

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