import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
from pathlib import Path
from io import StringIO
import requests
import lxml
import html5lib

st.set_page_config(page_title="AI Stock Scanner", layout="wide")
st.title("AI Stock Scanner — Breakout Scanner")

st.info("Note: This app requires 'lxml' and 'html5lib' libraries for HTML parsing. Please ensure they are installed.")

# ---------------------- Load Universe ---------------------- #
@st.cache_data(ttl=3600)
def load_universe(file_path="universe.txt", top_n_nasdaq=100, top_n_sp500=500, top_n_russell=2000, include_all=True):
    p = Path(file_path)
    tickers = []

    headers = {"User-Agent": "Mozilla/5.0"}

    # Fetch S&P 500 tickers
    url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response_sp500 = requests.get(url_sp500, headers=headers)
    tables_sp500 = pd.read_html(StringIO(response_sp500.text))
    sp500_df = tables_sp500[0]
    ticker_col_sp500 = next((col for col in sp500_df.columns if "Symbol" in col or "Ticker" in col), 'Symbol')
    sp500_tickers = sp500_df[ticker_col_sp500].str.upper().str.replace('.', '-').tolist()[:top_n_sp500]
    st.info(f"Loaded {len(sp500_tickers)} tickers from S&P 500")

    # Fetch NASDAQ-100 tickers
    url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
    response_nasdaq = requests.get(url_nasdaq, headers=headers)
    tables_nasdaq = pd.read_html(StringIO(response_nasdaq.text))
    # The table with tickers is the 4th table (index 3)
    nasdaq_df = tables_nasdaq[3]
    ticker_col_nasdaq = next((col for col in nasdaq_df.columns if "Symbol" in col or "Ticker" in col), None)
    nasdaq_tickers = nasdaq_df[ticker_col_nasdaq].str.upper().str.replace('.', '-').tolist()[:top_n_nasdaq] if ticker_col_nasdaq else []
    st.info(f"Loaded {len(nasdaq_tickers)} tickers from NASDAQ-100")

    # Fetch Russell 2000 tickers
    url_russell = "https://en.wikipedia.org/wiki/Russell_2000_Index"
    response_russell = requests.get(url_russell, headers=headers)
    tables_russell = pd.read_html(StringIO(response_russell.text))
    russell_tickers = []
    for table in tables_russell:
        ticker_col_russell = next((col for col in table.columns if "Ticker" in col or "Symbol" in col), None)
        if ticker_col_russell:
            russell_tickers = table[ticker_col_russell].str.upper().str.replace('.', '-').tolist()
            break
    russell_tickers = russell_tickers[:top_n_russell]
    st.info(f"Loaded {len(russell_tickers)} tickers from Russell 2000")

    combined_tickers = list(dict.fromkeys(nasdaq_tickers + sp500_tickers + russell_tickers))

    if include_all:
        tickers = combined_tickers

    p.write_text("\n".join(tickers))
    st.info(f"Auto-populated universe.txt with {len(tickers)} tickers")
    return tickers

scanner_index = st.sidebar.selectbox("Select Index to Scan:", ["NASDAQ 100", "S&P 500", "Russell 2000"])
all_tickers = load_universe()

# Filter tickers by selected index
def filter_tickers_by_index(tickers, index_name):
    index_name = index_name.lower()
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if index_name == "s&p 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            ticker_col = next((col for col in df.columns if "Symbol" in col or "Ticker" in col), None)
            if not ticker_col:
                return []
            index_tickers = set(df[ticker_col].str.upper().str.replace('.', '-').tolist())
        elif index_name == "nasdaq 100":
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            nasdaq_df = tables[3]
            ticker_col = next((col for col in nasdaq_df.columns if "Symbol" in col or "Ticker" in col), None)
            if not ticker_col:
                return []
            index_tickers = set(nasdaq_df[ticker_col].str.upper().str.replace('.', '-').tolist())
        elif index_name == "russell 2000":
            url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            index_tickers = set()
            for table in tables:
                ticker_col = next((col for col in table.columns if "Ticker" in col or "Symbol" in col), None)
                if ticker_col:
                    index_tickers = set(table[ticker_col].str.upper().str.replace('.', '-').tolist())
                    break
        else:
            return tickers
        filtered = [t for t in tickers if t in index_tickers]
        return filtered
    except Exception as e:
        st.warning(f"Failed to filter tickers by index {index_name}: {e}")
        return tickers

tickers = filter_tickers_by_index(all_tickers, scanner_index)
tickers = [t.replace('.', '-') for t in tickers]
st.sidebar.write(f"Loaded {len(tickers)} tickers for {scanner_index}")

# ---------------------- Helper Functions ---------------------- #
def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def sma(series, window):
    return series.rolling(window).mean()

def atr14(high, low, close):
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1)
    tr_max = tr.max(axis=1)
    return tr_max.rolling(14).mean().iloc[-1]

@st.cache_data(ttl=3600)
def fetch_history_safe(tickers, period_days=60):
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

# ---------------------- Scanners ---------------------- #
def scan_breakout(hist, min_price=1.0, max_price=1500.0, rsi_min=55, rsi_max=70):
    rows = []
    for t, df in hist.items():
        if len(df) < 25: continue
        price = df["Close"].iloc[-1]
        if pd.isna(price) or price < min_price or price > max_price: continue
        rsi_val = rsi(df["Close"]).iloc[-1]
        if pd.isna(rsi_val) or rsi_val < rsi_min or rsi_val > rsi_max: continue
        ma9 = sma(df["Close"],9).iloc[-1]
        ma21 = sma(df["Close"],21).iloc[-1]
        if pd.isna(ma9) or pd.isna(ma21): continue
        if (price <= ma9) or (price <= ma21): continue
        atr = atr14(df["High"], df["Low"], df["Close"])
        trend = "Strong Up" if rsi_val >= 67 else "Up" if rsi_val >= 60 else "Neutral"
        rows.append({"Ticker": t, "Price": round(price,2), "RSI14": round(rsi_val,2), "ATR14": round(atr,2), "Trend": trend})
    return pd.DataFrame(rows)

# ---------------------- Streamlit UI ---------------------- #
st.sidebar.header("Scanner Settings")
scanner_type = st.sidebar.radio("Select scanner type:", ["Breakout Momentum"])
min_price = st.sidebar.number_input("Min Price", 0.1, 100.0, 1.0)
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