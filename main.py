import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
from pathlib import Path

st.set_page_config(page_title="AI Stock Scanner", layout="wide")
st.title("AI Stock Scanner — Breakout Scanner")


# ---------------------- New Universe Loading Logic ---------------------- #
import requests
import io

# Helper: Get exchange tickers from NASDAQ Trader symbol directories
@st.cache_data(ttl=86400)
def get_exchange_tickers(exchange="nasdaq"):
    """
    Fetch tickers from NASDAQ Trader symbol directory for NASDAQ, NYSE, or AMEX.
    """
    urls = {
        "nasdaq": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "nyse": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "amex": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    if exchange.lower() not in urls:
        return []
    url = urls[exchange.lower()]
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        # Remove last summary row
        df = df.iloc[:-1]
        if exchange.lower() == "nasdaq":
            tickers = df["Symbol"].tolist()
        else:
            # NYSE/AMEX: filter by Exchange column
            exch_code = {"nyse": "N", "amex": "A"}[exchange.lower()]
            tickers = df[df["Exchange"] == exch_code]["ACT Symbol"].tolist()
        # Remove test/placeholder tickers
        tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
        return tickers
    except Exception as e:
        st.warning(f"Failed to fetch {exchange.upper()} tickers: {e}")
        return []

# Helper: Filter tickers by price using yfinance
@st.cache_data(ttl=3600)
def filter_by_price(tickers, min_price=1.0, max_price=1500.0):
    filtered = []
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="2d", interval="1d", progress=False, threads=True, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"].iloc[-1]
            else:
                closes = data["Close"]
            for t in batch:
                try:
                    price = closes[t] if isinstance(closes, pd.Series) and t in closes else closes.get(t, np.nan)
                    if pd.isna(price):
                        continue
                    if min_price <= price <= max_price:
                        filtered.append(t)
                except Exception:
                    continue
        except Exception as e:
            continue
    return filtered

# Main function to load universe
@st.cache_data(ttl=86400)
def load_universe(file_path="universe.txt", min_price=1.0, max_price=1500.0, exchange="all"):
    p = Path(file_path)
    if p.exists() and p.stat().st_size > 0:
        tickers = p.read_text().splitlines()
        st.info(f"Loaded universe from {file_path} with {len(tickers)} tickers")
        return tickers
    tickers = []
    # Determine which exchanges to fetch
    exchanges = []
    if exchange == "all":
        exchanges = ["nasdaq", "nyse", "amex"]
    else:
        exchanges = [exchange.lower()]
    all_tickers = []
    for exch in exchanges:
        all_tickers.extend(get_exchange_tickers(exch))
    # Remove duplicates and sort
    all_tickers = sorted(set(all_tickers))
    filtered = filter_by_price(all_tickers, min_price, max_price)
    filtered = [t.replace(".", "-") for t in filtered]
    p.write_text("\n".join(filtered))
    st.info(f"Universe loaded: {len(filtered)} tickers from {', '.join(exchanges).upper()} between ${min_price} and ${max_price}")
    return filtered

# Streamlit sidebar for exchange selection
exchange_options = {
    "NASDAQ": "nasdaq",
    "NYSE": "nyse",
    "AMEX": "amex",
    "All Exchanges": "all"
}
st.sidebar.header("Universe Selection")
selected_exchange = st.sidebar.selectbox("Select Exchange:", list(exchange_options.keys()), index=0)

# Price range for universe filter
min_price_univ = st.sidebar.number_input("Universe Min Price", 0.1, 100.0, 1.0, key="univ_min_price")
max_price_univ = st.sidebar.number_input("Universe Max Price", 0.5, 5000.0, 1500.0, key="univ_max_price")

tickers = load_universe(min_price=min_price_univ, max_price=max_price_univ, exchange=exchange_options[selected_exchange])
st.sidebar.write(f"Loaded {len(tickers)} tickers for {selected_exchange}")

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

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

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
def scan_breakout(hist, min_price=1.0, max_price=1500.0, rsi_min=55, rsi_max=70,
                  min_pct_chg=None, max_pct_chg=None, min_avg_vol=None):
    rows = []
    for t, df in hist.items():
        if len(df) < 25: continue
        price = df["Close"].iloc[-1]
        if pd.isna(price) or price < min_price or price > max_price: continue

        # Calculate RSI
        rsi_val = rsi(df["Close"]).iloc[-1]
        if pd.isna(rsi_val) or rsi_val < rsi_min or rsi_val > rsi_max: continue

        # Calculate SMA for 9 and 21 periods
        ma9 = sma(df["Close"],9).iloc[-1]
        ma21 = sma(df["Close"],21).iloc[-1]
        if pd.isna(ma9) or pd.isna(ma21): continue

        # Calculate EMA for 9 and 21 periods for new filter
        ema9 = ema(df["Close"],9).iloc[-1]
        ema21 = ema(df["Close"],21).iloc[-1]
        if pd.isna(ema9) or pd.isna(ema21): continue

        # Price must be above EMA9 and EMA21
        if price <= ema9 or price <= ema21: continue

        # % Change today filter
        pct_chg = (df["Close"].iloc[-1] - df["Open"].iloc[-1]) / df["Open"].iloc[-1] * 100
        if min_pct_chg is not None and pct_chg < min_pct_chg:
            continue
        if max_pct_chg is not None and pct_chg > max_pct_chg:
            continue

        # Average volume over last 10 days filter
        avg_vol = df["Volume"].tail(10).mean()
        if min_avg_vol is not None and avg_vol < min_avg_vol:
            continue

        # Price must be above SMA9 and SMA21
        if (price <= ma9) or (price <= ma21): continue

        atr = atr14(df["High"], df["Low"], df["Close"])
        trend = "Strong Up" if rsi_val >= 67 else "Up" if rsi_val >= 60 else "Neutral"
        rows.append({"Ticker": t, "Price": round(price,2), "RSI14": round(rsi_val,2), "ATR14": round(atr,2),
                     "Trend": trend, "%Chg_d": round(pct_chg,2), "AvgVol10d": int(avg_vol)})
    return pd.DataFrame(rows)

# ---------------------- Streamlit UI ---------------------- #
st.sidebar.header("Scanner Settings")
scanner_type = st.sidebar.radio("Select scanner type:", ["Breakout Momentum"])
min_price = st.sidebar.number_input("Scanner Min Price", 0.1, 100.0, 1.0, key="scanner_min_price")
max_price = st.sidebar.number_input("Scanner Max Price", 0.5, 5000.0, 1500.0, key="scanner_max_price")
days_history = st.sidebar.slider("History Days", 20, 120, 60)

# New filters for breakout momentum
min_pct_chg = st.sidebar.number_input("Min % Change Today", -100.0, 100.0, 0.0, key="min_pct_chg")
max_pct_chg = st.sidebar.number_input("Max % Change Today", -100.0, 100.0, 100.0, key="max_pct_chg")
min_avg_vol = st.sidebar.number_input("Min Avg Volume (10-day)", 0, 100000000, 0, step=1000, key="min_avg_vol")

if st.button("Run Scanner"):
    st.info(f"Fetching {days_history}d historical data for {len(tickers)} tickers...")
    hist = fetch_history_safe(tickers, days_history)
    st.success("Data fetched!")

    df = scan_breakout(hist, min_price, max_price, rsi_min=55, rsi_max=70,
                       min_pct_chg=min_pct_chg, max_pct_chg=max_pct_chg, min_avg_vol=min_avg_vol)

    st.dataframe(df)
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "scanner_results.csv", "text/csv")

        st.subheader("Top 5 ticker charts")
        for t in df["Ticker"].head(5):
            st.write(t)
            st.line_chart(hist[t]["Close"])