import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import io
import matplotlib.pyplot as plt
import requests

st.title("Breakout Scanner")

# Parameters
min_price = st.sidebar.number_input("Minimum Price", value=1.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price", value=1500.0, step=0.1)
universe_file = "ticker.txt"

@st.cache_data(ttl=3600)
def load_universe(file_path="ticker.txt"):
    p = Path(file_path)
    tickers = []
    if p.exists() and p.stat().st_size > 0:
        raw_lines = p.read_text().splitlines()
        # Keep only the first part before any whitespace or tab and replace '.' with '-'
        tickers = [line.split()[0].replace('.', '-') for line in raw_lines if line.strip()]
        st.info(f"Loaded universe from {file_path} with {len(tickers)} tickers")
        return tickers
    else:
        st.warning(f"{file_path} not found or empty.")
        return []

@st.cache_data(ttl=3600)
def fetch_price_data(tickers, period="5d", interval="1d"):
    """
    Fetch historical price data for each ticker individually.
    Skips tickers that fail or return empty data.
    Returns a dict of DataFrames keyed by ticker.
    """
    price_data = {}
    skipped_tickers = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df.empty:
                skipped_tickers.append(ticker)
                st.warning(f"No data found for {ticker}")
            else:
                price_data[ticker] = df
        except Exception as e:
            skipped_tickers.append(ticker)
            st.warning(f"Failed to fetch {ticker}: {e}")

    return price_data, skipped_tickers

tickers = load_universe()
if not tickers:
    st.error("No tickers found in ticker.txt.")
    st.stop()

with st.spinner(f"Fetching historical price data for {len(tickers)} tickers..."):
    price_data, skipped_tickers = fetch_price_data(tickers)

if skipped_tickers:
    st.sidebar.warning(f"Skipped {len(skipped_tickers)} tickers due to missing data or delisted: {', '.join(skipped_tickers)}")

# Filter tickers by price using last close price
valid_tickers = []
for ticker, data in price_data.items():
    try:
        price = data["Close"].iloc[-1]
        if min_price <= price <= max_price:
            valid_tickers.append(ticker)
    except Exception:
        continue

if not valid_tickers:
    st.warning("No tickers found within the specified price range.")
    st.stop()

# Download historical data for valid tickers for breakout scan
# Need at least 21 days to check breakout (latest close > previous 20-day max)
hist_days = 30
try:
    data = yf.download(valid_tickers, period=f"{hist_days}d", interval="1d", progress=False, threads=True, auto_adjust=False)
except Exception as e:
    st.error(f"Failed to download historical data: {e}")
    st.stop()

breakouts = []
if isinstance(data.columns, pd.MultiIndex):
    closes = data["Close"]
    for ticker in valid_tickers:
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()
        if len(series) < 21:
            continue
        latest_close = series.iloc[-1]
        prev_20_max = series.iloc[:-1].max()
        if latest_close > prev_20_max:
            breakout_pct = (latest_close - prev_20_max) / prev_20_max * 100
            breakouts.append({
                "Ticker": ticker,
                "Latest Close": latest_close,
                "Prev 20-day Max": prev_20_max,
                "Breakout %": breakout_pct
            })
else:
    # Single ticker case
    series = data["Close"].dropna()
    if len(series) >= 21:
        latest_close = series.iloc[-1]
        prev_20_max = series.iloc[:-1].max()
        if latest_close > prev_20_max:
            breakout_pct = (latest_close - prev_20_max) / prev_20_max * 100
            breakouts.append({
                "Ticker": valid_tickers[0],
                "Latest Close": latest_close,
                "Prev 20-day Max": prev_20_max,
                "Breakout %": breakout_pct
            })

if not breakouts:
    st.info("No breakout candidates found.")
    st.stop()

df_breakouts = pd.DataFrame(breakouts)
df_breakouts = df_breakouts.sort_values(by="Breakout %", ascending=False)
st.subheader(f"Breakout Candidates ({len(df_breakouts)})")
st.dataframe(df_breakouts.style.format({"Latest Close": "${:,.2f}", "Prev 20-day Max": "${:,.2f}", "Breakout %": "{:.2f}%"}))

# CSV download
csv_buffer = io.StringIO()
df_breakouts.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode()

st.download_button(
    label="Download Breakouts as CSV",
    data=csv_bytes,
    file_name="breakouts.csv",
    mime="text/csv"
)

# Plot top 5 breakout charts
top5 = df_breakouts.head(5)["Ticker"].tolist()
st.subheader("Top 5 Breakout Price Charts")

for ticker in top5:
    st.markdown(f"### {ticker}")
    hist = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=False)
    if hist.empty:
        st.write("No data available.")
        continue
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="Close")
    ax.set_title(f"{ticker} Close Price (last 3 months)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)