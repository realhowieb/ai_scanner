import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path

# Load tickers
@st.cache_data(ttl=3600)
def load_sp600_tickers(file_path="sp600.txt"):
    p = Path(file_path)
    tickers = []
    if p.exists() and p.stat().st_size > 0:
        raw_lines = p.read_text().splitlines()
        tickers = [line.split()[0].replace('.', '-') for line in raw_lines if line.strip()]
    return tickers

# Fetch price data
@st.cache_data(ttl=3600)
def fetch_price_data(tickers, period="60d", interval="1d"):
    price_data = {}
    skipped = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df.empty:
                skipped.append(ticker)
            else:
                price_data[ticker] = df
        except:
            skipped.append(ticker)
    return price_data, skipped

# Filter tickers by price
def filter_tickers_by_price(price_data, min_price, max_price):
    filtered = []
    for t, df in price_data.items():
        if df.empty or 'Close' not in df.columns:
            continue
        try:
            price = float(df['Close'].iloc[-1])
        except:
            continue
        if min_price <= price <= max_price:
            filtered.append(t)
    return filtered

# Breakout scanner
def breakout_scanner(price_data, min_price=5, max_price=1000):
    results = []
    for ticker, df in price_data.items():
        if df.empty or len(df) < 21 or 'Close' not in df.columns:
            continue
        try:
            latest_close = float(df['Close'].iloc[-1])
            prev_max = float(df['Close'].iloc[-21:-1].max())
        except:
            continue
        if min_price <= latest_close <= max_price and latest_close > prev_max:
            pct_breakout = (latest_close - prev_max) / prev_max * 100 if prev_max > 0 else np.nan
            results.append({'Ticker': ticker, 'Latest Close': latest_close,
                            'Previous 20d Max': prev_max, 'Breakout %': round(pct_breakout, 2)})
    if results:
        df_results = pd.DataFrame(results).sort_values('Breakout %', ascending=False).reset_index(drop=True)
    else:
        df_results = pd.DataFrame(columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %'])
    return df_results

# Automatic scanner function
def auto_scan(min_price=5, max_price=1000):
    tickers = load_sp600_tickers()
    price_data, skipped = fetch_price_data(tickers)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results.csv", index=False)
    return breakout_df, skipped, filtered_data

# Streamlit UI
st.title("Automatic S&P 600 Breakout Scanner")

min_price = st.sidebar.number_input("Minimum Price ($)", min_value=0.0, value=5.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0.0, value=1000.0, step=1.0)

if st.sidebar.button("Run Automatic Scan"):
    with st.spinner("Scanning S&P 600 tickers..."):
        breakout_df, skipped, filtered_data = auto_scan(min_price, max_price)

    if skipped:
        st.warning(f"Skipped {len(skipped)} tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

    if breakout_df.empty:
        st.info("No breakout candidates found.")
    else:
        st.success(f"Found {len(breakout_df)} breakout candidates.")
        st.dataframe(breakout_df)

        st.download_button(
            "Download Breakout Results",
            data=breakout_df.to_csv(index=False),
            file_name="breakout_results.csv",
            mime="text/csv"
        )

        st.subheader("Top 5 Breakout Charts")
        for ticker in breakout_df['Ticker'].head(5):
            df = filtered_data.get(ticker)
            if df is None or df.empty or 'Close' not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_title(f"{ticker} Close Price - Last 60 Days")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)