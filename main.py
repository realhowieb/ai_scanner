import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import asyncio
from pyppeteer import launch

st.title("AI Scanner - Breakout Scanner (Finviz Top Stocks)")

# Sidebar filters
min_price = st.sidebar.number_input("Minimum Price ($)", min_value=0.0, value=5.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0.0, value=1500.0, step=1.0)
market_option = st.sidebar.selectbox("Select Market", ["S&P 500", "NASDAQ 100", "AMEX"])

TICKER_FILE = "ticker.txt"

def fetch_top_finviz_tickers(top_n=100, market="S&P 500"):
    """
    Fetch top tickers from Finviz using Pyppeteer.
    Must be run in the main thread due to asyncio signal handling.
    """
    import asyncio
    from pyppeteer import launch
    from bs4 import BeautifulSoup
    from pathlib import Path

    async def fetch_async(url):
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.goto(url)
        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        tickers = [a.text.strip() for a in soup.find_all('a', class_='screener-link-primary')]
        return tickers[:top_n]

    if market == "S&P 500":
        url = "https://finviz.com/screener.ashx?v=111&f=idx_sp500"
    elif market == "NASDAQ 100":
        url = "https://finviz.com/screener.ashx?v=111&f=idx_nasdaq100"
    elif market == "AMEX":
        url = "https://finviz.com/screener.ashx?v=111&f=exch_amex"
    else:
        url = "https://finviz.com/screener.ashx?v=111"

    try:
        tickers = asyncio.run(fetch_async(url))
    except Exception as e:
        st.error(f"Failed to fetch tickers from Finviz: {e}")
        return []

    if tickers:
        Path("ticker.txt").write_text("\n".join(tickers))
        st.info(f"Auto-populated ticker.txt with {len(tickers)} tickers from {market}")

    return tickers

@st.cache_data(ttl=3600)
def fetch_price_data(tickers, period="60d", interval="1d"):
    price_data = {}
    skipped_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df.empty:
                skipped_tickers.append(ticker)
            else:
                price_data[ticker] = df
        except Exception:
            skipped_tickers.append(ticker)
    return price_data, skipped_tickers

def filter_by_price(price_data, min_price, max_price):
    filtered = []
    for t, df in price_data.items():
        if df.empty or 'Close' not in df.columns:
            continue
        price = df['Close'].iloc[-1]
        if min_price <= price <= max_price:
            filtered.append(t)
    return filtered

def breakout_scan(price_data, min_price, max_price):
    results = []
    for ticker, df in price_data.items():
        if df.empty or len(df) < 21 or 'Close' not in df.columns:
            continue
        latest_close = df['Close'].iloc[-1]
        if latest_close < min_price or latest_close > max_price:
            continue
        prev_max = df['Close'].iloc[-21:-1].max()
        if latest_close > prev_max:
            results.append({
                "Ticker": ticker,
                "Latest Close": latest_close,
                "Prev 20-day Max": prev_max,
                "Breakout %": round((latest_close - prev_max) / prev_max * 100, 2)
            })
    return pd.DataFrame(results).sort_values("Breakout %", ascending=False)

if "tickers" not in st.session_state:
    st.session_state.tickers = []

if st.sidebar.button("Fetch & Scan"):
    tickers = fetch_top_finviz_tickers(top_n=100, market=market_option)
    if not tickers:
        st.error("No tickers fetched from Finviz.")
    else:
        st.session_state.tickers = tickers

if st.session_state.tickers:
    price_data, skipped = fetch_price_data(st.session_state.tickers)
    if skipped:
        st.warning(f"Skipped {len(skipped)} tickers (no data or delisted): {', '.join(skipped)}")

    filtered = filter_by_price(price_data, min_price, max_price)
    if not filtered:
        st.error("No tickers within the price range.")
    else:
        filtered_data = {t: price_data[t] for t in filtered}
        breakout_df = breakout_scan(filtered_data, min_price, max_price)

        if breakout_df.empty:
            st.info("No breakout candidates found.")
        else:
            st.success(f"Found {len(breakout_df)} breakout candidates.")
            st.dataframe(breakout_df)

            csv_buffer = io.StringIO()
            breakout_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download CSV",
                data=csv_buffer.getvalue(),
                file_name="breakouts.csv",
                mime="text/csv"
            )

            st.subheader("Top 5 Breakout Charts")
            for ticker in breakout_df['Ticker'].head(5):
                df = filtered_data[ticker]
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(df.index, df['Close'], label='Close Price')
                ax.set_title(f"{ticker} Close Price - Last 60 Days")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)