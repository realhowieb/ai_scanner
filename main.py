import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Fetch hot stocks from Yahoo Finance top gainers
@st.cache_data(ttl=1800)
def fetch_hot_stocks():
    url = "https://finance.yahoo.com/gainers"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', attrs={'data-reactid': '32'})
        if table is None:
            table = soup.find('table')
        tickers = []
        if table:
            rows = table.find_all('tr')[1:]  # skip header
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    ticker = cols[0].text.strip().replace('.', '-')
                    tickers.append(ticker)
        return tickers
    except Exception as e:
        return []

# Fetch most active stocks from Yahoo Finance
@st.cache_data(ttl=1800)
def fetch_most_active_stocks():
    url = "https://finance.yahoo.com/most-active"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        tickers = []
        if table:
            rows = table.find_all('tr')[1:]  # skip header
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    ticker = cols[0].text.strip().replace('.', '-')
                    tickers.append(ticker)
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch most active stocks: {e}")
        return []

# Load tickers
@st.cache_data(ttl=3600)
def load_sp600_tickers(file_path="sp600.txt"):
    p = Path(file_path)
    tickers = []
    if p.exists() and p.stat().st_size > 0:
        raw_lines = p.read_text().splitlines()
        tickers = [line.split()[0].replace('.', '-') for line in raw_lines if line.strip()]
    return tickers

# Load S&P 500 tickers
@st.cache_data(ttl=3600)
def load_sp500_tickers(file_path="sp500.txt"):
    p = Path(file_path)
    tickers = []
    if p.exists() and p.stat().st_size > 0:
        raw_lines = p.read_text().splitlines()
        tickers = [line.split()[0].replace('.', '-') for line in raw_lines if line.strip()]
    return tickers

# New function to fetch and save S&P 500 tickers from Wikipedia
def fetch_and_save_sp500(file_path="sp500.txt"):
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        p = Path(file_path)
        p.write_text('\n'.join(tickers))
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 tickers: {e}")
        return []

# Remove delisted tickers
def remove_delisted_tickers(file_path="sp600.txt"):
    tickers = load_sp600_tickers(file_path)
    valid_tickers = []
    removed_count = 0
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_tickers.append(ticker)
            else:
                removed_count += 1
        except:
            removed_count += 1
    p = Path(file_path)
    p.write_text('\n'.join(valid_tickers))
    return removed_count, valid_tickers

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

# Helper function to format large numbers
def format_volume(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.3f}M"
    elif v >= 1_000:
        return f"{v/1_000:.1f}K"
    else:
        return str(v)

# Breakout scanner with formatted Volume
def breakout_scanner(price_data, min_price=5, max_price=1000):
    results = []
    for ticker, df in price_data.items():
        if df.empty or len(df) < 21 or 'Close' not in df.columns:
            continue
        try:
            latest_close = float(df['Close'].iloc[-1])
            prev_max = float(df['Close'].iloc[-21:-1].max())
            latest_volume = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else np.nan
            latest_volume_fmt = format_volume(latest_volume)
        except:
            continue
        if min_price <= latest_close <= max_price and latest_close > prev_max:
            pct_breakout = (latest_close - prev_max) / prev_max * 100 if prev_max > 0 else np.nan
            results.append({
                'Ticker': ticker,
                'Latest Close': latest_close,
                'Previous 20d Max': prev_max,
                'Breakout %': round(pct_breakout, 2),
                'Volume': latest_volume_fmt
            })
    if results:
        df_results = pd.DataFrame(results).sort_values('Breakout %', ascending=False).reset_index(drop=True)
    else:
        df_results = pd.DataFrame(columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %', 'Volume'])
    return df_results

# Automatic scanner function for full S&P 600 tickers or uploaded tickers
def auto_scan(min_price=5, max_price=1000):
    tickers = st.session_state.get("tickers", [])
    price_data, skipped = fetch_price_data(tickers)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results.csv", index=False)
    return breakout_df, skipped, filtered_data

# Automatic scanner function for hot stocks
def auto_scan_hot_stocks(min_price=5, max_price=1000):
    hot_tickers = fetch_hot_stocks()
    if not hot_tickers:
        return pd.DataFrame(columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %']), [], {}
    price_data, skipped = fetch_price_data(hot_tickers)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results_hot.csv", index=False)
    return breakout_df, skipped, filtered_data

# Automatic scanner function for S&P 500 tickers
def auto_scan_sp500(min_price=5, max_price=1000):
    tickers = load_sp500_tickers()
    price_data, skipped = fetch_price_data(tickers)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results_sp500.csv", index=False)
    return breakout_df, skipped, filtered_data

# Streamlit UI
st.title("Automatic S&P 600 Breakout Scanner")

min_price = st.sidebar.number_input("Minimum Price ($)", min_value=0.0, value=5.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0.0, value=1000.0, step=1.0)

uploaded_file = st.sidebar.file_uploader("Upload tickers file (.txt or .csv)", type=['txt', 'csv'])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode("utf-8")
            lines = content.splitlines()
            tickers = [line.strip().replace('.', '-') for line in lines if line.strip()]
        else:
            df_uploaded = pd.read_csv(uploaded_file, header=None)
            tickers = []
            for col in df_uploaded.columns:
                tickers.extend([str(t).strip().replace('.', '-') for t in df_uploaded[col] if str(t).strip()])
        st.session_state.tickers = tickers

        # Run auto_scan immediately after upload
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

    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
else:
    if "tickers" not in st.session_state:
        st.session_state.tickers = load_sp600_tickers()

# New: Stock ticker search input and display
st.subheader("Search Stock by Ticker")
search_ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):").upper().strip()

if search_ticker:
    try:
        stock = yf.Ticker(search_ticker)
        hist = stock.history(period="60d", interval="1d")
        if hist.empty:
            st.error(f"No historical data found for ticker '{search_ticker}'. It may be invalid or delisted.")
        else:
            st.success(f"Showing closing price chart for {search_ticker} (last 60 days)")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(hist.index, hist['Close'], label='Close Price')
            ax.set_title(f"{search_ticker} Close Price - Last 60 Days")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Run breakout scan on this single ticker
            breakout_df = breakout_scanner({search_ticker: hist}, min_price, max_price)
            if breakout_df.empty:
                st.info(f"No breakout detected for {search_ticker} based on the last 60 days data.")
            else:
                st.success(f"Breakout detected for {search_ticker}:")
                st.dataframe(breakout_df)
    except Exception as e:
        st.error(f"Error fetching data for ticker '{search_ticker}': {e}")



if st.sidebar.button("Run Hot Stocks Scan"):
    with st.spinner("Fetching and scanning hot stocks..."):
        breakout_df, skipped, filtered_data = auto_scan_hot_stocks(min_price, max_price)

    if skipped:
        st.warning(f"Skipped {len(skipped)} hot tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

    if breakout_df.empty:
        st.info("No breakout candidates found among hot stocks.")
    else:
        st.success(f"Found {len(breakout_df)} breakout candidates among hot stocks.")
        st.dataframe(breakout_df)

        st.download_button(
            "Download Hot Stocks Breakout Results",
            data=breakout_df.to_csv(index=False),
            file_name="breakout_results_hot.csv",
            mime="text/csv"
        )

        st.subheader("Top 5 Hot Stocks Breakout Charts")
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

# Most Active Stocks Scan Button
if st.sidebar.button("Run Most Active Stocks Scan"):
    with st.spinner("Fetching and scanning most active stocks..."):
        most_active_tickers = fetch_most_active_stocks()
        if most_active_tickers:
            price_data, skipped = fetch_price_data(most_active_tickers)
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            breakout_df = breakout_scanner(filtered_data, min_price, max_price)

            if skipped:
                st.warning(f"Skipped {len(skipped)} most active tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

            if breakout_df.empty:
                st.info("No breakout candidates found among most active stocks.")
            else:
                st.success(f"Found {len(breakout_df)} breakout candidates among most active stocks.")
                st.dataframe(breakout_df)

                st.download_button(
                    "Download Most Active Stocks Breakout Results",
                    data=breakout_df.to_csv(index=False),
                    file_name="breakout_results_most_active.csv",
                    mime="text/csv"
                )

                st.subheader("Top 5 Most Active Breakout Charts")
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
        else:
            st.error("Failed to fetch most active stocks.")

if st.sidebar.button("Run Automatic Scan"):
    with st.spinner("Scanning S&P 500 tickers..."):
        breakout_df_sp500, skipped_sp500, filtered_data_sp500 = auto_scan_sp500(min_price, max_price)

    if skipped_sp500:
        st.warning(f"Skipped {len(skipped_sp500)} S&P 500 tickers due to missing data or delisted: {', '.join(skipped_sp500[:10])}...")

    st.subheader("S&P 500 Breakout Scan Results")
    if breakout_df_sp500.empty:
        st.info("No breakout candidates found in S&P 500.")
    else:
        st.success(f"Found {len(breakout_df_sp500)} breakout candidates in S&P 500.")
        st.dataframe(breakout_df_sp500)

        st.download_button(
            "Download S&P 500 Breakout Results",
            data=breakout_df_sp500.to_csv(index=False),
            file_name="breakout_results_sp500.csv",
            mime="text/csv"
        )

        st.subheader("Top 5 S&P 500 Breakout Charts")
        for ticker in breakout_df_sp500['Ticker'].head(5):
            df = filtered_data_sp500.get(ticker)
            if df is None or df.empty or 'Close' not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_title(f"{ticker} Close Price - Last 60 Days")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    with st.spinner("Scanning S&P 600 tickers..."):
        breakout_df_sp600, skipped_sp600, filtered_data_sp600 = auto_scan(min_price, max_price)

    if skipped_sp600:
        st.warning(f"Skipped {len(skipped_sp600)} S&P 600 tickers due to missing data or delisted: {', '.join(skipped_sp600[:10])}...")

    st.subheader("S&P 600 Breakout Scan Results")
    if breakout_df_sp600.empty:
        st.info("No breakout candidates found in S&P 600.")
    else:
        st.success(f"Found {len(breakout_df_sp600)} breakout candidates in S&P 600.")
        st.dataframe(breakout_df_sp600)

        st.download_button(
            "Download S&P 600 Breakout Results",
            data=breakout_df_sp600.to_csv(index=False),
            file_name="breakout_results.csv",
            mime="text/csv"
        )

        st.subheader("Top 5 S&P 600 Breakout Charts")
        for ticker in breakout_df_sp600['Ticker'].head(5):
            df = filtered_data_sp600.get(ticker)
            if df is None or df.empty or 'Close' not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_title(f"{ticker} Close Price - Last 60 Days")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

if st.sidebar.button("Fetch Latest S&P 500 Tickers"):
    with st.spinner("Fetching latest S&P 500 tickers from Wikipedia..."):
        new_tickers = fetch_and_save_sp500()
        if new_tickers:
            st.session_state.tickers = new_tickers
            st.success(f"Fetched and saved {len(new_tickers)} S&P 500 tickers to sp500.txt.")
        else:
            st.error("Failed to fetch S&P 500 tickers from Wikipedia.")

if st.sidebar.button("Clean delisted tickers"):
    with st.spinner("Cleaning delisted tickers..."):
        removed_count, valid_tickers = remove_delisted_tickers()
        st.session_state.tickers = valid_tickers
    st.success(f"Removed {removed_count} delisted tickers from sp600.txt.")