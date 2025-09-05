import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from bs4 import BeautifulSoup

def premarket_scan(tickers):
    """
    Perform a pre-market scan for a list of tickers.
    Returns a DataFrame with columns: Ticker, Premarket First Price, Premarket Last Price, Premarket % Change.
    """
    import pandas as pd
    import yfinance as yf
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", interval="5m", prepost=True)
            if df.empty or "Close" not in df.columns:
                continue
            # Pre-market session is before 09:30:00 in NY time
            # yfinance returns index in UTC, so convert to US/Eastern and filter
            df_local = df.copy()
            if not df_local.index.tz:
                df_local.index = df_local.index.tz_localize("UTC")
            df_local.index = df_local.index.tz_convert("America/New_York")
            premarket_df = df_local[df_local.index.time < pd.to_datetime("09:30:00").time()]
            if premarket_df.empty:
                continue
            first_price = float(premarket_df["Close"].iloc[0])
            last_price = float(premarket_df["Close"].iloc[-1])
            pct_change = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0.0
            results.append({
                "Ticker": ticker,
                "Premarket First Price": round(first_price, 4),
                "Premarket Last Price": round(last_price, 4),
                "Premarket % Change": round(pct_change, 2)
            })
        except Exception:
            continue
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=["Ticker", "Premarket First Price", "Premarket Last Price", "Premarket % Change"])

# Post-market scan function
def postmarket_scan(tickers):
    """
    Perform a post-market scan for a list of tickers.
    Returns a DataFrame with columns: Ticker, Postmarket First Price, Postmarket Last Price, Postmarket % Change.
    """
    import pandas as pd
    import yfinance as yf
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", interval="5m", prepost=True)
            if df.empty or "Close" not in df.columns:
                continue
            df_local = df.copy()
            if not df_local.index.tz:
                df_local.index = df_local.index.tz_localize("UTC")
            df_local.index = df_local.index.tz_convert("America/New_York")
            # Post-market session is >= 16:00:00
            post_df = df_local[df_local.index.time >= pd.to_datetime("16:00:00").time()]
            if post_df.empty:
                continue
            first_price = float(post_df["Close"].iloc[0])
            last_price = float(post_df["Close"].iloc[-1])
            pct_change = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0.0
            results.append({
                "Ticker": ticker,
                "Postmarket First Price": round(first_price, 4),
                "Postmarket Last Price": round(last_price, 4),
                "Postmarket % Change": round(pct_change, 2)
            })
        except Exception:
            continue
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=["Ticker", "Postmarket First Price", "Postmarket Last Price", "Postmarket % Change"])


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

# Fetch trending stocks from Yahoo Finance
@st.cache_data(ttl=1800)
def fetch_trending_stocks():
    url = "https://finance.yahoo.com/markets/stocks/trending/"
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
        st.error(f"Failed to fetch trending stocks: {e}")
        return []
#Fetch and Save Nasdaq Tickers
def fetch_and_save_nasdaq(file_path="nasdaq.txt"):
    try:
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df = pd.read_csv(url, sep='|')

        # Clean up tickers to avoid floats/NaN
        tickers = df['Symbol'].dropna().astype(str).tolist()
        tickers = [t.strip() for t in tickers if t.strip()]

        Path(file_path).write_text('\n'.join(tickers))
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch Nasdaq tickers: {e}")
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


# Batch price data fetcher
@st.cache_data(ttl=3600)
def fetch_price_data_batch(tickers, period="60d", interval="1d", batch_size=50):
    """
    Fetch price data for tickers in batches to avoid throttling.
    Returns dict of DataFrames keyed by ticker, and a list of tickers skipped (no data).
    """
    import yfinance as yf
    import pandas as pd
    price_data = {}
    skipped = []
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, period=period, interval=interval, group_by="ticker", progress=False, threads=False)
            # If only one ticker, df is not multi-indexed
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in batch:
                    if ticker in df.columns.get_level_values(0):
                        subdf = df[ticker]
                        if not subdf.empty and 'Close' in subdf.columns:
                            price_data[ticker] = subdf
                        else:
                            skipped.append(ticker)
                    else:
                        skipped.append(ticker)
            else:
                # Only one ticker
                if not df.empty and 'Close' in df.columns:
                    price_data[batch[0]] = df
                else:
                    skipped.append(batch[0])
        except Exception:
            skipped.extend(batch)
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
    price_data, skipped = fetch_price_data_batch(tickers, period="60d", interval="1d", batch_size=50)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results.csv", index=False)
    return breakout_df, skipped, filtered_data


# Streamlit UI
st.title("Money Moves Breakout Scanner")

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
        hot_tickers = fetch_hot_stocks()
        if not hot_tickers:
            breakout_df, skipped, filtered_data = pd.DataFrame(columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %']), [], {}
        else:
            price_data, skipped = fetch_price_data_batch(hot_tickers, period="60d", interval="1d", batch_size=50)
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            breakout_df = breakout_scanner(filtered_data, min_price, max_price)
            if not breakout_df.empty:
                breakout_df.to_csv("breakout_results_hot.csv", index=False)

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
            price_data, skipped = fetch_price_data_batch(most_active_tickers, period="60d", interval="1d", batch_size=50)
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

# Trending Stocks Scan Button
if st.sidebar.button("Run Trending Scan"):
    with st.spinner("Fetching and scanning trending stocks..."):
        trending_tickers = fetch_trending_stocks()
        if trending_tickers:
            price_data, skipped = fetch_price_data_batch(trending_tickers, period="60d", interval="1d", batch_size=50)
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            breakout_df = breakout_scanner(filtered_data, min_price, max_price)

            if skipped:
                st.warning(f"Skipped {len(skipped)} trending tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

            st.subheader("Trending Stocks Breakout Scan Results")
            if breakout_df.empty:
                st.info("No breakout candidates found among trending stocks.")
            else:
                st.success(f"Found {len(breakout_df)} breakout candidates among trending stocks.")
                st.dataframe(breakout_df)

                st.download_button(
                    "Download Trending Stocks Breakout Results",
                    data=breakout_df.to_csv(index=False),
                    file_name="breakout_results_trending.csv",
                    mime="text/csv"
                )

                st.subheader("Top 5 Trending Breakout Charts")
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
            st.error("Failed to fetch trending stocks.")

if st.sidebar.button("Run S&P 500 Scan"):
    with st.spinner("Fetching S&P 500 tickers..."):
        tickers = load_sp500_tickers()

    if not tickers:
        st.error("No S&P 500 tickers available to scan.")
    else:
        st.subheader("S&P 500 Breakout Scan Results (updating live...)")
        results_placeholder = st.empty()
        progress = st.progress(0)
        skipped = []
        breakout_rows = []
        batch_size = 50
        total = len(tickers)
        for i in range(0, total, batch_size):
            batch = tickers[i:i+batch_size]
            price_data, skipped_batch = fetch_price_data_batch(batch, period="60d", interval="1d", batch_size=batch_size)
            skipped.extend(skipped_batch)
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            breakout_df = breakout_scanner(filtered_data, min_price, max_price)
            if not breakout_df.empty:
                for row in breakout_df.itertuples(index=False):
                    breakout_rows.append(list(row))
                df_partial = pd.DataFrame(breakout_rows, columns=breakout_df.columns)
                results_placeholder.dataframe(df_partial)
            progress.progress(min((i + batch_size) / total, 1.0))

        st.success(f"Scan complete ✅ Found {len(breakout_rows)} breakout candidates in S&P 500.")
        if skipped:
            st.warning(f"Skipped {len(skipped)} tickers due to missing data or delisted.")

#Run Nasdaq Button
if st.sidebar.button("Run Nasdaq Scan"):
    with st.spinner("Fetching Nasdaq tickers..."):
        tickers = fetch_and_save_nasdaq()

    if not tickers:
        st.error("No Nasdaq tickers available to scan.")
    else:
        st.subheader("Nasdaq Breakout Scan Results (updating live...)")
        results_placeholder = st.empty()  # For live table updates
        progress = st.progress(0)         # Progress bar
        skipped = []
        breakout_rows = []
        batch_size = 50
        total = len(tickers)
        for i in range(0, total, batch_size):
            batch = tickers[i:i+batch_size]
            price_data, skipped_batch = fetch_price_data_batch(batch, period="60d", interval="1d", batch_size=batch_size)
            skipped.extend(skipped_batch)
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            breakout_df = breakout_scanner(filtered_data, min_price, max_price)
            if not breakout_df.empty:
                for row in breakout_df.itertuples(index=False):
                    breakout_rows.append(list(row))
                if breakout_rows:
                    df_partial = pd.DataFrame(breakout_rows, columns=breakout_df.columns)
                    results_placeholder.dataframe(df_partial)
            progress.progress(min((i + batch_size) / total, 1.0))

        st.success(f"Scan complete ✅ Found {len(breakout_rows)} breakout candidates.")
        if skipped:
            st.warning(f"Skipped {len(skipped)} tickers due to missing data or delisted.")

# Fetch S&P 500 Tickers Button
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
# Add Pre-market Scan Button
if st.sidebar.button("Run Pre-market Scan"):
    tickers = st.session_state.get("tickers", [])
    if not tickers:
        st.error("No tickers available for pre-market scan.")
    else:
        with st.spinner(f"Running pre-market scan on {len(tickers)} tickers..."):
            pm_df = premarket_scan(tickers)
        st.subheader("Pre-market Scan Results")
        if pm_df.empty:
            st.info("No pre-market candidates found or no pre-market data available.")
        else:
            st.success(f"Found {len(pm_df)} tickers with pre-market data.")
            st.dataframe(pm_df)
            st.download_button(
                label="Download Pre-market Results as CSV",
                data=pm_df.to_csv(index=False),
                file_name="premarket_results.csv",
                mime="text/csv"
            )

# Add Post-market Scan Button
if st.sidebar.button("Run Post-market Scan"):
    tickers = st.session_state.get("tickers", [])
    if not tickers:
        st.error("No tickers available for post-market scan.")
    else:
        with st.spinner(f"Running post-market scan on {len(tickers)} tickers..."):
            post_df = postmarket_scan(tickers)
        st.subheader("Post-market Scan Results")
        if post_df.empty:
            st.info("No post-market candidates found or no post-market data available.")
        else:
            st.success(f"Found {len(post_df)} tickers with post-market data.")
            st.dataframe(post_df)
            st.download_button(
                label="Download Post-market Results as CSV",
                data=post_df.to_csv(index=False),
                file_name="postmarket_results.csv",
                mime="text/csv"
            )