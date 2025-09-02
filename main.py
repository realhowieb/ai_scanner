import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import io
import matplotlib.pyplot as plt

st.title("Breakout Scanner")

# Parameters
min_price = st.sidebar.number_input("Minimum Price", value=1.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price", value=1500.0, step=0.1)
universe_file = "universe.txt"

@st.cache_data(ttl=3600)
def load_universe(file_path="universe.txt"):
    p = Path(file_path)
    tickers = []
    # Try reading existing file
    if p.exists() and p.stat().st_size > 0:
        tickers = p.read_text().splitlines()
        st.info(f"Loaded universe from {file_path} with {len(tickers)} tickers")
        return tickers
    else:
        st.warning(f"{file_path} not found or empty. Auto-populating tickers using yfinance...")
        try:
            import yfinance as yf
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp500_tickers = sp500.constituents.index.tolist() if hasattr(sp500, "constituents") else []

            # NASDAQ 100
            nasdaq100 = yf.Ticker("^NDX")
            nasdaq100_tickers = nasdaq100.constituents.index.tolist() if hasattr(nasdaq100, "constituents") else []

            tickers = sorted(list(set(sp500_tickers + nasdaq100_tickers)))
            if tickers:
                p.write_text("\n".join(tickers))
                st.success(f"Saved {len(tickers)} tickers to {file_path}")
            else:
                st.error("Failed to fetch tickers via yfinance.")
            return tickers
        except Exception as e:
            st.error(f"Failed to auto-populate tickers: {e}")
            return []

tickers = load_universe()
if not tickers:
    st.stop()

# Filter tickers by price using last close price
valid_tickers = []
skipped_tickers = []

batch_size = 100
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, period="5d", interval="1d", progress=False, threads=True, auto_adjust=False)
    except Exception:
        skipped_tickers.extend(batch)
        continue

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].iloc[-1]
        for t in batch:
            try:
                price = closes[t] if t in closes else np.nan
                if pd.isna(price):
                    skipped_tickers.append(t)
                elif min_price <= price <= max_price:
                    valid_tickers.append(t)
            except Exception:
                skipped_tickers.append(t)
    else:
        # Single ticker case
        try:
            price = data["Close"].iloc[-1]
            if min_price <= price <= max_price:
                valid_tickers.append(batch[0])
        except Exception:
            skipped_tickers.append(batch[0])

if skipped_tickers:
    st.sidebar.warning(f"Skipped {len(skipped_tickers)} tickers due to missing data or delisted.")

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