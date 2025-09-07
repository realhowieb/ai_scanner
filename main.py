import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import time
import warnings

# --- Parallel download defaults (UI removed) ---
use_parallel = True        # set False to force batch mode
parallel_chunk = 800       # tickers per worker in parallel mode
parallel_workers = 4       # number of worker threads for chunked downloads


# --- Ticker normalization helpers ---
import re

def normalize_ticker(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s:
        return ""
    if s.startswith("$"):
        s = s[1:]
    # Replace dot with dash only for US class shares like BRK.B; keep international suffix dots (e.g., 0700.HK)
    if "." in s:
        parts = s.split(".")
        if len(parts[-1]) == 1:  # class-share notation
            s = "-".join(parts)
        else:
            # keep dot for markets like .HK, .TO, etc.
            pass
    s = s.replace(" ", "")
    return s

def sanitize_ticker_list(tickers):
    seen = set()
    cleaned = []
    for t in tickers:
        s = normalize_ticker(t)
        if not s:
            continue
        # drop obvious non-symbols
        if any(ch in s for ch in (":", "/", "\\")):
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned

# --- US-only filter ---
def filter_us_tickers(tickers):
    """
    Keep likely U.S.-listed tickers.
    Heuristic: drop symbols with a dot suffix used by many non-U.S. exchanges
    (e.g., '0700.HK', 'SHOP.TO', 'RY.TO', 'INFY.NS').
    U.S. class shares like 'BRK.B' are normalized to 'BRK-B' in normalize_ticker,
    so anything that still contains a '.' is treated as non-U.S. and removed.
    """
    result = []
    for t in tickers:
        s = normalize_ticker(t)
        if not s:
            continue
        # If there is still a dot in the symbol after normalization, treat as non-U.S.
        if "." in s:
            parts = s.split(".")
            # if it's a class-share pattern (single-letter suffix), normalize it to dash just in case
            if len(parts[-1]) == 1:
                s = "-".join(parts)
            else:
                # foreign exchange suffix -> skip
                continue
        result.append(s)
    return result

# --- Diagnostics helper ---
def show_diagnostics(title: str, *,
                     universe_before: int,
                     universe_after: int,
                     chunk_size: int,
                     workers: int,
                     downloaded_count: int,
                     skipped_count: int,
                     elapsed_s: float):
    removed = max(universe_before - universe_after, 0)
    st.subheader(f"{title} Diagnostics")
    st.markdown(
        f"""
- **Universe size (before filters):** {universe_before}
- **After US-only & cleaning:** {universe_after}  *(removed {removed})*
- **Chunk size / workers:** {chunk_size} / {workers}
- **Downloaded successfully:** {downloaded_count}
- **Skipped / no data:** {skipped_count}
- **Download time:** {elapsed_s:.2f}s
        """
    )

# --- Rolling log panel helper (thread-safe, no Session State dependencies) ---
from collections import deque
_global_log_buffers = {}
_global_log_locks = {}
_global_master_lock = Lock()

def new_log_panel(title: str = "Diagnostics Log", key: str = None, expanded: bool = False):
    if key is None:
        key = f"log_{title.replace(' ', '_')}"
    # Initialize per-key buffer and lock once, safely
    with _global_master_lock:
        if key not in _global_log_buffers:
            _global_log_buffers[key] = deque(maxlen=500)
        if key not in _global_log_locks:
            _global_log_locks[key] = Lock()

    def logger(msg: str):
        ts = time.strftime('%H:%M:%S')
        with _global_log_locks[key]:
            _global_log_buffers[key].append(f"[{ts}] {msg}")

    def render():
        exp = st.expander(title, expanded=expanded)
        with exp:
            with _global_log_locks[key]:
                lines = list(_global_log_buffers[key])
            st.code("\n".join(lines), language='text')

    return logger, render

# --- No-op helper for conditional diagnostics UI ---
def _noop(*args, **kwargs):
    return None

# --- Gap / Unusual Volume Scanner ---
def gap_unusual_volume_scanner(price_data):
    """
    Scan for tickers with an opening gap > 4% or unusual volume (>2x 20d avg volume).
    Returns a DataFrame.
    """
    results = []
    for ticker, df in price_data.items():
        if df.empty or len(df) < 21 or 'Close' not in df.columns or 'Open' not in df.columns or 'Volume' not in df.columns:
            continue
        try:
            prev_close = df['Close'].iloc[-2].item()
            today_open = df['Open'].iloc[-1].item()
            today_close = df['Close'].iloc[-1].item()
            today_volume = df['Volume'].iloc[-1].item()
            avg_20d_vol = df['Volume'].iloc[-21:-1].mean().item()
            gap_pct = ((today_open - prev_close) / prev_close) * 100 if prev_close else np.nan
            unusual_vol = today_volume > 2 * avg_20d_vol if avg_20d_vol else False
            if abs(gap_pct) >= 4 or unusual_vol:
                results.append({
                    'Ticker': ticker,
                    'Prev Close': round(prev_close, 3),
                    'Today Open': round(today_open, 3),
                    'Today Close': round(today_close, 3),
                    'Gap %': round(gap_pct, 2),
                    'Today Vol': int(today_volume),
                    'Avg 20d Vol': int(avg_20d_vol),
                    'Unusual Vol (>2x avg)': unusual_vol
                })
        except Exception:
            continue
    if results:
        return pd.DataFrame(results).sort_values('Gap %', key=abs, ascending=False).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['Ticker', 'Prev Close', 'Today Open', 'Today Close', 'Gap %', 'Today Vol', 'Avg 20d Vol', 'Unusual Vol (>2x avg)'])

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
            first_price = premarket_df["Close"].iloc[0].item()
            last_price = premarket_df["Close"].iloc[-1].item()
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
            first_price = post_df["Close"].iloc[0].item()
            last_price = post_df["Close"].iloc[-1].item()
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
                    ticker = normalize_ticker(cols[0].text)
                    if ticker:
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
                    ticker = normalize_ticker(cols[0].text)
                    if ticker:
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
                    ticker = normalize_ticker(cols[0].text)
                    if ticker:
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
        tickers = sanitize_ticker_list(tickers)

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
        tickers = [normalize_ticker(line.split()[0]) for line in raw_lines if line.strip()]
    return tickers

# Load S&P 500 tickers
@st.cache_data(ttl=3600)
def load_sp500_tickers(file_path="sp500.txt"):
    p = Path(file_path)
    tickers = []
    if p.exists() and p.stat().st_size > 0:
        raw_lines = p.read_text().splitlines()
        tickers = [normalize_ticker(line.split()[0]) for line in raw_lines if line.strip()]
    return tickers

# New function to fetch and save S&P 500 tickers from Wikipedia
def fetch_and_save_sp500(file_path="sp500.txt"):
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df['Symbol'].astype(str).tolist()
        tickers = [normalize_ticker(t) for t in tickers]
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

# --- Robust splitter for multi-ticker DataFrames from yfinance ---
def _split_multi_ticker_df(df, chunk):
    """Return dict[ticker] -> DataFrame regardless of MultiIndex orientation.
    yfinance sometimes returns MultiIndex with either level-0=ticker or level-0=field.
    This helper tries both and returns only non-empty frames containing a 'Close' column.
    """
    import pandas as pd
    out = {}
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker case already handled upstream; do nothing here.
        return out

    lv0 = list(df.columns.levels[0])
    lv1 = list(df.columns.levels[1]) if df.columns.nlevels > 1 else []

    # Try orientation A: level-0 == ticker
    tickers_lv0 = set(df.columns.get_level_values(0))
    found_any = False
    for t in chunk:
        if t in tickers_lv0:
            sub = df[t]
            if not sub.empty and 'Close' in sub.columns:
                out[t] = sub
                found_any = True
    if found_any:
        return out

    # Try orientation B: level-1 == ticker (level-0=field: Open/High/...)
    if df.columns.nlevels >= 2:
        tickers_lv1 = set(df.columns.get_level_values(1))
        for t in chunk:
            if t in tickers_lv1:
                try:
                    sub = df.xs(t, axis=1, level=1)
                except Exception:
                    continue
                if not sub.empty and 'Close' in sub.columns:
                    out[t] = sub
        if out:
            return out

    return out

# Parallel price data fetcher (ThreadPoolExecutor)
def fetch_price_data_parallel(
    tickers,
    period="60d",
    interval="1d",
    chunk_size=800,
    max_workers=4,
    max_retries=3,
    retry_sleep=1.0,
    logger=None,
):
    import yfinance as yf
    import pandas as pd

    # Clean & dedupe
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if logger:
        logger(f"Prepared {len(tickers)} symbols (chunk_size={chunk_size}, max_workers={max_workers})")

    def fetch_chunk(chunk):
        if logger:
            logger(f"Downloading chunk of {len(chunk)} symbols…")
        # Retry the whole chunk a few times
        for attempt in range(max_retries):
            try:
                warnings.filterwarnings(
                    "ignore",
                    message=r"YF\.download\(\) has changed argument auto_adjust default to True",
                    category=FutureWarning,
                )
                df = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    threads=False,        # we already parallelize chunks
                    auto_adjust=False     # keep raw OHLC for candlesticks
                )
                return chunk, df
            except Exception:
                if attempt < max_retries - 1:
                    if logger:
                        logger(f"Chunk retry {attempt+1}/{max_retries} after backoff {retry_sleep * (2 ** attempt):.1f}s")
                    time.sleep(retry_sleep * (2 ** attempt))
                    continue
                return chunk, None

    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(fetch_chunk, chunks):
            results.append(res)

    if logger:
        logger(f"All chunks returned; processing per-ticker frames…")
    price_data, skipped = {}, []

    for chunk, df in results:
        # If the whole chunk failed, try per-ticker fallback with retries
        if df is None:
            if logger:
                logger(f"Chunk failed; attempting per-ticker fallback for {len(chunk)} symbols…")
            for t in chunk:
                got = False
                for attempt in range(max_retries):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            price_data[t] = df_t
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        time.sleep(retry_sleep * (2 ** attempt))
                        continue
                if not got:
                    if logger:
                        logger(f"Skipped {t}: no data after retries")
                    skipped.append(t)
            continue

        # Split multi-ticker DataFrame robustly
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            split = _split_multi_ticker_df(df, chunk)
            present = set(split.keys())
            for t in chunk:
                if t in present:
                    price_data[t] = split[t]
                else:
                    # Fallback per-ticker download with retries
                    got = False
                    for attempt in range(max_retries):
                        try:
                            df_t = yf.download(
                                t, period=period, interval=interval,
                                progress=False, threads=False, auto_adjust=False
                            )
                            if not df_t.empty and 'Close' in df_t.columns:
                                price_data[t] = df_t
                                got = True
                                if logger:
                                    logger(f"Recovered {t} via fallback")
                                break
                        except Exception:
                            time.sleep(retry_sleep * (2 ** attempt))
                            continue
                    if not got:
                        if logger:
                            logger(f"Skipped {t}: no data after retries")
                        skipped.append(t)
        else:
            # Single ticker returned (or completely empty)
            if not df.empty and 'Close' in df.columns:
                price_data[chunk[0]] = df
            else:
                t = chunk[0]
                got = False
                for attempt in range(max_retries):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            price_data[t] = df_t
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        time.sleep(retry_sleep * (2 ** attempt))
                        continue
                if not got:
                    if logger:
                        logger(f"Skipped {t}: no data after retries")
                    skipped.append(t)

    return price_data, skipped

# --- Streaming price data fetcher (process UI per-chunk) ---
def fetch_price_data_streaming(
    tickers,
    period="60d",
    interval="1d",
    chunk_size=800,
    max_workers=4,
    max_retries=3,
    retry_sleep=1.0,
    logger=None,
    progress_cb=None,
    per_chunk_cb=None,
):
    import yfinance as yf
    import pandas as pd

    # Clean & dedupe
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if logger:
        logger(f"Prepared {len(tickers)} symbols (chunk_size={chunk_size}, max_workers={max_workers})")

    def fetch_chunk(chunk):
        if logger:
            logger(f"Downloading chunk of {len(chunk)} symbols…")
        for attempt in range(max_retries):
            try:
                warnings.filterwarnings(
                    "ignore",
                    message=r"YF\.download\(\) has changed argument auto_adjust default to True",
                    category=FutureWarning,
                )
                df = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    auto_adjust=False,
                )
                return chunk, df
            except Exception:
                if attempt < max_retries - 1:
                    if logger:
                        logger(f"Chunk retry {attempt+1}/{max_retries} after backoff {retry_sleep * (2 ** attempt):.1f}s")
                    time.sleep(retry_sleep * (2 ** attempt))
                    continue
                return chunk, None

    def process_chunk(chunk, df):
        added, skipped_local = {}, []
        if df is None:
            if logger:
                logger(f"Chunk failed; attempting per-ticker fallback for {len(chunk)} symbols…")
            for t in chunk:
                got = False
                for attempt in range(max_retries):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            added[t] = df_t
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        time.sleep(retry_sleep * (2 ** attempt))
                        continue
                if not got:
                    skipped_local.append(t)
                    if logger:
                        logger(f"Skipped {t}: no data after retries")
            return added, skipped_local

        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            split = _split_multi_ticker_df(df, chunk)
            present = set(split.keys())
            for t in chunk:
                if t in present:
                    added[t] = split[t]
                else:
                    # try single fallback
                    got = False
                    for attempt in range(max_retries):
                        try:
                            df_t = yf.download(
                                t, period=period, interval=interval,
                                progress=False, threads=False, auto_adjust=False
                            )
                            if not df_t.empty and 'Close' in df_t.columns:
                                added[t] = df_t
                                got = True
                                if logger:
                                    logger(f"Recovered {t} via fallback")
                                break
                        except Exception:
                            time.sleep(retry_sleep * (2 ** attempt))
                            continue
                    if not got:
                        skipped_local.append(t)
                        if logger:
                            logger(f"Skipped {t}: no data after retries")
        else:
            # Single ticker returned (or empty)
            if not df.empty and 'Close' in df.columns:
                added[chunk[0]] = df
            else:
                t = chunk[0]
                got = False
                for attempt in range(max_retries):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            added[t] = df_t
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        time.sleep(retry_sleep * (2 ** attempt))
                        continue
                if not got:
                    skipped_local.append(t)
                    if logger:
                        logger(f"Skipped {t}: no data after retries")
        return added, skipped_local

    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    total = len(chunks)
    price_data, skipped = {}, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {executor.submit(fetch_chunk, ch): ch for ch in chunks}
        processed = 0
        for fut in as_completed(fut_map):
            ch = fut_map[fut]
            try:
                chunk, df = fut.result()
            except Exception:
                chunk, df = ch, None
            added, skipped_local = process_chunk(chunk, df)
            price_data.update(added)
            skipped.extend(skipped_local)
            processed += 1

            if per_chunk_cb is not None:
                try:
                    per_chunk_cb(added)
                except Exception:
                    pass

            if progress_cb is not None:
                try:
                    progress_cb(processed, total)
                except Exception:
                    pass

    return price_data, skipped

# Filter tickers by price
#st.sidebar.markdown("### Filter by Price")
def filter_tickers_by_price(price_data, min_price, max_price):
    filtered = []
    for t, df in price_data.items():
        if df.empty or 'Close' not in df.columns:
            continue
        try:
            price = df['Close'].iloc[-1].item()
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
            latest_close = df['Close'].iloc[-1].item()
            prev_max = df['Close'].iloc[-21:-1].max().item()
            latest_volume = df['Volume'].iloc[-1].item() if 'Volume' in df.columns else np.nan
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


st.sidebar.markdown("### Filter by Price")
min_price = st.sidebar.number_input("Minimum Price ($)", min_value=0.0, value=5.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0.0, value=1000.0, step=1.0)


# Upload Ticker Files
st.sidebar.markdown("### Upload Tickers")
uploaded_file = st.sidebar.file_uploader("Upload tickers file (.txt or .csv)", type=['txt', 'csv'])

# Sidebar checkbox for gap/unusual volume filter
st.sidebar.markdown("### Sort Modifier")
apply_gap_filter = st.sidebar.checkbox("Apply Gap / Unusual Volume Filter", value=True)
us_only = st.sidebar.checkbox("US-only filter", value=True)
# --- Diagnostics toggle ---
show_diagnostics_ui = st.sidebar.checkbox("Show Diagnostics", value=False)



if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode("utf-8")
            lines = content.splitlines()
            tickers = sanitize_ticker_list(lines)
        else:
            df_uploaded = pd.read_csv(uploaded_file, header=None)
            raw = []
            for col in df_uploaded.columns:
                raw.extend([str(t) for t in df_uploaded[col] if str(t).strip()])
            tickers = sanitize_ticker_list(raw)
        st.session_state.tickers = tickers

        uni_before = len(st.session_state.tickers)
        if us_only:
            st.session_state.tickers = filter_us_tickers(st.session_state.tickers)
        uni_after = len(st.session_state.tickers)

        # Run auto_scan immediately after upload
        t0 = time.perf_counter()
        breakout_df, skipped, filtered_data = auto_scan(min_price, max_price)
        t1 = time.perf_counter()
        show_diagnostics(
            "Uploaded Tickers",
            universe_before=uni_before,
            universe_after=uni_after,
            chunk_size=parallel_chunk,
            workers=parallel_workers,
            downloaded_count=len(filtered_data),
            skipped_count=len(skipped),
            elapsed_s=t1 - t0,
        )

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
                if (
                    df is None
                    or df.empty
                    or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
                ):
                    continue
                # mplfinance expects columns: Open, High, Low, Close, Volume
                fig, axlist = mpf.plot(
                    df,
                    type='candle',
                    style='yahoo',
                    mav=(20,50),
                    volume=True,
                    figratio=(12,4),
                    figscale=1.2,
                    title=f"{ticker} Candlestick Chart",
                    returnfig=True,
                    tight_layout=True
                )
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
else:
    if "tickers" not in st.session_state:
        st.session_state.tickers = load_sp600_tickers()
        if us_only:
            st.session_state.tickers = filter_us_tickers(st.session_state.tickers)

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
            st.success(f"Showing candlestick chart for {search_ticker} (last 60 days)")
            # Ensure columns for mplfinance
            if all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                fig, axlist = mpf.plot(
                    hist,
                    type='candle',
                    style='yahoo',
                    mav=(20,50),
                    volume=True,
                    figratio=(12,4),
                    figscale=1.2,
                    title=f"{search_ticker} Candlestick Chart",
                    returnfig=True,
                    tight_layout=True
                )
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

#Run Quick Stocks
st.sidebar.markdown("### Quick Stocks")
if st.sidebar.button("Run Hot Stocks Scan"):
    with st.spinner("Fetching and scanning hot stocks..."):
        hot_tickers = fetch_hot_stocks()
        uni_before = len(hot_tickers)
        if us_only:
            hot_tickers = filter_us_tickers(hot_tickers)
        uni_after = len(hot_tickers)
        if not hot_tickers:
            breakout_df, skipped, filtered_data = pd.DataFrame(columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %']), [], {}
        else:
            if show_diagnostics_ui:
                hot_log, hot_render = new_log_panel("Hot Stocks — Diagnostics Log", expanded=False)
            else:
                hot_log, hot_render = None, _noop
            t0 = time.perf_counter()
            if use_parallel:
                price_data, skipped = fetch_price_data_parallel(
                    hot_tickers, period="60d", interval="1d",
                    chunk_size=parallel_chunk, max_workers=parallel_workers,
                    logger=hot_log
                )
            else:
                price_data, skipped = fetch_price_data_batch(
                    hot_tickers, period="60d", interval="1d", batch_size=50
                )
            t1 = time.perf_counter()
            downloaded_count = len({t for t in price_data if t in hot_tickers})
            if show_diagnostics_ui:
                show_diagnostics(
                    "Hot Stocks",
                    universe_before=uni_before,
                    universe_after=uni_after,
                    chunk_size=parallel_chunk,
                    workers=parallel_workers,
                    downloaded_count=downloaded_count,
                    skipped_count=len(skipped),
                    elapsed_s=t1 - t0,
                )
                hot_render()
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            # Gap / Unusual Volume Filter
            if apply_gap_filter:
                gap_df = gap_unusual_volume_scanner(filtered_data)
                if not gap_df.empty:
                    st.subheader("Gap / Unusual Volume Candidates")
                    st.dataframe(gap_df)
                    st.download_button(
                        "Download Gap / Unusual Volume Results",
                        data=gap_df.to_csv(index=False),
                        file_name="gap_unusual_volume_results.csv",
                        mime="text/csv"
                    )
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
            if (
                df is None
                or df.empty
                or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            ):
                continue
            fig, axlist = mpf.plot(
                df,
                type='candle',
                style='yahoo',
                mav=(20,50),
                volume=True,
                figratio=(12,4),
                figscale=1.2,
                title=f"{ticker} Candlestick Chart",
                returnfig=True,
                tight_layout=True
            )
            st.pyplot(fig)

# Most Active Stocks Scan Button
if st.sidebar.button("Run Most Active Stocks Scan"):
    with st.spinner("Fetching and scanning most active stocks..."):
        most_active_tickers = fetch_most_active_stocks()
        uni_before = len(most_active_tickers)
        if us_only:
            most_active_tickers = filter_us_tickers(most_active_tickers)
        uni_after = len(most_active_tickers)
        if most_active_tickers:
            if show_diagnostics_ui:
                ma_log, ma_render = new_log_panel("Most Active — Diagnostics Log", expanded=False)
            else:
                ma_log, ma_render = None, _noop
            t0 = time.perf_counter()
            if use_parallel:
                price_data, skipped = fetch_price_data_parallel(
                    most_active_tickers, period="60d", interval="1d",
                    chunk_size=parallel_chunk, max_workers=parallel_workers,
                    logger=ma_log
                )
            else:
                price_data, skipped = fetch_price_data_batch(
                    most_active_tickers, period="60d", interval="1d", batch_size=50
                )
            t1 = time.perf_counter()
            downloaded_count = len({t for t in price_data if t in most_active_tickers})
            if show_diagnostics_ui:
                show_diagnostics(
                    "Most Active",
                    universe_before=uni_before,
                    universe_after=uni_after,
                    chunk_size=parallel_chunk,
                    workers=parallel_workers,
                    downloaded_count=downloaded_count,
                    skipped_count=len(skipped),
                    elapsed_s=t1 - t0,
                )
                ma_render()
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            # Gap / Unusual Volume Filter
            if apply_gap_filter:
                gap_df = gap_unusual_volume_scanner(filtered_data)
                if not gap_df.empty:
                    st.subheader("Gap / Unusual Volume Candidates")
                    st.dataframe(gap_df)
                    st.download_button(
                        "Download Gap / Unusual Volume Results",
                        data=gap_df.to_csv(index=False),
                        file_name="gap_unusual_volume_results.csv",
                        mime="text/csv"
                    )
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
                    if (
                        df is None
                        or df.empty
                        or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
                    ):
                        continue
                    fig, axlist = mpf.plot(
                        df,
                        type='candle',
                        style='yahoo',
                        mav=(20,50),
                        volume=True,
                        figratio=(12,4),
                        figscale=1.2,
                        title=f"{ticker} Candlestick Chart",
                        returnfig=True,
                        tight_layout=True
                    )
                    st.pyplot(fig)
        else:
            st.error("Failed to fetch most active stocks.")

# Trending Stocks Scan Button
if st.sidebar.button("Run Trending Scan"):
    with st.spinner("Fetching and scanning trending stocks..."):
        trending_tickers = fetch_trending_stocks()
        uni_before = len(trending_tickers)
        if us_only:
            trending_tickers = filter_us_tickers(trending_tickers)
        uni_after = len(trending_tickers)
        if trending_tickers:
            if show_diagnostics_ui:
                tr_log, tr_render = new_log_panel("Trending — Diagnostics Log", expanded=False)
            else:
                tr_log, tr_render = None, _noop
            t0 = time.perf_counter()
            if use_parallel:
                price_data, skipped = fetch_price_data_parallel(
                    trending_tickers, period="60d", interval="1d",
                    chunk_size=parallel_chunk, max_workers=parallel_workers,
                    logger=tr_log
                )
            else:
                price_data, skipped = fetch_price_data_batch(
                    trending_tickers, period="60d", interval="1d", batch_size=50
                )
            t1 = time.perf_counter()
            downloaded_count = len({t for t in price_data if t in trending_tickers})
            if show_diagnostics_ui:
                show_diagnostics(
                    "Trending",
                    universe_before=uni_before,
                    universe_after=uni_after,
                    chunk_size=parallel_chunk,
                    workers=parallel_workers,
                    downloaded_count=downloaded_count,
                    skipped_count=len(skipped),
                    elapsed_s=t1 - t0,
                )
                tr_render()
            filtered = filter_tickers_by_price(price_data, min_price, max_price)
            filtered_data = {t: price_data[t] for t in filtered if t in price_data}
            # Gap / Unusual Volume Filter
            if apply_gap_filter:
                gap_df = gap_unusual_volume_scanner(filtered_data)
                if not gap_df.empty:
                    st.subheader("Gap / Unusual Volume Candidates")
                    st.dataframe(gap_df)
                    st.download_button(
                        "Download Gap / Unusual Volume Results",
                        data=gap_df.to_csv(index=False),
                        file_name="gap_unusual_volume_results.csv",
                        mime="text/csv"
                    )
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
                    if (
                        df is None
                        or df.empty
                        or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
                    ):
                        continue
                    fig, axlist = mpf.plot(
                        df,
                        type='candle',
                        style='yahoo',
                        mav=(20,50),
                        volume=True,
                        figratio=(12,4),
                        figscale=1.2,
                        title=f"{ticker} Candlestick Chart",
                        returnfig=True,
                        tight_layout=True
                    )
                    st.pyplot(fig)
        else:
            st.error("Failed to fetch trending stocks.")


# --- Helper for S&P 500 scan (for both button and live update) ---
st.sidebar.markdown("### S&P and Nasdaq Stocks")
def run_sp500_scan(us_only, parallel_chunk, parallel_workers, min_price, max_price, apply_gap_filter):
    with st.spinner("Fetching S&P 500 tickers..."):
        tickers = load_sp500_tickers()
        uni_before = len(tickers)
        if us_only:
            tickers = filter_us_tickers(tickers)
        uni_after = len(tickers)

    if not tickers:
        st.error("No S&P 500 tickers available to scan.")
        return

    st.subheader("S&P 500 Breakout Scan Results (Live)")
    if show_diagnostics_ui:
        sp_log, sp_render = new_log_panel("S&P 500 — Diagnostics Log", expanded=False)
    else:
        sp_log, sp_render = None, _noop

    # Live UI placeholders
    progress = st.progress(0)
    live_table = st.empty()
    status = st.empty()

    # Use smaller chunk size for better streaming granularity on ~500 symbols
    sp_chunk = max(50, min(parallel_chunk, 100))
    t0 = time.perf_counter()

    def progress_cb(done, total):
        if total:
            progress.progress(done / total)
        status.markdown(f"Processed **{done}/{total}** chunks…")

    running_rows = []
    seen_tickers = set()

    def per_chunk_cb(added_price_data):
        # Filter by price and run breakout on just-arrived data
        filtered = filter_tickers_by_price(added_price_data, min_price, max_price)
        filtered_chunk = {t: added_price_data[t] for t in filtered}
        if not filtered_chunk:
            return
        chunk_df = breakout_scanner(filtered_chunk, min_price, max_price)
        if chunk_df.empty:
            return
        # Append only new tickers to the running result to avoid duplicates
        for _, row in chunk_df.iterrows():
            tkr = row['Ticker']
            if tkr in seen_tickers:
                continue
            seen_tickers.add(tkr)
            running_rows.append([
                row['Ticker'], row['Latest Close'], row['Previous 20d Max'], row['Breakout %'], row['Volume']
            ])
        if running_rows:
            live_df = pd.DataFrame(
                running_rows,
                columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %', 'Volume']
            ).sort_values('Breakout %', ascending=False).reset_index(drop=True)
            live_table.dataframe(live_df)

    # Choose streaming or batch path based on use_parallel flag
    if use_parallel:
        price_data, skipped = fetch_price_data_streaming(
            tickers,
            period="60d",
            interval="1d",
            chunk_size=sp_chunk,
            max_workers=parallel_workers,
            logger=sp_log,
            progress_cb=progress_cb,
            per_chunk_cb=per_chunk_cb,
        )
    else:
        # Batch path: still show progress per batch
        total_batches = max(1, (len(tickers) + 49) // 50)
        done_batches = 0
        price_data, skipped = {}, []
        for i in range(0, len(tickers), 50):
            batch = tickers[i:i+50]
            pdict, skipped_batch = fetch_price_data_batch(batch, period="60d", interval="1d", batch_size=50)
            price_data.update(pdict)
            skipped.extend(skipped_batch)
            done_batches += 1
            progress_cb(done_batches, total_batches)
            per_chunk_cb(pdict)

    t1 = time.perf_counter()
    downloaded_count = len({t for t in price_data if t in tickers})
    if show_diagnostics_ui:
        show_diagnostics(
            "S&P 500",
            universe_before=uni_before,
            universe_after=uni_after,
            chunk_size=parallel_chunk,
            workers=parallel_workers,
            downloaded_count=downloaded_count,
            skipped_count=len(skipped),
            elapsed_s=t1 - t0,
        )
        sp_render()

    # Final consolidated view
    filtered_all = filter_tickers_by_price(price_data, min_price, max_price)
    filtered_data = {t: price_data[t] for t in filtered_all if t in price_data}

    # Gap / Unusual Volume (final table)
    if apply_gap_filter:
        gap_df = gap_unusual_volume_scanner(filtered_data)
        if not gap_df.empty:
            st.subheader("Gap / Unusual Volume Candidates")
            st.dataframe(gap_df)
            st.download_button(
                "Download Gap / Unusual Volume Results",
                data=gap_df.to_csv(index=False),
                file_name="gap_unusual_volume_results.csv",
                mime="text/csv"
            )

    breakout_df = breakout_scanner(filtered_data, min_price, max_price)
    if not breakout_df.empty:
        st.success(f"Scan complete ✅ Found {len(breakout_df)} breakout candidates in S&P 500.")
        final_df = breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True)
        live_table.dataframe(final_df)  # update the same live table
    else:
        st.info("No breakout candidates found in S&P 500.")

    if skipped:
        st.warning(f"Skipped {len(skipped)} tickers due to missing data or delisted.")

if st.sidebar.button("Run S&P 500 Scan"):
    run_sp500_scan(us_only, parallel_chunk, parallel_workers, min_price, max_price, apply_gap_filter)


#Run Nasdaq Button
if st.sidebar.button("Run Nasdaq Scan"):
    with st.spinner("Fetching Nasdaq tickers..."):
        tickers = fetch_and_save_nasdaq()
        uni_before = len(tickers)
        if us_only:
            tickers = filter_us_tickers(tickers)
        uni_after = len(tickers)

    if not tickers:
        st.error("No Nasdaq tickers available to scan.")
    else:
        st.subheader("Nasdaq Breakout Scan Results (Live)")
        if show_diagnostics_ui:
            ndq_log, ndq_render = new_log_panel("Nasdaq — Diagnostics Log", expanded=False)
        else:
            ndq_log, ndq_render = None, _noop

        # Live UI placeholders
        progress = st.progress(0)
        live_table = st.empty()
        status = st.empty()

        # Use a smaller chunk for better streaming granularity on large universe
        ndq_chunk = max(100, min(parallel_chunk, 200))
        t0 = time.perf_counter()

        def progress_cb(done, total):
            if total:
                progress.progress(done / total)
            status.markdown(f"Processed **{done}/{total}** chunks…")

        running_rows = []
        seen_tickers = set()

        def per_chunk_cb(added_price_data):
            # Filter by price and run breakout on just-arrived data
            filtered = filter_tickers_by_price(added_price_data, min_price, max_price)
            filtered_chunk = {t: added_price_data[t] for t in filtered}
            if not filtered_chunk:
                return
            chunk_df = breakout_scanner(filtered_chunk, min_price, max_price)
            if chunk_df.empty:
                return
            # Append only new tickers to the running result to avoid duplicates
            for _, row in chunk_df.iterrows():
                tkr = row['Ticker']
                if tkr in seen_tickers:
                    continue
                seen_tickers.add(tkr)
                running_rows.append([
                    row['Ticker'], row['Latest Close'], row['Previous 20d Max'], row['Breakout %'], row['Volume']
                ])
            if running_rows:
                live_df = pd.DataFrame(
                    running_rows,
                    columns=['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %', 'Volume']
                ).sort_values('Breakout %', ascending=False).reset_index(drop=True)
                live_table.dataframe(live_df)

        # Choose streaming or batch path based on use_parallel flag
        if use_parallel:
            price_data, skipped = fetch_price_data_streaming(
                tickers,
                period="60d",
                interval="1d",
                chunk_size=ndq_chunk,
                max_workers=parallel_workers,
                logger=ndq_log,
                progress_cb=progress_cb,
                per_chunk_cb=per_chunk_cb,
            )
        else:
            # Batch path: still show progress per batch
            total_batches = max(1, (len(tickers) + 49) // 50)
            done_batches = 0
            price_data, skipped = {}, []
            for i in range(0, len(tickers), 50):
                batch = tickers[i:i+50]
                pdict, skipped_batch = fetch_price_data_batch(batch, period="60d", interval="1d", batch_size=50)
                price_data.update(pdict)
                skipped.extend(skipped_batch)
                done_batches += 1
                progress_cb(done_batches, total_batches)
                per_chunk_cb(pdict)

        t1 = time.perf_counter()
        downloaded_count = len({t for t in price_data if t in tickers})
        if show_diagnostics_ui:
            show_diagnostics(
                "Nasdaq",
                universe_before=uni_before,
                universe_after=uni_after,
                chunk_size=ndq_chunk if use_parallel else 50,
                workers=parallel_workers if use_parallel else 1,
                downloaded_count=downloaded_count,
                skipped_count=len(skipped),
                elapsed_s=t1 - t0,
            )
            ndq_render()

        # Final consolidated view
        filtered_all = filter_tickers_by_price(price_data, min_price, max_price)
        filtered_data = {t: price_data[t] for t in filtered_all if t in price_data}

        # Gap / Unusual Volume (final table)
        if apply_gap_filter:
            gap_df = gap_unusual_volume_scanner(filtered_data)
            if not gap_df.empty:
                st.subheader("Gap / Unusual Volume Candidates")
                st.dataframe(gap_df)
                st.download_button(
                    "Download Gap / Unusual Volume Results",
                    data=gap_df.to_csv(index=False),
                    file_name="gap_unusual_volume_results.csv",
                    mime="text/csv"
                )

        breakout_df = breakout_scanner(filtered_data, min_price, max_price)
        if not breakout_df.empty:
            st.success(f"Scan complete ✅ Found {len(breakout_df)} breakout candidates in Nasdaq.")
            final_df = breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True)
            live_table.dataframe(final_df)  # update the same live table
        else:
            st.info("No breakout candidates found in Nasdaq.")

        if skipped:
            st.warning(f"Skipped {len(skipped)} tickers due to missing data or delisted.")

# Add Pre-market Scan Button
st.sidebar.markdown("### During Pre-Market Hour")
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
st.sidebar.markdown("### During Post-Market Hour")
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

# Fetch S&P 500 Tickers Button
st.sidebar.markdown("### Maintenance Only")
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
