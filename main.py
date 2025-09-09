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
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import time
import warnings


# --- Parallel download defaults (UI removed) ---
use_parallel = True        # set False to force batch mode
parallel_chunk = 800       # tickers per worker in parallel mode
parallel_workers = 4       # number of worker threads for chunked downloads

# -- DB Helpers --
# === Persistence (SQLite) ===
import os, json, sqlite3

DB_PATH = st.secrets.get("DB_PATH", "scanner.sqlite")

def _db_conn():
    # check_same_thread=False to allow Streamlit callbacks/threads to write
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with _db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            universe_before INTEGER,
            universe_after INTEGER,
            downloaded_count INTEGER,
            skipped_count INTEGER,
            elapsed_s REAL,
            params_json TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            ticker TEXT,
            latest_close REAL,
            prev_20d_max REAL,
            breakout_pct REAL,
            volume TEXT,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            atr REAL,
            rs20 REAL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )""")
        conn.commit()

def save_run(run_type: str, meta: dict, results_df: pd.DataFrame) -> int:
    """Persist one scan (header in 'runs' + rows in 'results'). Returns run_id."""
    init_db()
    with _db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO runs
               (run_type, universe_before, universe_after, downloaded_count, skipped_count, elapsed_s, params_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_type,
                int(meta.get("universe_before", 0)),
                int(meta.get("universe_after", 0)),
                int(meta.get("downloaded_count", 0)),
                int(meta.get("skipped_count", 0)),
                float(meta.get("elapsed_s", 0.0)),
                json.dumps(meta.get("params", {})),
            ),
        )
        run_id = cur.lastrowid

        if results_df is not None and not results_df.empty:
            df = results_df.copy()
            df["run_id"] = run_id
            def _col(df, name): return df[name] if name in df.columns else np.nan
            out = pd.DataFrame({
                "run_id": df["run_id"],
                "ticker": _col(df, "Ticker"),
                "latest_close": _col(df, "Latest Close"),
                "prev_20d_max": _col(df, "Previous 20d Max"),
                "breakout_pct": _col(df, "Breakout %"),
                "volume": _col(df, "Volume"),
                "rsi": _col(df, "RSI(14)"),
                "macd": _col(df, "MACD"),
                "macd_signal": _col(df, "MACD Signal"),
                "atr": _col(df, "ATR(14)"),
                "rs20": _col(df, "RS 20d vs SPY (%)"),
            })
            out.to_sql("results", conn, if_exists="append", index=False)
        conn.commit()
        return run_id

def list_runs(limit: int = 200) -> pd.DataFrame:
    init_db()
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT id, run_type, started_at, universe_before, universe_after, "
            "downloaded_count, skipped_count, elapsed_s FROM runs "
            "ORDER BY id DESC LIMIT ?",
            conn, params=(limit,)
        )

def load_run_results(run_id: int) -> pd.DataFrame:
    init_db()
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT ticker AS 'Ticker', latest_close AS 'Latest Close', "
            "prev_20d_max AS 'Previous 20d Max', breakout_pct AS 'Breakout %', "
            "volume AS 'Volume', rsi AS 'RSI(14)', macd AS 'MACD', "
            "macd_signal AS 'MACD Signal', atr AS 'ATR(14)', rs20 AS 'RS 20d vs SPY (%)' "
            "FROM results WHERE run_id=? ORDER BY CAST([Breakout %] AS REAL) DESC",
            conn, params=(run_id,)
        )

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

# --- Derivative/Problem instrument filters (Units / Warrants / Rights) ---
# Many raw universes contain SPAC Units (U), Warrants (W/WS/WT), or Rights (R)
# which frequently cause Yahoo 404/no-data. We filter those out by symbol pattern.

def _looks_like_unit_warrant_right(sym: str) -> bool:
    s = normalize_ticker(sym)
    if not s:
        return False
    # Do not block genuine one-letter tickers like 'W'
    if len(s) == 1:
        return False
    # Dash-separated suffix token, e.g., ABC-WS, ABC-WT, ABC-U, ABC-R
    parts = s.split('-')
    last = parts[-1]
    if last in {"U", "W", "WS", "WT", "R"}:
        # Allow exception: the entire symbol 'W' is legit, but handled above by len==1
        return True
    # SPAC-style without dash, e.g., ABCU, ABCW, ABCR
    if s.endswith(('U', 'W', 'R')) and len(s) > 1:
        # Avoid false positive for genuine single-letter 'W' already excluded
        return True
    return False

# --- Filter Ticker Helper ---
def filter_problem_tickers(tickers):
    out = []
    for t in tickers:
        s = normalize_ticker(t)
        if not s:
            continue
        # Units/Warrants/ Rights
        if _looks_like_unit_warrant_right(s):
            continue
        # Preferreds like ABC-PA, ABC-PB
        if re.search(r"-P[A-Z]$", s):
            continue
        # When-issued like ABC.WI
        if s.endswith(".WI"):
            continue
        out.append(s)
    return out

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

# --- Download Zip Button ---
def download_zip_button(label: str, files: dict, filename: str = "scan_bundle.zip"):
    """files: dict[name] = CSV string"""
    buf = io.BytesIO()
    with ZipFile(buf, 'w') as zf:
        for name, csv_data in files.items():
            zf.writestr(name, csv_data)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime='application/zip')

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

def premarket_scan(tickers, progress_cb=None, per_row_cb=None):
    """
    Perform a pre-market scan for a list of tickers.
    Streams results via per_row_cb as each qualifying ticker is found, and progress via progress_cb.
    Returns a DataFrame with columns: Ticker, Premarket First Price, Premarket Last Price, Premarket % Change.
    """
    import pandas as pd
    import yfinance as yf
    results = []
    total = len(tickers)
    done = 0
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", interval="5m", prepost=True)
            if df.empty or "Close" not in df.columns:
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass
                continue
            # Pre-market session is before 09:30:00 in NY time (df index is UTC)
            df_local = df.copy()
            if not df_local.index.tz:
                df_local.index = df_local.index.tz_localize("UTC")
            df_local.index = df_local.index.tz_convert("America/New_York")
            premarket_df = df_local[df_local.index.time < pd.to_datetime("09:30:00").time()]
            if premarket_df.empty:
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass
                continue
            first_price = premarket_df["Close"].iloc[0].item()
            last_price = premarket_df["Close"].iloc[-1].item()
            pct_change = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0.0
            row = {
                "Ticker": ticker,
                "Premarket First Price": round(first_price, 4),
                "Premarket Last Price": round(last_price, 4),
                "Premarket % Change": round(pct_change, 2),
            }
            results.append(row)
            if per_row_cb:
                try:
                    per_row_cb(row)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=["Ticker", "Premarket First Price", "Premarket Last Price", "Premarket % Change"])

# Post-market scan function (streaming)

def postmarket_scan(tickers, progress_cb=None, per_row_cb=None):
    """
    Perform a post-market scan for a list of tickers.
    Streams results via per_row_cb as each qualifying ticker is found, and progress via progress_cb.
    Returns a DataFrame with columns: Ticker, Postmarket First Price, Postmarket Last Price, Postmarket % Change.
    """
    import pandas as pd
    import yfinance as yf
    results = []
    total = len(tickers)
    done = 0
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", interval="5m", prepost=True)
            if df.empty or "Close" not in df.columns:
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass
                continue
            df_local = df.copy()
            if not df_local.index.tz:
                df_local.index = df_local.index.tz_localize("UTC")
            df_local.index = df_local.index.tz_convert("America/New_York")
            # Post-market session is >= 16:00:00
            post_df = df_local[df_local.index.time >= pd.to_datetime("16:00:00").time()]
            if post_df.empty:
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass
                continue
            first_price = post_df["Close"].iloc[0].item()
            last_price = post_df["Close"].iloc[-1].item()
            pct_change = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0.0
            row = {
                "Ticker": ticker,
                "Postmarket First Price": round(first_price, 4),
                "Postmarket Last Price": round(last_price, 4),
                "Postmarket % Change": round(pct_change, 2)
            }
            results.append(row)
            if per_row_cb:
                try:
                    per_row_cb(row)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass
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
        # Use HTTPS (FTP is often blocked/slow). Schema documented by Nasdaq Trader.
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        df = pd.read_csv(url, sep='|')

        # 1) Drop the trailer row and rows without a proper Symbol
        df = df[df['Symbol'].notna() & (df['Symbol'] != 'Symbol')]

        # 2) Remove Nasdaq test issues and problematic instruments
        #    - 'Test Issue' == 'Y' are synthetic symbols like ZAZZT, ZBZZT, ZCZZT, ZJZZT, ZWZZT, ZZZ, etc.
        #    - Security Name containing Unit / Warrant / Right often maps to tickers Yahoo doesn't serve (e.g., YHNAU).
        #    - Also filter out explicit Z*ZZT patterns as an extra guard.
        df = df[df.get('Test Issue', 'N') != 'Y']
        df = df[~df.get('Security Name', '').str.contains(r"\b(Unit|Warrant|Right)\b", case=False, na=False)]
        df = df[~df['Symbol'].astype(str).str.match(r"^[A-Z]+Z{2,}T$", na=False)]

        # 3) Optional: drop symbols flagged as deficient (Financial Status == 'D')
        if 'Financial Status' in df.columns:
            df = df[df['Financial Status'].fillna('N') != 'D']

        # Normalize & dedupe
        tickers = [normalize_ticker(t) for t in df['Symbol'].astype(str).tolist()]
        tickers = sanitize_ticker_list(tickers)
        tickers = filter_problem_tickers(tickers)

        # Persist
        Path(file_path).write_text('\n'.join(tickers))
        return tickers
    except Exception as e:
        st.warning(f"Failed to fetch Nasdaq tickers cleanly; falling back to any existing file. Error: {e}")
        try:
            p = Path(file_path)
            if p.exists() and p.stat().st_size > 0:
                return [normalize_ticker(line.split()[0]) for line in p.read_text().splitlines() if line.strip()]
        except Exception:
            pass
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
                        subdf = df[ticker].copy()
                        if not subdf.empty and 'Close' in subdf.columns:
                            price_data[ticker] = _downcast_ohlcv(subdf)
                        else:
                            skipped.append(ticker)
                    else:
                        skipped.append(ticker)
            else:
                # Only one ticker
                if not df.empty and 'Close' in df.columns:
                    price_data[batch[0]] = _downcast_ohlcv(df)
                else:
                    skipped.append(batch[0])
        except Exception:
            skipped.extend(batch)
    return price_data, skipped

# --- Robust splitter for multi-ticker DataFrames from yfinance ---
def _split_multi_ticker_df(df, chunk=None):
    import pandas as pd
    out = {}
    if df is None or df.empty:
        return out

    # If not a MultiIndex, nothing to split
    if not isinstance(df.columns, pd.MultiIndex):
        return out

    FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

    lvl0_vals = list(df.columns.get_level_values(0))
    lvl1_vals = list(df.columns.get_level_values(1)) if df.columns.nlevels > 1 else []

    def _harvest(level_as_ticker: int) -> dict:
        got = {}
        lvl = level_as_ticker
        try:
            tickers_here = [x for x in df.columns.get_level_values(lvl) if isinstance(x, str)]
            # Preserve order & dedupe
            seen = set()
            ordered = []
            for t in tickers_here:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            for t in ordered:
                try:
                    sub = df.xs(t, axis=1, level=lvl).copy()
                    if not sub.empty and any(col in sub.columns for col in ("Close", "close", "Adj Close")):
                        # Normalize column names in case Yahoo lowercases or uses spaces
                        cols = {c: c.title().replace("Adj close", "Adj Close") for c in sub.columns}
                        sub.rename(columns=cols, inplace=True)
                        got[str(t)] = sub
                except Exception:
                    continue
        except Exception:
            pass
        return got

    # Heuristic A: level-1 looks like fields -> level-0 are tickers
    if set(x for x in set(lvl1_vals) if isinstance(x, str)) & FIELDS:
        out = _harvest(0)
        if out:
            return out

    # Heuristic B: level-0 looks like fields -> level-1 are tickers
    if set(x for x in set(lvl0_vals) if isinstance(x, str)) & FIELDS:
        out = _harvest(1)
        if out:
            return out

    # Last resort: try both orientations and intersect with requested chunk if provided
    cand = {}
    cand.update(_harvest(0))
    cand.update(_harvest(1))
    if cand:
        if chunk:
            subset = {t: cand[t] for t in cand.keys() if t in set(chunk)}
            return subset if subset else cand
        return cand

    # Nothing matched cleanly
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


        # --- Helpers to rescue missing tickers in small mini-batches to reduce mass skips ---
    def _download_batch_yf(batch):
        """Robust yfinance batch download with normalized columns."""
        import yfinance as yf
        import pandas as pd
        df = yf.download(
            batch,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            threads=False,
            auto_adjust=False,
        )
        try:
            if isinstance(df.columns, pd.MultiIndex):
                new_levels = []
                for level in range(df.columns.nlevels):
                    vals = list(df.columns.get_level_values(level))
                    vals_norm = [str(v).strip() if isinstance(v, str) else v for v in vals]
                    new_levels.append(pd.Index(vals_norm))
                df.columns = pd.MultiIndex.from_arrays(new_levels)
            else:
                df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
        except Exception:
            pass
        return df

    def _split_multi_ticker_df_simple(df, want=None):
        """Split a MultiIndex df returned by yfinance into {ticker: subdf} quickly."""
        if not isinstance(df.columns, pd.MultiIndex):
            return {}
        # Try both orientations; keep whichever yields 'Close' column frames
        out = {}
        levels = [0, 1] if df.columns.nlevels == 2 else list(range(df.columns.nlevels))
        for lvl in levels:
            tickers_here = [x for x in df.columns.get_level_values(lvl) if isinstance(x, str)]
            seen = set()
            ordered = []
            for t in tickers_here:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            for t in ordered:
                try:
                    sub = df.xs(t, axis=1, level=lvl)
                    if 'Close' in sub.columns:
                        out[str(t)] = sub
                except Exception:
                    continue
        if want:
            want_set = set(want)
            out = {k: v for k, v in out.items() if k in want_set}
        return out

    def _rescue_missing_in_minibatches(missing_list, add_to_dict, tries=3, mini_size=8, sleep_base=1.0):
        """Attempt to refetch missing tickers in small batches to avoid throttling."""
        import time, random
        if not missing_list:
            return []
        still_missing = []
        for attempt in range(tries):
            # split into mini-batches
            for i in range(0, len(missing_list), mini_size):
                mb = missing_list[i:i+mini_size]
                try:
                    df_mb = _download_batch_yf(mb)
                    if isinstance(df_mb, pd.DataFrame) and not df_mb.empty and isinstance(df_mb.columns, pd.MultiIndex):
                        split = _split_multi_ticker_df_simple(df_mb, want=mb)
                        for t, sub in split.items():
                            try:
                                add_to_dict[t] = _downcast_ohlcv(sub)
                            except Exception:
                                continue
                except Exception:
                    pass
                # short jitter between mini-batches to ease rate limits
                time.sleep(sleep_base * (0.5 + random.random()))
            # build remaining set
            remaining = [t for t in missing_list if t not in add_to_dict]
            if not remaining:
                return []
            missing_list = remaining
        return missing_list

    def fetch_chunk(chunk):
        import warnings
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
                    threads=False,        # we already parallelize chunks
                    auto_adjust=False     # keep raw OHLC for candlesticks
                )
                # Normalize column names once to avoid case/spacing surprises
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        new_levels = []
                        for level in range(df.columns.nlevels):
                            vals = list(df.columns.get_level_values(level))
                            vals_norm = [str(v).strip() if isinstance(v, str) else v for v in vals]
                            new_levels.append(pd.Index(vals_norm))
                        df.columns = pd.MultiIndex.from_arrays(new_levels)
                    else:
                        df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
                except Exception:
                    pass
                return chunk, df
            except Exception:
                if attempt < max_retries - 1:
                    if logger:
                        logger(f"Chunk retry {attempt+1}/{max_retries} after backoff {retry_sleep * (2 ** attempt):.1f}s")
                    sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                    time.sleep(sleep_s)
                    continue
                return chunk, None
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
                # Normalize column names once to avoid case/spacing surprises
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        new_levels = []
                        for level in range(df.columns.nlevels):
                            vals = list(df.columns.get_level_values(level))
                            vals_norm = [str(v).strip() if isinstance(v, str) else v for v in vals]
                            new_levels.append(pd.Index(vals_norm))
                        df.columns = pd.MultiIndex.from_arrays(new_levels)
                    else:
                        df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
                except Exception:
                    pass
                return chunk, df
            except Exception:
                if attempt < max_retries - 1:
                    if logger:
                        logger(f"Chunk retry {attempt+1}/{max_retries} after backoff {retry_sleep * (2 ** attempt):.1f}s")
                    sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                    time.sleep(sleep_s)
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
                            price_data[t] = _downcast_ohlcv(df_t)
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                        time.sleep(sleep_s)
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
            # First add what we got
            for t in present:
                price_data[t] = _downcast_ohlcv(split[t])
            # Try a light-weight mini-batch rescue for the missing ones before per-ticker
            missing = [t for t in chunk if t not in present]
            missing = _rescue_missing_in_minibatches(missing, price_data, tries=3, mini_size=8, sleep_base=0.8)
            # Any still missing -> per-ticker fallback with retries (last resort)
            for t in list(missing):
                got = False
                for attempt in range(max_retries):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            price_data[t] = _downcast_ohlcv(df_t)
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                        time.sleep(sleep_s)
                        continue
                if not got:
                    if logger:
                        logger(f"Skipped {t}: no data after retries")
                    skipped.append(t)
        else:
            # Single ticker returned (or completely empty)
            if not df.empty and 'Close' in df.columns:
                price_data[chunk[0]] = _downcast_ohlcv(df)
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
                            price_data[t] = _downcast_ohlcv(df_t)
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                        time.sleep(sleep_s)
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

    # --- Helpers (streaming) to rescue missing tickers in small mini-batches ---
    def _download_batch_yf_stream(batch):
        import yfinance as yf
        import pandas as pd
        df = yf.download(
            batch,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            threads=False,
            auto_adjust=False,
        )
        try:
            if isinstance(df.columns, pd.MultiIndex):
                new_levels = []
                for level in range(df.columns.nlevels):
                    vals = list(df.columns.get_level_values(level))
                    vals_norm = [str(v).strip() if isinstance(v, str) else v for v in vals]
                    new_levels.append(pd.Index(vals_norm))
                df.columns = pd.MultiIndex.from_arrays(new_levels)
            else:
                df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
        except Exception:
            pass
        return df

    def _split_multi_ticker_df_simple_stream(df, want=None):
        if not isinstance(df.columns, pd.MultiIndex):
            return {}
        out = {}
        levels = [0, 1] if df.columns.nlevels == 2 else list(range(df.columns.nlevels))
        for lvl in levels:
            tickers_here = [x for x in df.columns.get_level_values(lvl) if isinstance(x, str)]
            seen = set()
            ordered = []
            for t in tickers_here:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            for t in ordered:
                try:
                    sub = df.xs(t, axis=1, level=lvl)
                    if 'Close' in sub.columns:
                        out[str(t)] = sub
                except Exception:
                    continue
        if want:
            want_set = set(want)
            out = {k: v for k, v in out.items() if k in want_set}
        return out

    def _rescue_missing_in_minibatches_stream(missing_list, add_to_dict, tries=3, mini_size=8, sleep_base=0.8):
        import time, random
        if not missing_list:
            return []
        for attempt in range(tries):
            for i in range(0, len(missing_list), mini_size):
                mb = missing_list[i:i+mini_size]
                try:
                    df_mb = _download_batch_yf_stream(mb)
                    if isinstance(df_mb, pd.DataFrame) and not df_mb.empty and isinstance(df_mb.columns, pd.MultiIndex):
                        split = _split_multi_ticker_df_simple_stream(df_mb, want=mb)
                        for t, sub in split.items():
                            try:
                                add_to_dict[t] = _downcast_ohlcv(sub)
                            except Exception:
                                continue
                except Exception:
                    pass
                time.sleep(sleep_base * (0.5 + random.random()))
            remaining = [t for t in missing_list if t not in add_to_dict]
            if not remaining:
                return []
            missing_list = remaining
        return missing_list

    def fetch_chunk(chunk):
        import warnings
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
                # Normalize column names once to avoid case/spacing surprises
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        new_levels = []
                        for level in range(df.columns.nlevels):
                            vals = list(df.columns.get_level_values(level))
                            vals_norm = [str(v).strip() if isinstance(v, str) else v for v in vals]
                            new_levels.append(pd.Index(vals_norm))
                        df.columns = pd.MultiIndex.from_arrays(new_levels)
                    else:
                        df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
                except Exception:
                    pass
                return chunk, df
            except Exception:
                if attempt < max_retries - 1:
                    if logger:
                        logger(f"Chunk retry {attempt+1}/{max_retries} after backoff {retry_sleep * (2 ** attempt):.1f}s")
                    sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                    time.sleep(sleep_s)
                    continue
                return chunk, None

    def process_chunk(chunk, df):
        # Scale fallback budget with chunk length so big chunks don't exhaust too fast
        fallback_budget_s = max(60.0, 1.0 * len(chunk))
        per_ticker_attempts = 4
        deadline = time.time() + fallback_budget_s
        added, skipped_local = {}, []
        if df is None:
            if logger:
                logger(f"Chunk failed; attempting per-ticker fallback for {len(chunk)} symbols…")
            for t in chunk:
                if time.time() > deadline:
                    # Out of time budget; skip the rest to keep UI moving
                    remaining = [x for x in chunk if x not in added and x not in skipped_local]
                    skipped_local.extend(remaining)
                    if logger:
                        logger(f"Fallback budget exceeded; skipped {len(remaining)} remaining symbols in chunk")
                    break
                got = False
                for attempt in range(per_ticker_attempts):
                    try:
                        df_t = yf.download(
                            t, period=period, interval=interval,
                            progress=False, threads=False, auto_adjust=False
                        )
                        if not df_t.empty and 'Close' in df_t.columns:
                            added[t] = _downcast_ohlcv(df_t)
                            got = True
                            if logger:
                                logger(f"Recovered {t} via fallback")
                            break
                    except Exception:
                        sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                        time.sleep(sleep_s)
                        continue
                if not got:
                    skipped_local.append(t)
                    if logger:
                        logger(f"Skipped {t}: no data after limited fallback")
            return added, skipped_local

        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            split = _split_multi_ticker_df(df, chunk)
            # Add what we already received
            for t, sub in split.items():
                try:
                    added[t] = _downcast_ohlcv(sub)
                except Exception:
                    continue
            # Try a mini-batch rescue for the missing ones first
            present = set(split.keys())
            missing = [t for t in chunk if t not in present]
            if time.time() <= deadline:
                missing = _rescue_missing_in_minibatches_stream(missing, added, tries=3, mini_size=8, sleep_base=0.8)
            # Any still missing -> per-ticker last resort within remaining budget
            for t in list(missing):
                got = False
                if time.time() <= deadline:
                    for attempt in range(per_ticker_attempts):
                        try:
                            df_t = yf.download(
                                t, period=period, interval=interval,
                                progress=False, threads=False, auto_adjust=False
                            )
                            if not df_t.empty and 'Close' in df_t.columns:
                                added[t] = _downcast_ohlcv(df_t)
                                got = True
                                if logger:
                                    logger(f"Recovered {t} via fallback")
                                break
                        except Exception:
                            sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                            time.sleep(sleep_s)
                            continue
                if not got:
                    skipped_local.append(t)
                    if logger:
                        logger(f"Skipped {t}: no data after limited fallback or budget")
        else:
            # Single ticker returned (or empty)
            if not df.empty and 'Close' in df.columns:
                added[chunk[0]] = _downcast_ohlcv(df)
            else:
                t = chunk[0]
                got = False
                if time.time() <= deadline:
                    for attempt in range(per_ticker_attempts):
                        try:
                            df_t = yf.download(
                                t, period=period, interval=interval,
                                progress=False, threads=False, auto_adjust=False
                            )
                            if not df_t.empty and 'Close' in df_t.columns:
                                added[t] = _downcast_ohlcv(df_t)
                                got = True
                                if logger:
                                    logger(f"Recovered {t} via fallback")
                                break
                        except Exception:
                            sleep_s = retry_sleep * (2 ** attempt) * (1 + random.random() * 0.25)
                            time.sleep(sleep_s)
                            continue
                if not got:
                    skipped_local.append(t)
                    if logger:
                        logger(f"Skipped {t}: no data after limited fallback or budget")
        return added, skipped_local

    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    total = len(chunks)
    price_data, skipped = {}, []

    from concurrent.futures import wait, FIRST_COMPLETED

    chunk_timeout = 240  # seconds watchdog per chunk (raised further to allow mini-batch rescues)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        attempts = {tuple(ch): 0 for ch in chunks}
        fut_map = {}
        start_time = {}

        def submit_chunk(ch):
            fut = executor.submit(fetch_chunk, ch)
            fut_map[fut] = tuple(ch)
            start_time[fut] = time.time()
            return fut

        for ch in chunks:
            submit_chunk(ch)

        processed = 0
        while fut_map:
            done, not_done = wait(list(fut_map.keys()), timeout=chunk_timeout, return_when=FIRST_COMPLETED)

            if not done:
                now = time.time()
                for fut in list(not_done):
                    ch_key = fut_map[fut]
                    if now - start_time[fut] >= chunk_timeout:
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        del start_time[fut]
                        del fut_map[fut]
                        attempts[ch_key] += 1
                        if logger:
                            logger(f"Watchdog: chunk {len(ch_key)} symbols timed out; attempt {attempts[ch_key]}/{max_retries}")
                        if attempts[ch_key] <= max_retries:
                            submit_chunk(list(ch_key))
                        else:
                            added, skipped_local = process_chunk(list(ch_key), None)
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
                continue

            for fut in done:
                ch_key = fut_map.pop(fut)
                start_time.pop(fut, None)
                try:
                    chunk, df = fut.result()
                except Exception:
                    chunk, df = list(ch_key), None

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

                # Add a small throttle after each completed chunk to ease rate limits
                try:
                    time.sleep(0.2)  # small inter-chunk throttle to avoid rate-limits
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
# --- Filter by Dollar Volume ---
def filter_by_dollar_volume(price_data, min_dollar_vol=2_000_000):
    """Return tickers whose latest dollar volume (Close * Volume) meets the threshold."""
    keep = []
    for t, df in price_data.items():
        if df is None or df.empty:
            continue
        if {'Close', 'Volume'}.issubset(df.columns):
            try:
                c = float(df['Close'].iloc[-1])
                v = float(df['Volume'].iloc[-1])
                if c * v >= float(min_dollar_vol):
                    keep.append(t)
            except Exception:
                continue
    return keep

def _downcast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint for OHLCV frames."""
    if df is None or df.empty:
        return df
    for c in ('Open','High','Low','Close'):
        if c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    if 'Volume' in df.columns:
        try:
            df.loc[:, 'Volume'] = pd.to_numeric(df['Volume'], errors='coerce').astype('int32')
        except Exception:
            df.loc[:, 'Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    return df
# Helper function to format large numbers
def format_volume(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.3f}M"
    elif v >= 1_000:
        return f"{v/1_000:.1f}K"
    else:
        return str(v)

# --- TA & SPY  Helpers
@st.cache_data(ttl=3600)
def get_spy_history(period="60d"):
    return yf.download("SPY", period=period, interval="1d",
                       progress=False, threads=False, auto_adjust=False)

# --- Technical indicator helpers ---
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series):
    macd = _ema(series, 12) - _ema(series, 26)
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _rs_20d_vs_spy(df_ticker: pd.DataFrame, df_spy: pd.DataFrame) -> float:
    """Relative strength vs SPY in percent over up to the last 20 trading days.
    If fewer than 21 points are available for either series, fall back to 10 or 5 days.
    Robust to index misalignment and missing days.
    """
    try:
        ct = pd.Series(df_ticker['Close']).astype(float)
        cs = pd.Series(df_spy['Close']).astype(float)
        # Align on union of dates and ffill to handle gaps/holidays
        idx = ct.index.union(cs.index)
        ct = ct.reindex(idx).ffill()
        cs = cs.reindex(idx).ffill()
        if ct.dropna().empty or cs.dropna().empty:
            return np.nan
        # Choose the largest feasible lookback from [20, 10, 5]
        for lb in (20, 10, 5):
            # need lb+1 observations to compute pct_change(lb)
            if ct.notna().sum() >= lb + 1 and cs.notna().sum() >= lb + 1:
                rt = ct.pct_change(lb)
                rs = cs.pct_change(lb)
                # evaluate at the last available date of the ticker
                last_date = df_ticker.index[-1]
                rel = (rt - rs).loc[:last_date].dropna()
                if not rel.empty:
                    return round(float(rel.iloc[-1] * 100), 2)
        return np.nan
    except Exception:
        return np.nan


# --- Database (Postgres first; fallback to SQLite) ---
import os, json, sqlite3
import streamlit as st
# --- Diagnostics toggle (define early so it's available everywhere) ---
if "show_diagnostics_ui" not in st.session_state:
    st.session_state["show_diagnostics_ui"] = False

show_diagnostics_ui = st.sidebar.checkbox(
    "Show Diagnostics",
    value=st.session_state["show_diagnostics_ui"],
    key="show_diagnostics_top",
)
st.session_state["show_diagnostics_ui"] = show_diagnostics_ui

# --- Scheduler (optional; uses scheduler.py module) ---
try:
    import scheduler as sched

    # Wire up headless runners if present; otherwise fall back to harmless st.toast
    def _dummy_run(name: str):
        def _f():
            try:
                st.toast(f"{name} run triggered (no headless runner wired)", icon="⏱️")
            except Exception:
                pass
            return -1
        return _f

    _run_pre = globals().get("run_premarket_headless") or _dummy_run("Pre-market")
    _run_post = globals().get("run_postmarket_headless") or _dummy_run("Post-market")
    _run_sp500_fn = globals().get("run_sp500_headless")

    # Support either run_sp500_headless(session_label="regular") or a simple callable
    def _run_sp500_wrapper():
        fn = _run_sp500_fn or _dummy_run("S&P 500")
        try:
            # Prefer a function that accepts session_label, else just call it
            return fn(session_label="regular") if fn.__code__.co_argcount else fn()
        except Exception:
            return fn()

    # Render scheduler controls in the sidebar
    sched.render_sidebar_controls(
        sched.RunFns(
            run_premarket=_run_pre,
            run_postmarket=_run_post,
            run_sp500=_run_sp500_wrapper,
        ),
        tz_str="America/New_York",
        pre_time=(8, 30),      # 08:30 ET
        intraday_time=(9, 45), # 09:45 ET
        post_time=(16, 10),    # 16:10 ET
    )
except ModuleNotFoundError as _e:
    # Graceful if scheduler.py or APScheduler isn't available
    st.sidebar.markdown("### Scheduler")
    st.sidebar.caption(f"Scheduler unavailable: {_e}. Add scheduler.py and install apscheduler to enable.")
except Exception as _e:
    st.sidebar.markdown("### Scheduler")
    st.sidebar.caption(f"Scheduler error: {_e}")
from pathlib import Path
from sqlalchemy import create_engine, text

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:
    class StreamlitSecretNotFoundError(Exception):
        pass
# Prefer hosted Postgres/Neon/Supabase via DB_URL secret/env; fallback to local SQLite file

def _normalize_postgres_url(url: str) -> str:
    """Normalize postgres URL and ensure sslmode=require for Neon/Supabase."""
    if not isinstance(url, str) or not url:
        return url
    # Accept old scheme postgres:// and rewrite to postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Ensure sslmode=require if not present (case-insensitive)
    if "postgresql://" in url or "postgresql+" in url:
        if "sslmode=" not in url.lower():
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}sslmode=require"
    return url

def _build_engine():
    # 1) Try DB_URL from env or secrets; also accept DATABASE_URL
    raw_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not raw_url:
        try:
            # Accessing st.secrets can raise if secrets.toml doesn't exist
            raw_url = st.secrets.get("DB_URL", None)  # type: ignore[attr-defined]
        except StreamlitSecretNotFoundError:
            raw_url = None
        except Exception:
            raw_url = None
    if raw_url:
        db_url = _normalize_postgres_url(raw_url)
        try:
            # First try default driver (psycopg2 if installed)
            return create_engine(db_url, pool_pre_ping=True)
        except ModuleNotFoundError as e:
            # psycopg2 not installed; retry with psycopg3 driver
            if "psycopg2" in str(e):
                alt = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
                return create_engine(alt, pool_pre_ping=True)
            raise
        except Exception as e:
            st.warning(f"Postgres engine failed: {e}. Falling back to SQLite.")
    # 2) Fallback to SQLite (local file)
    db_path = os.getenv("DB_PATH")
    if not db_path:
        try:
            db_path = st.secrets.get("DB_PATH", "scanner.sqlite")  # type: ignore[attr-defined]
        except StreamlitSecretNotFoundError:
            db_path = "scanner.sqlite"
        except Exception:
            db_path = "scanner.sqlite"
    try:
        p = Path(db_path)
        if p.parent and str(p.parent) not in (".", ""):
            p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return create_engine(f"sqlite:///{db_path}")

engine = _build_engine()


def _init_db():
    """Create tables if they don't exist. Uses portable DDL across Postgres/SQLite."""
    try:
        with engine.begin() as conn:
            dialect = engine.dialect.name
            if dialect == "postgresql":
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id BIGSERIAL PRIMARY KEY,
                        run_type TEXT NOT NULL,
                        started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        downloaded_count INTEGER,
                        skipped_count INTEGER,
                        elapsed_s DOUBLE PRECISION,
                        params_json TEXT
                    );
                    """
                ))
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS results_json (
                        run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
                        row_json TEXT
                    );
                    """
                ))
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS run_deltas (
                        run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
                        change TEXT NOT NULL,  -- 'added' or 'removed'
                        row_json TEXT
                    );
                    """
                ))
                # --- Migrations: add hot columns if missing ---
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN IF NOT EXISTS ticker TEXT"))
                except Exception:
                    pass
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN IF NOT EXISTS breakout_pct REAL"))
                except Exception:
                    pass
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN IF NOT EXISTS session TEXT"))
                except Exception:
                    pass
            else:  # sqlite
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_type TEXT NOT NULL,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        downloaded_count INTEGER,
                        skipped_count INTEGER,
                        elapsed_s REAL,
                        params_json TEXT
                    );
                    """
                ))
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS results_json (
                        run_id INTEGER,
                        row_json TEXT,
                        FOREIGN KEY(run_id) REFERENCES runs(id)
                    );
                    """
                ))
                conn.execute(text(
                    """
                    CREATE TABLE IF NOT EXISTS run_deltas (
                        run_id INTEGER,
                        change TEXT NOT NULL,  -- 'added' or 'removed'
                        row_json TEXT,
                        FOREIGN KEY(run_id) REFERENCES runs(id)
                    );
                    """
                ))
                # --- Migrations: add hot columns if missing (SQLite lacks IF NOT EXISTS on ADD COLUMN in older versions) ---
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN ticker TEXT"))
                except Exception:
                    pass
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN breakout_pct REAL"))
                except Exception:
                    pass
                try:
                    conn.execute(text("ALTER TABLE results_json ADD COLUMN session TEXT"))
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"Database init issue: {e}")


# --- Index helper ---
def _ensure_indexes():
    """Create helpful indexes (safe to call each start)."""
    try:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text("CREATE INDEX IF NOT EXISTS runs_type_time_idx ON runs (run_type, started_at DESC);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS results_json_run_idx ON results_json (run_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS run_deltas_run_idx ON run_deltas (run_id, change);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS results_json_ticker_idx ON results_json (ticker);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS results_json_breakout_idx ON results_json (breakout_pct DESC);"))
    except Exception as e:
        st.warning(f"Index setup issue: {e}")

_init_db()
_ensure_indexes()


def save_run(run_type: str, meta: dict, results_df: pd.DataFrame) -> int:
    """Persist a scan run and its rows; returns run id."""
    params_json = json.dumps(meta.get("params", {}), ensure_ascii=False)
    downloaded_count = int(meta.get("downloaded_count", 0))
    skipped_count = int(meta.get("skipped_count", 0))
    elapsed_s = float(meta.get("elapsed_s", 0.0))

    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "postgresql":
            run_id = conn.execute(
                text(
                    """
                    INSERT INTO runs (run_type, downloaded_count, skipped_count, elapsed_s, params_json)
                    VALUES (:run_type, :downloaded_count, :skipped_count, :elapsed_s, :params_json)
                    RETURNING id
                    """
                ),
                {
                    "run_type": run_type,
                    "downloaded_count": downloaded_count,
                    "skipped_count": skipped_count,
                    "elapsed_s": elapsed_s,
                    "params_json": params_json,
                },
            ).scalar_one()
        else:
            res = conn.execute(
                text(
                    """
                    INSERT INTO runs (run_type, downloaded_count, skipped_count, elapsed_s, params_json)
                    VALUES (:run_type, :downloaded_count, :skipped_count, :elapsed_s, :params_json)
                    """
                ),
                {
                    "run_type": run_type,
                    "downloaded_count": downloaded_count,
                    "skipped_count": skipped_count,
                    "elapsed_s": elapsed_s,
                    "params_json": params_json,
                },
            )
            # sqlite lastrowid
            run_id = res.lastrowid  # type: ignore[attr-defined]

        # Store results as JSON rows for flexible schema
        if results_df is not None and not results_df.empty:
            rows = results_df.to_dict(orient="records")
            payload = []
            session_label = (meta.get("params", {}) or {}).get("session")
            for r in rows:
                rj = json.dumps(r, default=str)
                ticker = r.get("Ticker")
                breakout_pct = None
                for key in ("Breakout %", "Premarket % Change", "Postmarket % Change"):
                    if key in r and r.get(key) is not None:
                        try:
                            breakout_pct = float(r.get(key))
                            break
                        except Exception:
                            continue
                payload.append({
                    "run_id": int(run_id),
                    "row_json": rj,
                    "ticker": ticker,
                    "breakout_pct": breakout_pct,
                    "session": session_label,
                })
            try:
                conn.execute(
                    text("INSERT INTO results_json (run_id, row_json, ticker, breakout_pct, session) "
                         "VALUES (:run_id, :row_json, :ticker, :breakout_pct, :session)"),
                    payload,
                )
            except Exception:
                # Fallback to legacy schema if hot columns are unavailable
                conn.execute(
                    text("INSERT INTO results_json (run_id, row_json) VALUES (:run_id, :row_json)"),
                    [{"run_id": int(run_id), "row_json": p['row_json']} for p in payload],
                )

        # --- Persist added/removed deltas vs previous run of the same type ---
        try:
            # Find previous run of the same type
            prev_id = conn.execute(
                text(
                    "SELECT id FROM runs WHERE run_type = :rt AND id < :rid ORDER BY id DESC LIMIT 1"
                ),
                {"rt": run_type, "rid": int(run_id)},
            ).scalar()

            if prev_id is not None:
                # Load previous run results
                prev_rows = conn.execute(
                    text("SELECT row_json FROM results_json WHERE run_id = :rid"),
                    {"rid": int(prev_id)},
                ).scalars().all()
                older_df = pd.DataFrame([json.loads(r) for r in prev_rows]) if prev_rows else pd.DataFrame()

                # Helper to get ticker set and row map
                def _ticker_set_and_map(df: pd.DataFrame):
                    if df is None or df.empty:
                        return set(), {}
                    col = "Ticker" if "Ticker" in df.columns else (df.columns[0] if len(df.columns) else None)
                    if col is None:
                        return set(), {}
                    tickers = list(map(str, df[col].astype(str)))
                    row_map = {str(t): df.iloc[i].to_dict() for i, t in enumerate(tickers)}
                    return set(tickers), row_map

                cur_set, cur_map = _ticker_set_and_map(results_df)
                old_set, old_map = _ticker_set_and_map(older_df)

                added = sorted(cur_set - old_set)
                removed = sorted(old_set - cur_set)

                payload = []
                for t in added:
                    try:
                        payload.append({
                            "run_id": int(run_id),
                            "change": "added",
                            "row_json": json.dumps(cur_map.get(t, {"Ticker": t}), default=str),
                        })
                    except Exception:
                        continue
                for t in removed:
                    try:
                        payload.append({
                            "run_id": int(run_id),
                            "change": "removed",
                            "row_json": json.dumps(old_map.get(t, {"Ticker": t}), default=str),
                        })
                    except Exception:
                        continue

                if payload:
                    conn.execute(
                        text("INSERT INTO run_deltas (run_id, change, row_json) VALUES (:run_id, :change, :row_json)"),
                        payload,
                    )
        except Exception as _e:
            # Non-fatal: continue even if delta persistence fails
            pass
    return int(run_id)


def list_runs(limit: int = 300) -> pd.DataFrame:
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, run_type, started_at, downloaded_count, skipped_count, elapsed_s, params_json "
                    "FROM runs ORDER BY id DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            ).mappings().all()
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Could not list runs: {e}")
        return pd.DataFrame(columns=["id","run_type","started_at","downloaded_count","skipped_count","elapsed_s","params_json"])


def load_run_results(run_id: int) -> pd.DataFrame:
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT row_json FROM results_json WHERE run_id = :rid"),
                {"rid": int(run_id)},
            ).scalars().all()
        if not rows:
            return pd.DataFrame()
        dicts = [json.loads(r) for r in rows]
        return pd.DataFrame(dicts)
    except Exception as e:
        st.warning(f"Could not load run #{run_id}: {e}")
        return pd.DataFrame()

# Breakout scanner with formatted Volume
def breakout_scanner(price_data, min_price=5, max_price=1000, include_ta: bool = False, spy_df: pd.DataFrame = None):
    results = []
    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 21 or 'Close' not in df.columns:
            continue
        try:
            latest_close = float(df['Close'].iloc[-1])
            prev_max = float(pd.Series(df['Close'].iloc[-21:-1]).max())
            latest_volume = float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else float('nan')
            latest_volume_fmt = format_volume(latest_volume) if pd.notna(latest_volume) else ""
        except Exception:
            continue

        if min_price <= latest_close <= max_price and prev_max > 0 and latest_close > prev_max:
            pct_breakout = (latest_close - prev_max) / prev_max * 100.0
            row = {
                'Ticker': ticker,
                'Latest Close': round(latest_close, 4),
                'Previous 20d Max': round(prev_max, 4),
                'Breakout %': round(pct_breakout, 2),
                'Volume': latest_volume_fmt
            }
            # classify after row exists
            row['Class'] = _classify_breakout(row['Breakout %'])

            if include_ta:
                try:
                    rsi = _rsi(df['Close']).iloc[-1]
                    macd, signal, hist = _macd(df['Close'])
                    atr14 = _atr(df).iloc[-1]
                    rs20 = _rs_20d_vs_spy(df, spy_df) if (spy_df is not None and 'Close' in spy_df.columns) else np.nan
                    row.update({
                        'RSI(14)': round(float(rsi), 2) if pd.notna(rsi) else np.nan,
                        'MACD': round(float(macd.iloc[-1]), 4) if pd.notna(macd.iloc[-1]) else np.nan,
                        'MACD Signal': round(float(signal.iloc[-1]), 4) if pd.notna(signal.iloc[-1]) else np.nan,
                        'ATR(14)': round(float(atr14), 4) if pd.notna(atr14) else np.nan,
                        'RS 20d vs SPY (%)': rs20 if pd.notna(rs20) else np.nan,
                    })
                except Exception:
                    pass
            results.append(row)

    if results:
        return pd.DataFrame(results).sort_values('Breakout %', ascending=False).reset_index(drop=True)
    else:
        cols = ['Ticker', 'Latest Close', 'Previous 20d Max', 'Breakout %', 'Volume', 'Class']
        if include_ta:
            cols += ['RSI(14)', 'MACD', 'MACD Signal', 'ATR(14)', 'RS 20d vs SPY (%)']
        return pd.DataFrame(columns=cols)

# Automatic scanner function for full S&P 600 tickers or uploaded tickers
def auto_scan(min_price=5, max_price=1000):
    tickers = st.session_state.get("tickers", [])
    price_data, skipped = fetch_price_data_batch(tickers, period="60d", interval="1d", batch_size=50)
    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    liquid = filter_by_dollar_volume(price_data, min_dollar_vol)
    filtered = [t for t in filtered if t in liquid]
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}
    breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
    # Save CSV automatically
    if not breakout_df.empty:
        breakout_df.to_csv("breakout_results.csv", index=False)
    return breakout_df, skipped, filtered_data

# --- UI helpers: market session chip & breakout class mapping ---
import datetime as _dt
import pytz as _pytz

def _us_market_session(now_utc=None):
    """Return ('Pre-Market'|'Regular'|'Post-Market'|'Closed', is_open_bool)."""
    try:
        if now_utc is None:
            now_utc = _dt.datetime.utcnow().replace(tzinfo=_pytz.UTC)
        et = now_utc.astimezone(_pytz.timezone("America/New_York"))
        wd = et.weekday()  # 0=Mon
        if wd >= 5:
            return "Closed", False
        t = et.time()
        pre_start = _dt.time(4, 0)
        reg_start = _dt.time(9, 30)
        reg_end   = _dt.time(16, 0)
        post_end  = _dt.time(20, 0)
        if pre_start <= t < reg_start:
            return "Pre-Market", True
        if reg_start <= t < reg_end:
            return "Regular", True
        if reg_end <= t < post_end:
            return "Post-Market", True
        return "Closed", False
    except Exception:
        return "Unknown", False

def _chip(label, tone="info"):
    """Render a small pill badge."""
    color = {"info":"#2f7ed8","success":"#22863a","warn":"#b08800","error":"#d73a49"}.get(tone,"#2f7ed8")
    return f"<span style='display:inline-block;padding:3px 8px;border-radius:999px;background:{color};color:white;font-size:12px;'>{label}</span>"

def _classify_breakout(pct):
    try:
        if pct is None or pd.isna(pct):
            return ""
        if pct >= 5:
            return "A"
        if pct >= 2:
            return "B"
        return "C"
    except Exception:
        return ""

import streamlit as st

# --- Shared column config for scan result tables ---
COMMON_COLCFG = {
    "Latest Close": st.column_config.NumberColumn(format="$%.2f"),
    "Previous 20d Max": st.column_config.NumberColumn(format="$%.2f"),
    "Breakout %": st.column_config.NumberColumn(format="%.2f"),
    "Premarket % Change": st.column_config.NumberColumn(format="%.2f"),
    "Postmarket % Change": st.column_config.NumberColumn(format="%.2f"),
    "Premarket $ Change": st.column_config.NumberColumn(format="$%.2f"),
    "Postmarket $ Change": st.column_config.NumberColumn(format="$%.2f"),
}

# Streamlit UI
st.title("Money Moves Breakout Scanner")

# --- Chip and Tabs ---
# Session chip
_sess_label, _is_open = _us_market_session()
st.markdown(_chip(f"Market: {_sess_label}", "success" if _is_open else "warn"), unsafe_allow_html=True)

# Tabs for app sections
tab_scan, tab_history = st.tabs(["🔎 Scans", "📜 History"])

# --- Filter by Price ---
st.sidebar.markdown("### Filter by Price")
min_price = st.sidebar.number_input("Minimum Price ($)", min_value=0.0, value=5.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0.0, value=1000.0, step=1.0)

# --- Liquidity ---
st.sidebar.markdown("### Liquidity Filter")
min_dollar_vol = st.sidebar.slider(
    "Minimum Dollar Volume (USD)",
    min_value=0, max_value=50_000_000, value=2_000_000, step=250_000,
    help="Filters tickers whose latest Close*Volume is below this threshold."
)

# --- Analytics ---
st.sidebar.markdown("### Analytics")
include_ta = st.sidebar.checkbox("Include TA columns (RSI/MACD/ATR/RS vs SPY)", value=False)
spy_df = get_spy_history("60d") if include_ta else None
if include_ta and (spy_df is None or spy_df.empty or 'Close' not in spy_df.columns):
    st.warning("TA enabled, but SPY data unavailable — RS 20d vs SPY will be blank.")

# --- Upload Ticker Files ---
st.sidebar.markdown("### Upload Tickers")
uploaded_file = st.sidebar.file_uploader("Upload tickers file (.txt or .csv)", type=['txt', 'csv'])

# --- Sidebar checkbox for gap/unusual volume filter ---
st.sidebar.markdown("### Sort Modifier")
apply_gap_filter = st.sidebar.checkbox("Apply Gap / Unusual Volume Filter", value=True)
us_only = st.sidebar.checkbox("US-only filter", value=True)
# --- Diagnostics toggle ---



if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode("utf-8")
            lines = content.splitlines()
            tickers = sanitize_ticker_list(lines)
            tickers = filter_problem_tickers(tickers)
        else:
            df_uploaded = pd.read_csv(uploaded_file, header=None)
            raw = []
            for col in df_uploaded.columns:
                raw.extend([str(t) for t in df_uploaded[col] if str(t).strip()])
            tickers = sanitize_ticker_list(raw)
            tickers = filter_problem_tickers(tickers)
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
            st.dataframe(breakout_df, column_config=COMMON_COLCFG, width="stretch")

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
        st.session_state.tickers = filter_problem_tickers(st.session_state.tickers)

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
            breakout_df = breakout_scanner({search_ticker: hist}, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
            if breakout_df.empty:
                st.info(f"No breakout detected for {search_ticker} based on the last 60 days data.")
            else:
                st.success(f"Breakout detected for {search_ticker}:")
                st.dataframe(breakout_df, column_config=COMMON_COLCFG, width="stretch")
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
            liquid = filter_by_dollar_volume(price_data, min_dollar_vol)
            filtered = [t for t in filtered if t in liquid]
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
            breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
            if not breakout_df.empty:
                breakout_df.to_csv("breakout_results_hot.csv", index=False)

    if skipped:
        st.warning(f"Skipped {len(skipped)} hot tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

    if breakout_df.empty:
        st.info("No breakout candidates found among hot stocks.")
    else:
        st.success(f"Found {len(breakout_df)} breakout candidates among hot stocks.")
        st.dataframe(breakout_df, column_config=COMMON_COLCFG, use_container_width=True)
        # --- DB Helper ---
        # Persist run to DB
        meta = {
            "universe_before": uni_before,
            "universe_after": uni_after,
            "downloaded_count": downloaded_count,
            "skipped_count": len(skipped),
            "elapsed_s": (t1 - t0),
            "params": {
                "min_price": float(min_price),
                "max_price": float(max_price),
                "include_ta": bool(include_ta),
                "min_dollar_vol": int(min_dollar_vol),
                "use_parallel": bool(use_parallel),
                "parallel_workers": int(parallel_workers),
                "parallel_chunk": int(parallel_chunk if use_parallel else 50),
                "us_only": bool(us_only),
                "apply_gap_filter": bool(apply_gap_filter),
            },
        }
        run_id = save_run("hot", meta, breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True))
        st.caption(f"Saved to history as run #{run_id}")

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
            liquid = filter_by_dollar_volume(price_data, min_dollar_vol)
            filtered = [t for t in filtered if t in liquid]
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
            breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)

            if skipped:
                st.warning(f"Skipped {len(skipped)} most active tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

            if breakout_df.empty:
                st.info("No breakout candidates found among most active stocks.")
            else:
                st.success(f"Found {len(breakout_df)} breakout candidates among most active stocks.")
                st.dataframe(breakout_df, column_config=COMMON_COLCFG, use_container_width=True)
                # --- DB Most Active ---
                # Persist run to DB
                meta = {
                    "universe_before": uni_before,
                    "universe_after": uni_after,
                    "downloaded_count": downloaded_count,
                    "skipped_count": len(skipped),
                    "elapsed_s": (t1 - t0),
                    "params": {
                        "min_price": float(min_price),
                        "max_price": float(max_price),
                        "include_ta": bool(include_ta),
                        "min_dollar_vol": int(min_dollar_vol),
                        "use_parallel": bool(use_parallel),
                        "parallel_workers": int(parallel_workers),
                        "parallel_chunk": int(parallel_chunk if use_parallel else 50),
                        "us_only": bool(us_only),
                        "apply_gap_filter": bool(apply_gap_filter),
                    },
                }
                run_id = save_run("most_active", meta,
                                  breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True))
                st.caption(f"Saved to history as run #{run_id}")

                # --- Download Button ---
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
            liquid = filter_by_dollar_volume(price_data, min_dollar_vol)
            filtered = [t for t in filtered if t in liquid]
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
            breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)

            if skipped:
                st.warning(f"Skipped {len(skipped)} trending tickers due to missing data or delisted: {', '.join(skipped[:10])}...")

            st.subheader("Trending Stocks Breakout Scan Results")
            if breakout_df.empty:
                st.info("No breakout candidates found among trending stocks.")
            else:
                st.success(f"Found {len(breakout_df)} breakout candidates among trending stocks.")
                st.dataframe(breakout_df, column_config=COMMON_COLCFG, use_container_width=True)

                # -- DB Helper Trending ---
                # Persist run to DB
                meta = {
                    "universe_before": uni_before,
                    "universe_after": uni_after,
                    "downloaded_count": downloaded_count,
                    "skipped_count": len(skipped),
                    "elapsed_s": (t1 - t0),
                    "params": {
                        "min_price": float(min_price),
                        "max_price": float(max_price),
                        "include_ta": bool(include_ta),
                        "min_dollar_vol": int(min_dollar_vol),
                        "use_parallel": bool(use_parallel),
                        "parallel_workers": int(parallel_workers),
                        "parallel_chunk": int(parallel_chunk if use_parallel else 50),
                        "us_only": bool(us_only),
                        "apply_gap_filter": bool(apply_gap_filter),
                    },
                }
                run_id = save_run("trending", meta,
                                  breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True))
                st.caption(f"Saved to history as run #{run_id}")

                # --- Download Trending Stocks Breakout ---
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
# Network / retry settings
net_max_retries = st.sidebar.slider(
    "Max retries per chunk", min_value=1, max_value=6, value=3,
    help="How many times to retry a failing data chunk before falling back per-ticker."
)
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

    # Use a conservative fixed chunk for stable MultiIndex returns & smoother streaming
    sp_chunk = 50
    t0 = time.perf_counter()

    def progress_cb(done, total):
        if total:
            progress.progress(done / total)
        status.markdown(f"Processed **{done}/{total}** chunks…")

    running_rows = []
    seen_tickers = set()

    def per_chunk_cb(added_price_data):
        # Filter by price + liquidity and run breakout on just-arrived data
        filtered = filter_tickers_by_price(added_price_data, min_price, max_price)
        liquid = filter_by_dollar_volume(added_price_data, min_dollar_vol)
        filtered = [t for t in filtered if t in liquid]
        filtered_chunk = {t: added_price_data[t] for t in filtered}
        if not filtered_chunk:
            return
        chunk_df = breakout_scanner(filtered_chunk, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
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
            live_table.dataframe(live_df, column_config=COMMON_COLCFG, use_container_width=True)

    # Choose streaming or batch path based on use_parallel flag
    if use_parallel:
        price_data, skipped = fetch_price_data_streaming(
            tickers,
            period="60d",
            interval="1d",
            chunk_size=sp_chunk,
            max_workers=parallel_workers,
            max_retries=net_max_retries,
            retry_sleep=1.2,
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
            chunk_size=sp_chunk,
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

    breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
    if not breakout_df.empty:
        st.success(f"Scan complete ✅ Found {len(breakout_df)} breakout candidates in S&P 500.")
        final_df = breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True)
        final_df = final_df.drop_duplicates(subset=["Ticker"], keep="first")
        live_table.dataframe(final_df, column_config=COMMON_COLCFG, use_container_width=True)  # update the same live table
        files = {"breakout_sp500.csv": final_df.to_csv(index=False)}
        try:
            if apply_gap_filter and 'gap_df' in locals() and not gap_df.empty:
                files["gap_unusual_volume_sp500.csv"] = gap_df.to_csv(index=False)
        except Exception:
            pass
        # Persist run to DB
        meta = {
            "universe_before": uni_before,
            "universe_after": uni_after,
            "downloaded_count": downloaded_count,
            "skipped_count": len(skipped),
            "elapsed_s": (t1 - t0),
            "params": {
                "min_price": float(min_price),
                "max_price": float(max_price),
                "include_ta": bool(include_ta),
                "min_dollar_vol": int(min_dollar_vol),
                "use_parallel": bool(use_parallel),
                "parallel_workers": int(parallel_workers),
                "parallel_chunk": int(parallel_chunk),
                "us_only": bool(us_only),
                "apply_gap_filter": bool(apply_gap_filter),
            },
        }
        run_id = save_run("sp500", meta, final_df)
        st.caption(f"Saved to history as run #{run_id}")
        if files:
            download_zip_button("Download S&P 500 Bundle (ZIP)", files, filename="sp500_scan_bundle.zip")
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
        t0 = time.perf_counter()

        # Use smaller chunks and more workers for better throughput on large universes
        ndq_chunk = 60 if use_parallel else 50
        ndq_workers = max(4, min(parallel_workers, 6)) if use_parallel else 1

        def progress_cb(done, total):
            if total:
                progress.progress(done / total)
            status.markdown(f"Processed **{done}/{total}** chunks…")

        running_rows = []
        seen_tickers = set()

        def per_chunk_cb(added_price_data):
            # Filter by price + liquidity and run breakout on just-arrived data
            filtered = filter_tickers_by_price(added_price_data, min_price, max_price)
            liquid = filter_by_dollar_volume(added_price_data, min_dollar_vol)
            filtered = [t for t in filtered if t in liquid]
            filtered_chunk = {t: added_price_data[t] for t in filtered}
            if not filtered_chunk:
                return
            chunk_df = breakout_scanner(filtered_chunk, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
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
                live_table.dataframe(live_df, column_config=COMMON_COLCFG, use_container_width=True)

        # Choose streaming or batch path based on use_parallel flag
        if use_parallel:
            price_data, skipped = fetch_price_data_streaming(
                tickers,
                period="60d",
                interval="1d",
                chunk_size=ndq_chunk,
                max_workers=ndq_workers,
                max_retries=net_max_retries,
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
                workers=ndq_workers if use_parallel else 1,
                downloaded_count=downloaded_count,
                skipped_count=len(skipped),
                elapsed_s=(t1 - t0),
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

        breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
        if not breakout_df.empty:
            st.success(f"Scan complete ✅ Found {len(breakout_df)} breakout candidates in Nasdaq.")
            final_df = breakout_df.sort_values('Breakout %', ascending=False).reset_index(drop=True)
            live_table.dataframe(final_df, column_config=COMMON_COLCFG, use_container_width=True) # update the same live table
            # Persist run to DB
            meta = {
                "universe_before": uni_before,
                "universe_after": uni_after,
                "downloaded_count": downloaded_count,
                "skipped_count": len(skipped),
                "elapsed_s": (t1 - t0),
                "params": {
                    "min_price": float(min_price),
                    "max_price": float(max_price),
                    "include_ta": bool(include_ta),
                    "min_dollar_vol": int(min_dollar_vol),
                    "use_parallel": bool(use_parallel),
                    "parallel_workers": int(parallel_workers),
                    "parallel_chunk": int(ndq_chunk if use_parallel else 50),
                    "us_only": bool(us_only),
                    "apply_gap_filter": bool(apply_gap_filter),
                },
            }
            run_id = save_run("nasdaq", meta, final_df)
            st.caption(f"Saved to history as run #{run_id}")
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
        st.subheader("Pre-market Scan Results (Live)")
        progress = st.progress(0)
        live_table = st.empty()
        status = st.empty()

        running_rows = []

        def progress_cb(done, total):
            if total:
                progress.progress(done / total)
                status.markdown(f"Processed **{done}/{total}** tickers…")

        def per_row_cb(row):
            try:
                first = float(row.get("Premarket First Price", float("nan")))
                last = float(row.get("Premarket Last Price", float("nan")))
                dollar_chg = (last - first) if np.isfinite(first) and np.isfinite(last) else np.nan
            except Exception:
                dollar_chg = np.nan
                first = row.get("Premarket First Price", np.nan)
                last = row.get("Premarket Last Price", np.nan)

            running_rows.append([
                row.get("Ticker", ""),
                first,
                last,
                dollar_chg,
                row.get("Premarket % Change", np.nan),
            ])

            live_df = pd.DataFrame(
                running_rows,
                columns=["Ticker", "Premarket First Price", "Premarket Last Price",
                         "Premarket $ Change", "Premarket % Change"]
            )

            for c in ["Premarket First Price", "Premarket Last Price", "Premarket $ Change"]:
                live_df[c] = pd.to_numeric(live_df[c], errors="coerce").round(2)

            live_df = live_df.sort_values("Premarket % Change", ascending=False).reset_index(drop=True)
            live_table.dataframe(live_df, column_config=COMMON_COLCFG, use_container_width=True)

        with st.spinner(f"Running pre-market scan on {len(tickers)} tickers…"):
            pm_df = premarket_scan(tickers, progress_cb=progress_cb, per_row_cb=per_row_cb)
            # Compute absolute pre-market dollar change and order columns
            if pm_df is not None and not pm_df.empty:
                pm_df["Premarket $ Change"] = (
                    pd.to_numeric(pm_df["Premarket Last Price"], errors="coerce") -
                    pd.to_numeric(pm_df["Premarket First Price"], errors="coerce")
                ).round(2)

                wanted_cols = [
                    "Ticker",
                    "Premarket First Price",
                    "Premarket Last Price",
                    "Premarket $ Change",
                    "Premarket % Change",
                ]
                existing_cols = [c for c in wanted_cols if c in pm_df.columns]
                pm_df = pm_df[existing_cols + [c for c in pm_df.columns if c not in existing_cols]]

        # Final consolidated view
        if pm_df.empty:
            st.info("No pre-market candidates found or no pre-market data available.")
        else:
            st.success(f"Scan complete ✅ Found {len(pm_df)} tickers with pre-market data.")
            final_df = pm_df.sort_values("Premarket % Change", ascending=False).reset_index(drop=True)
            live_table.dataframe(final_df, column_config=COMMON_COLCFG, use_container_width=True)  # update the same live table

            # --- Pre-Market DB Helper
            meta = {
                "universe_before": len(tickers),
                "universe_after": len(tickers),
                "downloaded_count": int(len(final_df)),
                "skipped_count": 0,
                "elapsed_s": 0.0,
                "params": {"session": "premarket"}
            }
            run_id = save_run("premarket", meta, final_df)
            st.caption(f"Saved to history as run #{run_id}")

            # --- Download Pre-Market Result as CSV
            st.download_button(
                label="Download Pre-market Results as CSV",
                data=final_df.to_csv(index=False),
                file_name="premarket_results.csv",
                mime="text/csv"
            )

# Helper to render the entire Post-market UI (scoped to Scans tab)
def render_postmarket_section():
    # Add Post-market Scan Button (scoped rendering via tab container)
    tickers = st.session_state.get("tickers", [])
    if st.sidebar.button("Run Post-market Scan"):
        if not tickers:
            st.error("No tickers available for post-market scan.")
            return
        st.subheader("Post-market Scan Results (Live)")
        progress = st.progress(0)
        live_table = st.empty()
        status = st.empty()

        running_rows = []

        def progress_cb(done, total):
            if total:
                progress.progress(done / total)
                status.markdown(f"Processed **{done}/{total}** tickers…")

        def per_row_cb(row):
            try:
                first = float(row.get("Postmarket First Price", float("nan")))
                last = float(row.get("Postmarket Last Price", float("nan")))
                dollar_chg = (last - first) if np.isfinite(first) and np.isfinite(last) else np.nan
            except Exception:
                dollar_chg = np.nan
                first = row.get("Postmarket First Price", np.nan)
                last = row.get("Postmarket Last Price", np.nan)

            running_rows.append([
                row.get("Ticker", ""),
                first,
                last,
                dollar_chg,
                row.get("Postmarket % Change", np.nan),
            ])

            live_df = pd.DataFrame(
                running_rows,
                columns=["Ticker", "Postmarket First Price", "Postmarket Last Price",
                         "Postmarket $ Change", "Postmarket % Change"]
            )

            for c in ["Postmarket First Price", "Postmarket Last Price", "Postmarket $ Change"]:
                live_df[c] = pd.to_numeric(live_df[c], errors="coerce").round(2)

            live_df = live_df.sort_values("Postmarket % Change", ascending=False).reset_index(drop=True)
            live_table.dataframe(live_df, column_config=COMMON_COLCFG, use_container_width=True)

        with st.spinner(f"Running post-market scan on {len(tickers)} tickers…"):
            t0 = time.perf_counter()
            post_df = postmarket_scan(tickers, progress_cb=progress_cb, per_row_cb=per_row_cb)
            t1 = time.perf_counter()
            elapsed_s = t1 - t0

        # Compute absolute post-market dollar change and order columns
        if post_df is not None and not post_df.empty:
            post_df["Postmarket $ Change"] = (
                pd.to_numeric(post_df["Postmarket Last Price"], errors="coerce") -
                pd.to_numeric(post_df["Postmarket First Price"], errors="coerce")
            ).round(2)

            wanted_cols = [
                "Ticker",
                "Postmarket First Price",
                "Postmarket Last Price",
                "Postmarket $ Change",
                "Postmarket % Change",
            ]
            existing_cols = [c for c in wanted_cols if c in post_df.columns]
            post_df = post_df[existing_cols + [c for c in post_df.columns if c not in existing_cols]]

        # Final consolidated view
        if post_df.empty:
            st.info("No post-market candidates found or no post-market data available.")
        else:
            st.success(f"Scan complete ✅ Found {len(post_df)} tickers with post-market data.")
            final_df = post_df.sort_values("Postmarket % Change", ascending=False).reset_index(drop=True)
            live_table.dataframe(final_df, column_config=COMMON_COLCFG, use_container_width=True)  # update the same live table

            # --- Post-Market Results ---
            meta = {
                "universe_before": len(tickers),
                "universe_after": len(tickers),
                "downloaded_count": int(len(final_df)),
                "skipped_count": 0,
                "elapsed_s": elapsed_s,
                "params": {"session": "postmarket"}
            }
            run_id = save_run("postmarket", meta, final_df)
            st.caption(f"Saved to history as run #{run_id}")

            # --- Download Post-Market Results ---
            st.download_button(
                label="Download Post-market Results as CSV",
                data=final_df.to_csv(index=False),
                file_name="postmarket_results.csv",
                mime="text/csv"
            )

with tab_scan:
    st.sidebar.markdown("### During Post-Market Hour")
    render_postmarket_section()

# --- History Viewer ---
with tab_history:
    st.header("History (DB)")
    try:
        runs_df = list_runs(limit=300)
    except Exception as e:
        runs_df = pd.DataFrame()
        st.warning(f"History unavailable: {e}")

    if runs_df.empty:
        st.info("No saved runs yet.")
    else:
        st.dataframe(runs_df)
        run_choices = runs_df.apply(
            lambda r: f"#{int(r['id'])} • {r['run_type']} • {r['started_at']} • "
                      f"{int(r['downloaded_count'])} dl / {int(r['skipped_count'])} sk • {float(r['elapsed_s']):.1f}s",
            axis=1
        ).tolist()
        run_ids = runs_df["id"].tolist()
        sel = st.selectbox(
            "Load a past run:",
            options=list(range(len(run_choices))),
            format_func=lambda i: run_choices[i]
        )
        selected_run_id = int(run_ids[sel])

        # Re-run with same settings
        if st.button("Re-run with these settings", key="rerun_same_settings"):
            try:
                row = runs_df[runs_df["id"] == selected_run_id].iloc[0]
                params = {}
                try:
                    params = json.loads(row.get("params_json") or "{}")
                except Exception:
                    params = {}
                # Restore a few common settings if present
                st.session_state["include_ta"] = bool(params.get("include_ta", st.session_state.get("include_ta", False)))
                st.session_state["apply_gap_filter"] = bool(params.get("apply_gap_filter", st.session_state.get("apply_gap_filter", True)))
                # Dispatch by run_type
                rt = str(row.get("run_type"))
                if rt == "sp500":
                    run_sp500_scan(us_only, parallel_chunk, parallel_workers, min_price, max_price, apply_gap_filter)
                elif rt == "nasdaq":
                    # Trigger Nasdaq path by programmatically pressing the sidebar action
                    # (For now, reuse the button handler code by calling the underlying scanning routine if you factor it.)
                    st.info("To re-run Nasdaq, click 'Run Nasdaq Scan' in the sidebar (direct programmatic dispatch can be factored similarly to S&P).")
                elif rt in ("hot", "most_active", "trending", "premarket", "postmarket"):
                    st.info(f"Re-run for '{rt}' is not yet wired programmatically. Use the sidebar action for now.")
            except Exception as e:
                st.warning(f"Could not re-run: {e}")

        # Load run results but don't render live tables here
        prev_df = load_run_results(selected_run_id)
        st.subheader(f"Saved Results for Run #{selected_run_id}")
        if prev_df.empty:
            st.info("This run has no saved rows.")
        else:
            with st.expander(f"Expand to view saved run #{selected_run_id} results", expanded=False):
                st.dataframe(prev_df, width="stretch")
            st.download_button(
                "Download This Run (CSV)",
                data=prev_df.to_csv(index=False),
                file_name=f"scan_run_{selected_run_id}.csv",
                mime="text/csv"
            )

        # --- Delta view: compare to previous run of the same type ---
        st.markdown("#### Compare to previous run of the same type")
        do_delta = st.checkbox("Show delta (added / removed tickers)", key="history_delta_checkbox")

        if do_delta:
            try:
                this_row = runs_df[runs_df["id"] == selected_run_id].iloc[0]
                same_type = runs_df[(runs_df["run_type"] == this_row["run_type"]) & (runs_df["id"] < selected_run_id)].sort_values("id", ascending=False)
                if same_type.empty:
                    st.info("No previous run of the same type to compare.")
                else:
                    prev_id = int(same_type.iloc[0]["id"])
                    older_df = load_run_results(prev_id)

                    def _tickers(df):
                        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
                        return set(map(str, df[col].astype(str))) if not df.empty else set()

                    cur_t = _tickers(prev_df)
                    old_t = _tickers(older_df)
                    added = sorted(cur_t - old_t)
                    removed = sorted(old_t - cur_t)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Added vs run #{prev_id}** ({len(added)})")
                        st.dataframe(pd.DataFrame({"Ticker": added}), width="stretch")
                    with c2:
                        st.markdown(f"**Removed vs run #{prev_id}** ({len(removed)})")
                        st.dataframe(pd.DataFrame({"Ticker": removed}), width="stretch")
            except Exception as _e:
                st.warning(f"Delta view error: {_e}")

        # Persisted deltas (if available)
        with st.expander("Show saved deltas for this run (if any)", expanded=False):
            try:
                with engine.begin() as conn:
                    rows = conn.execute(
                        text("SELECT change, row_json FROM run_deltas WHERE run_id = :rid"),
                        {"rid": int(selected_run_id)},
                    ).mappings().all()
                if not rows:
                    st.caption("No deltas persisted for this run.")
                else:
                    df_d = pd.DataFrame([{"change": r["change"], **json.loads(r["row_json"])} for r in rows])
                    st.dataframe(df_d, width="stretch")
            except Exception as _e:
                st.caption(f"No deltas available: {_e}")

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
