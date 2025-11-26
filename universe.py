from __future__ import annotations

import os
import time
from typing import List

import pandas as pd
import streamlit as st

try:
    import requests
except Exception:
    requests = None

from scan.engine import safe_yf_download


def _try_import(path: str, attr: str | None = None):
    """Local copy of safe import helper for universe loaders."""
    try:
        mod = __import__(path, fromlist=[path.split(".")[-1]])
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


# ---------- Universe loaders ----------
# Try your real loaders first, fallback to tiny defaults.

_load_sp500 = (
    _try_import("ai_scanner.ui.universe", "load_sp500_universe")
    or _try_import("ai_scanner.ui.universe", "get_sp500")
    or _try_import("ui.universe", "load_sp500_universe")
    or _try_import("ui.universe", "get_sp500")
)

_load_nasdaq = (
    _try_import("ai_scanner.ui.universe", "load_nasdaq_universe")
    or _try_import("ai_scanner.ui.universe", "get_nasdaq")
    or _try_import("ui.universe", "load_nasdaq_universe")
    or _try_import("ui.universe", "get_nasdaq")
)


def _fetch_yahoo_universe(scr_id: str, count: int = 1000) -> List[str]:
    """Fetch predefined Yahoo Finance screener tickers."""
    if requests is None:
        raise RuntimeError("requests not available")

    url = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"scrIds": scr_id, "count": count, "start": 0}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.yahoo.com/",
    }

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=12)
            if r.status_code == 429:
                # backoff and retry
                time.sleep(1.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            quotes = (
                data.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
            )
            tickers = [q.get("symbol") for q in quotes if q.get("symbol")]
            # De-dupe while preserving order
            seen = set()
            out: List[str] = []
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))

    raise last_err or RuntimeError("Yahoo universe fetch failed")


def _note_yahoo_fail(which: str, err: Exception):
    key = f"yahoo_fail_noted_{which}"
    if not st.session_state.get(key):
        st.session_state[key] = True
        st.caption(
            f"Yahoo {which} universe fallback unavailable (rate-limited). Using Wikipedia instead."
        )


def _fetch_wikipedia_table(url: str, col: str) -> List[str]:
    """Fallback if Yahoo endpoint changes; pulls ticker lists from Wikipedia."""
    try:
        tables = pd.read_html(url)
        for t in tables:
            if col in t.columns:
                tickers = (
                    t[col]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                return tickers
    except Exception:
        return []
    return []


def _fetch_nasdaq_official_listings() -> List[str]:
    """Fetch NASDAQ-listed tickers from NASDAQ Trader official symbol directory."""
    if requests is None:
        raise RuntimeError("requests not available")

    urls = [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    tickers: List[str] = []
    for url in urls:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        lines = r.text.splitlines()
        if not lines:
            continue

        header = lines[0].split("|")
        for line in lines[1:]:
            if line.startswith("File Creation Time"):
                break
            parts = line.split("|")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            sym = row.get("Symbol") or row.get("ACT Symbol")
            if sym:
                sym = sym.strip().replace(".", "-")
                if sym and sym[0].isalnum():
                    tickers.append(sym)

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_sp500_universe() -> List[str]:
    # 0) Preferred: local sp500.txt (same folder as app.py or in ./data)
    local_paths = [
        os.path.join(os.path.dirname(__file__), "sp500.txt"),
        os.path.join(os.path.dirname(__file__), "data", "sp500.txt"),
        os.path.join(os.getcwd(), "sp500.txt"),
        os.path.join(os.getcwd(), "data", "sp500.txt"),
    ]
    for path in local_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    tickers = [
                        ln.strip()
                        for ln in f
                        if ln.strip() and not ln.startswith("#")
                    ]
                seen = set()
                out: List[str] = []
                for t in tickers:
                    if t not in seen:
                        seen.add(t)
                        out.append(t)
                if out and len(out) >= 400:
                    st.caption(
                        f"Loaded SP500 universe from {os.path.basename(path)} ({len(out)} tickers)."
                    )
                    return out
                else:
                    st.caption(
                        f"Local {os.path.basename(path)} returned {len(out)} tickers; expecting full list."
                    )
        except Exception as e:
            st.caption(f"Failed loading local SP500 file at {path}: {e}")

    # 1) Prefer your local/custom loader if it exists
    if callable(_load_sp500):
        try:
            local = list(_load_sp500())
            if local and len(local) >= 100:
                return local
            else:
                st.caption(
                    f"Local SP500 universe returned {len(local) if local else 0} tickers; using Wikipedia instead."
                )
        except Exception as e:
            st.caption(
                f"Local SP500 universe loader failed: {e}. Using Wikipedia instead."
            )

    # 2) Stable primary fallback: Wikipedia S&P 500 list
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        col="Symbol",
    )
    if wiki and len(wiki) >= 450:
        return wiki

    # 3) Optional Yahoo fallback (may rate-limit)
    try:
        tickers = _fetch_yahoo_universe("sp500", count=520)
        if tickers:
            return tickers
    except Exception as e:
        _note_yahoo_fail("SP500", e)

    # 4) Tiny last-resort default
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_nasdaq_universe() -> List[str]:
    if callable(_load_nasdaq):
        return list(_load_nasdaq())

    # ✅ Official NASDAQ Trader listings (NASDAQ Composite universe)
    try:
        tickers = _fetch_nasdaq_official_listings()
        if tickers:
            return tickers
    except Exception as e:
        st.caption(f"Official NASDAQ listings fallback failed: {e}")

    # Yahoo Finance predefined screener fallback (Nasdaq 100)
    try:
        tickers = _fetch_yahoo_universe("nasdaq100", count=120)
        if tickers:
            return tickers
    except Exception as e:
        _note_yahoo_fail("NASDAQ", e)

    # Wikipedia fallback (Nasdaq-100)
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        col="Ticker",
    )
    if wiki:
        return wiki

    return ["TSLA", "PLTR", "AMD", "SOFI", "SNOW", "CRWD"]


def filter_universe(tickers: List[str]) -> List[str]:
    """Basic cleaning for ticker universes (remove obvious junk)."""
    cleaned = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        sym = t.strip().upper()
        if not sym:
            continue
        # Ignore symbols that contain spaces or obvious non-equity prefixes
        if " " in sym:
            continue
        if sym.startswith("^"):
            continue
        if len(sym) > 10:
            continue
        cleaned.append(sym)
    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for sym in cleaned:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def apply_liquidity_filter_batch(
    tickers: List[str],
    *,
    min_price: float,
    min_avg_dollar_vol: float,
) -> List[str]:
    """Filter tickers by approximate price and 20D dollar volume using yfinance."""
    if not tickers:
        return []

    # Use the shared safe_yf_download from scan.engine
    prices = safe_yf_download(tickers, period="1mo", interval="1d", group_by="ticker")
    if prices is None or prices.empty:
        return tickers  # fallback: do not filter if data missing

    keep: List[str] = []

    def _get_series(sym: str, field: str):
        if prices is None or prices.empty:
            return None
        if isinstance(prices.columns, pd.MultiIndex):
            try:
                return prices[(sym, field)].dropna()
            except Exception:
                try:
                    return prices[(field, sym)].dropna()
                except Exception:
                    return None
        try:
            return prices[field].dropna()
        except Exception:
            return None

    for sym in tickers:
        try:
            close = _get_series(sym, "Close")
            vol = _get_series(sym, "Volume")
            if close is None or vol is None:
                continue
            close = close.dropna()
            vol = vol.dropna()
            if close.empty or vol.empty:
                continue

            last_close = float(close.iloc[-1])
            if last_close < min_price:
                continue

            window_v = min(20, len(vol))
            avg_vol20 = float(vol.tail(window_v).mean()) if window_v > 0 else float(vol.iloc[-1])
            dollar_vol20 = avg_vol20 * last_close

            if dollar_vol20 >= min_avg_dollar_vol:
                keep.append(sym)
        except Exception:
            continue

    return keep if keep else tickers