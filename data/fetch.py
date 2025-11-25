"""
Data fetching utilities for price history with robust batching, retries, and optional
real‑time streaming of results.

These helpers are intentionally UI-agnostic. Pass a `logger` callable if you want
to surface progress in Streamlit; otherwise they are safe to use in headless jobs.

Dependencies: pandas, numpy, yfinance
"""
from __future__ import annotations

__all__ = [
    "fetch_price_data_parallel",
    "fetch_price_data_batch",
    "fetch_price_data_streaming",
    "fetch_hot_stocks",
    "fetch_and_save_nasdaq",
    "fetch_and_save_sp500",
    # Back-compat wrappers
    "fetch_most_active_stocks",
    "fetch_trending_stocks",
    "fetch_top_gainers",
    "fetch_top_losers",
    "load_sp500_tickers",
    "load_nasdaq_tickers",
    "load_sp400_tickers",
    "load_sp600_tickers",
    "fetch_and_save_sp600",
    "remove_delisted_tickers",
]

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import time
import math
import pandas as pd
import numpy as np
import requests
import re

try:
    import yfinance as yf
except Exception:  # pragma: no cover - allow the module to import without yfinance during linting
    yf = None  # type: ignore


# -----------------------------
# Helpers
# -----------------------------

LogFn = Optional[Callable[[str], None]]


def _noop_log(_: str) -> None:
    """Default no-op logger."""
    return None


def _chunk_list(items: Sequence[str], size: int) -> List[List[str]]:
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _is_rate_limited(err: Exception) -> bool:
    msg = str(err).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg


# Back-compat shim: remove_delisted_tickers used to live in a monolithic module.
# Prefer the canonical implementation in data.filters if present.
try:
    from .filters import remove_delisted_tickers as _remove_delisted_tickers  # type: ignore
except Exception:
    _remove_delisted_tickers = None  # type: ignore

def remove_delisted_tickers(tickers: Sequence[str]) -> List[str]:
    """
    Normalize symbols and remove obvious delisted/placeholder symbols.

    If ai_scanner.data.filters.remove_delisted_tickers exists, we delegate to it.
    Otherwise we use a conservative fallback that:
      - strips a leading '$'
      - uppercases
      - keeps only symbols matching ^[A-Z][A-Z0-9.\-]*$
      - drops placeholders like *ZZT tickers (e.g., ZAZZT, ZBZZT)
      - drops obvious non-tickers seen in logs (FILE, FORL, FERA)
    """
    if _remove_delisted_tickers is not None:
        return _remove_delisted_tickers(tickers)

    out: List[str] = []
    allowed = re.compile("^[A-Z][A-Z0-9.\\-]*$")
    for t in tickers:
        if not t:
            continue
        s = str(t).strip()
        if not s:
            continue
        if s.startswith("$"):
            s = s[1:]
        s = s.upper()
        # filter invalid character sets
        if not allowed.match(s):
            continue
        # filter known Yahoo placeholder/delisted markers
        if s.endswith("ZZT"):
            continue
        # filter a few observed noise tokens
        if s in {"FILE", "FORL", "FERA"}:
            continue
        out.append(s)
    return out


def _download_batch_yf(
    symbols: Sequence[str],
    period: str,
    interval: str,
    logger: LogFn = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download a batch of tickers via yfinance.download and normalize to per-symbol DataFrames.

    Returns dict of {symbol: DataFrame}. Symbols that fail are omitted.
    """
    log = logger or _noop_log
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    if not symbols:
        return {}

    # yfinance.download supports list of tickers and returns a wide MultiIndex column DataFrame.
    try:
        data = yf.download(
            tickers=list(symbols),
            period=period,
            interval=interval,
            group_by="ticker",
            threads=False,  # we are handling threading outside
            progress=False,
            auto_adjust=False,
            prepost=False,
            repair=False,
        )
    except Exception as e:
        # If the whole batch error'd, nothing to return
        log(f"[download] batch failed ({len(symbols)} symbols): {e}")
        return {}

    out: Dict[str, pd.DataFrame] = {}

    # If only one ticker, yfinance returns a single-index columns frame.
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                df = data[sym].dropna(how="all")
                if not df.empty:
                    out[sym] = _coerce_ohlcv_dtypes(df)
    else:
        # Single symbol result; infer name from input order
        df_single = data.dropna(how="all")
        if not df_single.empty:
            out[symbols[0]] = _coerce_ohlcv_dtypes(df_single)

    return out


def _coerce_ohlcv_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure we have standard columns + types (float32, int32)."""
    # Normalize column names case
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)

    # Keep only known OHLCV if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    float_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("Int64")

    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()
    return df


def _rescue_in_minibatches(
    missing: List[str],
    period: str,
    interval: str,
    logger: LogFn,
    max_retries: int = 3,
    mini_size: int = 8,
    sleep_base: float = 0.8,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Try to recover missing symbols by downloading in smaller minibatches with backoff."""
    log = logger or _noop_log
    rescued: Dict[str, pd.DataFrame] = {}
    still_missing: List[str] = list(missing)

    for attempt in range(1, max_retries + 1):
        if not still_missing:
            break
        minibatches = _chunk_list(still_missing, mini_size)
        log(f"[rescue] attempt {attempt}/{max_retries} in {len(minibatches)} mini-batches…")
        next_missing: List[str] = []
        for mini in minibatches:
            got = _download_batch_yf(mini, period, interval, logger=logger)
            rescued.update(got)
            # anything not returned stays missing
            for s in mini:
                if s not in got:
                    next_missing.append(s)
        still_missing = next_missing
        if still_missing:
            # Backoff a bit, increasing each attempt
            time.sleep(sleep_base * attempt)

    return rescued, still_missing


# -----------------------------
# Public API
# -----------------------------

def fetch_price_data_parallel(
    tickers: Sequence[str],
    period: str = "60d",
    interval: str = "1d",
    chunk_size: int = 500,
    max_workers: int = 8,
    logger: LogFn = None,
    rescue_minibatch: bool = True,
    minibatch_size: int = 8,
    max_retries: int = 3,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Download price history for many tickers using thread-pooled batches.

    Returns:
        (data_dict, skipped_list)
    """
    log = logger or _noop_log
    symbols = [s for s in map(str.strip, tickers) if s]
    if not symbols:
        return {}, []

    chunks = _chunk_list(symbols, chunk_size)
    log(f"[prepare] {len(symbols)} symbols (chunk_size={chunk_size}, max_workers={max_workers})")

    data: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_download_batch_yf, chunk, period, interval, logger): chunk for chunk in chunks}
        for fut in as_completed(future_map):
            batch_syms = future_map[fut]
            try:
                got = fut.result()
            except Exception as e:
                log(f"[error] batch {batch_syms[:3]}… failed: {e}")
                got = {}

            data.update(got)
            # everything not returned from that chunk is considered missing for now
            for s in batch_syms:
                if s not in got:
                    skipped.append(s)

    # Deduplicate skipped
    if skipped:
        # keep only those genuinely missing after combining all chunks
        skipped = [s for s in symbols if s not in data]

    if rescue_minibatch and skipped:
        log(f"[rescue] {len(skipped)} symbols missing from initial pass. Trying minibatches…")
        rescued, still_missing = _rescue_in_minibatches(
            skipped, period, interval, logger=logger, max_retries=max_retries, mini_size=minibatch_size
        )
        data.update(rescued)
        skipped = still_missing
        for s in rescued.keys():
            log(f"[rescued] {s}")

    # Final summary
    log(f"[done] fetched {len(data)} / {len(symbols)} symbols. Skipped: {len(skipped)}")
    return data, skipped


def fetch_price_data_batch(
    tickers: Sequence[str],
    period: str = "60d",
    interval: str = "1d",
    batch_size: int = 50,
    logger: LogFn = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Sequential batch download (no threads). Useful for very small universes,
    debugging, or environments where thread pools are discouraged.
    """
    log = logger or _noop_log
    symbols = [s for s in map(str.strip, tickers) if s]
    data: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []

    for i, batch in enumerate(_chunk_list(symbols, batch_size), start=1):
        log(f"[batch] {i}/{math.ceil(len(symbols)/batch_size)}: downloading {len(batch)} symbols…")
        got = _download_batch_yf(batch, period, interval, logger=logger)
        data.update(got)
        for s in batch:
            if s not in got:
                skipped.append(s)

    skipped = [s for s in symbols if s not in data]
    log(f"[done] fetched {len(data)} / {len(symbols)} symbols. Skipped: {len(skipped)}")
    return data, skipped


def fetch_price_data_streaming(
    tickers: Sequence[str],
    period: str = "60d",
    interval: str = "1d",
    chunk_size: int = 200,
    max_workers: int = 8,
    logger: LogFn = None,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Generator that yields (symbol, df) as results become available.
    Ideal for showing incremental results in the UI.

    Note: This still fetches in batches for efficiency, then yields per-symbol.
    """
    log = logger or _noop_log
    symbols = [s for s in map(str.strip, tickers) if s]
    if not symbols:
        return
        yield  # pragma: no cover (generator formality)

    chunks = _chunk_list(symbols, chunk_size)
    log(f"[stream] {len(symbols)} symbols (chunk_size={chunk_size}, max_workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_download_batch_yf, chunk, period, interval, logger): chunk for chunk in chunks}
        for fut in as_completed(future_map):
            try:
                got = fut.result()
            except Exception as e:
                log(f"[stream error] {e}")
                got = {}
            for sym, df in got.items():
                yield sym, df


def fetch_hot_stocks(
    categories: Sequence[str] = ("trending", "most_active", "gainers", "losers"),
    count: int = 50,
    region: str = "US",
    logger: LogFn = None,
) -> Dict[str, List[str]]:
    """
    Fetch lists of hot stocks (trending, most active, gainers, losers) from Yahoo Finance.

    Returns a dict mapping category -> list of ticker symbols.

    This is UI-agnostic and safe for headless use. Requires network access.
    """
    log = logger or _noop_log
    # Map our friendly names to Yahoo's predefined screener IDs
    scr_ids = {
        "most_active": "most_actives",
        "gainers": "day_gainers",
        "losers": "day_losers",
        "trending": "trending_tickers",
        # Optional extras you might want later:
        "undervalued_growth": "undervalued_growth_stocks",
        "aggressive_small_caps": "aggressive_small_caps",
    }

    base_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    out: Dict[str, List[str]] = {}

    for cat in categories:
        scr = scr_ids.get(cat, cat)
        try:
            resp = requests.get(
                base_url,
                params={
                    "formatted": "false",
                    "lang": "en-US",
                    "region": region,
                    "scrIds": scr,
                    "start": 0,
                    "count": int(count),
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            quotes = (
                payload.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )
            symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
            out[cat] = symbols
            log(f"[hot] {cat}: fetched {len(symbols)} tickers")
        except Exception as e:
            log(f"[hot] failed {cat}: {e}")
            out[cat] = []

    return out

# -----------------------------
# Back-compat convenience wrappers
# -----------------------------

def fetch_most_active_stocks(count: int = 50, region: str = "US", logger: LogFn = None) -> List[str]:
    """Return a flat list of the most active tickers.

    Backward-compatible wrapper around `fetch_hot_stocks`.
    """
    res = fetch_hot_stocks(categories=("most_active",), count=count, region=region, logger=logger)
    return res.get("most_active", [])


def fetch_trending_stocks(count: int = 50, region: str = "US", logger: LogFn = None) -> List[str]:
    """Return a flat list of trending tickers.

    Backward-compatible wrapper around `fetch_hot_stocks`.
    """
    res = fetch_hot_stocks(categories=("trending",), count=count, region=region, logger=logger)
    return res.get("trending", [])


def fetch_top_gainers(count: int = 50, region: str = "US", logger: LogFn = None) -> List[str]:
    """Return a flat list of day gainers.

    Backward-compatible wrapper around `fetch_hot_stocks`.
    """
    res = fetch_hot_stocks(categories=("gainers",), count=count, region=region, logger=logger)
    return res.get("gainers", [])


def fetch_top_losers(count: int = 50, region: str = "US", logger: LogFn = None) -> List[str]:
    """Return a flat list of day losers.

    Backward-compatible wrapper around `fetch_hot_stocks`.
    """
    res = fetch_hot_stocks(categories=("losers",), count=count, region=region, logger=logger)
    return res.get("losers", [])


# ---------------------------------------------
# Back-compat wrapper for NASDAQ universe fetch
# ---------------------------------------------
try:
    # Prefer the canonical implementation in data.universe if present
    from .prices import fetch_and_save_nasdaq as _fetch_and_save_nasdaq  # type: ignore
except Exception:
    _fetch_and_save_nasdaq = None  # type: ignore


def fetch_and_save_nasdaq(*args, **kwargs):
    """Backward-compatible import shim.

    Historically this lived in a monolithic module and callers imported it from
    `ai_scanner.data.fetch`. During the package split it moved to
    `ai_scanner.data.universe`. This thin wrapper preserves old imports.

    If `ai_scanner.data.universe.fetch_and_save_nasdaq` is available, this
    delegates to it. Otherwise it raises a clear error telling the caller to
    update imports or implement the function.
    """
    if _fetch_and_save_nasdaq is None:
        raise ImportError(
            "fetch_and_save_nasdaq is not implemented in ai_scanner.data.fetch. "
            "Either implement it in ai_scanner.data.universe and keep this shim, "
            "or import it directly from ai_scanner.data.universe."
        )
    return _fetch_and_save_nasdaq(*args, **kwargs)

# ---------------------------------------------
# Back-compat wrapper for S&P 500 universe fetch
# ---------------------------------------------
try:
    from .prices import fetch_and_save_sp500 as _fetch_and_save_sp500  # type: ignore
except Exception:
    _fetch_and_save_sp500 = None  # type: ignore


def fetch_and_save_sp500(*args, **kwargs):
    """Backward-compatible import shim.

    Historically this lived in a monolithic module and callers imported it from
    `ai_scanner.data.fetch`. During the package split it moved to
    `ai_scanner.data.universe`. This thin wrapper preserves old imports.

    If `ai_scanner.data.universe.fetch_and_save_sp500` is available, this
    delegates to it. Otherwise it raises a clear error telling the caller to
    update imports or implement the function.
    """
    if _fetch_and_save_sp500 is None:
        raise ImportError(
            "fetch_and_save_sp500 is not implemented in ai_scanner.data.fetch. "
            "Either implement it in ai_scanner.data.universe and keep this shim, "
            "or import it directly from ai_scanner.data.universe."
        )
    return _fetch_and_save_sp500(*args, **kwargs)

# ---------------------------------------------
# Back-compat wrappers for loading universes
# ---------------------------------------------
try:
    from .prices import load_sp500_tickers as _load_sp500_tickers  # type: ignore
except Exception:
    _load_sp500_tickers = None  # type: ignore

try:
    from .prices import load_nasdaq_tickers as _load_nasdaq_tickers  # type: ignore
except Exception:
    _load_nasdaq_tickers = None  # type: ignore

# Optional additional universes (back-compat shims)
try:
    from .prices import load_sp400_tickers as _load_sp400_tickers  # type: ignore
except Exception:
    _load_sp400_tickers = None  # type: ignore

try:
    from .prices import load_sp600_tickers as _load_sp600_tickers  # type: ignore
except Exception:
    _load_sp600_tickers = None  # type: ignore

try:
    from .prices import fetch_and_save_sp600 as _fetch_and_save_sp600  # type: ignore
except Exception:
    _fetch_and_save_sp600 = None  # type: ignore

def load_sp500_tickers(*args, **kwargs):
    """Backward-compatible import shim for loading S&P 500 symbols.

    Prefer importing from `ai_scanner.data.universe`. This wrapper exists to
    avoid breaking older imports from `ai_scanner.data.fetch` during the
    module split.
    """
    if _load_sp500_tickers is None:
        raise ImportError(
            "load_sp500_tickers is not implemented in ai_scanner.data.fetch. "
            "Implement it in ai_scanner.data.universe and import from there, "
            "or keep this shim and ensure the function exists in universe."
        )
    return _load_sp500_tickers(*args, **kwargs)

def load_nasdaq_tickers(*args, **kwargs):
    """Backward-compatible import shim for loading NASDAQ symbols.

    Prefer importing from `ai_scanner.data.universe`. This wrapper exists to
    avoid breaking older imports from `ai_scanner.data.fetch` during the
    module split.
    """
    if _load_nasdaq_tickers is None:
        raise ImportError(
            "load_nasdaq_tickers is not implemented in ai_scanner.data.fetch. "
            "Implement it in ai_scanner.data.universe and import from there, "
            "or keep this shim and ensure the function exists in universe."
        )
    return _load_nasdaq_tickers(*args, **kwargs)

def load_sp400_tickers(*args, **kwargs):
    """Backward-compatible import shim for loading S&P 400 (MidCap) symbols.

    Prefer importing from `ai_scanner.data.universe`. This wrapper exists to
    avoid breaking older imports from `ai_scanner.data.fetch` during the
    module split.
    """
    if _load_sp400_tickers is None:
        raise ImportError(
            "load_sp400_tickers is not implemented in ai_scanner.data.fetch. "
            "Implement it in ai_scanner.data.universe and import from there, "
            "or keep this shim and ensure the function exists in universe."
        )
    return _load_sp400_tickers(*args, **kwargs)

def load_sp600_tickers(*args, **kwargs):
    """Backward-compatible import shim for loading S&P 600 (SmallCap) symbols.

    Prefer importing from `ai_scanner.data.universe`. This wrapper exists to
    avoid breaking older imports from `ai_scanner.data.fetch` during the
    module split.
    """
    if _load_sp600_tickers is None:
        raise ImportError(
            "load_sp600_tickers is not implemented in ai_scanner.data.fetch. "
            "Implement it in ai_scanner.data.universe and import from there, "
            "or keep this shim and ensure the function exists in universe."
        )
    return _load_sp600_tickers(*args, **kwargs)

def fetch_and_save_sp600(*args, **kwargs):
    """Backward-compatible import shim for fetching & saving S&P 600 universe.

    Delegates to `ai_scanner.data.universe.fetch_and_save_sp600` if available.
    """
    if _fetch_and_save_sp600 is None:
        raise ImportError(
            "fetch_and_save_sp600 is not implemented in ai_scanner.data.fetch. "
            "Either implement it in ai_scanner.data.universe and keep this shim, "
            "or import it directly from ai_scanner.data.universe."
        )
    return _fetch_and_save_sp600(*args, **kwargs)
