"""ai_scanner.prices

Robust, parallel price fetching utilities used across the app.
- Parallel batched downloads via Alpaca Market Data when configured (fallback to yfinance), with adaptive chunking
- Exponential backoff + jitter on errors
- Missing-ticker rescue using mini-batches
- Clean return format: (dict[str, pd.DataFrame], list[tuple[str, str]])
- No Streamlit dependency; optional `logger` callable for UI/console

All functions are safe to import in headless scheduler jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Iterator
import concurrent.futures as _fut
import datetime as _dt
import time as _time
import random as _random
import os as _os
import hashlib as _hashlib

import numpy as _np
import pandas as _pd

# --- In-memory price DataFrame cache ---
from typing import Dict
_PRICE_CACHE: Dict[str, _pd.DataFrame] = {}

def clear_price_cache() -> None:
    """Clear the in-memory price DataFrame cache.

    This is useful when debugging or when you suspect stale OHLCV data.
    """
    try:
        _PRICE_CACHE.clear()
    except Exception:
        # Extremely defensive: if anything goes wrong, just ignore.
        pass

try:
    import requests as _req
except Exception:  # pragma: no cover
    _req = None

# yfinance import (used for historical OHLCV). We keep the import error so
# the UI can surface a clear message when running in restricted environments.
try:
    import yfinance as _yf  # type: ignore
    _YF_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment-specific
    _yf = None  # type: ignore
    _YF_IMPORT_ERROR = e

# ---------- Public API ----------
__all__ = [
    "PriceFetchConfig",
    "fetch_price_data_parallel",
    "fetch_price_data_batch",
    "fetch_price_data_streaming",
    "clear_price_cache",
]


@dataclass
class PriceFetchConfig:
    tickers: Sequence[str]
    period: str = "60d"
    interval: str = "1d"
    prepost: bool = False
    max_workers: int = 4
    chunk_size: int = 70
    timeout_s: float = 10.0
    # How many full-batch retries when the API rate-limits
    batch_retries: int = 2
    # After full batches: try to rescue missing tickers in tiny mini-batches
    rescue_missing: bool = True
    mini_size: int = 8
    mini_retries: int = 2
    # Backoff base seconds for retries
    backoff_base: float = 1.2
    # Optional budget limiter (max count of successful tickers)
    success_budget: int | None = None


# Diagnostic helper for yfinance import and network status
def debug_yfinance_status(symbol: str = "AAPL") -> dict:
    """Return a small diagnostic snapshot about yfinance health.

    This is used by the Streamlit UI to help explain why price_data might
    be empty (e.g. yfinance not installed, import error, or network
    failures when calling Yahoo).
    """
    status: dict[str, object] = {
        "available": bool(_yf is not None),
        "import_error": str(_YF_IMPORT_ERROR) if "._YF_IMPORT_ERROR".split(".")[-1] in globals() and _YF_IMPORT_ERROR else None,
        "test_symbol": symbol,
        "test_rows": None,
        "test_error": None,
    }

    if _yf is None:
        return status

    try:
        df = _yf.download(symbol, period="60d", interval="1d", progress=False)
        status["test_rows"] = int(len(df))
    except Exception as e:  # pragma: no cover - depends on network
        status["test_error"] = str(e)

    return status


# ----------------- Alpaca helpers -----------------


def _get_alpaca_config() -> Dict[str, str] | None:
    """Return Alpaca API configuration if available, otherwise None.

    We try, in order:
      1) streamlit.secrets (if Streamlit is installed)
      2) environment variables
    """
    api_key: str | None = None
    api_secret: str | None = None
    data_url: str | None = None

    # Try Streamlit secrets first
    try:  # pragma: no cover - optional dependency
        import streamlit as _st  # type: ignore

        secrets = getattr(_st, "secrets", {})
        api_key = secrets.get("ALPACA_API_KEY_ID") or api_key
        api_secret = secrets.get("ALPACA_API_SECRET_KEY") or api_secret
        data_url = secrets.get("ALPACA_DATA_URL") or data_url
    except Exception:
        pass

    # Fallback to environment
    api_key = api_key or _os.getenv("ALPACA_API_KEY_ID")
    api_secret = api_secret or _os.getenv("ALPACA_API_SECRET_KEY")
    data_url = data_url or _os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets"

    if not api_key or not api_secret or _req is None:
        return None

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "data_url": data_url.rstrip("/"),
    }


def _alpaca_timeframe_from_interval(interval: str) -> str | None:
    """Map yfinance-style intervals to Alpaca timeframes (subset only).

    We currently support daily-only (1d) in this prices module.
    """
    mapping = {
        "1d": "1Day",
        "1D": "1Day",
    }
    return mapping.get(interval)


def _alpaca_limit_from_period(period: str) -> int:
    """Convert a yfinance-style period string into an approximate bar limit."""
    default = 60
    try:
        p = period.lower()
        if p.endswith("d"):
            return max(1, int(p[:-1]))
        if p.endswith("mo"):
            months = int(p[:-2])
            return max(1, months * 21)  # ~21 trading days per month
        if p.endswith("y"):
            years = int(p[:-1])
            return max(1, years * 252)  # ~252 trading days per year
    except Exception:
        pass
    return default


def _download_multi_alpaca(
    tickers: Sequence[str],
    period: str,
    interval: str,
    prepost: bool,  # kept for signature compatibility; daily bars ignore it
    timeout_s: float,
) -> Dict[str, _pd.DataFrame]:
    """Download bars for multiple symbols from Alpaca Market Data.

    Returns a dict[symbol -> DataFrame] with OHLCV columns shaped like
    yfinance output (Open, High, Low, Close, Adj Close, Volume).
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        raise RuntimeError("Alpaca configuration is not available.")

    timeframe = _alpaca_timeframe_from_interval(interval)
    if timeframe is None:
        raise RuntimeError(f"Unsupported interval for Alpaca bars: {interval!r}")

    limit = _alpaca_limit_from_period(period)
    symbols = [s for s in dict.fromkeys(tickers) if isinstance(s, str) and s.strip()]
    if not symbols:
        return {}

    url = f"{cfg['data_url']}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }

    out: Dict[str, _pd.DataFrame] = {}

    # Alpaca has practical limits on how many symbols can be requested in a
    # single /v2/stocks/bars call. To stay well under those limits (and to keep
    # responses fast for large universes like NASDAQ/Combo), we further chunk
    # the provided tickers into modest sub-batches.
    max_symbols_per_call = 150
    for chunk in _chunks(symbols, max_symbols_per_call):
        params = {
            "symbols": ",".join(sorted(chunk)),
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "raw",
            "feed": "iex",
        }

        try:
            resp = _req.get(url, headers=headers, params=params, timeout=timeout_s)
        except Exception:
            # Network / transport error for this chunk: skip it and continue with others.
            continue

        if resp.status_code != 200:
            # Do not raise for a single-chunk failure; just skip this chunk so that
            # partial results can still be used. The caller will fall back to
            # yfinance only if *all* chunks fail to return any data.
            continue

        try:
            payload = resp.json()
        except Exception:
            # Malformed JSON for this chunk; skip it.
            continue

        # Alpaca responds with a dict under 'bars': {symbol: [bars...]}, but we
        # defensively handle a top-level mapping as well.
        bars_by_symbol = payload.get("bars") if isinstance(payload, dict) else {}
        if not bars_by_symbol and isinstance(payload, dict):
            # Some legacy formats may use a different key; treat payload itself
            bars_by_symbol = {
                k: v for k, v in payload.items() if isinstance(v, list)
            }

        for sym, bars in (bars_by_symbol or {}).items():
            if not bars:
                continue
            df = _pd.DataFrame(bars)
            if df.empty:
                continue

            # Expect columns like t/o/h/l/c/v
            if "t" in df.columns:
                df["t"] = _pd.to_datetime(df["t"], errors="coerce")
                df = df.set_index("t")

            rename: Dict[str, str] = {}
            for old, new in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("v", "Volume")):
                if old in df.columns:
                    rename[old] = new
            if rename:
                df = df.rename(columns=rename)

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            df = _normalize_df(df)
            try:
                df.attrs["source"] = "alpaca_multi"
                df.attrs["symbol"] = str(sym).upper()
            except Exception:
                pass
            out[str(sym).upper()] = df

    return out


# ----------------- Helpers -----------------


def _log(logger: Callable[[str], None] | None, msg: str) -> None:
    if logger:
        try:
            logger(msg)
        except Exception:
            pass


# --- Helper for cache keys ---
def _cache_key(sym: str, cfg: PriceFetchConfig) -> str:
    """Build a stable cache key for a given symbol + config."""
    return f"{str(sym).upper()}|{cfg.period}|{cfg.interval}|{'1' if cfg.prepost else '0'}"


def _chunks(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    return (list(seq[i : i + n]) for i in range(0, len(seq), n))


def _backoff_sleep(base: float, attempt: int) -> None:
    jitter = _random.uniform(0.25, 0.75)
    delay = max(0.05, base * (2 ** max(0, attempt - 1)) * jitter)
    _time.sleep(delay)


# ----------------- Core download routines -----------------

def _download_multi(
    tickers: Sequence[str],
    period: str,
    interval: str,
    prepost: bool,
    timeout_s: float,
) -> Dict[str, _pd.DataFrame]:
    """Safe multi-symbol download wrapper.

    Historically this used yfinance's native *multi-ticker* download, but on
    some hosts that path can silently "poison" results (misaligned OHLCV,
    duplicate values across symbols, etc.).

    To avoid that, this helper now delegates to `_download_batch`, which
    performs **per-symbol** yfinance downloads plus `_normalize_df`. The
    signature is preserved for older call sites, but the behavior is now
    equivalent to a single robust batch fetch.
    """
    if not _yf:
        raise RuntimeError(
            "yfinance is not available in this environment (import failed)."
        )

    # Normalize and deduplicate symbols up-front.
    norm_syms: List[str] = [
        str(s).upper()
        for s in (tickers or [])
        if isinstance(s, str) and s.strip()
    ]
    # Preserve order but drop duplicates.
    norm_syms = list(dict.fromkeys(norm_syms))

    if not norm_syms:
        return {}

    # Build a minimal config that mirrors the original arguments but forces
    # single-threaded behavior (since `_download_batch` loops over symbols).
    cfg = PriceFetchConfig(
        tickers=norm_syms,
        period=period,
        interval=interval,
        prepost=prepost,
        max_workers=1,
        chunk_size=max(1, len(norm_syms)),
        timeout_s=timeout_s,
        batch_retries=0,
        rescue_missing=False,
        mini_size=1,
        mini_retries=0,
        backoff_base=1.2,
        success_budget=None,
    )

    data, _skipped = _download_batch(norm_syms, cfg)
    return data


def _normalize_df(df: _pd.DataFrame) -> _pd.DataFrame:
    """Best-effort normalization of an OHLCV DataFrame.

    This helper is intentionally defensive: any failure to coerce dtypes will
    leave the original column values in place rather than raising. This avoids
    crashes like "TypeError: arg must be a list, tuple, 1-d array, or Series"
    that can occur with some yfinance/pandas combinations.
    """
    if df is None or not isinstance(df, _pd.DataFrame):
        return df

    df = df.copy()

    # Normalize column names to canonical OHLCV labels.
    # This handles common variants like 'open', 'Open ', 'adj_close', etc.
    try:
        col_map: Dict[object, str] = {}
        for col in list(df.columns):
            name = str(col).strip()
            lower = name.lower().replace("_", " ")

            if lower in ("open", "o"):
                new = "Open"
            elif lower in ("high", "h"):
                new = "High"
            elif lower in ("low", "l"):
                new = "Low"
            elif lower in ("close", "c"):
                new = "Close"
            elif lower in ("adj close", "adjclose", "adjusted close"):
                new = "Adj Close"
            elif lower in ("volume", "vol", "v"):
                new = "Volume"
            else:
                new = name

            col_map[col] = new

        if col_map:
            df = df.rename(columns=col_map)
            # Ensure column names are unique (drop duplicate labels, keep first).
            try:
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]
            except Exception:
                # If anything goes wrong during deduplication, keep the original columns.
                pass
    except Exception:
        # If anything goes wrong during renaming, keep the original columns.
        pass

    # Float-like OHLC columns
    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
        if c not in df.columns:
            continue
        try:
            df[c] = _pd.to_numeric(df[c], errors="coerce")
            try:
                df[c] = df[c].astype("float32")
            except TypeError:
                pass
        except Exception:
            continue

    # Volume column
    if "Volume" in df.columns:
        try:
            df["Volume"] = _pd.to_numeric(df["Volume"], errors="coerce")
            try:
                df["Volume"] = df["Volume"].astype("int64")
            except TypeError:
                pass
        except Exception:
            pass

    # Sort index defensively
    try:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
    except Exception:
        pass

    return df


# --- Frame fingerprinting helper ---
def _frame_fingerprint(df: _pd.DataFrame) -> str | None:
    """
    Build a lightweight fingerprint for a price DataFrame.

    This is used as a defensive guardrail against the rare case where
    multiple *different* tickers end up with the exact same OHLCV data
    due to upstream API quirks. We intentionally only sample a few rows
    so this stays inexpensive even for large universes.
    """
    if not isinstance(df, _pd.DataFrame) or df.empty:
        return None

    # Focus on core OHLCV columns only.
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not cols:
        return None

    try:
        # Take a few rows from the head and tail to build a sample.
        sample_head = df[cols].head(4)
        sample_tail = df[cols].tail(4)
        sample = _pd.concat([sample_head, sample_tail], axis=0)
        # Convert to CSV bytes and hash them.
        payload = sample.to_csv(index=False).encode("utf-8")
        return _hashlib.sha1(payload).hexdigest()
    except Exception:
        # If anything goes wrong, just skip fingerprinting.
        return None


def _download_batch(
    batch: Sequence[str],
    cfg: PriceFetchConfig,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """Download a batch of symbols using per-symbol yfinance calls only.

    This intentionally bypasses the multi-ticker `_download_multi` path,
    which can return empty results on restricted hosts even when single-
    symbol downloads work. It is slower but much more reliable for
    environments like Streamlit Cloud.
    """
    data: Dict[str, _pd.DataFrame] = {}
    skipped: List[Tuple[str, str]] = []

    if not batch:
        return data, skipped

    for sym in batch:
        sym_u = str(sym).upper()
        if _yf is None:
            skipped.append((sym_u, "yf_not_installed"))
            continue

        # 1) Pull raw bars via per-symbol Ticker().history
        try:
            ticker_obj = _yf.Ticker(sym_u)
            df = ticker_obj.history(
                period=cfg.period,
                interval=cfg.interval,
                prepost=cfg.prepost,
                auto_adjust=False,
            )
            # Defensive: ensure each symbol gets its own independent DataFrame
            try:
                df = _pd.DataFrame(df).copy(deep=True)
            except Exception:
                pass
        except Exception as e:
            skipped.append((sym_u, f"error_download:{type(e).__name__}:{e}"))
            continue

        if df is None or df.empty:
            skipped.append((sym_u, "empty_single"))
            continue

        # 2) Normalize the DataFrame but keep the DateTimeIndex semantics
        try:
            norm = _normalize_df(df)

            # Safeguard: ensure columns are unique after normalization
            try:
                if norm.columns.duplicated().any():
                    norm = norm.loc[:, ~norm.columns.duplicated()]
            except Exception:
                pass

            # Tag the frame with its symbol and make sure it is not shared
            try:
                norm.attrs["symbol"] = sym_u
                norm = norm.copy(deep=True)
            except Exception:
                norm = norm.copy()

            data[sym_u] = norm
        except Exception as e:
            # If normalization blows up, fall back to the raw df so the caller
            # still has something to work with.
            data[sym_u] = df
            skipped.append((sym_u, f"error_normalize:{type(e).__name__}:{e}"))

    return data, skipped


def _rescue_missing_in_minibatches(
    missing_syms: Sequence[str],
    cfg: PriceFetchConfig,
    logger: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """
    Rescue pass for symbols that failed in the main parallel batches.

    IMPORTANT:
    - Uses _download_batch (per-symbol yfinance) instead of the
      multi-ticker _download_multi to avoid any chance of misaligned /
      duplicated data coming from yfinance's multi-symbol mode.
    """
    data: Dict[str, _pd.DataFrame] = {}
    skipped: List[Tuple[str, str]] = []

    if not missing_syms:
        return data, skipped

    for mini in _chunks(missing_syms, cfg.mini_size):
        if not mini:
            continue

        _log(logger, f"[prices] Rescue mini-batch of {len(mini)} symbols…")

        # We will retry this mini-batch a few times using _download_batch,
        # shrinking the 'still_missing' set as we successfully rescue symbols.
        still_missing: List[str] = list(mini)

        for attempt in range(1, cfg.mini_retries + 2):
            try:
                got_batch, skipped_batch = _download_batch(still_missing, cfg)
            except Exception as e:  # very defensive
                got_batch = {}
                skipped_batch = [
                    (s, f"rescue_error:{type(e).__name__}:{e}") for s in still_missing
                ]

            # Tag rescued frames
            for sym, df in got_batch.items():
                try:
                    df.attrs["source"] = "rescue_single"
                    df.attrs["symbol"] = str(sym).upper()
                except Exception:
                    pass

            for _sym, _df in got_batch.items():
                try:
                    data[_sym] = _df.copy(deep=True)
                except Exception:
                    data[_sym] = _df
            skipped.extend(skipped_batch)

            # Recompute what is still missing after this attempt
            still_missing = [s for s in still_missing if s not in got_batch]

            if not still_missing:
                break  # this mini-batch is fully rescued

            if attempt <= cfg.mini_retries and still_missing:
                _backoff_sleep(cfg.backoff_base, attempt)

        # Any symbols that are still missing after all attempts are marked as such
        for s in still_missing:
            skipped.append((s, "missing_after_rescue"))

    return data, skipped


# ----------------- Public Orchestrator -----------------


def fetch_price_data_parallel(
    tickers: Sequence[str],
    *,
    period: str = "60d",
    interval: str = "1d",
    prepost: bool = False,
    max_workers: int = 4,
    chunk_size: int = 70,
    timeout_s: float = 10.0,
    batch_retries: int = 2,
    rescue_missing: bool = True,
    mini_size: int = 8,
    mini_retries: int = 2,
    backoff_base: float = 1.2,
    success_budget: int | None = None,
    logger: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """
    Fetch price data for `tickers` in parallel batches.

    Returns (price_data, skipped), where:
      - price_data: dict[symbol -> DataFrame]
      - skipped: list of (symbol, reason)
    """
    tickers = [t for t in (tickers or []) if isinstance(t, str) and t.strip()]
    if not tickers:
        return {}, []

    cfg = PriceFetchConfig(
        tickers=tickers,
        period=period,
        interval=interval,
        prepost=prepost,
        max_workers=max_workers,
        chunk_size=chunk_size,
        timeout_s=timeout_s,
        batch_retries=batch_retries,
        rescue_missing=rescue_missing,
        mini_size=mini_size,
        mini_retries=mini_retries,
        backoff_base=backoff_base,
        success_budget=success_budget,
    )

    _log(
        logger,
        f"[prices] Prepared {len(tickers)} symbols (chunk_size={cfg.chunk_size}, max_workers={cfg.max_workers})",
    )

    price_data: Dict[str, _pd.DataFrame] = {}
    skipped: List[Tuple[str, str]] = []

    # --- Cache lookups -------------------------------------------------
    remaining: List[str] = []
    cache_hits = 0
    for sym in cfg.tickers:
        key = _cache_key(sym, cfg)
        cached = _PRICE_CACHE.get(key)

        # Validate cached frame: must be a non-empty DataFrame and must have
        # a `.attrs["symbol"]` tag that matches this ticker. Older cache
        # entries without a symbol tag are treated as stale and discarded.
        use_cached = False
        if isinstance(cached, _pd.DataFrame) and not cached.empty:
            try:
                sym_u = str(sym).upper()
                attrs = getattr(cached, "attrs", {}) or {}
                cached_sym = attrs.get("symbol", None)
                if isinstance(cached_sym, str) and cached_sym.upper() == sym_u:
                    use_cached = True
                else:
                    # Stale or mismatched cache entry: remove so it can't be reused.
                    _PRICE_CACHE.pop(key, None)
            except Exception:
                # If anything goes wrong while validating, treat as a cache miss.
                use_cached = False

        if use_cached:
            try:
                df_cached = cached.copy(deep=True)  # type: ignore[assignment]
            except Exception:
                # Fallback: construct a new DataFrame from the cached object
                df_cached = _pd.DataFrame(cached)  # type: ignore[arg-type]
            price_data[str(sym).upper()] = df_cached
            cache_hits += 1
        else:
            remaining.append(sym)

    if cache_hits:
        _log(logger, f"[prices] Cache hits: {cache_hits} symbols")

    download_tickers: List[str] = remaining

    # -------------------------------------------------------------------
    chunks = list(_chunks(download_tickers, cfg.chunk_size))

    # --- Limited parallelism over _download_batch ---
    if chunks:
        # If only one chunk or caller forces max_workers=1, keep it simple/serial.
        if len(chunks) == 1 or cfg.max_workers <= 1:
            for batch in chunks:
                _log(logger, f"[prices] Downloading chunk of {len(batch)} symbols…")
                try:
                    data, sk = _download_batch(batch, cfg)
                except Exception as e:  # very defensive
                    data, sk = {}, [(s, f"batch_error:{type(e).__name__}:{e}") for s in batch]
                price_data.update(data)
                skipped.extend(sk)
        else:
            # Use a small cap on workers to avoid hammering yfinance too hard.
            worker_count = min(max(1, cfg.max_workers), 4, len(chunks))
            _log(
                logger,
                f"[prices] Downloading {len(chunks)} chunks with {worker_count} workers (chunk_size={cfg.chunk_size})…",
            )
            with _fut.ThreadPoolExecutor(max_workers=worker_count) as ex:
                future_map = {ex.submit(_download_batch, batch, cfg): batch for batch in chunks}
                done = 0
                total = len(future_map)
                for fut in _fut.as_completed(future_map):
                    batch = future_map[fut]
                    try:
                        data, sk = fut.result()
                    except Exception as e:
                        data, sk = {}, [(s, f"batch_error:{type(e).__name__}:{e}") for s in batch]
                    done += 1
                    _log(
                        logger,
                        f"[prices] Chunk {done}/{total} ready ({len(data)} ok, {len(sk)} skipped)",
                    )
                    price_data.update(data)
                    skipped.extend(sk)
    # --- End limited parallelism block ---

    # Optionally rescue missing
    if cfg.rescue_missing:
        missing = [s for s in tickers if s not in price_data]
        if missing:
            _log(logger, f"[prices] Rescuing {len(missing)} missing symbols in mini-batches of {cfg.mini_size}…")
            got, sk2 = _rescue_missing_in_minibatches(missing, cfg, logger)
            price_data.update(got)
            skipped.extend(sk2)

    # Success budget
    if cfg.success_budget is not None and len(price_data) > cfg.success_budget:
        # Trim deterministically by ticker name to keep logic simple
        keep = sorted(price_data.keys())[: cfg.success_budget]
        drop = [k for k in price_data.keys() if k not in keep]
        for k in drop:
            price_data.pop(k, None)
            skipped.append((k, "over_budget"))

    # Final log of missing
    still_missing = [s for s in tickers if s not in price_data]
    for s in still_missing:
        skipped.append((s, "missing_final"))

    if still_missing:
        _log(
            logger,
            f"[prices] Missing in final result: "
            f"{', '.join(still_missing[:8])}"
            f"{'…' if len(still_missing) > 8 else ''}"
        )

    # Final cleanup: only return symbols that have a non-empty DataFrame.
    # This prevents downstream scanners from operating on placeholders or
    # partially-constructed frames.
    valid_price_data: Dict[str, _pd.DataFrame] = {}
    for sym, df in price_data.items():
        if isinstance(df, _pd.DataFrame) and not df.empty:
            valid_price_data[sym] = df
        else:
            skipped.append((sym, "invalid_frame"))

    # Defensive de-duplication: if multiple tickers end up with *identical*
    # OHLCV data fingerprints, treat the later ones as poisoned duplicates
    # and drop them. This is extremely unlikely to affect legitimate data
    # but protects against upstream multi-ticker bugs.
    fingerprints: Dict[str, str] = {}
    deduped_price_data: Dict[str, _pd.DataFrame] = {}
    for sym, df in valid_price_data.items():
        fp = _frame_fingerprint(df)
        if fp is None:
            # No fingerprint available; keep the frame as-is.
            deduped_price_data[sym] = df
            continue

        owner = fingerprints.get(fp)
        if owner is None:
            fingerprints[fp] = sym
            deduped_price_data[sym] = df
        else:
            # This frame looks identical to another symbol's data; skip it.
            skipped.append((sym, f"duplicate_frame_like:{owner}"))

    # Update cache with validated, de-duplicated frames so future scans
    # can reuse them safely.
    for sym, df in deduped_price_data.items():
        key = _cache_key(sym, cfg)
        try:
            _PRICE_CACHE[key] = df.copy(deep=True)
        except Exception:
            _PRICE_CACHE[key] = df

    return deduped_price_data, skipped


# ----------------- Back-compat shims -----------------

def fetch_price_data_batch(
    tickers: Sequence[str],
    *,
    period: str = "60d",
    interval: str = "1d",
    prepost: bool = False,
    timeout_s: float = 10.0,
    batch_retries: int = 2,
    backoff_base: float = 1.2,
    logger: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """Backwards-compatible single-batch download.

    Uses the same robust logic as a normal batch, without rescue mini-batches
    or extra parallelism. Suitable where callers expect a one-shot call.
    """
    syms = [t for t in (tickers or []) if isinstance(t, str) and t.strip()]
    if not syms:
        return {}, []

    cfg = PriceFetchConfig(
        tickers=syms,
        period=period,
        interval=interval,
        prepost=prepost,
        max_workers=1,               # single logical batch
        chunk_size=max(1, len(syms)),
        timeout_s=timeout_s,
        batch_retries=batch_retries,
        rescue_missing=False,        # no mini-rescue in strict batch mode
        backoff_base=backoff_base,
    )
    _log(logger, f"[prices] Batch download {len(syms)} symbols…")
    return _download_batch(syms, cfg)


def fetch_price_data_streaming(
    tickers: Sequence[str],
    *,
    period: str = "60d",
    interval: str = "1d",
    prepost: bool = False,
    max_workers: int = 4,
    chunk_size: int = 70,
    timeout_s: float = 10.0,
    batch_retries: int = 2,
    backoff_base: float = 1.2,
    logger: Callable[[str], None] | None = None,
) -> Iterator[Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]], int, int]]:
    """Stream results chunk-by-chunk.

    Yields tuples: (partial_data, partial_skipped, chunks_done, chunks_total)
    Rescue of missing tickers is intentionally *not* done here to keep
    streaming responsive; callers can decide to do a final rescue pass if
    desired using `fetch_price_data_parallel`.
    """
    syms = [t for t in (tickers or []) if isinstance(t, str) and t.strip()]
    if not syms:
        return

    cfg = PriceFetchConfig(
        tickers=syms,
        period=period,
        interval=interval,
        prepost=prepost,
        max_workers=max_workers,
        chunk_size=chunk_size,
        timeout_s=timeout_s,
        batch_retries=batch_retries,
        rescue_missing=False,
        backoff_base=backoff_base,
    )

    chunks = list(_chunks(cfg.tickers, cfg.chunk_size))
    total = len(chunks)
    if total == 0:
        return

    _log(logger, f"[prices] Streaming {len(syms)} symbols in {total} chunks (size={cfg.chunk_size})…")

    done = 0
    with _fut.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        future_map = {ex.submit(_download_batch, ch, cfg): ch for ch in chunks}
        for fut in _fut.as_completed(future_map):
            try:
                data, skipped = fut.result()
            except Exception as e:
                data, skipped = {}, [(s, f"stream_error:{type(e).__name__}") for s in future_map[fut]]
            done += 1
            _log(logger, f"[prices] Chunk {done}/{total} ready ({len(data)} ok, {len(skipped)} skipped)")
            yield data, skipped, done, total
