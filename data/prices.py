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

import concurrent.futures as _fut
import datetime as _dt
import os
import time as _time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import numpy as _np
import pandas as _pd

from .price_alpaca import (
    download_multi_alpaca as _download_multi_alpaca,
)
from .price_alpaca import (
    get_alpaca_config as _get_alpaca_config,
)
from .price_utils import (
    backoff_sleep as _backoff_sleep,
)
from .price_utils import (
    cache_key as _cache_key,
)
from .price_utils import (
    chunks as _chunks,
)
from .price_utils import (
    frame_fingerprint as _frame_fingerprint,
)
from .price_utils import (
    normalize_price_frame as _normalize_df,
)

try:
    from .provider_diagnostics import summarize_provider_skips
except ImportError:  # pragma: no cover - keep import resilient in legacy paths
    summarize_provider_skips = None  # type: ignore


_YFINANCE_BASE_ERRORS = (RuntimeError, TimeoutError, ConnectionError, OSError, ValueError)

# --- In-memory price DataFrame cache ---
# We keep a short TTL so repeated scans within a few seconds can reuse
# results, but fresh data is fetched on subsequent runs.
_PRICE_CACHE_TTL_SECONDS: float = 60.0  # ~1 minute; tuned for daily bars
_PRICE_CACHE: Dict[str, tuple[_pd.DataFrame, float]] = {}

def clear_price_cache() -> None:
    """Clear the in-memory price DataFrame cache.

    This is useful when debugging or when you suspect stale OHLCV data.
    """
    _PRICE_CACHE.clear()

# yfinance import (used for historical OHLCV). We keep the import error so
# the UI can surface a clear message when running in restricted environments.
try:
    import yfinance as _yf  # type: ignore
    from yfinance import exceptions as _yf_exceptions  # type: ignore

    _YF_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as e:  # pragma: no cover - environment-specific
    _yf = None  # type: ignore
    _yf_exceptions = None  # type: ignore
    _YF_IMPORT_ERROR = e


def _build_yfinance_errors() -> tuple[type[Exception], ...]:
    """Return provider exceptions that should be treated as non-fatal skips."""
    extra_names = (
        "YFException",
        "YFRateLimitError",
        "YFPricesMissingError",
        "YFTzMissingError",
    )
    extra_types = []
    for name in extra_names:
        cls = getattr(_yf_exceptions, name, None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            extra_types.append(cls)
    return _YFINANCE_BASE_ERRORS + tuple(extra_types)


_YFINANCE_ERRORS = _build_yfinance_errors()


def _build_alpaca_errors() -> tuple[type[Exception], ...]:
    base = (RuntimeError, OSError, ValueError, KeyError, TypeError, ConnectionError, TimeoutError)
    try:
        from requests import exceptions as _req_exc  # type: ignore
        return base + (_req_exc.RequestException,)
    except ImportError:
        return base


_ALPACA_ERRORS = _build_alpaca_errors()

# ---------- Public API ----------
__all__ = [
    "PriceFetchConfig",
    "fetch_price_data_parallel",
    "fetch_price_data_batch",
    "fetch_price_data_streaming",
    "debug_yfinance_status",
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
        "import_error": str(_YF_IMPORT_ERROR) if _YF_IMPORT_ERROR else None,
        "test_symbol": symbol,
        "test_rows": None,
        "test_error": None,
    }

    if _yf is None:
        return status

    try:
        df = _yf.download(
            symbol,
            period="60d",
            interval="1d",
            progress=False,
            threads=False,
            timeout=10,
        )
        status["test_rows"] = int(len(df))
    except _YFINANCE_ERRORS as e:  # pragma: no cover - depends on network
        status["test_error"] = str(e)

    return status


# ----------------- Helpers -----------------


def _log(logger: Callable[[str], None] | None, msg: str) -> None:
    if logger:
        try:
            logger(msg)
        except (RuntimeError, TypeError, ValueError):
            pass


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

    remaining = [str(s).upper() for s in batch if str(s).strip()]

    # --- Primary provider: Alpaca (bulk), when configured ---
    # Alpaca does not IP-throttle like yfinance, so it handles large universes
    # (e.g. the 2400+ COMBO list) reliably. Whatever Alpaca doesn't return falls
    # through to the per-symbol yfinance loop below — unless PRICE_SKIP_YF_FALLBACK
    # is set, in which case stragglers (mostly micro-caps/warrants not on Alpaca's
    # IEX feed, which never pass filters) are skipped for a big speedup.
    alpaca_returned_data = False
    if _get_alpaca_config() is not None:
        try:
            alpaca_data = _download_multi_alpaca(
                remaining, cfg.period, cfg.interval, cfg.prepost, cfg.timeout_s
            )
            for sym, adf in (alpaca_data or {}).items():
                if adf is None or adf.empty:
                    continue
                try:
                    norm = _normalize_df(adf)
                except _YFINANCE_ERRORS:
                    norm = adf
                data[str(sym).upper()] = norm
            got = set(data.keys())
            remaining = [s for s in remaining if s not in got]
            alpaca_returned_data = len(got) > 0
        except _ALPACA_ERRORS as e:
            print(f"[prices] Alpaca batch failed, falling back to yfinance: {type(e).__name__}: {e}")

    # Skip the slow per-symbol yfinance fallback when Alpaca already returned
    # data and the operator opted into Alpaca-only mode for speed.
    if alpaca_returned_data and os.getenv("PRICE_SKIP_YF_FALLBACK", "").strip() == "1":
        for sym in remaining:
            skipped.append((str(sym).upper(), "skipped_yf_fallback"))
        remaining = []

    for sym in remaining:
        sym_u = str(sym).upper()
        if _yf is None:
            skipped.append((sym_u, "yf_not_installed"))
            continue

        # 1) Pull raw bars via per-symbol Ticker().history
        try:
            ticker_obj = _yf.Ticker(sym_u)
            # IMPORTANT:
            # - auto_adjust=False: do not apply total-return adjustments
            # - actions=False: avoid mixing in dividends/splits as adjustments
            df = ticker_obj.history(
                period=cfg.period,
                interval=cfg.interval,
                prepost=cfg.prepost,
                auto_adjust=False,
                actions=False,
            )
            # Defensive: ensure each symbol gets its own independent DataFrame
            try:
                df = _pd.DataFrame(df).copy(deep=True)
            except (TypeError, ValueError):
                pass

            # yfinance may still return OHLC values that are effectively
            # split-adjusted. If we have a Stock Splits column, we can
            # reconstruct raw Yahoo-style prices by undoing the cumulative
            # split factor. This keeps scanner calculations aligned with
            # what you see on finance.yahoo.com.
            try:
                if (
                    isinstance(df, _pd.DataFrame)
                    and not df.empty
                    and "Stock Splits" in df.columns
                ):
                    # Build a cumulative split factor series. Stock Splits
                    # is usually 0 when no split occurred; replace 0 with 1
                    # so the cumulative product behaves correctly.
                    split_series = df["Stock Splits"].replace(0, 1)
                    # Newer rows first => reverse, cumprod, reverse back.
                    cumulative_split = split_series.iloc[::-1].cumprod().iloc[::-1]

                    # Only adjust columns that actually exist.
                    for col in ["Open", "High", "Low", "Close"]:
                        if col in df.columns:
                            df[col] = df[col] * cumulative_split

                    # Keep Adj Close in sync with the raw Close that the
                    # rest of the app expects.
                    if "Close" in df.columns:
                        df["Adj Close"] = df["Close"]
            except (AttributeError, KeyError, TypeError, ValueError):
                # If anything goes wrong while de-adjusting, keep df as-is.
                pass
        except _YFINANCE_ERRORS as e:
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
            except (AttributeError, TypeError, ValueError):
                pass

            # Tag the frame with its symbol and make sure it is not shared
            try:
                norm.attrs["symbol"] = sym_u
                norm = norm.copy(deep=True)
            except (AttributeError, TypeError, ValueError):
                norm = norm.copy()

            data[sym_u] = norm
        except (AttributeError, KeyError, TypeError, ValueError) as e:
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
            except _YFINANCE_ERRORS as e:
                got_batch = {}
                skipped_batch = [
                    (s, f"rescue_error:{type(e).__name__}:{e}") for s in still_missing
                ]

            # Tag rescued frames
            for sym, df in got_batch.items():
                try:
                    df.attrs["source"] = "rescue_single"
                    df.attrs["symbol"] = str(sym).upper()
                except (AttributeError, TypeError, ValueError):
                    pass

            for _sym, _df in got_batch.items():
                try:
                    data[_sym] = _df.copy(deep=True)
                except (AttributeError, TypeError, ValueError):
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
    use_cache: bool = True,
    logger: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """
    Fetch price data for `tickers` in parallel batches.

    Returns (price_data, skipped), where:
      - price_data: dict[symbol -> DataFrame]
      - skipped: list of (symbol, reason)

    If `use_cache` is False, cache lookups and writes are skipped and all
    symbols are fetched fresh for this call.
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
    if use_cache:
        remaining: List[str] = []
        cache_hits = 0
        for sym in cfg.tickers:
            key = _cache_key(sym, cfg)
            cached_entry = _PRICE_CACHE.get(key)
            cached_df: _pd.DataFrame | None = None

            if isinstance(cached_entry, tuple) and len(cached_entry) == 2:
                candidate_df, ts = cached_entry
                # Enforce a simple TTL so that intraday re-runs still get fresh data
                if (ts is not None) and (_time.time() - float(ts) <= _PRICE_CACHE_TTL_SECONDS):
                    cached = candidate_df
                else:
                    # Stale entry; drop it
                    _PRICE_CACHE.pop(key, None)
                    cached = None
            else:
                # Backwards-compat: old-style entries without a timestamp are treated as stale
                cached = None

            if isinstance(cached, _pd.DataFrame) and not cached.empty:
                try:
                    sym_u = str(sym).upper()
                    attrs = getattr(cached, "attrs", {}) or {}
                    cached_sym = attrs.get("symbol", None)
                    if isinstance(cached_sym, str) and cached_sym.upper() == sym_u:
                        cached_df = cached
                    else:
                        _PRICE_CACHE.pop(key, None)
                except (AttributeError, TypeError, ValueError):
                    cached_df = None

            if cached_df is not None:
                try:
                    df_cached = cached_df.copy(deep=True)
                except (AttributeError, TypeError, ValueError):
                    df_cached = _pd.DataFrame(cached_df)
                price_data[str(sym).upper()] = df_cached
                cache_hits += 1
            else:
                remaining.append(sym)

        if cache_hits:
            _log(logger, f"[prices] Cache hits: {cache_hits} symbols")

        download_tickers: List[str] = remaining
    else:
        # Bypass cache entirely: every symbol is downloaded fresh.
        _log(logger, "[prices] Cache disabled for this call (use_cache=False)")
        download_tickers = list(cfg.tickers)

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
                except _YFINANCE_ERRORS as e:
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
                    except _YFINANCE_ERRORS as e:
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
    if use_cache:
        for sym, df in deduped_price_data.items():
            key = _cache_key(sym, cfg)
            try:
                _PRICE_CACHE[key] = (df.copy(deep=True), _time.time())
            except (AttributeError, TypeError, ValueError):
                _PRICE_CACHE[key] = (df, _time.time())

    if summarize_provider_skips is not None:
        try:
            summary = summarize_provider_skips(
                requested=len(tickers),
                returned=len(deduped_price_data),
                skipped=skipped,
            )
            _log(logger, f"[prices] {summary.message}")
        except (RuntimeError, TypeError, ValueError):
            pass

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
            except _YFINANCE_ERRORS as e:
                data, skipped = {}, [(s, f"stream_error:{type(e).__name__}") for s in future_map[fut]]
            done += 1
            _log(logger, f"[prices] Chunk {done}/{total} ready ({len(data)} ok, {len(skipped)} skipped)")
            yield data, skipped, done, total
