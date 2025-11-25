"""ai_scanner.prices

Robust, parallel price fetching utilities used across the app.
- Parallel batched downloads via yfinance, with adaptive chunking
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

import numpy as _np
import pandas as _pd

try:
    import yfinance as _yf
except Exception:  # pragma: no cover
    _yf = None  # allows unit testing without yfinance installed

# ---------- Public API ----------
__all__ = [
    "PriceFetchConfig",
    "fetch_price_data_parallel",
    "fetch_price_data_batch",
    "fetch_price_data_streaming",
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


# ----------------- Helpers -----------------

def _log(logger: Callable[[str], None] | None, msg: str) -> None:
    if logger:
        try:
            logger(msg)
        except Exception:
            pass


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
    if not _yf:
        raise RuntimeError("yfinance is not installed.")

    # yfinance.download with list returns multiindex columns; `group_by='ticker'`
    data = _yf.download(
        tickers=list(tickers),
        period=period,
        interval=interval,
        prepost=prepost,
        timeout=timeout_s,
        group_by="ticker",
        threads=False,  # we parallelize at a higher level
        progress=False,
    )

    # Normalize to dict[ticker -> DataFrame]
    out: Dict[str, _pd.DataFrame] = {}

    # If only a single ticker, yfinance returns regular columns; handle both shapes.
    if isinstance(data.columns, _pd.MultiIndex):
        for sym in set(k for k, _ in data.columns):
            df = data[sym].copy()
            if not df.empty:
                out[sym] = _normalize_df(df)
    else:
        # Single ticker path: we don't know its name reliably; caller will split
        # batches so this path is rare; we leave to per-ticker fallback if needed.
        pass

    return out


def _normalize_df(df: _pd.DataFrame) -> _pd.DataFrame:
    # Ensure expected columns exist; coerce numerics; avoid SettingWithCopy
    cols_f32 = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    if cols_f32:
        df = df.copy()
        for c in cols_f32:
            df[c] = _pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "Volume" in df.columns:
        df["Volume"] = _pd.to_numeric(df["Volume"], errors="coerce").astype("int64")
    # Sort by index if not sorted
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _download_batch(
    batch: Sequence[str],
    cfg: PriceFetchConfig,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    """Download a batch of symbols; returns (data, skipped)"""
    data: Dict[str, _pd.DataFrame] = {}
    skipped: List[Tuple[str, str]] = []

    if not batch:
        return data, skipped

    # Try a few times for the whole batch
    for attempt in range(1, cfg.batch_retries + 2):
        try:
            multi = _download_multi(batch, cfg.period, cfg.interval, cfg.prepost, cfg.timeout_s)
            data.update(multi)
            # Mark any symbols from batch that didn't appear as missing
            missing = [s for s in batch if s not in multi]
            for s in missing:
                skipped.append((s, "missing_in_batch"))
            return data, skipped
        except Exception as e:
            if attempt <= cfg.batch_retries:
                _backoff_sleep(cfg.backoff_base, attempt)
            else:
                # mark whole batch as failed; try per-ticker fallback below
                _ = str(e)

    # Per-ticker fallback (slower, but more reliable)
    for sym in batch:
        try:
            df = _yf.download(
                sym, period=cfg.period, interval=cfg.interval, prepost=cfg.prepost,
                timeout=cfg.timeout_s, progress=False, threads=False
            )
            if not df.empty:
                data[sym] = _normalize_df(df)
            else:
                skipped.append((sym, "empty_after_fallback"))
        except Exception as e:
            skipped.append((sym, f"error_fallback:{type(e).__name__}"))

    return data, skipped


def _rescue_missing_in_minibatches(
    missing_syms: Sequence[str],
    cfg: PriceFetchConfig,
) -> Tuple[Dict[str, _pd.DataFrame], List[Tuple[str, str]]]:
    data: Dict[str, _pd.DataFrame] = {}
    skipped: List[Tuple[str, str]] = []
    if not missing_syms:
        return data, skipped

    for mini in _chunks(missing_syms, cfg.mini_size):
        # Retry each mini-batch with its own small attempts
        got: Dict[str, _pd.DataFrame] = {}
        mini_missing: List[str] = []
        for attempt in range(1, cfg.mini_retries + 2):
            try:
                multi = _download_multi(mini, cfg.period, cfg.interval, cfg.prepost, cfg.timeout_s)
                got.update(multi)
                mini_missing = [s for s in mini if s not in multi]
                if not mini_missing:
                    break
            except Exception:
                mini_missing = list(mini)  # treat all as missing this round
            if attempt <= cfg.mini_retries:
                _backoff_sleep(cfg.backoff_base, attempt)
        # Record results
        data.update(got)
        for s in mini_missing:
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

    chunks = list(_chunks(cfg.tickers, cfg.chunk_size))
    # Download batches in parallel
    with _fut.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        for batch in chunks:
            _log(logger, f"[prices] Downloading chunk of {len(batch)} symbols…")
        results = list(ex.map(lambda b: _download_batch(b, cfg), chunks))

    # Aggregate
    for data, sk in results:
        price_data.update(data)
        skipped.extend(sk)

    # Optionally rescue missing
    if cfg.rescue_missing:
        missing = [s for s in tickers if s not in price_data]
        if missing:
            _log(logger, f"[prices] Rescuing {len(missing)} missing symbols in mini-batches of {cfg.mini_size}…")
            got, sk2 = _rescue_missing_in_minibatches(missing, cfg)
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
        _log(logger, f"[prices] Missing in final result: {', '.join(still_missing[:8])}{'…' if len(still_missing) > 8 else ''}")

    return price_data, skipped


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
