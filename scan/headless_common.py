"""Shared helpers for headless scanner entrypoints."""

from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd

from data.filters import filter_by_dollar_volume, filter_tickers_by_price
from data.prices import fetch_price_data_batch, fetch_price_data_parallel
from scan.breakout import run_breakout_scan
from scan.gap_unusual import gap_unusual_volume_scanner

HEADLESS_BOUNDARY_ERRORS = (
    RuntimeError,
    TimeoutError,
    ConnectionError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    ImportError,
)


def fetch_headless_prices(
    universe: Iterable[str],
    *,
    period: str = "60d",
    interval: str = "1d",
    use_parallel: bool = True,
    parallel_chunk: int = 800,
    parallel_workers: int = 4,
    batch_size: int = 50,
) -> tuple[dict[str, pd.DataFrame], list[object], float]:
    """Fetch price data with backwards-compatible provider signatures."""
    symbols = list(universe)
    t0 = time.perf_counter()
    if use_parallel:
        try:
            price_data, skipped = fetch_price_data_parallel(
                symbols,
                period=period,
                interval=interval,
                chunk_size=parallel_chunk,
                max_workers=parallel_workers,
                logger=None,
            )
        except TypeError:
            price_data, skipped = fetch_price_data_parallel(
                symbols,
                period=period,
                interval=interval,
                chunks=parallel_chunk,
                max_workers=parallel_workers,
            )
    else:
        try:
            price_data, skipped = fetch_price_data_batch(
                symbols,
                period=period,
                interval=interval,
                batch_size=batch_size,
            )
        except TypeError:
            price_data, skipped = fetch_price_data_batch(symbols, period=period, interval=interval)
    elapsed = time.perf_counter() - t0
    return dict(price_data or {}), list(skipped or []), elapsed


def build_filtered_price_data(
    price_data: dict[str, pd.DataFrame],
    *,
    min_price: float,
    max_price: float,
    min_dollar_vol: int,
) -> dict[str, pd.DataFrame]:
    """Apply the shared price and liquidity filters for headless scans.

    Both filters take (tickers, lookup, thresholds) — the symbols to check and a
    lookup of their frames. Passing the price_data dict as `tickers` and the
    threshold as the `lookup` made every per-symbol lookup fail, so the filters
    returned nothing and headless (premarket/postmarket) scans always produced 0
    results.
    """
    symbols = list(price_data.keys())
    priced = set(filter_tickers_by_price(symbols, price_data, min_price, max_price))
    liquid = set(filter_by_dollar_volume(symbols, price_data, min_dollar_vol))
    return {t: price_data[t] for t in symbols if t in priced and t in liquid}


def maybe_run_gap_filter(price_data: dict[str, pd.DataFrame], *, enabled: bool) -> None:
    """Run the legacy gap scanner for side effects without blocking headless runs."""
    if not enabled:
        return
    try:
        gap_unusual_volume_scanner(price_data)
    except HEADLESS_BOUNDARY_ERRORS:
        return


def run_headless_breakout(
    price_data: dict[str, pd.DataFrame],
    *,
    spy_df: pd.DataFrame | None,
    session_label: str | None,
    min_price: float,
    max_price: float,
    top_n: int,
) -> pd.DataFrame:
    """Run the legacy breakout scanner with headless-safe fallback behavior."""
    try:
        df = run_breakout_scan(
            price_data=price_data,
            spy_df=spy_df,
            premarket=session_label == "premarket",
            afterhours=session_label == "postmarket",
            unusual_volume=False,
            min_gap=0.0,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
            diagnostics=False,
        )
    except HEADLESS_BOUNDARY_ERRORS:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def run_headless_pipeline(
    run_type: str,
    universe: Iterable[str],
    *,
    min_price: float,
    max_price: float,
    min_dollar_vol: int,
    use_parallel: bool,
    parallel_workers: int,
    parallel_chunk: int,
    apply_gap_filter: bool,
    top_n: int,
    spy_df: pd.DataFrame | None = None,
    session_label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch, filter, scan, and return results plus run metadata."""
    symbols = list(universe)
    price_data, skipped, elapsed_fetch = fetch_headless_prices(
        symbols,
        period="60d",
        interval="1d",
        use_parallel=use_parallel,
        parallel_chunk=parallel_chunk,
        parallel_workers=parallel_workers,
    )
    filtered_data = build_filtered_price_data(
        price_data,
        min_price=min_price,
        max_price=max_price,
        min_dollar_vol=min_dollar_vol,
    )
    maybe_run_gap_filter(filtered_data, enabled=apply_gap_filter)
    breakout_df = run_headless_breakout(
        filtered_data,
        spy_df=spy_df,
        session_label=session_label or run_type,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
    )
    meta = {
        "downloaded_count": len({ticker for ticker in price_data if ticker in set(symbols)}),
        "skipped_count": len(skipped),
        "elapsed_s": float(elapsed_fetch),
    }
    return breakout_df, meta
