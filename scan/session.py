from __future__ import annotations
import time
import pandas as pd
from typing import Iterable, Tuple, Dict
from data.prices import fetch_price_data_parallel, fetch_price_data_batch
from data.filters import filter_tickers_by_price, filter_by_dollar_volume, filter_problem_tickers, filter_us_tickers
from data.fetch import load_sp500_tickers, load_sp600_tickers, fetch_and_save_sp500
from scan.breakout import breakout_scanner
from scan.gap_unusual import gap_unusual_volume_scanner
from scan.indicators import rs_20d_vs_spy  # only if needed by breakout_scanner

def headless_scan_run(run_type: str,
                      universe: Iterable[str],
                      params: dict,
                      spy_df: pd.DataFrame | None = None,
                      parallel: bool = True) -> Tuple[int, pd.DataFrame, dict]:
    """
    Returns (run_id placeholder -1, results_df, meta). Caller should save via db.runs.save_run().
    """
    min_price    = float(params.get("min_price", 5.0))
    max_price    = float(params.get("max_price", 1000.0))
    include_ta   = bool(params.get("include_ta", False))
    min_dv       = int(params.get("min_dollar_vol", 2_000_000))
    workers      = int(params.get("parallel_workers", 4))
    chunk        = int(params.get("parallel_chunk", 800))
    apply_gap    = bool(params.get("apply_gap_filter", True))

    t0 = time.perf_counter()
    if parallel:
        price_data, skipped = fetch_price_data_parallel(universe, period="60d", interval="1d",
                                                        chunk_size=chunk, max_workers=workers, logger=None)
    else:
        price_data, skipped = fetch_price_data_batch(universe, period="60d", interval="1d", batch_size=50)
    t1 = time.perf_counter()

    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    liquid   = filter_by_dollar_volume(price_data, min_dv)
    filtered_data = {t: price_data[t] for t in filtered if t in liquid}

    if apply_gap:
        try: _ = gap_unusual_volume_scanner(filtered_data)
        except Exception: pass

    df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)

    meta = {
        "downloaded_count": len(price_data),
        "skipped_count": len(skipped),
        "elapsed_s": (t1 - t0),
        "params": params,
    }
    return -1, (df if df is not None else pd.DataFrame()), meta

def premarket_headless(params: dict) -> Tuple[int, pd.DataFrame, dict]:
    uni = params.get("universe") or load_sp600_tickers()
    if params.get("us_only", True): uni = filter_us_tickers(uni)
    uni = filter_problem_tickers(uni)
    return headless_scan_run("premarket", uni, params, spy_df=None, parallel=params.get("use_parallel", True))

def postmarket_headless(params: dict) -> Tuple[int, pd.DataFrame, dict]:
    uni = params.get("universe") or load_sp600_tickers()
    if params.get("us_only", True): uni = filter_us_tickers(uni)
    uni = filter_problem_tickers(uni)
    return headless_scan_run("postmarket", uni, params, spy_df=None, parallel=params.get("use_parallel", True))

def sp500_headless(params: dict) -> Tuple[int, pd.DataFrame, dict]:
    try:
        uni = load_sp500_tickers() or fetch_and_save_sp500()
    except Exception:
        uni = load_sp500_tickers()
    uni = filter_problem_tickers(uni)
    return headless_scan_run("sp500", uni, params, spy_df=None, parallel=params.get("use_parallel", True))