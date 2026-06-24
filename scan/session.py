from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from data.fetch import fetch_and_save_sp500, load_sp500_tickers, load_sp600_tickers
from data.filters import filter_problem_tickers, filter_us_tickers
from scan.headless_common import HEADLESS_BOUNDARY_ERRORS, run_headless_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _read_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_sp500() -> list[str]:
    try:
        symbols = load_sp500_tickers()
    except HEADLESS_BOUNDARY_ERRORS:
        symbols = []
    return list(symbols or _read_symbols(ROOT / "sp500.txt"))


def _load_sp600_or_sp500() -> list[str]:
    try:
        symbols = load_sp600_tickers()
    except HEADLESS_BOUNDARY_ERRORS:
        symbols = []
    return list(symbols or _load_sp500())

def headless_scan_run(
    run_type: str,
    universe: Iterable[str],
    params: dict,
    spy_df: pd.DataFrame | None = None,
    parallel: bool = True,
) -> tuple[int, pd.DataFrame, dict]:
    """
    Returns (run_id placeholder -1, results_df, meta). Caller should save via db.runs.save_run().
    """
    min_price = float(params.get("min_price", 5.0))
    max_price = float(params.get("max_price", 1000.0))
    min_dv = int(params.get("min_dollar_vol", 2_000_000))
    workers = int(params.get("parallel_workers", 4))
    chunk = int(params.get("parallel_chunk", 800))
    apply_gap = bool(params.get("apply_gap_filter", True))

    df, meta = run_headless_pipeline(
        run_type,
        universe,
        min_price=min_price,
        max_price=max_price,
        min_dollar_vol=min_dv,
        use_parallel=parallel,
        parallel_workers=workers,
        parallel_chunk=chunk,
        apply_gap_filter=apply_gap,
        top_n=int(params.get("top_n", 100)),
        spy_df=spy_df,
        session_label=run_type,
    )

    meta["params"] = params
    return -1, (df if df is not None else pd.DataFrame()), meta


def premarket_headless(params: dict) -> tuple[int, pd.DataFrame, dict]:
    uni = params.get("universe") or _load_sp600_or_sp500()
    if params.get("us_only", True):
        uni = filter_us_tickers(uni)
    uni = filter_problem_tickers(uni)
    return headless_scan_run("premarket", uni, params, spy_df=None, parallel=params.get("use_parallel", True))


def postmarket_headless(params: dict) -> tuple[int, pd.DataFrame, dict]:
    uni = params.get("universe") or _load_sp600_or_sp500()
    if params.get("us_only", True):
        uni = filter_us_tickers(uni)
    uni = filter_problem_tickers(uni)
    return headless_scan_run("postmarket", uni, params, spy_df=None, parallel=params.get("use_parallel", True))


def sp500_headless(params: dict) -> tuple[int, pd.DataFrame, dict]:
    try:
        uni = _load_sp500() or fetch_and_save_sp500()
    except HEADLESS_BOUNDARY_ERRORS:
        uni = _load_sp500()
    uni = filter_problem_tickers(uni)
    return headless_scan_run("sp500", uni, params, spy_df=None, parallel=params.get("use_parallel", True))
