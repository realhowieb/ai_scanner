"""Headless pre/post-market scan helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.fetch import fetch_and_save_sp500, load_sp500_tickers, load_sp600_tickers
from data.filters import (
    filter_problem_tickers,
    filter_us_tickers,
)
from db.runs import save_run
from scan.headless_common import HEADLESS_BOUNDARY_ERRORS, fetch_headless_prices, run_headless_pipeline

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

try:
    from scan.spy import get_spy_history  # type: ignore
except ImportError:
    def get_spy_history(*args, **kwargs):  # type: ignore
        return None

def _fetch_prices(universe, *, period: str = "60d", interval: str = "1d", use_parallel: bool = True,
                  parallel_chunk: int = 800, parallel_workers: int = 4):
    return fetch_headless_prices(
        universe,
        period=period,
        interval=interval,
        use_parallel=use_parallel,
        parallel_chunk=parallel_chunk,
        parallel_workers=parallel_workers,
    )

def run_scan(
    run_type: str,
    universe: list[str] | None,
    *,
    min_price: float = 5.0,
    max_price: float = 1000.0,
    include_ta: bool = False,
    min_dollar_vol: int = 2_000_000,
    use_parallel: bool = True,
    parallel_workers: int = 4,
    parallel_chunk: int = 800,
    us_only: bool = True,
    apply_gap_filter: bool = True,
    session_label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if universe is None:
        universe = _load_sp600_or_sp500()
    if us_only:
        universe = filter_us_tickers(universe)
    universe = filter_problem_tickers(universe)
    if not universe:
        return pd.DataFrame(), {"params": {}, "skipped_count": 0, "downloaded_count": 0, "elapsed_s": 0.0}

    spy_df = get_spy_history("60d") if include_ta else None

    breakout_df, meta = run_headless_pipeline(
        run_type,
        universe,
        min_price=min_price,
        max_price=max_price,
        min_dollar_vol=min_dollar_vol,
        use_parallel=use_parallel,
        parallel_workers=parallel_workers,
        parallel_chunk=parallel_chunk,
        apply_gap_filter=apply_gap_filter,
        top_n=100,
        spy_df=spy_df,
        session_label=session_label,
    )

    meta["params"] = {
        "min_price": float(min_price),
        "max_price": float(max_price),
        "include_ta": bool(include_ta),
        "min_dollar_vol": int(min_dollar_vol),
        "use_parallel": bool(use_parallel),
        "parallel_workers": int(parallel_workers),
        "parallel_chunk": int(parallel_chunk if use_parallel else 50),
        "us_only": bool(us_only),
        "apply_gap_filter": bool(apply_gap_filter),
        "session": session_label,
    }
    return breakout_df, meta

def run_and_save(run_type: str, universe: list[str] | None, **kwargs) -> int:
    df, meta = run_scan(run_type, universe, **kwargs)
    try:
        if not df.empty and "Breakout %" in df.columns:
            to_save = df.sort_values("Breakout %", ascending=False).reset_index(drop=True)
        else:
            to_save = df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else pd.DataFrame()
        results_json = to_save.to_json(orient="records", date_format="iso")
        save_run(
            name=f"{run_type} | {len(to_save)} results",
            label=run_type,
            username="scheduler",
            row_count=len(to_save),
            duration_sec=float(meta.get("elapsed_s", 0.0)),
            results_json=results_json or json.dumps([]),
            is_snapshot=False,
            allow_sqlite_fallback=False,
        )
        return 0
    except HEADLESS_BOUNDARY_ERRORS:
        return -1

def run_premarket_headless() -> int:
    return run_and_save("premarket", None, session_label="premarket")

def run_postmarket_headless() -> int:
    return run_and_save("postmarket", None, session_label="postmarket")

def run_sp500_headless(session_label: str = "regular") -> int:
    try:
        uni = _load_sp500() or fetch_and_save_sp500()
    except HEADLESS_BOUNDARY_ERRORS:
        uni = _load_sp500()
    uni = uni or []
    uni = filter_problem_tickers(uni)
    return run_and_save("sp500", uni, session_label=session_label)
