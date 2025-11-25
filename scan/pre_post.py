# ai_scanner/scan/pre_post.py
from __future__ import annotations
import time
import pandas as pd
from ai_scanner.data.universe import load_sp600_tickers, load_sp500_tickers, fetch_and_save_sp500
from ai_scanner.data.prices import fetch_price_data_parallel, fetch_price_data_batch
from ai_scanner.data.filters import filter_us_tickers, filter_problem_tickers, filter_tickers_by_price, filter_by_dollar_volume
from ai_scanner.scan.breakout import breakout_scanner
from ai_scanner.db.runs import save_run

try:
    from ai_scanner.scan.gap_unusual import gap_unusual_volume_scanner  # type: ignore
except Exception:
    def gap_unusual_volume_scanner(*args, **kwargs):  # type: ignore
        return None

try:
    from ai_scanner.scan.spy import get_spy_history  # type: ignore
except Exception:
    def get_spy_history(*args, **kwargs):  # type: ignore
        return None

def _fetch_prices(universe, *, period: str = "60d", interval: str = "1d", use_parallel: bool = True,
                  parallel_chunk: int = 800, parallel_workers: int = 4):
    t0 = time.perf_counter()
    if use_parallel:
        try:
            # Preferred signature
            price_data, skipped = fetch_price_data_parallel(
                universe, period=period, interval=interval,
                chunk_size=parallel_chunk, max_workers=parallel_workers, logger=None
            )
        except TypeError:
            # Alternate arg name "chunks"
            price_data, skipped = fetch_price_data_parallel(
                universe, period=period, interval=interval,
                chunks=parallel_chunk, max_workers=parallel_workers
            )
    else:
        try:
            price_data, skipped = fetch_price_data_batch(universe, period=period, interval=interval, batch_size=50)
        except TypeError:
            price_data, skipped = fetch_price_data_batch(universe, period=period, interval=interval)
    t1 = time.perf_counter()
    return price_data, skipped, (t1 - t0)

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
        universe = load_sp600_tickers()
    if us_only:
        universe = filter_us_tickers(universe)
    universe = filter_problem_tickers(universe)
    if not universe:
        return pd.DataFrame(), {"params": {}, "skipped_count": 0, "downloaded_count": 0, "elapsed_s": 0.0}

    spy_df = get_spy_history("60d") if include_ta else None

    price_data, skipped, elapsed_fetch = _fetch_prices(
        universe, period="60d", interval="1d", use_parallel=use_parallel,
        parallel_chunk=parallel_chunk, parallel_workers=parallel_workers
    )

    filtered = filter_tickers_by_price(price_data, min_price, max_price)
    liquid = filter_by_dollar_volume(price_data, min_dollar_vol)
    filtered = [t for t in filtered if t in liquid]
    filtered_data = {t: price_data[t] for t in filtered if t in price_data}

    if apply_gap_filter:
        try:
            _ = gap_unusual_volume_scanner(filtered_data)
        except Exception:
            pass

    try:
        breakout_df = breakout_scanner(filtered_data, min_price, max_price, include_ta=include_ta, spy_df=spy_df)
    except Exception:
        breakout_df = pd.DataFrame()

    meta = {
        "downloaded_count": len({t for t in price_data if t in universe}),
        "skipped_count": len(skipped),
        "elapsed_s": float(elapsed_fetch),
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
            "session": session_label,
        },
    }
    return breakout_df, meta

def run_and_save(run_type: str, universe: list[str] | None, **kwargs) -> int:
    df, meta = run_scan(run_type, universe, **kwargs)
    try:
        if not df.empty and "Breakout %" in df.columns:
            to_save = df.sort_values("Breakout %", ascending=False).reset_index(drop=True)
        else:
            to_save = df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else pd.DataFrame()
        run_id = save_run(run_type, meta, to_save)
        return int(run_id)
    except Exception:
        return -1

def run_premarket_headless() -> int:
    return run_and_save("premarket", None, session_label="premarket")

def run_postmarket_headless() -> int:
    return run_and_save("postmarket", None, session_label="postmarket")

def run_sp500_headless(session_label: str = "regular") -> int:
    try:
        uni = load_sp500_tickers() or fetch_and_save_sp500()
    except Exception:
        uni = load_sp500_tickers()
    uni = uni or []
    uni = filter_problem_tickers(uni)
    return run_and_save("sp500", uni, session_label=session_label)