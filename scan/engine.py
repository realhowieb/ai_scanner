from typing import List, Callable, TypeVar, Any
import pandas as pd
import streamlit as st

T = TypeVar("T")

def safe_call(
    fn: Callable[..., T],
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> T | None:
    """Generic wrapper to safely call a function used by the scans.

    If the underlying function raises, we log a warning to the Streamlit UI
    (when available) and return None instead of crashing the app.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        try:
            if label:
                st.warning(f"{label} failed: {e}")
            else:
                st.warning(f"{fn.__name__} failed: {e}")
        except Exception:
            # If Streamlit isn't available (e.g. during offline tests), just ignore.
            pass
        return None

@st.cache_data(show_spinner=False)
def cached_real_scan(
    tickers: tuple[str, ...],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    """Cached wrapper around the real breakout scan.

    This matches the signature used in ui/scans.py, which passes a tuple of
    tickers and the filter parameters. The implementation just delegates to
    `run_breakout_scan`, which currently uses the legacy breakout engine.
    """
    return run_breakout_scan(
        list(tickers),
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )

def run_breakout_scan(
    tickers: List[str],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    """Public entry point for breakout scans.

    This function is responsible for:
    - Extending the ticker list with SPY for relative-strength style metrics.
    - Fetching OHLCV price data for the requested universe.
    - Delegating to `scan.breakout.run_breakout_scan`, which expects a
      mapping of symbol -> DataFrame and an optional SPY DataFrame.
    """
    from . import breakout as legacy_breakout

    # Ensure SPY is included for RS calculations.
    tickers_plus_spy = sorted(set(list(tickers) + ["SPY"]))

    price_data: dict[str, pd.DataFrame] = {}
    spy_df: pd.DataFrame | None = None

    # Prefer the parallel fetcher, but fall back to the batch implementation
    # if it returns no data or raises.
    try:
        from ai_scanner.data.prices import fetch_price_data_parallel  # type: ignore

        price_data, _skipped = fetch_price_data_parallel(tickers_plus_spy)
        if not price_data:
            raise RuntimeError("parallel price fetch returned no data")
    except Exception:
        try:
            from ai_scanner.data.prices import fetch_price_data_batch  # type: ignore

            price_data, _skipped = fetch_price_data_batch(tickers_plus_spy)
            if not price_data:
                raise RuntimeError("batch price fetch returned no data")
        except Exception:
            # As a last resort, return an empty DataFrame so the caller can
            # handle the "no data" condition gracefully.
            return pd.DataFrame()

    spy_df = price_data.get("SPY")

    return legacy_breakout.run_breakout_scan(
        price_data=price_data,
        spy_df=spy_df,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )