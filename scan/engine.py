from __future__ import annotations

from typing import List, Callable, TypeVar, Any
import pandas as pd
import streamlit as st

T = TypeVar("T")

def safe_call(
    fn: Callable[..., T],
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> T:
    """Debug helper: call a function and surface any exceptions.

    Instead of swallowing errors and returning None, this version logs the full
    traceback to Streamlit (when available) and then re-raises so that the app
    shows a clear error box. This is useful while we are debugging the scan engine.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        msg_label = label or fn.__name__

        # Try to surface the error inside the Streamlit app
        try:
            st.error(f"❌ {msg_label} crashed: {e}")
            st.code(tb, language="python")
        except Exception:
            # Fallback to console log if Streamlit UI is not available
            print(f"{msg_label} crashed: {e}\n{tb}")

        # Re-raise so Streamlit shows the red error box with the traceback
        raise

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
    st.caption(f"🚀 run_breakout_scan called with {len(tickers)} tickers")
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
    # if it returns no data or raises. While debugging, we explicitly surface
    # any exceptions so we can see why price_data is empty.
    parallel_error = None
    batch_error = None

    # --- Parallel fetch attempt ---
    try:
        from data.prices import fetch_price_data_parallel  # type: ignore

        price_data, _skipped = fetch_price_data_parallel(tickers_plus_spy)
        if not price_data:
            raise RuntimeError("parallel price fetch returned no data")
    except Exception as e:
        import traceback

        parallel_error = traceback.format_exc()
        try:
            st.error(f"❌ fetch_price_data_parallel failed: {e}")
            st.code(parallel_error, language="python")
        except Exception:
            print("fetch_price_data_parallel failed:", parallel_error)
        price_data = {}

    # --- Batch fetch attempt (only if parallel failed or returned nothing) ---
    if not price_data:
        try:
            from data.prices import fetch_price_data_batch  # type: ignore

            price_data, _skipped = fetch_price_data_batch(tickers_plus_spy)
            if not price_data:
                raise RuntimeError("batch price fetch returned no data")
        except Exception as e:
            import traceback

            batch_error = traceback.format_exc()
            try:
                st.error(f"❌ fetch_price_data_batch failed: {e}")
                st.code(batch_error, language="python")
            except Exception:
                print("fetch_price_data_batch failed:", batch_error)
            price_data = {}

    # If both attempts failed, bail out with an empty DataFrame but only after
    # surfacing the underlying errors above.
    if not price_data:
        try:
            st.error("❌ Price fetch failed: no data returned from either parallel or batch.")
        except Exception:
            pass
        return pd.DataFrame()

    # Diagnostics: show what the fetch layer actually returned
    if diagnostics:
        try:
            st.caption(
                f"🧩 engine.run_breakout_scan: fetched price_data for "
                f"{len(price_data)} symbols; sample keys: {list(price_data.keys())[:10]}"
            )
        except Exception:
            pass

    spy_df = price_data.get("SPY")
    if "SPY" in price_data:
        price_data_no_spy = {k: v for k, v in price_data.items() if k != "SPY"}
    else:
        price_data_no_spy = price_data

    if diagnostics:
        try:
            st.caption(
                f"➡️ engine.run_breakout_scan: calling breakout with "
                f"{len(price_data_no_spy)} symbols (excluding SPY)"
            )
        except Exception:
            pass

    df = legacy_breakout.run_breakout_scan(
        price_data=price_data_no_spy,
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

    if diagnostics:
        try:
            st.caption(
                f"⬅️ engine.run_breakout_scan: breakout returned "
                f"{0 if df is None else len(df)} rows"
            )
        except Exception:
            pass

    return df