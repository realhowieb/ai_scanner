from __future__ import annotations

from typing import List, Callable, TypeVar, Any
import pandas as pd
import streamlit as st
import time

T = TypeVar("T")

SCAN_PROFILES: dict[str, dict[str, float | bool]] = {
    "regular": {
        "min_gap_factor": 1.0,
        "force_unusual_volume": False,
    },
    "aggressive": {
        "min_gap_factor": 0.75,
        "force_unusual_volume": False,
    },
    "conservative": {
        "min_gap_factor": 1.25,
        "force_unusual_volume": True,
    },
}

def _apply_scan_profile(
    profile: str,
    *,
    min_gap: float,
    unusual_volume: bool,
) -> tuple[float, bool]:
    """Return adjusted (min_gap, unusual_volume) based on the chosen scan profile.

    Profiles are intentionally simple:
    - "regular": pass-through
    - "aggressive": slightly lower the min_gap, keep unusual-volume as requested
    - "conservative": slightly raise the min_gap and always enable unusual-volume
    """
    config = SCAN_PROFILES.get(profile, SCAN_PROFILES["regular"])
    factor = float(config.get("min_gap_factor", 1.0))
    force_unusual = bool(config.get("force_unusual_volume", False))

    effective_min_gap = max(0.0, min_gap * factor)
    effective_unusual_volume = unusual_volume or force_unusual
    return effective_min_gap, effective_unusual_volume

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

def _make_progress_ui(total: int, *, title: str) -> tuple[callable, callable]:
    """Create lightweight progress + heartbeat UI.

    Returns (tick, done) callables.

    - tick(i, note): updates bar/caption (throttled)
    - done(note): completes bar and final message

    Safe to call even when Streamlit UI is unavailable.
    """
    total = max(1, int(total))
    last = {"t": 0.0}

    try:
        bar = st.progress(0.0)
        line = st.empty()
        started = time.time()

        def tick(i: int, note: str = "") -> None:
            now = time.time()
            if now - last["t"] < 0.25:
                return
            last["t"] = now
            pct = min(1.0, max(0.0, float(i) / float(total)))
            try:
                bar.progress(pct)
                elapsed = now - started
                suffix = f" • {note}" if note else ""
                line.caption(f"{title}: {i:,}/{total:,} ({pct*100:.1f}%) • {elapsed:.1f}s{suffix}")
            except Exception:
                pass

        def done(note: str = "") -> None:
            try:
                bar.progress(1.0)
                suffix = f" • {note}" if note else ""
                line.caption(f"{title}: done{suffix}")
            except Exception:
                pass

        return tick, done
    except Exception:
        # Non-UI context (background runs/tests)
        def _noop(*_a: Any, **_k: Any) -> None:
            return None

        return _noop, _noop

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
    profile: str = "regular",
    diagnostics: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Cached wrapper around the real breakout scan.

    This matches the signature used in ui/scans.py, which passes a tuple of
    tickers and the filter parameters. The implementation just delegates to
    `run_breakout_scan`, which currently uses the legacy breakout engine.

    Accepts a `profile` parameter controlling scan behaviour, defaults to "regular".
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
        profile=profile,
        diagnostics=diagnostics,
        use_cache=use_cache,
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
    profile: str = "regular",
    diagnostics: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Public entry point for breakout scans.

    This function is responsible for:
    - Extending the ticker list with SPY for relative-strength style metrics.
    - Fetching OHLCV price data for the requested universe.
    - Delegating to `scan.breakout.run_breakout_scan`, which expects a
      mapping of symbol -> DataFrame and an optional SPY DataFrame.

    IMPORTANT: `price_data` must be defined on all code paths. Some scan modes
    (e.g., small SP500 scans) skip the large-scan branch; without a default
    initialization, referencing `price_data` would raise UnboundLocalError.
    """

    if diagnostics:
        try:
            st.caption(
                f"🚀 run_breakout_scan called with {len(tickers)} tickers "
                f"(use_cache={use_cache}, profile={profile!r})"
            )
        except Exception:
            # If the UI is not available (e.g., during background runs), just ignore.
            pass

    effective_min_gap, effective_unusual_volume = _apply_scan_profile(
        profile,
        min_gap=min_gap,
        unusual_volume=unusual_volume,
    )

    from . import breakout as legacy_breakout

    # Ensure SPY is included for RS calculations.
    tickers_plus_spy = sorted(set(list(tickers) + ["SPY"]))

    total_requested = len(tickers_plus_spy)
    # Large universes benefit from progress UI; small scans stay fast.
    show_progress = bool(st.session_state.get("show_scan_progress", True))
    large_scan = total_requested >= 800

    # Prefer the parallel fetcher for small scans; for large scans, do a
    # chunked batch fetch so we can show real progress in the UI.
    parallel_error = None
    batch_error = None

    # ✅ ALWAYS initialize so later `if not price_data:` checks are safe.
    price_data: dict[str, pd.DataFrame] = {}

    if large_scan and show_progress:
        tick, done = _make_progress_ui(total_requested, title="Fetching prices")
        try:
            from data.prices import fetch_price_data_batch  # type: ignore

            # Chunk size tuned to keep UI responsive without hammering Yahoo.
            chunk_size = int(st.session_state.get("price_fetch_chunk_size", 150))
            chunk_size = max(25, min(400, chunk_size))

            tick(0, note=f"chunk_size={chunk_size}")
            processed = 0
            for i in range(0, total_requested, chunk_size):
                chunk = tickers_plus_spy[i : i + chunk_size]
                # Keep the inner call isolated so one bad chunk doesn't kill the whole run.
                try:
                    chunk_data, _skipped = fetch_price_data_batch(chunk)
                    if chunk_data:
                        price_data.update(chunk_data)
                except Exception:
                    # Swallow chunk failures; we still want the scan to proceed.
                    pass

                processed = min(total_requested, i + len(chunk))
                tick(processed)

            done(note=f"symbols_with_data={len(price_data)}")

            if not price_data:
                raise RuntimeError("batch (chunked) price fetch returned no data")
        except Exception as e:
            import traceback

            batch_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ chunked fetch_price_data_batch failed: {e}")
                    st.code(batch_error, language="python")
                except Exception:
                    print("chunked fetch_price_data_batch failed:", batch_error)
            else:
                print("chunked fetch_price_data_batch failed:", batch_error)
            price_data = {}

    # --- Fast path: parallel fetch for smaller scans ---
    if not price_data:
        try:
            from data.prices import fetch_price_data_parallel  # type: ignore

            price_data, _skipped = fetch_price_data_parallel(
                tickers_plus_spy,
                use_cache=use_cache,
            )
            if not price_data:
                raise RuntimeError("parallel price fetch returned no data")
        except Exception as e:
            import traceback

            parallel_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ fetch_price_data_parallel failed: {e}")
                    st.code(parallel_error, language="python")
                except Exception:
                    print("fetch_price_data_parallel failed:", parallel_error)
            else:
                print("fetch_price_data_parallel failed:", parallel_error)
            price_data = {}

    # --- Fallback: single-shot batch fetch ---
    if not price_data:
        try:
            from data.prices import fetch_price_data_batch  # type: ignore

            price_data, _skipped = fetch_price_data_batch(tickers_plus_spy)
            if not price_data:
                raise RuntimeError("batch price fetch returned no data")
        except Exception as e:
            import traceback

            batch_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ fetch_price_data_batch failed: {e}")
                    st.code(batch_error, language="python")
                except Exception:
                    print("fetch_price_data_batch failed:", batch_error)
            else:
                print("fetch_price_data_batch failed:", batch_error)
            price_data = {}

    # Heartbeat before the breakout stage so users don't think the app froze.
    try:
        if show_progress:
            st.caption(
                f"🧠 Running breakout calculations on "
                f"{max(0, len(price_data) - (1 if 'SPY' in price_data else 0)):,}/"
                f"{max(0, total_requested - 1):,} symbols…"
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
                f"➡️ engine.run_breakout_scan: profile={profile!r}, "
                f"effective_min_gap={effective_min_gap:.4f}, "
                f"effective_unusual_volume={effective_unusual_volume}; "
                f"calling breakout with {len(price_data_no_spy)} symbols (excluding SPY)"
            )
        except Exception:
            pass

    df = legacy_breakout.run_breakout_scan(
        price_data=price_data_no_spy,
        spy_df=spy_df,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=effective_unusual_volume,
        min_gap=effective_min_gap,
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