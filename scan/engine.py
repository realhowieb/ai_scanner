from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    from types import SimpleNamespace as _NS
    st = _NS(  # type: ignore[assignment]
        cache_data=lambda **_kw: (lambda fn: fn),
        session_state={},
        caption=lambda *_a, **_kw: None,
        warning=lambda *_a, **_kw: None,
        progress=lambda *_a, **_kw: _NS(empty=lambda: None),
        empty=lambda: _NS(empty=lambda: None),
    )

from config import (
    DB_CACHE_MAX_AGE_MINUTES,
    DB_CACHE_MIN_TICKERS,
    PRICE_FETCH_CHUNK_MAX,
    PRICE_FETCH_CHUNK_MIN,
    PRICE_FETCH_CHUNK_SIZE,
    PROGRESS_UI_THROTTLE_SEC,
)

T = TypeVar("T")

_STREAMLIT_UI_ERRORS = (RuntimeError, TypeError, ValueError, AttributeError)
_ENGINE_BOUNDARY_ERRORS = (
    RuntimeError,
    TimeoutError,
    ConnectionError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
)

# Optional snapshot hooks used by ui/scans.py.
# We keep these as simple callables to avoid hard dependencies in the engine.

SnapshotLoader = Callable[[str], Dict[str, pd.DataFrame]]
SnapshotSaver = Callable[[str, Dict[str, pd.DataFrame]], None]


def _diag_caption(diagnostics: bool, message: str) -> None:
    """Best-effort diagnostic caption that is safe outside Streamlit."""
    if not diagnostics:
        return
    try:
        st.caption(message)
    except _STREAMLIT_UI_ERRORS:
        print(message)


def _diag_exception(diagnostics: bool, context: str, exc: BaseException) -> None:
    """Surface non-fatal scan issues when diagnostics are enabled."""
    _diag_caption(diagnostics, f"⚠️ {context}: {exc}")


# --- DB cache helpers for admin full-universe scans ---
def _db_cache_allowed_for_run(*, tickers_count: int, diagnostics: bool) -> bool:
    """Admin-only DB cache for full-universe scans.

    This is intentionally conservative:
    - only enabled for admin contexts
    - only enabled for large universes (to prevent adding DB overhead to small scans)
    - safe if DB module/functions are missing
    """
    if not _is_admin_context():
        return False
    # Only worth it for big runs
    if int(tickers_count) < DB_CACHE_MIN_TICKERS:
        return False
    try:
        ss = getattr(st, "session_state", None)
        if ss and ss.get("disable_db_price_cache", False):
            _diag_caption(diagnostics, "🧱 DB price cache disabled by session_state (disable_db_price_cache=True)")
            return False
    except _ENGINE_BOUNDARY_ERRORS as e:
        _diag_exception(diagnostics, "DB price cache admin/session check skipped", e)
    return True


def _db_load_price_cache(
    symbols: list[str],
    *,
    max_age_minutes: int = 30,
    diagnostics: bool = False,
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Best-effort: load cached OHLCV for symbols from DB.

    Returns (cached_data, stale_symbols).

    NOTE: We support multiple function names in db/prices.py so this engine stays
    backward-compatible while you iterate.
    """
    cached: dict[str, pd.DataFrame] = {}
    stale: set[str] = set(symbols)

    try:
        from db import prices as db_prices  # type: ignore

        # Preferred API
        fn = getattr(db_prices, "get_price_data_snapshot", None)
        if callable(fn):
            out = fn(symbols, max_age_minutes=max_age_minutes)
            if isinstance(out, tuple) and len(out) == 2:
                data, stale_list = out
                if isinstance(data, dict):
                    cached = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
                if isinstance(stale_list, (list, set, tuple)):
                    stale = set(str(x) for x in stale_list)
                else:
                    stale = set(symbols) - set(cached.keys())
            else:
                # If function returns only a dict
                if isinstance(out, dict):
                    cached = {k: v for k, v in out.items() if isinstance(v, pd.DataFrame)}
                    stale = set(symbols) - set(cached.keys())

            if diagnostics:
                _diag_caption(
                    diagnostics,
                    f"🧱 DB cache hit: {len(cached):,}/{len(symbols):,} symbols (stale={len(stale):,}, max_age={max_age_minutes}m)",
                )
            return cached, stale

        # Back-compat API: only returns dict
        fn2 = getattr(db_prices, "get_price_data", None)
        if callable(fn2):
            out2 = fn2(symbols)
            if isinstance(out2, dict):
                cached = {k: v for k, v in out2.items() if isinstance(v, pd.DataFrame)}
                stale = set(symbols) - set(cached.keys())
            if diagnostics:
                _diag_caption(diagnostics, f"🧱 DB cache hit: {len(cached):,}/{len(symbols):,} symbols (no staleness API)")
            return cached, stale

    except _ENGINE_BOUNDARY_ERRORS as e:
        _diag_exception(diagnostics, "DB cache load skipped", e)

    return cached, stale


def _db_save_price_cache(
    data: dict[str, pd.DataFrame],
    *,
    diagnostics: bool = False,
) -> None:
    """Best-effort: persist OHLCV data to DB for reuse."""
    if not data:
        return
    try:
        from db import prices as db_prices  # type: ignore

        fn = getattr(db_prices, "upsert_price_data_snapshot", None)
        if callable(fn):
            fn(data)
            _diag_caption(diagnostics, f"💾 DB cache saved: {len(data):,} symbols")
            return

        fn2 = getattr(db_prices, "upsert_price_data", None)
        if callable(fn2):
            fn2(data)
            _diag_caption(diagnostics, f"💾 DB cache saved: {len(data):,} symbols")
            return

    except _ENGINE_BOUNDARY_ERRORS as e:
        _diag_exception(diagnostics, "DB cache save skipped", e)


def _is_admin_context() -> bool:
    """Admin role check: session state is the fast path; DB is the authoritative gate.

    For cache-write operations (the only place this is called) we verify against
    the DB so a tampered session_state cannot unlock admin-only paths.
    """
    try:
        ss = getattr(st, "session_state", None)
        if not ss:
            return False

        # Fast-reject: if session says not admin, skip DB call entirely.
        session_claims_admin = bool(ss.get("is_admin", False))
        role = (ss.get("role") or ss.get("user_role") or ss.get("account_role") or "").strip().lower()
        if not session_claims_admin and role != "admin":
            return False

        # Session claims admin — verify against DB before granting privileged access.
        username = (ss.get("username") or ss.get("user") or "").strip().lower()
        if not username:
            return False
        try:
            from db.users import is_admin_from_db
        except ImportError:
            return session_claims_admin
        return is_admin_from_db(username)
    except _ENGINE_BOUNDARY_ERRORS:
        return False

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
        except _STREAMLIT_UI_ERRORS:
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
            if now - last["t"] < PROGRESS_UI_THROTTLE_SEC:
                return
            last["t"] = now
            pct = min(1.0, max(0.0, float(i) / float(total)))
            try:
                bar.progress(pct)
                elapsed = now - started
                suffix = f" • {note}" if note else ""
                line.caption(f"{title}: {i:,}/{total:,} ({pct*100:.1f}%) • {elapsed:.1f}s{suffix}")
            except _STREAMLIT_UI_ERRORS:
                pass

        def done(note: str = "") -> None:
            try:
                bar.progress(1.0)
                suffix = f" • {note}" if note else ""
                line.caption(f"{title}: done{suffix}")
            except _STREAMLIT_UI_ERRORS:
                pass

        return tick, done
    except _STREAMLIT_UI_ERRORS:
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
    snapshot_id: str | None = None,
    snapshot_loader: SnapshotLoader | None = None,
    snapshot_saver: SnapshotSaver | None = None,
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
        snapshot_id=snapshot_id,
        snapshot_loader=snapshot_loader,
        snapshot_saver=snapshot_saver,
    )

def run_breakout_scan(
    tickers: List[str],
    *,
    min_dollar_vol: float = 0.0,
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
    snapshot_id: str | None = None,
    snapshot_loader: SnapshotLoader | None = None,
    snapshot_saver: SnapshotSaver | None = None,
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
        _diag_caption(
            diagnostics,
            f"🚀 run_breakout_scan called with {len(tickers)} tickers "
            f"(use_cache={use_cache}, profile={profile!r})",
        )

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
    try:
        show_progress = bool(st.session_state.get("show_scan_progress", True))
    except _STREAMLIT_UI_ERRORS:
        show_progress = True

    # ✅ ALWAYS initialize so later `if not price_data:` checks are safe.
    price_data: dict[str, pd.DataFrame] = {}

    # --- DB-first pricing path (admin only, full-universe) ---
    # This is independent of the file-based snapshot_id hooks.
    use_db_cache = _db_cache_allowed_for_run(tickers_count=total_requested, diagnostics=diagnostics)
    if use_db_cache:
        cached, stale = _db_load_price_cache(
            tickers_plus_spy,
            max_age_minutes=int(getattr(st.session_state, "get", lambda *_a, **_k: 30)("db_price_cache_max_age_minutes", 30)),
            diagnostics=diagnostics,
        )
        if cached:
            price_data.update(cached)

    # --- Optional snapshot hooks (admin only) ---
    # These are secondary: they can override/augment DB cache if you pass a snapshot_id.
    allow_snapshot = bool(snapshot_id) and _is_admin_context()
    if allow_snapshot and snapshot_id and snapshot_loader:
        try:
            loaded = snapshot_loader(str(snapshot_id))
            if isinstance(loaded, dict) and loaded:
                price_data.update({k: v for k, v in loaded.items() if isinstance(v, pd.DataFrame)})
                if diagnostics:
                    _diag_caption(diagnostics, f"📦 Loaded pricing snapshot: {snapshot_id} ({len(loaded)} symbols)")
        except _ENGINE_BOUNDARY_ERRORS as e:
            # Snapshot load must never break scans.
            _diag_exception(diagnostics, f"Snapshot load skipped for {snapshot_id}", e)

    # Only fetch what we're missing (especially important for admin snapshot runs)
    tickers_to_fetch = [t for t in tickers_plus_spy if t not in price_data]
    # If we loaded a DB cache and it supports staleness, prefer the stale set.
    # This prevents us from skipping symbols that exist in DB but are too old.
    if use_db_cache:
        try:
            # 'stale' is defined only when use_db_cache=True; keep this guarded.
            tickers_to_fetch = [t for t in tickers_to_fetch if t in stale or t not in price_data]
        except _ENGINE_BOUNDARY_ERRORS as e:
            _diag_exception(diagnostics, "DB cache stale-symbol filtering skipped", e)
    fetch_total = len(tickers_to_fetch)
    large_scan = fetch_total >= DB_CACHE_MIN_TICKERS

    if not tickers_to_fetch:
        # All requested symbols were satisfied from the snapshot.
        if diagnostics:
            try:
                st.caption("✅ All pricing satisfied from snapshot; no Yahoo fetch needed")
            except _STREAMLIT_UI_ERRORS:
                pass

    # Prefer the parallel fetcher for small scans; for large scans, do a
    # chunked batch fetch so we can show real progress in the UI.
    parallel_error = None
    batch_error = None
    provider_skipped: list[tuple[str, str]] = []

    if large_scan and show_progress:
        tick, done = _make_progress_ui(fetch_total, title="Fetching prices")
        try:
            from data.prices import fetch_price_data_batch  # type: ignore

            # Chunk size tuned to keep UI responsive without hammering Yahoo.
            chunk_size = int(st.session_state.get("price_fetch_chunk_size", PRICE_FETCH_CHUNK_SIZE))
            chunk_size = max(PRICE_FETCH_CHUNK_MIN, min(PRICE_FETCH_CHUNK_MAX, chunk_size))

            tick(0, note=f"chunk_size={chunk_size}")
            processed = 0
            for i in range(0, fetch_total, chunk_size):
                chunk = tickers_to_fetch[i : i + chunk_size]
                # Keep the inner call isolated so one bad chunk doesn't kill the whole run.
                try:
                    chunk_data, _skipped = fetch_price_data_batch(chunk)
                    provider_skipped.extend(_skipped or [])
                    if chunk_data:
                        price_data.update(chunk_data)
                except _ENGINE_BOUNDARY_ERRORS as e:
                    # Swallow chunk failures; we still want the scan to proceed.
                    _diag_exception(
                        diagnostics,
                        f"Price fetch chunk skipped ({chunk[0] if chunk else '?'}-{chunk[-1] if chunk else '?'})",
                        e,
                    )

                processed = min(fetch_total, i + len(chunk))
                tick(processed, note=f"{chunk[-1]}" if chunk else "")

            done(note=f"symbols_with_data={len(price_data)}")

            if not price_data:
                raise RuntimeError("batch (chunked) price fetch returned no data")
        except _ENGINE_BOUNDARY_ERRORS as e:
            import traceback

            batch_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ chunked fetch_price_data_batch failed: {e}")
                    st.code(batch_error, language="python")
                except _STREAMLIT_UI_ERRORS:
                    print("chunked fetch_price_data_batch failed:", batch_error)
            else:
                print("chunked fetch_price_data_batch failed:", batch_error)
            _log_scan_error(e, context="chunked_fetch_price_data_batch", tickers=tickers_to_fetch)
            price_data = {}

    # --- Fast path: parallel fetch for smaller scans ---
    if not price_data and tickers_to_fetch:
        try:
            from data.prices import fetch_price_data_parallel  # type: ignore

            price_data_new, _skipped = fetch_price_data_parallel(
                tickers_to_fetch,
                use_cache=use_cache,
            )
            provider_skipped.extend(_skipped or [])
            if price_data_new:
                price_data.update(price_data_new)
            if not price_data:
                raise RuntimeError("parallel price fetch returned no data")
        except _ENGINE_BOUNDARY_ERRORS as e:
            import traceback

            parallel_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ fetch_price_data_parallel failed: {e}")
                    st.code(parallel_error, language="python")
                except _STREAMLIT_UI_ERRORS:
                    print("fetch_price_data_parallel failed:", parallel_error)
            else:
                print("fetch_price_data_parallel failed:", parallel_error)
            price_data = {}

    # --- Fallback: single-shot batch fetch ---
    if not price_data and tickers_to_fetch:
        try:
            from data.prices import fetch_price_data_batch  # type: ignore

            price_data_new, _skipped = fetch_price_data_batch(tickers_to_fetch)
            provider_skipped.extend(_skipped or [])
            if price_data_new:
                price_data.update(price_data_new)
            if not price_data:
                raise RuntimeError("batch price fetch returned no data")
        except _ENGINE_BOUNDARY_ERRORS as e:
            import traceback

            batch_error = traceback.format_exc()
            if diagnostics:
                try:
                    st.error(f"❌ fetch_price_data_batch failed: {e}")
                    st.code(batch_error, language="python")
                except _STREAMLIT_UI_ERRORS:
                    print("fetch_price_data_batch failed:", batch_error)
            else:
                print("fetch_price_data_batch failed:", batch_error)
            price_data = {}

    if tickers_to_fetch and provider_skipped:
        try:
            from data.provider_diagnostics import format_skip_examples, summarize_provider_skips

            fetched_count = len([t for t in tickers_to_fetch if t in price_data])
            provider_summary = summarize_provider_skips(
                requested=len(tickers_to_fetch),
                returned=fetched_count,
                skipped=provider_skipped,
            )
            if diagnostics:
                if provider_summary.severe:
                    st.warning(provider_summary.message)
                else:
                    st.caption(f"⚠️ {provider_summary.message}")
                examples = format_skip_examples(provider_skipped, limit=6)
                if examples:
                    st.caption(f"Provider skip sample: {examples}")
            elif provider_summary.severe:
                print(f"Price provider warning: {provider_summary.message}")
                try:
                    from telemetry import log_provider_warning
                    username = None
                    try:
                        ss = getattr(st, "session_state", None)
                        username = (ss.get("username") or None) if ss else None
                    except _ENGINE_BOUNDARY_ERRORS:
                        pass
                    log_provider_warning("yfinance", provider_summary.message, username=username)
                except _ENGINE_BOUNDARY_ERRORS:
                    pass
        except _ENGINE_BOUNDARY_ERRORS as e:
            _diag_exception(diagnostics, "Provider diagnostics skipped", e)

    # If we fetched prices and a snapshot_saver is provided, persist the snapshot.
    if allow_snapshot and snapshot_id and snapshot_saver and price_data:
        try:
            snapshot_saver(str(snapshot_id), price_data)
            if diagnostics:
                _diag_caption(diagnostics, f"💾 Saved pricing snapshot: {snapshot_id} ({len(price_data)} symbols)")
        except _ENGINE_BOUNDARY_ERRORS as e:
            # Snapshot persistence should never break a scan.
            _diag_exception(diagnostics, f"Snapshot save skipped for {snapshot_id}", e)

    # Persist DB cache for admin full-universe runs so subsequent runs reuse data.
    # This must never block scans.
    if use_db_cache and price_data:
        try:
            _db_save_price_cache(price_data, diagnostics=diagnostics)
        except _ENGINE_BOUNDARY_ERRORS as e:
            _diag_exception(diagnostics, "DB cache save wrapper skipped", e)

    # Heartbeat before the breakout stage so users don't think the app froze.
    try:
        if show_progress:
            st.caption(
                f"🧠 Running breakout calculations on "
                f"{max(0, len(price_data) - (1 if 'SPY' in price_data else 0)):,}/"
                f"{max(0, total_requested - 1):,} symbols…"
            )
    except _STREAMLIT_UI_ERRORS:
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
        except _STREAMLIT_UI_ERRORS:
            pass

    try:
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
            min_dollar_vol=min_dollar_vol,
        )
    except _ENGINE_BOUNDARY_ERRORS as e:
        _log_scan_error(e, context="legacy_breakout.run_breakout_scan", tickers=tickers)
        raise

    if diagnostics:
        try:
            st.caption(
                f"⬅️ engine.run_breakout_scan: breakout returned "
                f"{0 if df is None else len(df)} rows"
            )
        except _STREAMLIT_UI_ERRORS:
            pass

    return df


def _log_scan_error(exc: BaseException, *, context: str, tickers: list) -> None:
    """Best-effort telemetry call — never raises."""
    try:
        from telemetry import log_scan_error
        username = None
        try:
            ss = getattr(st, "session_state", None)
            username = (ss.get("username") or None) if ss else None
        except _ENGINE_BOUNDARY_ERRORS:
            pass
        log_scan_error(exc, context=context, username=username, ticker_count=len(tickers))
    except _ENGINE_BOUNDARY_ERRORS:
        pass
