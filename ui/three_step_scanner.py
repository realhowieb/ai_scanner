"""Premium three-step scanner flow.

This module owns the Market -> Strategy -> Profile scanner so ui.scans can
stay focused on the legacy manual scan controls.
"""

from __future__ import annotations

import time

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    from types import SimpleNamespace as _NS
    st = _NS(session_state={}, cache_data=lambda **_kw: (lambda fn: fn))  # type: ignore[assignment]
from auth.tiering import require_min_tier
from db.runs import list_runs, save_run
from ml_prebreakout import score_prebreakout
from scan.ai_confidence import score_ai_confidence
from scan.engine import run_breakout_scan, safe_call
from scan.options import (
    DEFAULT_MARKET,
    DEFAULT_PROFILE,
    DEFAULT_STRATEGY,
    build_scan_run_options,
    normalize_market,
    normalize_profile,
    normalize_strategy,
)
from scan.strategies import apply_strategy_filter
from scan.universe_selection import resolve_scan_universe
from ui.scan_providers import sanitize_universe_symbols

try:
    from ui.universe import (
        filter_universe,
        load_nasdaq_universe,
        load_sp500_universe,
    )
except ModuleNotFoundError:
    from ai_scanner.ui.universe import (  # type: ignore
        filter_universe,
        load_nasdaq_universe,
        load_sp500_universe,
    )


def _is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))


def _init_scan_session_state() -> None:
    """Ensure the 3-step scanner has stable defaults."""
    if "scan_market" not in st.session_state:
        st.session_state.scan_market = DEFAULT_MARKET
    if "scan_strategy" not in st.session_state:
        st.session_state.scan_strategy = DEFAULT_STRATEGY
    if "scan_profile" not in st.session_state:
        st.session_state.scan_profile = DEFAULT_PROFILE
    if "scan_live_mode" not in st.session_state:
        st.session_state.scan_live_mode = False
    if "scan_active_step" not in st.session_state:
        st.session_state.scan_active_step = 1


def run_scan_engine(
    market: str,
    strategy: str,
    profile: str,
    live_mode: bool = False,
) -> pd.DataFrame:
    """Run a scan based on Market / Strategy / Profile selections."""
    market = normalize_market(market)
    profile = normalize_profile(profile)
    strategy = normalize_strategy(strategy)

    tickers = resolve_scan_universe(
        market,
        st.session_state,
        is_admin=_is_admin(),
        safe_call=safe_call,
        load_sp500_universe=load_sp500_universe,
        load_nasdaq_universe=load_nasdaq_universe,
        filter_universe=filter_universe,
        sanitize_symbols=sanitize_universe_symbols,
    )

    if not tickers:
        return pd.DataFrame()

    opts = build_scan_run_options(profile, st.session_state, is_admin=_is_admin())

    snapshot_id = None
    if _is_admin():
        snapshot_id = st.session_state.get("price_snapshot_id") or st.session_state.get("snapshot_id")

    try:
        df = run_breakout_scan(
            tickers=list(tickers),
            premarket=opts.premarket,
            afterhours=opts.afterhours,
            unusual_volume=opts.unusual_volume,
            min_gap=opts.min_gap,
            min_price=opts.min_price,
            max_price=opts.max_price,
            top_n=opts.top_n,
            profile=profile or "regular",
            diagnostics=False,
            use_cache=not live_mode,
            snapshot_id=snapshot_id,
        )
    except TypeError:
        df = run_breakout_scan(
            tickers=list(tickers),
            premarket=opts.premarket,
            afterhours=opts.afterhours,
            unusual_volume=opts.unusual_volume,
            min_gap=opts.min_gap,
            min_price=opts.min_price,
            max_price=opts.max_price,
            top_n=opts.top_n,
            profile=profile or "regular",
            diagnostics=False,
            use_cache=not live_mode,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        df = apply_strategy_filter(strategy, df)
    except (KeyError, TypeError, ValueError):
        pass

    df = df.head(opts.top_n).reset_index(drop=True)

    try:
        df = score_prebreakout(df)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    df = score_ai_confidence(df)
    return df


def _step_status(active_step: int, step_num: int, done_map: dict[int, bool]) -> str:
    if active_step == step_num:
        return "active"
    if done_map.get(step_num, False):
        return "done"
    return "upcoming"


def _step_label(active_step: int, step_num: int, title: str, done_map: dict[int, bool]) -> str:
    icons = {"active": "🔵", "done": "🟢", "upcoming": "⚪"}
    status = _step_status(active_step, step_num, done_map)
    return f"{icons[status]} {step_num} {title}"


def _persist_three_step_run(df: pd.DataFrame | None, duration_sec: float) -> None:
    """Keep latest results and run history synced without breaking the UI."""
    try:
        st.session_state.results_df = df if df is not None else pd.DataFrame()
        if df is None:
            return

        filtered_count = len(df)
        results_json = df.to_json(orient="records")
        run_label = (
            f"3-Step | {st.session_state.scan_market} - "
            f"{st.session_state.scan_strategy.replace('_', ' ').title()} - "
            f"{st.session_state.scan_profile.title()}"
        )
        save_run(
            f"{run_label} | {filtered_count} results",
            results_json,
            label=run_label,
            username=st.session_state.get("username", "anonymous"),
            row_count=filtered_count,
            duration_sec=round(float(duration_sec), 2),
            is_snapshot=False,
        )

        try:
            list_runs.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass
    except (RuntimeError, TypeError, ValueError, OSError):
        pass


def render_three_step_scanner() -> None:
    """Render the premium three-step scanner layout."""
    tier = st.session_state.get("tier")
    if not require_min_tier(tier, "premium", "EZ 3-Step AI Scanner"):
        return

    _init_scan_session_state()

    done_map = {
        1: bool(st.session_state.get("scan_market")),
        2: bool(st.session_state.get("scan_strategy")),
        3: bool(st.session_state.get("scan_profile")),
    }
    active_step = int(st.session_state.get("scan_active_step", 1))

    with st.expander(
        _step_label(active_step, 1, "Select Market Universe", done_map),
        expanded=(active_step == 1),
    ):
        market_cols = st.columns(3)

        def _market_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"market_{value}"):
                st.session_state.scan_market = value
                st.session_state.scan_active_step = 2

        _market_button("SP500", "SP500", market_cols[0])
        _market_button("NASDAQ", "NASDAQ", market_cols[1])
        _market_button("Combo (SP500 + NASDAQ)", "COMBO", market_cols[2])
        st.caption(f"**Current market:** {st.session_state.scan_market}")

    with st.expander(
        _step_label(active_step, 2, "Select Strategy", done_map),
        expanded=(active_step == 2),
    ):
        strategy_cols_row1 = st.columns(3)
        strategy_cols_row2 = st.columns(3)

        def _strategy_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"strategy_{value}"):
                st.session_state.scan_strategy = value
                st.session_state.scan_active_step = 3

        _strategy_button("Gap-Up", "gap_up", strategy_cols_row1[0])
        _strategy_button("Gap-Down", "gap_down", strategy_cols_row1[1])
        _strategy_button("Most Active", "most_active", strategy_cols_row1[2])
        _strategy_button("Unusual Volume", "unusual_vol", strategy_cols_row2[0])
        _strategy_button("Momentum", "momentum", strategy_cols_row2[1])
        _strategy_button("Breakout-Only", "breakout_only", strategy_cols_row2[2])

        st.caption(
            f"**Current strategy:** "
            f"{st.session_state.scan_strategy.replace('_', ' ').title()}"
        )

    with st.expander(
        _step_label(active_step, 3, "Profile, Run Scan & View Results", done_map),
        expanded=(active_step == 3),
    ):
        profile_cols = st.columns(3)

        def _profile_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"profile_{value}"):
                st.session_state.scan_profile = value
                st.session_state.scan_active_step = 3

        _profile_button("Aggressive", "aggressive", profile_cols[0])
        _profile_button("Regular", "regular", profile_cols[1])
        _profile_button("Conservative", "conservative", profile_cols[2])

        st.caption(f"**Current profile:** {st.session_state.scan_profile.title()}")
        st.markdown("")

        run_cols = st.columns([2, 1, 1])
        run_clicked = run_cols[0].button("Run Scan", key="run_scan_button")
        st.session_state.scan_live_mode = run_cols[1].toggle(
            "Live (10s refresh)",
            value=st.session_state.scan_live_mode,
            key="live_toggle",
        )

        status_placeholder = st.empty()
        results_placeholder = st.empty()

    if run_clicked:
        started_at = time.perf_counter()
        with st.spinner("Scanning Markets"):
            df = run_scan_engine(
                market=st.session_state.scan_market,
                strategy=st.session_state.scan_strategy,
                profile=st.session_state.scan_profile,
                live_mode=st.session_state.scan_live_mode,
            )
        duration_sec = time.perf_counter() - started_at

        num_rows = 0 if df is None else len(df)
        status_placeholder.success(
            f"Scan complete in **{duration_sec:.1f}s**. Returned **{num_rows}** rows "
            f"for **{st.session_state.scan_market}** - "
            f"Strategy **{st.session_state.scan_strategy.replace('_', ' ').title()}** - "
            f"Profile **{st.session_state.scan_profile.title()}**."
        )

        if df is not None and not df.empty:
            results_placeholder.info(
                f"Results updated in **Latest scan results** panel ({len(df)} rows)."
            )
        else:
            results_placeholder.info(
                "No results matched the current filters. "
                "Try lowering the minimum gap %, price, or volume filters."
            )

        _persist_three_step_run(df, duration_sec=duration_sec)
        st.session_state.scan_active_step = 3
    else:
        status_placeholder.info(
            "Choose a **Market**, **Strategy**, and **Profile**, then click **Run Scan**."
        )


__all__ = ["render_three_step_scanner", "run_scan_engine"]
