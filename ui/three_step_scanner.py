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
    try:
        from scan.ranking import apply_default_ranking

        df = apply_default_ranking(df)
    except Exception:
        pass
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


def nl_to_scan_settings(description: str) -> dict | None:
    """Map a natural-language scan description to {market, strategy, profile}.

    Claude is constrained to a strict JSON object over the allowed values;
    anything unparseable or out-of-vocabulary returns None (caller keeps the
    current settings and tells the user).
    """
    try:
        import json
        import re

        from ui.ai import ask_claude

        system = (
            "You map trading-scan requests to JSON with exactly these keys and "
            "allowed values.\n"
            'market: one of "SP500", "NASDAQ", "COMBO"\n'
            'strategy: one of "gap_up", "gap_down", "most_active", '
            '"unusual_vol", "momentum", "breakout_only"\n'
            'profile: one of "aggressive", "regular", "conservative"\n'
            "Choose the closest fit; COMBO when breadth/small caps are implied; "
            "conservative when safety is implied, aggressive when speed/risk is. "
            "Reply with ONLY the JSON object."
        )
        text, err = ask_claude(
            system=system,
            user=f"Request: {description.strip()[:300]}",
            max_tokens=120,
            username=(st.session_state.get("username") or None),
            feature="nl_scan_setup",
        )
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        raw = json.loads(match.group(0))
        market = str(raw.get("market", "")).upper()
        strategy = str(raw.get("strategy", "")).lower()
        profile = str(raw.get("profile", "")).lower()
        if market not in ("SP500", "NASDAQ", "COMBO"):
            return None
        if strategy not in (
            "gap_up", "gap_down", "most_active", "unusual_vol", "momentum", "breakout_only"
        ):
            return None
        if profile not in ("aggressive", "regular", "conservative"):
            return None
        return {"market": market, "strategy": strategy, "profile": profile}
    except Exception:
        return None


def render_three_step_scanner() -> None:
    """Render the premium three-step scanner layout."""
    tier = st.session_state.get("tier")
    if not require_min_tier(tier, "premium", "EZ 3-Step AI Scanner"):
        return

    _init_scan_session_state()

    # One control row replaces the old three-expander wizard: the "steps" were
    # just three choices, and the ceremony hid the Run button behind clicks.
    STRATEGIES = {
        "gap_up": "Gap-Up",
        "gap_down": "Gap-Down",
        "most_active": "Most Active",
        "unusual_vol": "Unusual Volume",
        "momentum": "Momentum",
        "breakout_only": "Breakout-Only",
    }
    MARKETS = {"SP500": "SP500", "NASDAQ": "NASDAQ", "COMBO": "Combo (SP500+NASDAQ)"}
    PROFILES = {"aggressive": "Aggressive", "regular": "Regular", "conservative": "Conservative"}

    c1, c2, c3, c4, c5 = st.columns([1, 1.3, 1, 1, 0.9])
    with c1:
        st.selectbox(
            "Market", list(MARKETS), format_func=MARKETS.get, key="scan_market"
        )
    with c2:
        st.selectbox(
            "Strategy", list(STRATEGIES), format_func=STRATEGIES.get, key="scan_strategy"
        )
    with c3:
        st.selectbox(
            "Profile", list(PROFILES), format_func=PROFILES.get, key="scan_profile"
        )
    with c4:
        st.write("")
        run_clicked = st.button("🚀 Run Scan", key="run_scan_button", width="stretch")
    with c5:
        st.write("")
        st.session_state.scan_live_mode = st.toggle(
            "Live (10s)",
            value=st.session_state.scan_live_mode,
            key="live_toggle",
        )

    # ✨ AI setup: describe the scan in plain English; Claude picks the
    # dropdowns and the scan runs on the next pass (chip-prefill pattern —
    # widget-keyed state must be set before the widgets render).
    ai1, ai2 = st.columns([4, 1])
    with ai1:
        ai_desc = st.text_input(
            "Describe your scan",
            key="ai_scan_desc",
            placeholder='✨ e.g. "safe momentum plays across the whole market"',
            label_visibility="collapsed",
        )
    with ai2:
        ai_go = st.button("✨ AI setup & run", key="ai_scan_btn", width="stretch")
    if ai_go and ai_desc.strip():
        with st.spinner("Interpreting your scan…"):
            settings = nl_to_scan_settings(ai_desc)
        if settings:
            st.session_state["scan_market"] = settings["market"]
            st.session_state["scan_strategy"] = settings["strategy"]
            st.session_state["scan_profile"] = settings["profile"]
            st.session_state["_ai_run_pending"] = True
            st.rerun()
        else:
            st.warning("Couldn't map that to scan settings — adjust the dropdowns instead.")
    if st.session_state.pop("_ai_run_pending", False):
        run_clicked = True
        st.caption(
            f"✨ AI set: **{MARKETS.get(st.session_state.scan_market)}** · "
            f"**{STRATEGIES.get(st.session_state.scan_strategy)}** · "
            f"**{PROFILES.get(st.session_state.scan_profile)}** — running…"
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
        # AI read on the results, right where the trader is looking (module is
        # button-gated + per-scan cached + entitlement/limit aware).
        try:
            from ui.ai_summary import render_ai_summary

            if df is not None and len(df):
                render_ai_summary(df)
        except Exception:
            pass

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
