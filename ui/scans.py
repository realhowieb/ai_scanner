"""Scan controls UI module.

Contains the scan buttons (SP500, NASDAQ, Combo) and the core do_scan logic
that runs the breakout scan and persists results to the runs DB.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Any
from zoneinfo import ZoneInfo
import time
import traceback

import pandas as pd
import streamlit as st
from auth.tiering import require_min_tier

from db.runs import save_run, save_daily_snapshot, list_runs
from ml_prebreakout import score_prebreakout
from scan.engine import safe_call, run_breakout_scan
from scan.execution import run_manual_scan_execution
from scan.options import (
    DEFAULT_MARKET,
    DEFAULT_PROFILE,
    DEFAULT_STRATEGY,
    apply_admin_caps,
    build_scan_run_options,
    normalize_market,
    normalize_profile,
    normalize_strategy,
)
from scan.strategies import apply_strategy_filter
from scan.universe_selection import resolve_scan_universe
from ui.scan_diagnostics import render_data_provider_diagnostics
from ui.scan_providers import (
    apply_alpaca_extended_prices,
    sanitize_universe_symbols,
)
from ui.single_ticker import handle_single_ticker_actions, render_single_ticker_panel
from ui.watchlists import handle_active_watchlist_actions, render_active_watchlist_tools


# --- Admin helpers for scan caps ---
def _is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))


def _admin_override_caps(max_nasdaq_scan: int, max_combo_scan: int, top_n: int) -> tuple[int, int, int]:
    """Admin can scan bigger universes. Keep defaults for non-admin."""
    return apply_admin_caps(max_nasdaq_scan, max_combo_scan, top_n, is_admin=_is_admin())


# Universe loaders (imported here so this module is self-contained)
try:
    from ui.universe import (
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )
except ModuleNotFoundError:
    from ai_scanner.ui.universe import (  # type: ignore
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )


# --- Three-step scanner session helpers (universe / strategy / profile) ---

def _init_scan_session_state() -> None:
    """Ensure we have default selections in session_state for the 3-step scanner layout.

    These keys are safe to add even if the legacy button-based layout is still in use.
    """
    if "scan_market" not in st.session_state:
        st.session_state.scan_market = DEFAULT_MARKET
    if "scan_strategy" not in st.session_state:
        st.session_state.scan_strategy = DEFAULT_STRATEGY
    if "scan_profile" not in st.session_state:
        st.session_state.scan_profile = DEFAULT_PROFILE
    if "scan_live_mode" not in st.session_state:
        st.session_state.scan_live_mode = False
    # NEW: initialize scan_active_step
    if "scan_active_step" not in st.session_state:
        st.session_state.scan_active_step = 1



def run_scan_engine(
    market: str,
    strategy: str,
    profile: str,
    live_mode: bool = False,
) -> pd.DataFrame:
    """Run a scan based on Market / Strategy / Profile selections.

    This function reuses the same breakout engine used by the legacy button-based
    layout, but chooses a universe and simple parameter defaults based on the
    user's selections.
    """
    # 1) Resolve universe for the selected market
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

    # Optional: price snapshot reuse (admin-only). If the engine supports it, we pass it.
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
        # Older engine signature: ignore snapshot_id
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

    # 4) Strategy-specific post-filters (operate on the breakout results)
    try:
        df = apply_strategy_filter(strategy, df)
    except (KeyError, TypeError, ValueError):
        # On any filter error, just return the unfiltered results (capped below)
        pass

    # 5) Cap to Top N and return
    df = df.head(opts.top_n).reset_index(drop=True)

    # 6) Optional: score pre-breakout probabilities
    try:
        df = score_prebreakout(df)
    except (ImportError, RuntimeError, TypeError, ValueError):
        # Never break scans if the model is missing or fails
        pass
    return df


# --- 3-step scanner clean layout (experimental) ---
def render_three_step_scanner() -> None:
    """Clean 3-step scanner layout: Market → Strategy → Profile + Run.

    This does not replace the existing render_scan_controls() yet; it can be
    called from app.py alongside or instead of the legacy layout.
    """
    # Premium-only: EZ 3-Step AI Scanner is available on Premium and Admin tiers.
    tier = st.session_state.get("tier")
    if not require_min_tier(tier, "premium", "EZ 3-Step AI Scanner"):
        return

    _init_scan_session_state()

    # Step done flags: True if selection exists in session_state
    step1_done = bool(st.session_state.get("scan_market"))
    step2_done = bool(st.session_state.get("scan_strategy"))
    step3_done = bool(st.session_state.get("scan_profile"))

    # NEW: get active step from session state
    active_step = int(st.session_state.get("scan_active_step", 1))

    # Helper to render step headers with active/completed indicator
    def _step_header(step_num: int, title: str) -> None:
        # active step = blue, completed = green, upcoming = white
        active = active_step == step_num
        done_map = {1: step1_done, 2: step2_done, 3: step3_done}
        done = done_map.get(step_num, False)

        if active:
            icon = "🔵"
        elif done:
            icon = "🟢"
        else:
            icon = "⚪️"
        st.markdown(f"### {icon} {step_num} {title}")

    # Helper to build a label string for collapsible sections
    def _step_label(step_num: int, title: str) -> str:
        active = active_step == step_num
        done_map = {1: step1_done, 2: step2_done, 3: step3_done}
        done = done_map.get(step_num, False)

        if active:
            icon = "🔵"
        elif done:
            icon = "🟢"
        else:
            icon = "⚪️"
        return f"{icon} {step_num} {title}"

    # ─────────────────────────────
    # STEP 1 — SELECT MARKET (Collapsible)
    # ─────────────────────────────
    with st.expander(
        _step_label(1, "Select Market Universe"),
        expanded=(active_step == 1),
    ):
        market_cols = st.columns(3)

        def _market_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"market_{value}"):
                st.session_state.scan_market = value
                # NEW: advance to step 2
                st.session_state.scan_active_step = 2

        _market_button("SP500", "SP500", market_cols[0])
        _market_button("NASDAQ", "NASDAQ", market_cols[1])
        _market_button("Combo (SP500 + NASDAQ)", "COMBO", market_cols[2])

        st.caption(f"**Current market:** {st.session_state.scan_market}")

    # ─────────────────────────────
    # STEP 2 — SELECT STRATEGY (Collapsible)
    # ─────────────────────────────
    with st.expander(
        _step_label(2, "Select Strategy"),
        expanded=(active_step == 2),
    ):
        strategy_cols_row1 = st.columns(3)
        strategy_cols_row2 = st.columns(3)

        def _strategy_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"strategy_{value}"):
                st.session_state.scan_strategy = value
                # NEW: advance to step 3
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

    # ─────────────────────────────
    # STEP 3 — PROFILE + RUN + RESULTS (Collapsible)
    # ─────────────────────────────
    with st.expander(
        _step_label(3, "Profile, Run Scan & View Results"),
        expanded=(active_step == 3),
    ):
        profile_cols = st.columns(3)

        def _profile_button(label: str, value: str, col) -> None:
            if col.button(label, key=f"profile_{value}"):
                st.session_state.scan_profile = value
                # NEW: set active step to 3 (remain on step 3)
                st.session_state.scan_active_step = 3

        _profile_button("Aggressive", "aggressive", profile_cols[0])
        _profile_button("Regular", "regular", profile_cols[1])
        _profile_button("Conservative", "conservative", profile_cols[2])

        st.caption(f"**Current profile:** {st.session_state.scan_profile.title()}")

        st.markdown("")

        # Run / Live controls
        run_cols = st.columns([2, 1, 1])
        run_clicked = run_cols[0].button("🚀 Run Scan", key="run_scan_button")
        st.session_state.scan_live_mode = run_cols[1].toggle(
            "Live (10s refresh)",
            value=st.session_state.scan_live_mode,
            key="live_toggle",
        )

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        results_placeholder = st.empty()

    if run_clicked:
        # Use basic Streamlit spinner instead of GIF loader
        with st.spinner("Scanning Markets 💸"):
            df = run_scan_engine(
                market=st.session_state.scan_market,
                strategy=st.session_state.scan_strategy,
                profile=st.session_state.scan_profile,
                live_mode=st.session_state.scan_live_mode,
            )

        # Optionally clear the loader when done
        # progress_placeholder.empty()

        num_rows = 0 if df is None else len(df)
        status_placeholder.success(
            f"✅ Scan complete. Returned **{num_rows}** rows "
            f"for **{st.session_state.scan_market}** • "
            f"Strategy **{st.session_state.scan_strategy.replace('_', ' ').title()}** • "
            f"Profile **{st.session_state.scan_profile.title()}**."
        )

        if df is not None and not df.empty:
            # Do not render the table here; rely on the shared Latest scan results panel.
            results_placeholder.info(
                f"Results updated in **Latest scan results** "
                f"panel ({len(df)} rows)."
            )
        else:
            results_placeholder.info(
                "No results matched the current filters. "
                "Try lowering the minimum gap %, price, or volume filters."
            )

        # ⭐ NEW: keep Latest scan results in sync and log to history
        try:
            # Update the shared results_df so the Latest scan results panel shows this run
            st.session_state.results_df = df if df is not None else pd.DataFrame()

            # Persist this run to the history DB (mirrors do_scan behaviour)
            if df is not None:
                filtered_count = len(df)
                duration_sec = 0.0  # 3‑step engine already finished; no fine‑grained timing here
                results_json = df.to_json(orient="records")
                run_label = (
                    f"3-Step | {st.session_state.scan_market} • "
                    f"{st.session_state.scan_strategy.replace('_', ' ').title()} • "
                    f"{st.session_state.scan_profile.title()}"
                )
                run_name = f"{run_label} | {filtered_count} results"
                save_run(
                    run_name,
                    results_json,
                    label=run_label,
                    username=st.session_state.get("username", "anonymous"),
                    row_count=filtered_count,
                    duration_sec=duration_sec,
                    is_snapshot=False,
                )

                # Clear cached run list so the new run appears immediately in history
                try:
                    list_runs.clear()  # type: ignore
                except AttributeError:
                    pass
        except Exception:
            # Never break the UI because of logging issues
            pass
        # NEW: keep step 3 active after scan
        st.session_state.scan_active_step = 3
    else:
        status_placeholder.info(
            "Choose a **Market**, **Strategy**, and **Profile**, then click **Run Scan**."
        )


def _banner(msg: str, level: str = "info") -> None:
    """Local banner helper so this module does not depend on app.py."""
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def render_scan_controls(
    can_scan_sp500: bool,
    can_scan_nasdaq: bool,
    max_nasdaq_scan: int,
    max_combo_scan: int,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    premarket: bool,
    afterhours: bool,
    unusual_vol: bool,
    diagnostics: bool,
    username: str,
    apply_gap_filter: bool = False,
) -> None:
    """Render scan buttons and run scans when clicked.

    This function updates `st.session_state.results_df` with the latest scan results
    and also updates the universe-related keys used elsewhere in the app.
    """

    # Admin override: allow larger universe caps + result caps inside this module
    max_nasdaq_scan, max_combo_scan, top_n = _admin_override_caps(
        int(max_nasdaq_scan), int(max_combo_scan), int(top_n)
    )

    # Admin is a ROLE, not a tier: bypass plan-based button disabling.
    is_admin = _is_admin()

    # Scan profile selector (Regular / Aggressive / Conservative)
    profile_label = st.radio(
        "Scan profile",
        ["Regular", "Aggressive", "Conservative"],
        horizontal=True,
        key="scan_profile_choice",
    )
    scan_profile = profile_label.lower().strip()
    st.caption(
        f"Current scan profile: **{profile_label}** "
        "(tunes min gap and unusual volume behavior)."
    )

    st.subheader("⚡ Quick Market Scans")
    st.caption("Run SP500, NASDAQ, and combo scans using your current filters.")

    # Buttons (hard-wired universes)
    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        run_sp500_btn = st.button(
            "Run SP500 Scan",
            width="stretch",
            disabled=(not can_scan_sp500 and not is_admin),
        )
        st.caption("Runs SP500 regardless of sidebar universe.")

    with b2:
        run_nasdaq_btn = st.button(
            "Run NASDAQ Scan",
            width="stretch",
            disabled=(not can_scan_nasdaq and not is_admin),
        )
        st.caption("Runs NASDAQ regardless of sidebar universe.")

    with b3:
        run_combo_btn = st.button(
            "Run Combo Scan (SP500+NASDAQ)",
            width="stretch",
            disabled=(not (can_scan_sp500 and can_scan_nasdaq) and not is_admin),
        )
        if is_admin:
            st.caption("Runs Combo regardless of plan caps (admin override).")
        else:
            st.caption("Pro/Premium only.")

    # --- Earnings Calendar Debug (admin-only) ---
    # One-click test to verify Yahoo Finance -> DB writes without relying on scan timing or snapshot logic.
    if bool(st.session_state.get("is_admin")):
        with st.expander("🧪 Earnings Calendar Debug", expanded=False):
            st.caption(
                "Runs a small earnings refresh and shows the returned dates (best-effort)."
            )
            if st.button(
                "Fetch earnings for AAPL / MSFT / TSLA",
                key="btn_earnings_debug",
                    width="stretch",
            ):
                try:
                    try:
                        from earnings import populate_earnings_calendar  # type: ignore
                    except Exception:
                        from db.earnings import populate_earnings_calendar  # type: ignore

                    with st.spinner("Fetching earnings via Yahoo Finance..."):
                        result = populate_earnings_calendar(["AAPL", "MSFT", "TSLA"])

                    st.success("Earnings fetch attempted.")
                    st.write(result)
                    st.caption(
                        "If all dates are None, this is usually a network/VPN/captive-portal issue."
                    )
                except Exception as e:
                    st.error(f"Earnings debug failed: {e}")

    # Watchlist actions (uses active_watchlist_tickers from session_state)
    st.markdown("### 📋 Watchlist Tools")

    (
        view_watchlist_btn,
        run_watchlist_btn,
        clear_watchlist_btn,
        add_watchlist_btn,
        remove_watchlist_btn,
        watchlist_add_symbol,
    ) = render_active_watchlist_tools()

    single_ticker, show_chart_btn, run_single_scan_btn = render_single_ticker_panel()

    # Ensure results DataFrame exists in session state
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # --- Admin role check and universe cap overrides ---
    # Admin is a ROLE, not a tier. Admins bypass plan caps in UI + scan limits.
    if is_admin:
        st.caption("🛠️ Admin override: universe caps are disabled.")

    def _manual_combo_liquidity_filter(symbols: List[str]) -> List[str]:
        min_dollar_vol = st.session_state.get("min_dollar_vol")
        if min_dollar_vol is None:
            min_dollar_vol = 0.0
        try:
            combo_liquid = apply_liquidity_filter_batch(
                symbols,
                min_price=min_price,
                min_avg_dollar_vol=min_dollar_vol,
            )
        except Exception as e:
            _banner(f"⚠️ Combo liquidity filter failed: {e}", "warning")
            return symbols
        if combo_liquid is None or len(combo_liquid) == 0:
            return symbols
        return list(combo_liquid)

    def _resolve_manual_universe(market: str) -> List[str]:
        min_dollar_vol = st.session_state.get("min_dollar_vol")
        if min_dollar_vol is None:
            min_dollar_vol = 0.0
        combo_cache_key = ("manual_combo_liquidity", float(min_price), float(min_dollar_vol))
        return resolve_scan_universe(
            market,
            st.session_state,
            is_admin=_is_admin(),
            safe_call=safe_call,
            load_sp500_universe=load_sp500_universe,
            load_nasdaq_universe=load_nasdaq_universe,
            filter_universe=filter_universe,
            sanitize_symbols=sanitize_universe_symbols,
            label_suffix=market,
            combo_universe_transform=_manual_combo_liquidity_filter if market == "COMBO" else None,
            combo_cache_key=combo_cache_key if market == "COMBO" else None,
        )

    def do_scan(
        tickers: List[str],
        label: str,
        profile_override: Optional[str] = None,
    ):
        # Final safety net: regardless of caller, sanitize tickers before scanning.
        tickers = sanitize_universe_symbols(tickers)
        def _run_scan_body():
            n_input = len(tickers)
            t0 = time.time()

            # Decide which scan profile to use for this run
            effective_profile = (profile_override or scan_profile)
            effective_profile_label = (
                profile_label if profile_override is None else profile_override.capitalize()
            )

            # --- Clean status + progress bar UI ---
            # Show the current session mode (Regular / Premarket / After-hours) and profile
            mode_bits = []
            if premarket:
                mode_bits.append("Premarket")
            if afterhours:
                mode_bits.append("After-hours")
            if not mode_bits:
                mode_bits.append("Regular")
            mode_label = ", ".join(mode_bits)

            st.markdown(
                f"**Mode:** `{mode_label}` scan  •  Profile: `{effective_profile_label}`"
            )

            # Warn on very small universes (likely stub/fallback)
            if (
                len(tickers) < 50
                and not str(label).startswith("Watchlist")
                and not str(label).startswith("Search:")
            ):
                st.caption(
                    f"⚠️ {label} universe is very small ({len(tickers)} tickers). "
                    "This usually means a fallback/stub universe is still being used."
                )

            # Progress bar + status line
            progress = st.progress(0)
            status = st.empty()

            # Rough estimate of time based on universe size (tune as needed)
            est_seconds = len(tickers) * 0.015
            status.write(
                f"🔄 Preparing scan for **{len(tickers)}** tickers… "
                f"estimated ~{est_seconds:.1f}s"
            )

            # Live progress callback from the scan engine
            # Expected shapes:
            #   progress_cb(i, n, symbol, stage="download"|"score"|"done"|..., msg=str)
            # We keep it permissive so it works with different engine signatures.
            last_ui_ts = 0.0

            def progress_cb(i: Any = None, n: Any = None, symbol: Any = None, **kw: Any) -> None:
                nonlocal last_ui_ts
                try:
                    now = time.time()
                    # Throttle UI updates to avoid slowing the scan
                    if now - last_ui_ts < 0.15:
                        return
                    last_ui_ts = now

                    # Normalize inputs
                    ii = int(i) if i is not None else None
                    nn = int(n) if n is not None else None
                    sym = (str(symbol).strip().upper() if symbol is not None else "")
                    stage = str(kw.get("stage") or kw.get("phase") or "").strip()
                    msg = str(kw.get("msg") or kw.get("message") or "").strip()

                    # Compute progress: reserve 20% for preflight, 70% for engine, 10% for post
                    if ii is not None and nn is not None and nn > 0:
                        frac = min(max(ii / float(nn), 0.0), 1.0)
                        pct = 20 + int(frac * 70)
                        progress.progress(pct)
                        if sym:
                            status.write(
                                f"🔄 {label}: {ii}/{nn} • {sym} "
                                + (f"• {stage}" if stage else "")
                                + (f" — {msg}" if msg else "")
                            )
                        else:
                            status.write(
                                f"🔄 {label}: {ii}/{nn} "
                                + (f"• {stage}" if stage else "")
                                + (f" — {msg}" if msg else "")
                            )
                    else:
                        # If engine doesn't provide counts, still show heartbeat
                        if sym or stage or msg:
                            status.write(
                                f"🔄 {label}: "
                                + (f"{sym} " if sym else "")
                                + (f"• {stage}" if stage else "")
                                + (f" — {msg}" if msg else "")
                            )
                except Exception:
                    # Never allow progress UI to break the scan
                    return

            try:
                # Phase 1: pre-flight / parameters (0–20%)
                progress.progress(20)

                # Phase 2: run engine (20–90%)
                status.write("🚀 Running breakout engine… this may take a moment.")

                # For a clean UI, always disable engine-level diagnostics here
                engine_diagnostics = False

                # Optional: price snapshot reuse (admin-only). If the engine supports it, we pass it.
                snapshot_id = None
                if _is_admin():
                    snapshot_id = st.session_state.get("price_snapshot_id") or st.session_state.get("snapshot_id")

                df = run_manual_scan_execution(
                    runner=run_breakout_scan,
                    tickers=list(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    profile=effective_profile,
                    apply_gap_filter=apply_gap_filter,
                    diagnostics=engine_diagnostics,
                    progress_cb=progress_cb,
                    snapshot_id=snapshot_id,
                    extended_price_transform=apply_alpaca_extended_prices,
                )

                progress.progress(92)

                progress.progress(97)

                filtered_count = len(df) if df is not None else 0
                if diagnostics:
                    st.caption(
                        f"📊 Filtered down from {n_input} tickers to {filtered_count} results after filters."
                    )

                status.write(f"✨ Scan complete: **{filtered_count}** results.")
                progress.progress(100)

                dt = time.time() - t0
                st.session_state.results_df = df
                # Force the main app to re-render results immediately after a scan completes.
                # This avoids cases where Streamlit doesn't refresh the results table until a manual rerun.
                st.session_state["force_results_refresh"] = True

                # If a Watchlist scan returns 0 rows, show a hint about relaxing filters.
                if (str(label).startswith("Watchlist")) and (df is None or df.empty):
                    st.caption(
                        "No watchlist members passed your current filters. "
                        "Try lowering Min Gap %, widening the price range, or disabling Unusual Volume."
                    )

                _banner(
                    f"✅ {label} scan complete in {dt:.1f}s. Returned {filtered_count} rows.",
                    "success",
                )

                # Persist this scan to the runs DB (history + optional daily snapshot)
                try:
                    results_json = df.to_json(orient="records") if df is not None else "[]"
                    row_count = filtered_count
                    run_name = f"{label} | {row_count} results | {dt:.1f}s"
                    save_run(
                        run_name,
                        results_json,
                        label=label,
                        username=username,
                        row_count=row_count,
                        duration_sec=dt,
                        is_snapshot=False,
                    )

                    # Daily snapshot (keep existing rule if you want snapshots only before noon UTC)
                    # NOTE: snapshot timing is unrelated to earnings refresh.
                    try:
                        from datetime import datetime, timezone

                        now_utc = datetime.now(timezone.utc)
                        if now_utc.hour < 12:
                            save_daily_snapshot(label, results_json, username=username)
                    except Exception:
                        # Snapshot is best-effort only
                        pass
                except Exception:
                    # Never fail the UI just because DB logging failed
                    pass

                # Clear cached run list so new scan appears immediately in history
                try:
                    list_runs.clear()  # type: ignore
                except Exception:
                    pass

            except Exception as e:
                progress.progress(100)
                status.write("❌ Scan failed.")
                _banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

        # Run the scan with our custom progress bar UI (no extra spinner wrapper)
        _run_scan_body()

    handle_active_watchlist_actions(
        view_watchlist=view_watchlist_btn,
        run_watchlist=run_watchlist_btn,
        clear_watchlist=clear_watchlist_btn,
        add_symbol=add_watchlist_btn,
        remove_symbol=remove_watchlist_btn,
        symbol=watchlist_add_symbol,
        username=username,
        do_scan=do_scan,
        banner=_banner,
    )

    if run_sp500_btn:
        do_scan(_resolve_manual_universe("SP500"), "SP500")

    if run_nasdaq_btn:
        do_scan(_resolve_manual_universe("NASDAQ"), "NASDAQ")

    if run_combo_btn:
        do_scan(_resolve_manual_universe("COMBO"), "Combo")

    handle_single_ticker_actions(
        ticker=single_ticker,
        show_chart=show_chart_btn,
        run_scan=run_single_scan_btn,
        username=username,
        do_scan=do_scan,
        banner=_banner,
    )
