"""Scan controls UI module.

Contains the scan buttons (SP500, NASDAQ, Combo) and the core do_scan logic
that runs the breakout scan and persists results to the runs DB.
"""

import time
import traceback
from datetime import datetime, timedelta
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from db.runs import list_runs, save_daily_snapshot, save_run
from scan.engine import run_breakout_scan, safe_call
from scan.execution import run_manual_scan_execution
from scan.options import apply_admin_caps
from scan.universe_selection import resolve_scan_universe
from ui.scan_diagnostics import render_data_provider_diagnostics
from ui.scan_providers import (
    apply_alpaca_extended_prices,
    sanitize_universe_symbols,
)
from ui.single_ticker import handle_single_ticker_actions, render_single_ticker_panel
from ui.three_step_scanner import render_three_step_scanner
from ui.watchlists import handle_active_watchlist_actions


# --- Admin helpers for scan caps ---
def _is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))


def _admin_override_caps(max_nasdaq_scan: int, max_combo_scan: int, top_n: int) -> tuple[int, int, int]:
    """Admin can scan bigger universes. Keep defaults for non-admin."""
    return apply_admin_caps(max_nasdaq_scan, max_combo_scan, top_n, is_admin=_is_admin())


# Universe loaders (imported here so this module is self-contained)
try:
    from ui.universe import (
        apply_liquidity_filter_batch,
        filter_universe,
        load_nasdaq_universe,
        load_sp500_universe,
    )
except ModuleNotFoundError:
    from ai_scanner.ui.universe import (  # type: ignore
        apply_liquidity_filter_batch,
        filter_universe,
        load_nasdaq_universe,
        load_sp500_universe,
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

    # One compact header row: title + scan-profile selector side by side.
    hq1, hq2 = st.columns([2, 2])
    hq1.subheader("⚡ Quick Market Scans")
    with hq2:
        profile_label = st.radio(
            "Scan profile",
            ["Regular", "Aggressive", "Conservative"],
            horizontal=True,
            key="scan_profile_choice",
        )
    scan_profile = profile_label.lower().strip()

    # Buttons (hard-wired universes) — one shared caption instead of one each.
    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        run_sp500_btn = st.button(
            "Run SP500 Scan",
            width="stretch",
            disabled=(not can_scan_sp500 and not is_admin),
        )
    with b2:
        run_nasdaq_btn = st.button(
            "Run NASDAQ Scan",
            width="stretch",
            disabled=(not can_scan_nasdaq and not is_admin),
        )
    with b3:
        run_combo_btn = st.button(
            "Run Combo Scan (SP500+NASDAQ)" + ("" if is_admin else " · Pro+"),
            width="stretch",
            disabled=(not (can_scan_sp500 and can_scan_nasdaq) and not is_admin),
        )
    st.caption(
        f"Fixed universes, current filters, **{profile_label}** profile "
        "(profile tunes min gap and unusual-volume behavior)."
    )

    # Watchlist tool buttons are now rendered in the unified Watchlists area
    # (ui/watchlists.py); read their states here for the action handler below.
    (
        view_watchlist_btn,
        run_watchlist_btn,
        clear_watchlist_btn,
        add_watchlist_btn,
        remove_watchlist_btn,
        watchlist_add_symbol,
        watchlist_scan_all,
    ) = st.session_state.get(
        "_wl_tools_state", (False, False, False, False, False, "", False)
    )

    single_ticker, show_chart_btn, run_single_scan_btn = render_single_ticker_panel()

    # Ensure results DataFrame exists in session state
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # --- Admin role check and universe cap overrides ---
    # Admin is a ROLE, not a tier. Admins bypass plan caps in UI + scan limits.
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
        except (RuntimeError, TypeError, ValueError, OSError) as e:
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
        bypass_filters: bool = False,
    ):
        # Final safety net: regardless of caller, sanitize tickers before scanning.
        tickers = sanitize_universe_symbols(tickers)
        # "Score all" mode (watchlist): run the models on every symbol regardless
        # of the sidebar Min-Gap / price / Unusual-Volume screens, so nothing
        # silently drops to "Not in latest scan".
        eff_unusual_vol = False if bypass_filters else unusual_vol
        eff_min_gap = 0.0 if bypass_filters else min_gap
        eff_min_price = 0.0 if bypass_filters else min_price
        eff_max_price = 1_000_000.0 if bypass_filters else max_price
        eff_apply_gap_filter = False if bypass_filters else apply_gap_filter
        # Don't let Top-N cap a "score all" run below the watchlist size.
        eff_top_n = max(top_n, len(tickers)) if bypass_filters else top_n
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
                except (RuntimeError, TypeError, ValueError):
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

                # Apply the sidebar "Min Dollar Volume" as a liquidity floor so
                # in-app scans screen out illiquid micro-caps (same filter the
                # scheduled cron uses).
                sidebar_min_dollar_vol = (
                    0.0 if bypass_filters
                    else float(st.session_state.get("min_dollar_vol") or 0.0)
                )

                df = run_manual_scan_execution(
                    runner=run_breakout_scan,
                    tickers=list(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=eff_unusual_vol,
                    min_gap=eff_min_gap,
                    min_price=eff_min_price,
                    max_price=eff_max_price,
                    top_n=eff_top_n,
                    profile=effective_profile,
                    apply_gap_filter=eff_apply_gap_filter,
                    diagnostics=engine_diagnostics,
                    progress_cb=progress_cb,
                    snapshot_id=snapshot_id,
                    extended_price_transform=apply_alpaca_extended_prices,
                    min_dollar_vol=sidebar_min_dollar_vol,
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
                    except (RuntimeError, TypeError, ValueError, OSError):
                        # Snapshot is best-effort only
                        pass
                except (RuntimeError, TypeError, ValueError, OSError):
                    # Never fail the UI just because DB logging failed
                    pass

                # Clear cached run list so new scan appears immediately in history
                try:
                    list_runs.clear()  # type: ignore
                except AttributeError:
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
        scan_all=watchlist_scan_all,
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
