"""Scan controls UI module.

Contains the scan buttons (SP500, NASDAQ, Combo) and the core do_scan logic
that runs the breakout scan and persists results to the runs DB.
"""

from datetime import datetime
from typing import List
import time
import traceback

import pandas as pd
import streamlit as st

from db.runs import save_run, save_daily_snapshot, list_runs
from scan.engine import safe_call, cached_real_scan, _override_last_prices

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
) -> None:
    """Render scan buttons and run scans when clicked.

    This function updates `st.session_state.results_df` with the latest scan results
    and also updates the universe-related keys used elsewhere in the app.
    """

    # Buttons (hard-wired universes)
    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        run_sp500_btn = st.button("Run SP500 Scan", use_container_width=True, disabled=not can_scan_sp500)
        st.caption("Runs SP500 regardless of sidebar universe.")

    with b2:
        run_nasdaq_btn = st.button("Run NASDAQ Scan", use_container_width=True, disabled=not can_scan_nasdaq)
        st.caption("Runs NASDAQ regardless of sidebar universe.")

    with b3:
        run_combo_btn = st.button(
            "Run Combo Scan (SP500+NASDAQ)",
            use_container_width=True,
            disabled=not (can_scan_sp500 and can_scan_nasdaq),
        )
        st.caption("Pro/Premium only.")

    # Watchlist scan button (uses active_watchlist_tickers from session_state)
    watchlist_tickers = st.session_state.get("active_watchlist_tickers", []) or []
    has_watchlist = isinstance(watchlist_tickers, list) and len(watchlist_tickers) > 0

    bw, _ = st.columns([1, 3])
    with bw:
        run_watchlist_btn = st.button(
            "Run Watchlist Scan",
            use_container_width=True,
            disabled=not has_watchlist,
        )
        st.caption("Scans only tickers from your active watchlist.")

    # --- Single-ticker search & scan ---
    st.markdown("### 🔍 Search & Scan Single Ticker")

    c1, c2 = st.columns([3, 1])
    with c1:
        search_ticker = st.text_input(
            "Ticker symbol",
            key="single_search_ticker",
            placeholder="AAPL",
            help="Type a ticker symbol (e.g., AAPL, TSLA, NVDA) and run a focused breakout scan.",
        )
    with c2:
        run_single_search_btn = st.button(
            "Search & Scan",
            key="single_search_btn",
            use_container_width=True,
        )

    # Ensure results DataFrame exists in session state
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    def do_scan(tickers: List[str], label: str):
        def _run_scan_body():
            n_input = len(tickers)
            t0 = time.time()
            try:
                st.caption(f"🔎 Scanning {len(tickers)} tickers for {label}...")
                if (len(tickers) < 50) and not str(label).startswith("Watchlist") and not str(label).startswith("Search:"):
                    st.warning(
                        f"{label} universe is very small ({len(tickers)} tickers). "
                        "This usually means a fallback/stub universe is still being used."
                    )

                df = safe_call(
                    cached_real_scan,
                    tuple(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    diagnostics=diagnostics,
                    label="cached_real_scan",
                )

                # Apply Top N cap here to avoid doing last-price overrides on hundreds of rows.
                if df is not None and not df.empty:
                    df = df.head(top_n).reset_index(drop=True)

                    if premarket or afterhours:
                        df = _override_last_prices(df)

                filtered_count = len(df) if df is not None else 0
                if diagnostics:
                    st.caption(f"📊 Filtered down from {n_input} tickers to {filtered_count} results after filters.")

                st.caption(f"✅ {label}: {len(df)} results returned from scan.")
                dt = time.time() - t0
                st.session_state.results_df = df

                # If a Watchlist scan returns 0 rows, show a hint about relaxing filters.
                if (str(label).startswith("Watchlist")) and (df is None or df.empty):
                    st.caption(
                        "No watchlist members passed your current filters. "
                        "Try lowering Min Gap %, widening the price range, or disabling Unusual Volume."
                    )

                _banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")

                # Persist this scan to the runs DB (history + optional daily snapshot)
                try:
                    results_json = df.to_json(orient="records") if df is not None else "[]"
                    row_count = len(df) if df is not None else 0
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

                    # Morning snapshot: one per day per label (approx. before noon server time)
                    try:
                        current_hour = datetime.now().hour
                        if current_hour < 12:
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
                _banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

        # Some environments (e.g., restricted sandboxes, Python 3.13 runtimes) may not
        # allow starting new threads, which Streamlit's spinner uses internally.
        # Wrap the spinner in a try/except and fall back to running without it.
        try:
            with st.spinner(f"Scanning {label}..."):
                _run_scan_body()
        except Exception:
            _run_scan_body()

    # Handle button presses
    if run_sp500_btn:
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        sp500 = filter_universe(sp500)
        st.session_state["sp500_universe"] = sp500
        do_scan(sp500, "SP500")

    if run_nasdaq_btn:
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        do_scan(nasdaq_capped, "NASDAQ")

    if run_combo_btn:
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        sp500 = filter_universe(sp500)
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        combo_universe = sp500 + nasdaq_capped
        combo_liquid = apply_liquidity_filter_batch(combo_universe)
        combo_capped = combo_liquid[: int(max_combo_scan)]

        st.session_state["sp500_universe"] = sp500
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        st.session_state["combo_capped"] = combo_capped

        do_scan(combo_capped, "Combo")

    if run_watchlist_btn:
        # Normalize and validate tickers from the active watchlist
        tickers = [
            str(t).strip().upper()
            for t in (st.session_state.get("active_watchlist_tickers") or [])
            if str(t).strip()
        ]
        if not tickers:
            _banner("Active watchlist has no tickers to scan.", "warning")
        else:
            label = f"Watchlist ({len(tickers)} tickers)"
            do_scan(tickers, label)

    if run_single_search_btn:
        ticker = (search_ticker or "").strip().upper()
        if not ticker:
            _banner("Please enter a ticker symbol to search.", "warning")
        else:
            # Run the same breakout engine but on a single stock
            do_scan([ticker], f"Search: {ticker}")