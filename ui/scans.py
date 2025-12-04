"""Scan controls UI module.

Contains the scan buttons (SP500, NASDAQ, Combo) and the core do_scan logic
that runs the breakout scan and persists results to the runs DB.
"""

from datetime import datetime
from typing import List, Optional
import time
import traceback

import pandas as pd
import streamlit as st
import requests  # NEW

from db.runs import save_run, save_daily_snapshot, list_runs
from scan.engine import safe_call, run_breakout_scan
from db.watchlists import get_watchlist_tickers, set_watchlist_tickers

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



@st.cache_data(ttl=60)
def _get_live_quote(ticker: str) -> Optional[float]:
    """Best-effort live quote lookup for a single ticker.

    Uses yfinance if available. Returns a float price or None on failure.
    Cached briefly so repeated reruns don't hammer the API.
    """
    if not ticker:
        return None
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None

    try:
        t = yf.Ticker(ticker)
        # Prefer fast_info if available
        fast_info = getattr(t, "fast_info", None)
        last_price = None
        if fast_info is not None:
            last_price = getattr(fast_info, "last_price", None)
        if last_price is not None:
            return float(last_price)

        # Fallback: use last close from 1d history
        hist = t.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None

    return None


# --- Alpaca extended-hours helpers ---

ALPACA_MAX_SNAPSHOT_BATCH = 50  # keep batches small for free tier


def _get_alpaca_headers() -> Optional[dict]:
    """Return Alpaca auth headers from Streamlit secrets or None if not configured."""
    try:
        key = st.secrets["ALPACA_API_KEY_ID"]
        secret = st.secrets["ALPACA_API_SECRET_KEY"]
    except Exception:
        return None

    if not key or not secret:
        return None

    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _get_alpaca_extended_last_prices(symbols: List[str]) -> dict[str, float]:
    """
    Fetch extended-hours latest prices using Alpaca snapshots.

    Uses the free IEX feed; data is delayed but includes premarket / after-hours
    trades when available. If Alpaca is not configured or the request fails,
    returns an empty dict.
    """
    headers = _get_alpaca_headers()
    if not headers:
        return {}

    symbols = [str(s).strip().upper() for s in symbols if s]
    if not symbols:
        return {}

    out: dict[str, float] = {}
    base_url = "https://data.alpaca.markets/v2/stocks/snapshots"

    for i in range(0, len(symbols), ALPACA_MAX_SNAPSHOT_BATCH):
        batch = symbols[i : i + ALPACA_MAX_SNAPSHOT_BATCH]
        params = {
            "symbols": ",".join(batch),
            "feed": "iex",  # free tier feed
        }
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # If Alpaca is unavailable or misconfigured, skip this batch
            continue

        # Alpaca may return either:
        # 1) {"snapshots": {"AAPL": {...}, "TSLA": {...}}}
        # 2) {"AAPL": {...}, "TSLA": {...}}
        raw_snaps = data.get("snapshots")
        if isinstance(raw_snaps, dict) and raw_snaps:
            snapshots = raw_snaps
        elif isinstance(data, dict):
            snapshots = data
        else:
            snapshots = {}

        for sym in batch:
            snap = snapshots.get(sym)
            if not snap:
                continue

            last_trade = snap.get("latestTrade") or {}
            price = last_trade.get("p")

            # Fallback to minute bar close if needed
            if price is None:
                minute = snap.get("minuteBar") or {}
                c = minute.get("c")
                if c is not None:
                    price = c

            if price is not None:
                out[sym] = float(price)

    return out


def _apply_alpaca_extended_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Override the 'Last' (or similar) price column using Alpaca extended-hours prices.

    If Alpaca is not configured or returns no data, the original DataFrame is returned.
    """
    if df is None or df.empty:
        return df

    # Detect symbol column
    symbol_col = None
    for cand in ("Ticker", "Symbol", "symbol"):
        if cand in df.columns:
            symbol_col = cand
            break

    if symbol_col is None:
        return df

    symbols = df[symbol_col].astype(str).tolist()
    quotes = _get_alpaca_extended_last_prices(symbols)
    if not quotes:
        return df

    # Decide which price column to override; prefer 'Last'
    price_cols = ["Last", "Price", "Close"]
    target_col = next((c for c in price_cols if c in df.columns), None)
    if target_col is None:
        target_col = "Last"
        if target_col not in df.columns:
            df[target_col] = None

    def _apply_row(row):
        sym = str(row[symbol_col]).strip().upper()
        new_price = quotes.get(sym)
        return float(new_price) if new_price is not None else row[target_col]

    df[target_col] = df.apply(_apply_row, axis=1)
    return df


# --- Watchlist DataFrame builder for View Watchlist ---
def build_watchlist_df(tickers: List[str]) -> pd.DataFrame:
    """Build a rich watchlist DataFrame for the center 'View Watchlist' table."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        # If yfinance is unavailable, just return a minimal frame with symbols only.
        return pd.DataFrame(
            [
                {
                    "Symbol": str(sym).strip().upper(),
                    "Name": None,
                    "Last": None,
                    "Change": None,
                    "% Change": None,
                    "Prev Close": None,
                    "Open": None,
                    "High": None,
                    "Low": None,
                }
                for sym in tickers
            ]
        )

    rows = []
    for sym in tickers:
        sym = str(sym).strip().upper()
        last = prev_close = open_ = high = low = None
        name = None

        try:
            t = yf.Ticker(sym)
            fast = getattr(t, "fast_info", None)

            if fast is not None:
                # Try several common fast_info fields for current/regular price
                last = (
                    getattr(fast, "last_price", None)
                    or getattr(fast, "regular_market_price", None)
                )
                prev_close = getattr(fast, "previous_close", None)
                open_ = getattr(fast, "open", None)
                high = getattr(fast, "day_high", None)
                low = getattr(fast, "day_low", None)

            # Fallback with history if needed
            if last is None or prev_close is None:
                hist = t.history(period="2d")
                if not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].tolist()
                    if len(closes) >= 1 and last is None:
                        last = float(closes[-1])
                    if len(closes) >= 2 and prev_close is None:
                        prev_close = float(closes[-2])

            # Try to get a readable name
            try:
                info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {})
            except Exception:
                info = {}
            name = info.get("shortName") or info.get("longName") or ""
        except Exception:
            # Leave values as None on failure
            pass

        change = None
        change_pct = None
        if last is not None and prev_close not in (None, 0):
            change = float(last) - float(prev_close)
            change_pct = (change / float(prev_close)) * 100.0

        rows.append(
            {
                "Symbol": sym,
                "Name": name,
                "Last": last,
                "Change": change,
                "% Change": change_pct,
                "Prev Close": prev_close,
                "Open": open_,
                "High": high,
                "Low": low,
            }
        )

    return pd.DataFrame(rows)


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

    # Watchlist actions (uses active_watchlist_tickers from session_state)
    st.markdown("### 📋 Watchlist Tools")

    watchlist_tickers = st.session_state.get("active_watchlist_tickers", []) or []
    has_watchlist = isinstance(watchlist_tickers, list) and len(watchlist_tickers) > 0

    cw1, cw2, _ = st.columns([1, 1, 2])
    with cw1:
        view_watchlist_btn = st.button(
            "View Watchlist",
            key="btn_view_watchlist",
            use_container_width=True,
            disabled=not has_watchlist,
        )
    with cw2:
        run_watchlist_btn = st.button(
            "Run Watchlist Scan",
            key="btn_scan_watchlist",
            use_container_width=True,
            disabled=not has_watchlist,
        )

    st.caption("Use your active watchlist for viewing or scanning.")

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

    # Show a best-effort live quote under the search bar when a ticker is entered
    normalized_ticker = (search_ticker or "").strip().upper()
    if normalized_ticker:
        quote = _get_live_quote(normalized_ticker)
        if quote is not None:
            st.caption(f"{normalized_ticker} ~ ${quote:.2f}")
        else:
            st.caption(f"{normalized_ticker}: live quote unavailable.")

    # Ensure results DataFrame exists in session state
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    def do_scan(tickers: List[str], label: str):
        def _run_scan_body():
            n_input = len(tickers)
            t0 = time.time()
            try:
                # Debug: show universe size and a few sample tickers before calling the engine
                try:
                    st.caption(
                        f"🔍 Debug: running scan on {len(tickers)} tickers. "
                        f"Sample: {tickers[:10]}"
                    )
                except Exception:
                    # If something goes wrong with rendering the debug caption,
                    # ignore it so scans still run.
                    pass
                st.caption(f"🔎 Scanning {len(tickers)} tickers for {label}...")
                # Show the current session mode (Regular / Premarket / After-hours)
                mode_bits = []
                if premarket:
                    mode_bits.append("Premarket")
                if afterhours:
                    mode_bits.append("After-hours")
                if not mode_bits:
                    mode_bits.append("Regular")
                mode_label = ", ".join(mode_bits)
                st.markdown(f"**Mode:** `{mode_label}` scan")
                st.caption(f"Profile: {profile_label} ({scan_profile!r})")
                if (len(tickers) < 50) and not str(label).startswith("Watchlist") and not str(label).startswith("Search:"):
                    st.warning(
                        f"{label} universe is very small ({len(tickers)} tickers). "
                        "This usually means a fallback/stub universe is still being used."
                    )

                # TEMP: bypass cached_real_scan/safe_call and hit the engine directly
                df = run_breakout_scan(
                    tickers=list(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    profile=scan_profile,
                    diagnostics=diagnostics,
                )

                # Debug: show what the engine actually returned
                st.caption(f"🔥 Direct engine call returned: {0 if df is None else len(df)} rows")

                # Apply Top N cap here to avoid doing last-price overrides on hundreds of rows.
                if df is not None and not df.empty:
                    # Show a preview so we can see real rows, not just "0 results"
                    try:
                        st.dataframe(df.head(min(top_n, 20)))
                    except Exception:
                        pass

                    df = df.head(top_n).reset_index(drop=True)

                    if premarket or afterhours:
                        # Use Alpaca Market Data for extended-hours prices.
                        # If Alpaca is not configured or returns nothing, this is a no-op.
                        df = _apply_alpaca_extended_prices(df)

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

    if "view_watchlist_btn" in locals() and view_watchlist_btn:
        # Normalize and validate tickers from the active watchlist
        tickers = [
            str(t).strip().upper()
            for t in (st.session_state.get("active_watchlist_tickers") or [])
            if str(t).strip()
        ]
        if not tickers:
            _banner("Active watchlist has no tickers to view.", "warning")
        else:
            df_view = build_watchlist_df(tickers)
            st.session_state.results_df = df_view
            _banner(
                f"Showing active watchlist with {len(tickers)} tickers (with prices & daily change).",
                "info",
            )

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

        # Apply a liquidity filter to the combined universe using the same
        # min_price / min_dollar_vol filters as the main scan.
        # NOTE: apply_liquidity_filter_batch does not accept max_price, so we
        # only pass min_price and min_avg_dollar_vol here, and we call it directly
        # (not via safe_call) to avoid any time.sleep-based retry logic causing
        # issues in restricted environments.
        min_dollar_vol = st.session_state.get("min_dollar_vol")
        if min_dollar_vol is None:
            min_dollar_vol = 0.0

        try:
            combo_liquid = apply_liquidity_filter_batch(
                combo_universe,
                min_price=min_price,
                min_avg_dollar_vol=min_dollar_vol,
            )
        except Exception as e:
            _banner(
                f"⚠️ Combo liquidity filter failed: {e}",
                "warning",
            )
            combo_liquid = combo_universe

        # If the liquidity filter failed or returned nothing, fall back to the raw universe.
        if combo_liquid is None or len(combo_liquid) == 0:
            combo_liquid = combo_universe

        combo_capped = combo_liquid[: int(max_combo_scan)]

        st.session_state["sp500_universe"] = sp500
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        st.session_state["combo_capped"] = combo_capped

        do_scan(combo_capped, "Combo")

    # --- Alpaca Market Data self-test ---
    st.markdown("### 🧪 Data Provider Diagnostics")

    if st.button(
        "Test Alpaca Market Data (AAPL)",
        key="btn_test_alpaca",
        use_container_width=True,
    ):
        headers = _get_alpaca_headers()
        if not headers:
            _banner(
                "Alpaca API keys are not configured in Streamlit secrets. "
                "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .streamlit/secrets.toml.",
                "error",
            )
        else:
            with st.spinner("Contacting Alpaca for AAPL snapshot..."):
                quotes = _get_alpaca_extended_last_prices(["AAPL"])
            price = quotes.get("AAPL")
            if price is not None:
                st.success(f"✅ Alpaca Market Data OK. AAPL extended price: ${price:.2f}")
            else:
                import datetime
                now = datetime.datetime.utcnow()

                # Weekend-aware messaging
                if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    st.info("🟦 Market is closed (weekend). Extended-hours data is usually unavailable.")
                else:
                    st.info("Market may be outside active extended-hours windows.")

                _banner(
                    "Connected to Alpaca but no price was returned for AAPL. "
                    "Attempting to show raw Alpaca response for AAPL below for debugging.",
                    "warning",
                )

                # Detailed Alpaca debug request
                debug_url = "https://data.alpaca.markets/v2/stocks/snapshots"
                debug_params = {
                    "symbols": "AAPL",
                    "feed": st.secrets.get("ALPACA_FEED", "iex"),
                }
                try:
                    debug_resp = requests.get(
                        debug_url, headers=headers, params=debug_params, timeout=5
                    )
                    st.write("Alpaca debug HTTP status:", debug_resp.status_code)
                    try:
                        debug_json = debug_resp.json() or {}
                        st.write("Alpaca debug top-level keys:", list(debug_json.keys()))

                        # Handle both shapes: with/without 'snapshots' wrapper
                        raw_snaps = debug_json.get("snapshots")
                        if isinstance(raw_snaps, dict) and raw_snaps:
                            snaps = raw_snaps
                        elif isinstance(debug_json, dict):
                            snaps = debug_json
                        else:
                            snaps = {}

                        aapl_snap = snaps.get("AAPL") or {}

                        st.write(
                            "Alpaca debug 'snapshots' keys:",
                            list(snaps.keys()) if isinstance(snaps, dict) else snaps,
                        )
                        st.write(
                            "Alpaca debug AAPL snapshot keys:",
                            list(aapl_snap.keys()) if isinstance(aapl_snap, dict) else aapl_snap,
                        )
                        st.text_area(
                            "Raw Alpaca JSON (truncated)",
                            value=str(debug_json)[:1200],
                            height=200,
                        )
                    except Exception:
                        st.text_area(
                            "Raw Alpaca response (non-JSON, truncated)",
                            value=(debug_resp.text or "")[:1200],
                            height=200,
                        )
                except Exception as e:
                    st.error(f"Alpaca debug request failed: {e}")

    if run_single_search_btn:
        ticker = (search_ticker or "").strip().upper()
        if not ticker:
            _banner("Please enter a ticker symbol to search.", "warning")
        else:
            # Optionally auto-add this ticker to the active watchlist
            active_watchlist_id = st.session_state.get("active_watchlist_id")
            if active_watchlist_id is not None:
                try:
                    # Fetch current tickers for the active watchlist
                    existing = get_watchlist_tickers(active_watchlist_id, username) or []
                    norm_existing = {str(t).strip().upper() for t in existing}
                    if ticker not in norm_existing:
                        updated = sorted(norm_existing | {ticker})
                        set_watchlist_tickers(active_watchlist_id, username, list(updated))
                        st.caption(f"Added {ticker} to your active watchlist.")
                except Exception:
                    # Never break the UI if Neon/watchlists are unavailable
                    pass

            # Run the same breakout engine but on a single stock
            do_scan([ticker], f"Search: {ticker}")