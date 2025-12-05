"""Scan controls UI module.

Contains the scan buttons (SP500, NASDAQ, Combo) and the core do_scan logic
that runs the breakout scan and persists results to the runs DB.
"""

from datetime import datetime
from typing import List, Optional, Callable
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


# --- Three-step scanner session helpers (universe / strategy / profile) ---

DEFAULT_MARKET = "SP500"
DEFAULT_STRATEGY = "gap_up"
DEFAULT_PROFILE = "aggressive"


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
    market = (market or "").upper().strip()
    profile = (profile or "").lower().strip()
    strategy = (strategy or "").lower().strip()

    # Prefer existing universes stored in session_state (set by legacy scans)
    sp500: Optional[List[str]] = st.session_state.get("sp500_universe")
    nasdaq: Optional[List[str]] = st.session_state.get("nasdaq_universe")
    nasdaq_capped: Optional[List[str]] = st.session_state.get("nasdaq_capped")
    combo_capped: Optional[List[str]] = st.session_state.get("combo_capped")

    # Helper to safely load + filter a universe if not already cached
    def _ensure_sp500() -> List[str]:
        nonlocal sp500
        if not sp500:
            base = safe_call(load_sp500_universe, label="SP500 universe (3-step)")
            sp500 = filter_universe(base)
            st.session_state["sp500_universe"] = sp500
        return sp500 or []

    def _ensure_nasdaq() -> List[str]:
        nonlocal nasdaq, nasdaq_capped
        if not nasdaq:
            base = safe_call(load_nasdaq_universe, label="NASDAQ universe (3-step)")
            nasdaq = filter_universe(base)
            st.session_state["nasdaq_universe"] = nasdaq
        if not nasdaq_capped:
            max_nasdaq_scan = int(st.session_state.get("max_nasdaq_scan", 2000))
            nasdaq_capped = (nasdaq or [])[:max_nasdaq_scan]
            st.session_state["nasdaq_capped"] = nasdaq_capped
        return nasdaq_capped or []

    def _ensure_combo() -> List[str]:
        nonlocal combo_capped
        if combo_capped:
            return combo_capped
        base_sp = _ensure_sp500()
        base_nq = _ensure_nasdaq()
        max_combo_scan = int(st.session_state.get("max_combo_scan", 4000))
        universe = (base_sp or []) + (base_nq or [])
        combo_capped_local = universe[:max_combo_scan]
        st.session_state["combo_capped"] = combo_capped_local
        combo_capped = combo_capped_local
        return combo_capped or []

    if market == "SP500":
        tickers = _ensure_sp500()
    elif market == "NASDAQ":
        tickers = _ensure_nasdaq()
    else:  # "COMBO" or anything else
        tickers = _ensure_combo()

    if not tickers:
        return pd.DataFrame()

    # 2) Derive basic scan parameters from profile + sidebar settings (if present)
    # Pull existing sidebar config where available, otherwise fall back to sane defaults.
    min_price = float(st.session_state.get("min_price", 1.0))
    max_price = float(st.session_state.get("max_price", 1000.0))
    top_n = int(st.session_state.get("top_n", 150))
    premarket = bool(st.session_state.get("premarket", False))
    afterhours = bool(st.session_state.get("afterhours", False))

    # Profile-based defaults
    if profile == "aggressive":
        min_gap = float(st.session_state.get("min_gap_pct_aggressive", 0.0))
        unusual_volume = bool(st.session_state.get("unusual_vol_aggressive", False))
    elif profile == "conservative":
        min_gap = float(st.session_state.get("min_gap_pct_conservative", 3.0))
        unusual_volume = bool(st.session_state.get("unusual_vol_conservative", True))
    else:  # "regular" or unknown
        min_gap = float(st.session_state.get("min_gap_pct", 1.0))
        unusual_volume = bool(st.session_state.get("unusual_vol", True))

    # 3) Run the core breakout engine (no diagnostics for clean UI)
    df = run_breakout_scan(
        tickers=list(tickers),
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        profile=profile or "regular",
        diagnostics=False,
        use_cache=not live_mode,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # 4) Strategy-specific post-filters (operate on the breakout results)
    def _strategy_gap_up(frame: pd.DataFrame) -> pd.DataFrame:
        if "GapPct" not in frame.columns:
            return frame.head(0)
        return frame[frame["GapPct"] > 0].sort_values("GapPct", ascending=False)

    def _strategy_gap_down(frame: pd.DataFrame) -> pd.DataFrame:
        if "GapPct" not in frame.columns:
            return frame.head(0)
        return frame[frame["GapPct"] < 0].sort_values("GapPct", ascending=True)

    def _strategy_most_active(frame: pd.DataFrame) -> pd.DataFrame:
        col = "DollarVol20" if "DollarVol20" in frame.columns else (
            "Volume" if "Volume" in frame.columns else None
        )
        if col is None:
            return frame.head(0)
        return frame.sort_values(col, ascending=False)

    def _strategy_unusual_vol(frame: pd.DataFrame) -> pd.DataFrame:
        if "VolRel20" not in frame.columns:
            return frame.head(0)
        return frame[frame["VolRel20"] >= 2].sort_values("VolRel20", ascending=False)

    def _strategy_momentum(frame: pd.DataFrame) -> pd.DataFrame:
        if "Trend20D%" not in frame.columns or "Trend10D%" not in frame.columns:
            return frame.head(0)
        mask = (frame["Trend20D%"] > 0) & (frame["Trend10D%"] > 0)
        return frame[mask].sort_values("Trend20D%", ascending=False)

    def _strategy_breakout_only(frame: pd.DataFrame) -> pd.DataFrame:
        if "IsBreakout" not in frame.columns:
            return frame.head(0)
        base = frame[frame["IsBreakout"] == True]
        if "BreakoutScore" in base.columns:
            return base.sort_values("BreakoutScore", ascending=False)
        return base

    strategy_map = {
        "gap_up": _strategy_gap_up,
        "gap_down": _strategy_gap_down,
        "most_active": _strategy_most_active,
        "unusual_vol": _strategy_unusual_vol,
        "momentum": _strategy_momentum,
        "breakout_only": _strategy_breakout_only,
    }

    post = strategy_map.get(strategy)
    if post is not None:
        try:
            df = post(df)
        except Exception:
            # On any filter error, just return the unfiltered results (capped below)
            pass

    # 5) Cap to Top N and return
    df = df.head(top_n).reset_index(drop=True)
    return df


# --- 3-step scanner clean layout (experimental) ---
def render_three_step_scanner() -> None:
    """Clean 3-step scanner layout: Market → Strategy → Profile + Run.

    This does not replace the existing render_scan_controls() yet; it can be
    called from app.py alongside or instead of the legacy layout.
    """
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

    # ─────────────────────────────
    # STEP 1 — SELECT MARKET
    # ─────────────────────────────
    _step_header(1, "Select Market Universe")

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

    st.markdown("---")

    # ─────────────────────────────
    # STEP 2 — SELECT STRATEGY
    # ─────────────────────────────
    _step_header(2, "Select Strategy")

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

    st.markdown("---")

    # ─────────────────────────────
    # STEP 3 — PROFILE + RUN + RESULTS
    # ─────────────────────────────
    _step_header(3, "Profile, Run Scan & View Results")

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
        with progress_placeholder.container():
            # Show custom loading GIF from assets folder
            try:
                st.image("assets/hsf_spinner_3d.gif", width=150)
                st.caption("Scanning markets… Money Moves loading 💸")
            except Exception:
                # Fallback if the GIF path is wrong or unreadable
                st.write("🚀 Running scan...")

            # Run scan without Streamlit spinner since we now show GIF
            df = run_scan_engine(
                market=st.session_state.scan_market,
                strategy=st.session_state.scan_strategy,
                profile=st.session_state.scan_profile,
                live_mode=st.session_state.scan_live_mode,
            )

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
                except Exception:
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

    # --- Helper to build a combo (SP500 + NASDAQ) capped universe with liquidity filter ---
    def _build_combo_capped(universe_label: str) -> List[str]:
        """Build a Combo universe (SP500 + capped NASDAQ) with a liquidity filter applied."""
        sp500 = safe_call(load_sp500_universe, label=f"SP500 universe ({universe_label})")
        nasdaq = safe_call(load_nasdaq_universe, label=f"NASDAQ universe ({universe_label})")
        sp500 = filter_universe(sp500)
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        combo_universe = sp500 + nasdaq_capped

        # Liquidity filter: reuse the same min_price / min_dollar_vol logic as main Combo
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
                f"⚠️ {universe_label} liquidity filter failed: {e}",
                "warning",
            )
            combo_liquid = combo_universe

        if combo_liquid is None or len(combo_liquid) == 0:
            combo_liquid = combo_universe

        combo_capped = combo_liquid[: int(max_combo_scan)]

        # Persist universes in session_state for downstream use
        st.session_state["sp500_universe"] = sp500
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        st.session_state["combo_capped"] = combo_capped

        return combo_capped

    # --- Post-filter helpers for strategy scans (operate on breakout results) ---
    def _pf_gap_up(df: pd.DataFrame) -> pd.DataFrame:
        if "GapPct" not in df.columns:
            st.caption("⚠️ Gap-Up Scan: no 'GapPct' column in results.")
            return df.head(0)
        return df[df["GapPct"] > 0].sort_values("GapPct", ascending=False)

    def _pf_gap_down(df: pd.DataFrame) -> pd.DataFrame:
        if "GapPct" not in df.columns:
            st.caption("⚠️ Gap-Down Scan: no 'GapPct' column in results.")
            return df.head(0)
        return df[df["GapPct"] < 0].sort_values("GapPct", ascending=True)

    def _pf_most_active(df: pd.DataFrame) -> pd.DataFrame:
        col = None
        if "DollarVol20" in df.columns:
            col = "DollarVol20"
        elif "Volume" in df.columns:
            col = "Volume"
        if col is None:
            st.caption("⚠️ Most Active Scan: no 'DollarVol20' or 'Volume' column in results.")
            return df.head(0)
        return df.sort_values(col, ascending=False)

    def _pf_top_gainers(df: pd.DataFrame) -> pd.DataFrame:
        col = None
        if "Trend10D%" in df.columns:
            col = "Trend10D%"
        elif "GapPct" in df.columns:
            col = "GapPct"
        if col is None:
            st.caption("⚠️ Top Gainers Scan: no 'Trend10D%' or 'GapPct' column in results.")
            return df.head(0)
        return df.sort_values(col, ascending=False)

    def _pf_unusual_volume(df: pd.DataFrame) -> pd.DataFrame:
        if "VolRel20" not in df.columns:
            st.caption("⚠️ Unusual Volume Scan: no 'VolRel20' column in results.")
            return df.head(0)
        return df[df["VolRel20"] >= 2].sort_values("VolRel20", ascending=False)

    def _pf_momentum(df: pd.DataFrame) -> pd.DataFrame:
        if "Trend20D%" not in df.columns or "Trend10D%" not in df.columns:
            st.caption("⚠️ Momentum Scan: missing 'Trend20D%' or 'Trend10D%' columns.")
            return df.head(0)
        mask = (df["Trend20D%"] > 0) & (df["Trend10D%"] > 0)
        return df[mask].sort_values("Trend20D%", ascending=False)

    def _pf_breakout_only(df: pd.DataFrame) -> pd.DataFrame:
        if "IsBreakout" not in df.columns:
            st.caption("⚠️ Breakout-Only Scan: no 'IsBreakout' column in results.")
            return df.head(0)
        base = df[df["IsBreakout"] == True]
        if "BreakoutScore" in base.columns:
            return base.sort_values("BreakoutScore", ascending=False)
        return base

    def do_scan(
        tickers: List[str],
        label: str,
        profile_override: Optional[str] = None,
        post_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
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

            try:
                # Phase 1: pre-flight / parameters (0–20%)
                progress.progress(20)

                # Phase 2: run engine (20–90%)
                status.write("🚀 Running breakout engine… this may take a moment.")

                # For a clean UI, always disable engine-level diagnostics here
                engine_diagnostics = False

                df = run_breakout_scan(
                    tickers=list(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    profile=effective_profile,
                    diagnostics=engine_diagnostics,
                )

                progress.progress(70)

                # Phase 3: apply any strategy-specific post-filter (e.g., gap-up / most active)
                if df is not None and not df.empty and post_filter is not None:
                    try:
                        df = post_filter(df)
                    except Exception as pf_exc:
                        if diagnostics:
                            st.caption(f"⚠️ Post-filter for {label} failed: {pf_exc}")

                # Phase 4: cap to Top N + extended-hours price override (70–95%)
                if df is not None and not df.empty:
                    df = df.head(top_n).reset_index(drop=True)

                    if premarket or afterhours:
                        # Use Alpaca Market Data for extended-hours prices.
                        # If Alpaca is not configured or returns nothing, this is a no-op.
                        df = _apply_alpaca_extended_prices(df)

                progress.progress(95)

                filtered_count = len(df) if df is not None else 0
                if diagnostics:
                    st.caption(
                        f"📊 Filtered down from {n_input} tickers to {filtered_count} results after filters."
                    )

                status.write(f"✨ Scan complete: **{filtered_count}** results.")
                progress.progress(100)

                dt = time.time() - t0
                st.session_state.results_df = df

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
                progress.progress(100)
                status.write("❌ Scan failed.")
                _banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

        # Run the scan with our custom progress bar UI (no extra spinner wrapper)
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
        combo_capped = _build_combo_capped("Combo")
        do_scan(combo_capped, "Combo")


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


# --- Data Provider Diagnostics (moved from render_scan_controls) ---
def render_data_provider_diagnostics() -> None:
    """Render Alpaca / data provider diagnostics.

    Intended to be called from a dedicated Debug / Data Provider tab instead
    of the main scanner layout.
    """
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
                    st.info(
                        "🟦 Market is closed (weekend). "
                        "Extended-hours data is usually unavailable."
                    )
                else:
                    st.info("Market may be outside active extended-hours windows.")

                _banner(
                    "Connected to Alpaca but no price was returned for AAPL. "
                    "Attempting to show raw Alpaca response for AAPL below for debugging.",
                    "warning",
                )

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
                            list(aapl_snap.keys())
                            if isinstance(aapl_snap, dict)
                            else aapl_snap,
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