from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project base directory importable
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --------------- Charts import fallback ----------------
try:
    from charts import render_chart_for_ticker
except Exception:
    try:
        from ui.charts import render_chart_for_ticker  # type: ignore
    except Exception:

        def render_chart_for_ticker(ticker: str, *args, **kwargs):
            st.info("Chart module not available.")

# --------------- AI Notes fallback ----------------
try:
    from ai_notes import generate_ai_note
except Exception:
    try:
        from ui.ai_notes import generate_ai_note  # type: ignore
    except Exception:

        def generate_ai_note(row: pd.Series) -> str:
            return "AI notes module missing."


# --------------- Page config ----------------

st.set_page_config(
    page_title="Breakout Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- Market session helper (US/Eastern) ----------------

def get_market_session(now: datetime | None = None) -> str:
    """
    Return one of 'premarket', 'regular', 'afterhours', or 'closed'
    based on the current US/Eastern time.
    """
    tz = ZoneInfo("US/Eastern")
    if now is None:
        now = datetime.now(tz)
    else:
        now = now.astimezone(tz)

    # Weekend: Saturday (5) / Sunday (6)
    if now.weekday() >= 5:
        return "closed"

    t = now.time()

    # Premarket roughly 4:00–9:30 ET
    if time(4, 0) <= t < time(9, 30):
        return "premarket"

    # Regular session 9:30–16:00 ET
    if time(9, 30) <= t < time(16, 0):
        return "regular"

    # After-hours 16:00–20:00 ET
    if time(16, 0) <= t < time(20, 0):
        return "afterhours"

    # Everything else is treated as closed
    return "closed"

# --------------- Tiering ----------------
try:
    from auth.tiering import USERS_DB, ADMIN_USERS, get_user_tier, Tier
except Exception:
    from auth.tiering_fallback import USERS_DB, ADMIN_USERS, get_user_tier, Tier

# --------------- DB modules & UI modules ----------------
from db.users import seed_neon_users_from_local, load_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

# User settings (per-user defaults) – optional Neon-backed feature
try:
    from db.user_settings import get_user_settings, upsert_user_settings
except Exception:
    get_user_settings = None
    upsert_user_settings = None

from ui.auth import auth_ui
from ui.pricing import pricing_sidebar
from ui.admin_users import render_admin_users_panel
from ui.history import render_history_expander
from ui.results import render_results, get_results_df
from ui.scans import render_scan_controls
from ui.universe_panel import render_universe_panel, init_universe_state
from ui.filters import render_filters
from ui.db_status import render_db_status_badge
from auth.tiering_utils import derive_tier_flags
from ui.header import render_header
from ui.footer import render_footer
from ui.watchlists import render_watchlists_panel


# --------------- Cached Market Snapshot Helper ----------------
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_index_snapshot(symbol: str):
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if hist is None or hist.empty:
            return None, None
        closes = hist["Close"].tolist()
        last = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else last
        return last, prev
    except Exception:
        return None, None


def render_market_snapshot():
    st.subheader("Today's Market Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    def _idx(col, label, symbol):
        with col:
            last, prev = _fetch_index_snapshot(symbol)
            if last is None:
                st.metric(label, "—", "—")
                return
            pct = ((last - prev) / prev) * 100 if prev else 0
            st.metric(label, f"{last:.2f}", f"{pct:+.2f}%")

    _idx(c1, "S&P 500 (SPY)", "SPY")
    _idx(c2, "NASDAQ 100 (QQQ)", "QQQ")

    # Top Gainer / Most Active from last scan
    df = get_results_df()
    with c3:
        if df is None or df.empty:
            st.metric("Top Gainer", "—", "—")
        else:
            # Try to detect a suitable percent-change / gap / gain metric column
            # by scanning column names case-insensitively.
            lower_map = {col: col.lower() for col in df.columns}
            metric_col = None

            # 1) Prefer columns whose names clearly indicate change/gain
            for col, lower in lower_map.items():
                if any(
                    key in lower
                    for key in [
                        "% change",
                        "change %",
                        "pct_change",
                        "pct change",
                        "change",
                        "gain",
                        "gainer",
                        "chg",
                    ]
                ):
                    metric_col = col
                    break

            # 2) If none found, look for anything with "gap" in the name
            if metric_col is None:
                for col, lower in lower_map.items():
                    if "gap" in lower:
                        metric_col = col
                        break

            # 3) As a fallback, choose the first reasonable numeric column
            if metric_col is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                blacklist = {
                    "volume",
                    "vol",
                    "avg_volume",
                    "average_volume",
                    "dollar_volume",
                    "dollar vol",
                    "market_cap",
                    "market cap",
                }
                for col in numeric_cols:
                    if lower_map[col] not in blacklist:
                        metric_col = col
                        break

            if metric_col is None:
                # Truly no usable metric column found; fall back gracefully
                st.metric("Top Gainer", "—", "—")
                st.caption("Top Gainer: no suitable change/gain metric found in results.")
                # If you want one-time debug, you can uncomment:
                # st.write("Top Gainer DEBUG columns:", list(df.columns))
            else:
                # Sort by the detected gain metric descending and take the top row
                top = df.sort_values(metric_col, ascending=False).iloc[0]
                ticker = (
                    top.get("Ticker")
                    or top.get("symbol")
                    or top.get("Symbol")
                    or str(top.name)
                )

                try:
                    raw_val = float(top[metric_col])
                except Exception:
                    raw_val = 0.0

                # If the column already looks like a percent you still show it as %
                change_str = f"{raw_val:+.2f}%"
                st.metric("Top Gainer", ticker, change_str)
    with c4:
        if df is None or df.empty:
            st.metric("Most Active", "—", "—")
        else:
            vol_col = "DollarVol20" if "DollarVol20" in df else "Volume" if "Volume" in df else None
            if not vol_col:
                st.metric("Most Active", "—", "—")
            else:
                row = df.sort_values(vol_col, ascending=False).iloc[0]
                val = float(row[vol_col]) / 1_000_000
                suffix = "M" if vol_col == "DollarVol20" else "M sh"
                st.metric("Most Active", row.get("Ticker", "—"), f"{val:.1f}{suffix}")


# --------------- Price Ticker ----------------
TICKER_STRIP = ["SPY", "QQQ", "IWM", "DIA", "VIX", "AAPL", "MSFT", "NVDA", "TSLA"]


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_ticker_quotes(symbols: list[str]):
    """Return list of dicts: [{'symbol', 'last', 'change_pct'}, ...]."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []

    results = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            closes = hist["Close"].tolist()
            last = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else last
            change_pct = ((last - prev) / prev) * 100.0 if prev not in (0, None) else 0.0
            results.append({"symbol": sym, "last": last, "change_pct": change_pct})
        except Exception:
            continue
    return results


def render_price_ticker():
    """Render a scrolling ticker just under the main header."""
    data = _fetch_ticker_quotes(TICKER_STRIP)
    if not data:
        return  # nothing to show

    # Build HTML for ticker items
    items_html = ""
    for row in data:
        sym = row["symbol"]
        last = row["last"]
        cpct = row["change_pct"]
        color = "#2ecc71" if cpct >= 0 else "#e74c3c"
        items_html += (
            f"<span class='ticker__item'>"
            f"{sym}&nbsp;{last:.2f}&nbsp;"
            f"<span style='color:{color};'>{cpct:+.2f}%</span>"
            f"</span>"
        )

    ticker_html = f"""
    <style>
    .ticker-wrap {{
        width: 100%;
        overflow: hidden;
        background: #0f1117;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
        padding: 0;
        margin-top: 0;
        margin-bottom: 0;
        line-height: 1.0;
    }}
    .ticker {{
        display: inline-block;
        white-space: nowrap;
        animation: ticker-move 35s linear infinite;
        font-size: 14px;
        will-change: transform;
    }}
    .ticker__item {{
        display: inline-block;
        margin-right: 40px;
        color: #e0e0e0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    @keyframes ticker-move {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-50%, 0, 0); }}
    }}
    </style>
    <div class="ticker-wrap">
      <div class="ticker">
        {items_html} {items_html}  <!-- duplicate for seamless loop -->
      </div>
    </div>
    """

    st.markdown(
        f"<div style='margin:0;padding:0;'>{ticker_html}</div>",
        unsafe_allow_html=True,
    )

# --------------- User Settings Footer (Save Defaults) ----------------
def render_user_settings_footer(
    username: str | None,
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    diagnostics: bool | None = None,
) -> None:
    """
    Render a small footer in the sidebar that shows user/settings status
    and exposes a 'Save as my default settings' button when storage is wired.
    """
    # Normalize / persist username in session for downstream use
    session_username = (
        username
        or st.session_state.get("username")
        or st.session_state.get("user")
    )
    if session_username:
        st.session_state["username"] = session_username

    # Diagnostics caption so we always know why the button is or isn't visible
    st.sidebar.caption(
        f"User settings status — user: {session_username or 'not set'}, "
        f"storage: {'available' if callable(upsert_user_settings) else 'unavailable'}"
    )

    # Pull current filter values, preferring explicit args over session_state
    universe_val = st.session_state.get("universe")
    min_gap_val = st.session_state.get("min_gap")
    top_n_val = st.session_state.get("top_n")
    max_nasdaq_scan_val = st.session_state.get("max_nasdaq_scan")
    max_combo_scan_val = st.session_state.get("max_combo_scan")
    premarket_val = st.session_state.get("premarket")
    afterhours_val = st.session_state.get("afterhours")
    unusual_vol_val = st.session_state.get("unusual_vol")

    min_price_val = min_price if min_price is not None else st.session_state.get("min_price")
    max_price_val = max_price if max_price is not None else st.session_state.get("max_price")
    min_dollar_vol_val = st.session_state.get("min_dollar_vol")
    include_ta_val = st.session_state.get("include_ta")
    apply_gap_filter_val = st.session_state.get("apply_gap_filter")
    show_diag_val = diagnostics if diagnostics is not None else st.session_state.get("show_diagnostics_ui")

    if session_username and callable(upsert_user_settings):
        st.sidebar.caption(f"Signed in as: {session_username}")
        if st.sidebar.button("💾 Save as my default settings"):
            try:
                upsert_user_settings(
                    user_id=session_username,
                    universe=universe_val,
                    min_price=min_price_val,
                    max_price=max_price_val,
                    min_dollar_vol=min_dollar_vol_val,
                    include_ta=include_ta_val,
                    apply_gap_filter=apply_gap_filter_val,
                    show_diagnostics_ui=show_diag_val,
                    min_gap=min_gap_val,
                    top_n=top_n_val,
                    max_nasdaq_scan=max_nasdaq_scan_val,
                    max_combo_scan=max_combo_scan_val,
                    premarket=premarket_val,
                    afterhours=afterhours_val,
                    unusual_vol=unusual_vol_val,
                )
                st.sidebar.success("Default scan settings saved for your account.")
            except Exception as e:
                st.sidebar.error(f"Failed to save default settings: {e}")
    elif session_username:
        st.sidebar.caption(f"Signed in as: {session_username}")
        st.sidebar.caption("User settings storage is not available (Neon-only feature).")
    else:
        st.sidebar.caption(
            "No username set — defaults cannot be saved between sessions. "
            "If you want per-user defaults, ensure auth sets st.session_state['username']."
        )

# ============================================================
#                       MAIN UI
# ============================================================

def main():
    # Show ticker above the header (layout option B)
    render_price_ticker()
    render_header()

    # -------- AUTH FIRST --------
    authed, username, display_name = auth_ui()
    if not authed:
        # Not logged in: show only the login card (auth_ui handles it)
        st.stop()

    # At this point, auth_ui has decided we're logged in.
    # The login form might still be in the DOM for this rerun, so hide it with CSS.
    st.markdown(
        """
        <style>
        /* Hide the streamlit-authenticator login form once authenticated */
        div[data-testid="stForm"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Also check the raw authenticator state
    is_authed_state = st.session_state.get("authentication_status") is True

    # -------- ONE-TIME SPLASH SCREEN AFTER SUCCESSFUL LOGIN --------
    if is_authed_state and not st.session_state.get("just_logged_in"):
        st.session_state["just_logged_in"] = True

        st.markdown(
            """
            <div style='text-align:center; padding-top:70px; padding-bottom:40px;'>
                <h1 style='font-size: 42px; margin-bottom: 0;'>📈 Breakout Stock Scanner</h1>
                <p style='color:#cccccc; font-size:18px; margin-top:10px;'>Loading your dashboard…</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        # Stop here for this rerun so the *next* rerun loads the full dashboard
        st.stop()


    # -------- Load Users + Tier --------
    users_map = load_users()
    tier = get_user_tier(username, users_map)
    flags = derive_tier_flags(tier)

    # -------- Load Saved User Settings (if available) --------
    if username and callable(get_user_settings):
        try:
            saved = get_user_settings(username)
        except Exception:
            saved = None

        if saved:
            # Only seed session_state keys that are not already present,
            # so we don't override anything the user has already changed this session.
            if "min_price" not in st.session_state and saved.get("min_price") is not None:
                st.session_state["min_price"] = float(saved["min_price"])
            if "max_price" not in st.session_state and saved.get("max_price") is not None:
                st.session_state["max_price"] = float(saved["max_price"])
            if "min_dollar_vol" not in st.session_state and saved.get("min_dollar_vol") is not None:
                st.session_state["min_dollar_vol"] = float(saved["min_dollar_vol"])
            if "include_ta" not in st.session_state and saved.get("include_ta") is not None:
                st.session_state["include_ta"] = bool(saved["include_ta"])
            if "apply_gap_filter" not in st.session_state and saved.get("apply_gap_filter") is not None:
                st.session_state["apply_gap_filter"] = bool(saved["apply_gap_filter"])
            if "show_diagnostics_ui" not in st.session_state and saved.get("show_diagnostics_ui") is not None:
                st.session_state["show_diagnostics_ui"] = bool(saved["show_diagnostics_ui"])

    # -------- Sidebar Account Info --------
    st.sidebar.markdown(f"### 👤 {display_name}")
    st.sidebar.markdown(f"**Plan:** `{ 'Admin' if username in ADMIN_USERS else tier.name }`")
    #st.markdown("---")

    # -------- DB Status --------
    db_status = render_db_status_badge()

    # -------- Filters --------
    (
        min_gap,
        min_price,
        max_price,
        top_n,
        max_nasdaq_scan,
        max_combo_scan,
        premarket,
        afterhours,
        unusual_vol,
        diagnostics,
        min_dollar_vol,
        include_ta,
        apply_gap_filter,
    ) = render_filters(tier)

    # -------- Market session gating for extended-hours toggles --------
    session = get_market_session()
    st.sidebar.caption(f"Market session (US/Eastern): {session.capitalize()}")

    # Premarket toggle only takes effect during premarket session
    if premarket and session != "premarket":
        premarket = False
        st.session_state["premarket"] = False
        st.sidebar.info(
            "Premarket scans only run between 4:00–9:30am ET on trading days. "
            "The toggle has been reset to Regular mode."
        )

    # After-hours toggle only takes effect during after-hours session
    if afterhours and session != "afterhours":
        afterhours = False
        st.session_state["afterhours"] = False
        st.sidebar.info(
            "After-hours scans only run between 4:00–8:00pm ET on trading days. "
            "The toggle has been reset to Regular mode."
        )

    # -------- User Settings Footer (Save Defaults) --------
    render_user_settings_footer(
        username,
        min_price=float(min_price) if min_price is not None else None,
        max_price=float(max_price) if max_price is not None else None,
        diagnostics=bool(diagnostics) if diagnostics is not None else None,
    )


    # -------- Watchlists --------
    watch_id, watch_tickers = render_watchlists_panel(username)
    st.session_state["active_watchlist_id"] = watch_id
    st.session_state["active_watchlist_tickers"] = watch_tickers

    # -------- Market Snapshot --------
    try:
        with st.spinner("Loading market snapshot..."):
            render_market_snapshot()
    except Exception:
        st.warning("Market snapshot unavailable")

    # -------- Scan Controls --------
    render_scan_controls(
        can_scan_sp500=flags["can_scan_sp500"],
        can_scan_nasdaq=flags["can_scan_nasdaq"],
        max_nasdaq_scan=int(max_nasdaq_scan),
        max_combo_scan=int(max_combo_scan),
        min_gap=float(min_gap),
        min_price=float(min_price),
        max_price=float(max_price),
        top_n=int(top_n),
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        unusual_vol=bool(unusual_vol),
        diagnostics=bool(diagnostics),
        username=username,
    )

    st.markdown("---")

    # -------- Results --------
    df = get_results_df()
    rows = 0 if df is None else len(df)
    with st.expander(f"📊 Latest scan results ({rows} rows)", expanded=rows > 0):
        render_results(df, flags["can_export_csv"], flags["can_ai_notes"], render_chart_for_ticker, generate_ai_note)

    st.markdown("---")

    # -------- Scan History --------
    render_history_expander(db_status)

    # -------- Universe State --------
    init_universe_state()
    render_universe_panel()

    # -------- Admin Panel --------
    render_admin_users_panel(username, ADMIN_USERS, db_status)

    # -------- Pricing (Load Last) --------
    pricing_sidebar(username, users_map)

    # -------- Footer --------
    render_footer()

if __name__ == "__main__":
    main()