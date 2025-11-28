from __future__ import annotations

import sys
from pathlib import Path

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

# --------------- Tiering ----------------
try:
    from auth.tiering import USERS_DB, ADMIN_USERS, get_user_tier, Tier
except Exception:
    from auth.tiering_fallback import USERS_DB, ADMIN_USERS, get_user_tier, Tier

# --------------- DB modules & UI modules ----------------
from db.users import seed_neon_users_from_local, load_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

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
@st.cache_data(ttl=60)
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
        if df is None or df.empty or "% Change" not in df:
            st.metric("Top Gainer", "—", "—")
        else:
            top = df.sort_values("% Change", ascending=False).iloc[0]
            pct = float(top["% Change"])
            st.metric("Top Gainer", top.get("Ticker", "—"), f"{pct:+.2f}%")

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


@st.cache_data(ttl=30)
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

    st.markdown(ticker_html, unsafe_allow_html=True)

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

    # -------- Sidebar Account Info --------
    st.sidebar.markdown(f"### 👤 {display_name}")
    st.sidebar.markdown(f"**Plan:** `{ 'Admin' if username in ADMIN_USERS else tier.name }`")
    st.markdown("---")

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
    ) = render_filters(tier)


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