from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, time, date, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from types import SimpleNamespace

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
# --------------- Tiering ----------------
try:
    from auth.tiering import (
        USERS_DB,
        ADMIN_USERS,
        get_user_tier,
        Tier,
        require_min_tier,
        has_min_tier,
    )
except Exception:
    from auth.tiering_fallback import USERS_DB, ADMIN_USERS, get_user_tier, Tier

# --- Normalize ADMIN_USERS to be username-only (no implicit premium/admin coercion) ---
# Some legacy configs treat admin users as premium implicitly; we want DB tier to win.
try:
    if isinstance(ADMIN_USERS, (list, set, tuple)):
        ADMIN_USERS = {str(u).strip().lower() for u in ADMIN_USERS}
    elif isinstance(ADMIN_USERS, dict):
        ADMIN_USERS = {str(u).strip().lower() for u in ADMIN_USERS.keys()}
except Exception:
    pass

# --------------- DB modules & UI modules ----------------
from db.users import seed_neon_users_from_local, load_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

# User settings (per-user defaults) – optional Neon-backed feature
try:
    from db.user_settings import get_user_settings, upsert_user_settings
except Exception:
    get_user_settings = None
    upsert_user_settings = None

from ui.auth import auth_ui, logout_and_reset_session
from ui.pricing import pricing_sidebar
from ui.admin_users import render_admin_users_panel
from ui.history import render_history_expander
from ui.results import render_results, get_results_df
from ui.scans import render_scan_controls, render_three_step_scanner
from ui.universe_panel import render_universe_panel, init_universe_state
from ui.filters import render_filters
from ui.db_status import render_db_status_badge
## from auth.tiering_utils import derive_tier_flags
from ui.header import render_header
from ui.prebreakout_tab import render_prebreakout_tab
from ui.footer import render_footer
from ui.watchlists import render_watchlists_panel

from market_data import get_latest_quotes


# --------------- DB connection helper (app.py only) ----------------
# We intentionally do NOT depend on db/core.py (it may not exist in this repo).
# Prefer psycopg (v3). Fall back to psycopg2 if present.

def _get_database_url() -> str | None:
    try:
        # Env vars first
        import os

        v = os.getenv("DATABASE_URL") or os.getenv("database_url")
        if v:
            return str(v).strip()
    except Exception:
        pass

    try:
        # Streamlit secrets (case-sensitive)
        if hasattr(st, "secrets"):
            v = st.secrets.get("DATABASE_URL") or st.secrets.get("database_url")
            if v:
                return str(v).strip()
    except Exception:
        pass

    return None


# --------------- DB connection helper (app.py only) ----------------
# We intentionally do NOT depend on db/core.py (it may not exist in this repo).
# Prefer psycopg (v3). Fall back to psycopg2 if present.

def _get_db_conn_for_app():
    """Return a DB connection using DATABASE_URL/database_url.

    Uses psycopg (v3) if available; falls back to psycopg2.
    Caller is responsible for closing.
    """
    dsn = _get_database_url()
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    # Prefer psycopg (v3)
    try:
        import psycopg  # type: ignore

        return psycopg.connect(dsn)
    except Exception:
        pass

    # Fallback psycopg2
    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(dsn)
    except Exception as e:
        raise RuntimeError(f"No DB driver available (install psycopg or psycopg2). Last error: {e}")

# --------------- Earnings Calendar Enrichment (app.py only) ----------------
EARN_COL_DAYS = "📅 Earnings in X days"

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_earnings_dates_for_symbols(symbols: tuple[str, ...]) -> dict[str, date | None]:
    """Fetch earnings_date for a batch of symbols in one DB query.

    Returns dict: {"AAPL": date|None, ...}
    """
    if not symbols:
        return {}

    # Normalize
    syms = tuple(sorted({str(s).strip().upper() for s in symbols if str(s).strip()}))
    if not syms:
        return {}

    conn = _get_db_conn_for_app()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT upper(symbol) AS symbol, earnings_date
                FROM earnings_calendar
                WHERE upper(symbol) = ANY(%s)
                """,
                (list(syms),),
            )
            rows = cur.fetchall() or []

        out: dict[str, date | None] = {}
        for sym, ed in rows:
            if sym:
                out[str(sym).strip().upper()] = ed
        # Ensure all requested symbols exist in dict (even if missing in DB)
        for s in syms:
            out.setdefault(s, None)
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


def add_earnings_days_column(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add a single computed column: 📅 Earnings in X days.

    Uses ONE DB query for all symbols in the table.
    """
    if results_df is None or results_df.empty:
        return results_df

    if "Ticker" not in results_df.columns:
        return results_df

    tickers = (
        results_df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )
    tickers = [t for t in tickers if t]
    if not tickers:
        out = results_df.copy()
        if EARN_COL_DAYS not in out.columns:
            out[EARN_COL_DAYS] = None
        return out

    earn_map = _fetch_earnings_dates_for_symbols(tuple(tickers))
    today = date.today()

    out = results_df.copy()

    def _days_until(sym: object) -> int | None:
        s = str(sym).strip().upper()
        ed = earn_map.get(s)
        if not ed:
            return None
        try:
            return int((ed - today).days)
        except Exception:
            return None

    out[EARN_COL_DAYS] = out["Ticker"].apply(_days_until)

    # Move earnings column right after Ticker for usability
    try:
        cols = list(out.columns)
        if "Ticker" in cols and EARN_COL_DAYS in cols:
            cols.remove(EARN_COL_DAYS)
            ticker_idx = cols.index("Ticker")
            cols.insert(ticker_idx + 1, EARN_COL_DAYS)
            out = out[cols]
    except Exception:
        pass

    return out

# --------------- Tier/Admin Helper Functions ----------------

def _norm_str(v: object | None) -> str:
    """Normalize user-provided / DB-provided strings to a safe canonical form."""
    try:
        return str(v or "").strip()
    except Exception:
        return ""


def _norm_lower(v: object | None) -> str:
    return _norm_str(v).lower()


def _is_admin_user(username: str | None, tier_obj: object | None) -> bool:
    """Admin check that is resilient to whitespace/case/enum differences."""
    u = _norm_lower(username)
    if u and u in ADMIN_USERS:
        return True

    # Tier objects may expose `.key` (preferred) or `.name`.
    try:
        key = getattr(tier_obj, "key", None)
        if _norm_lower(key) == "admin":
            return True
    except Exception:
        pass

    try:
        name = getattr(tier_obj, "name", None)
        if _norm_lower(name) == "admin":
            return True
    except Exception:
        pass

    # Last resort: string form
    if _norm_lower(tier_obj) == "admin":
        return True

    return False


def _tier_key(tier_obj: object | None) -> str:
    """Return a stable tier key string for logging/comparisons/UI."""
    try:
        key = getattr(tier_obj, "key", None)
        if key is not None:
            return _norm_lower(key)
    except Exception:
        pass

    try:
        name = getattr(tier_obj, "name", None)
        if name is not None:
            return _norm_lower(name)
    except Exception:
        pass

    return _norm_lower(tier_obj) or "basic"


# --------------- Entitlements (Centralized) ----------------
# Define the minimum tier required for each feature.
FEATURE_MIN_TIER: dict[str, str] = {
    # scans
    "can_scan_sp500": "basic",
    "can_scan_nasdaq": "pro",

    # results
    "can_export_csv": "pro",
    "can_ai_notes": "premium",

    # pro features
    "can_scan_history": "pro",

    # premium-only
    "can_early_breakout": "premium",
    "can_full_universe": "premium",

    # admin-only
    "can_diagnostics": "admin",
    "can_admin_panel": "admin",
}


def compute_entitlements(*, tier_obj: object | None, is_admin: bool) -> dict[str, bool]:
    """Central, deterministic feature flags based on tier."""
    flags: dict[str, bool] = {}

    # Admin overrides everything
    if bool(is_admin):
        for k in FEATURE_MIN_TIER.keys():
            flags[k] = True
        return flags

    for feature, min_tier in FEATURE_MIN_TIER.items():
        if min_tier == "admin":
            flags[feature] = False
            continue
        try:
            flags[feature] = bool(has_min_tier(tier_obj, min_tier))
        except Exception:
            flags[feature] = False

    return flags


# Day 6: Upgrade CTA card for Basic users only
def render_sidebar_upgrade_card(tier_obj: object | None) -> None:
    """Show a simple upgrade CTA card for Basic users in the sidebar."""
    try:
        # Show only for Basic (no Pro access)
        if has_min_tier(tier_obj, "pro"):
            return
    except Exception:
        return

    with st.sidebar.container(border=True):
        st.markdown("### 💡 You’re on Basic")
        st.caption(
            "You’re seeing a limited scan.\n"
            "Upgrade to unlock advanced filters, exports, and AI signals."
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Upgrade to Pro", key="upgrade_to_pro", use_container_width=True):
                st.session_state["pricing_focus"] = "pro"
                st.switch_page("pages/billing.py")
        with c2:
            if st.button("⭐ Upgrade to Premium", key="upgrade_to_premium", use_container_width=True):
                st.session_state["pricing_focus"] = "premium"
                st.switch_page("pages/billing.py")

        st.caption(
            "Tip: Pro unlocks exports + advanced filters. Premium unlocks full-universe + Early Breakout."
        )

# --------------- Cached Market Snapshot Helper ----------------
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_index_snapshot(symbol: str) -> tuple[float | None, float | None]:
    """
    Fetch the last and previous close for a single index/ETF symbol.

    Tries Alpaca snapshots first; if that fails or returns no data,
    falls back to yfinance-based history.
    """
    # --- First attempt: Alpaca via market_data.get_latest_quotes ---
    try:
        quotes = get_latest_quotes([symbol])
    except Exception:
        quotes = {}

    if quotes:
        q = quotes.get(symbol.upper())
        if isinstance(q, dict):
            last = q.get("last")
            prev = q.get("prev_close")
            try:
                if last is not None:
                    last_f = float(last)
                    prev_f = float(prev) if prev is not None else last_f
                    return last_f, prev_f
            except Exception:
                pass

    # --- Fallback: yfinance download / history ---
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None, None

    hist = None
    try:
        hist = yf.download(
            symbol,
            period="2d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        hist = None

    if hist is not None and not hist.empty and "Close" in hist.columns:
        close_block = hist["Close"]
        # If Close is a DataFrame (multi-column), select the first column
        if isinstance(close_block, pd.DataFrame):
            close_block = close_block.iloc[:, 0]
        closes = close_block.dropna().to_list()
        if closes:
            last = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else last
            return last, prev

    # --- Fallback: legacy Ticker().history() ---
    try:
        t = yf.Ticker(symbol)
        hist_single = t.history(period="2d")
        if hist_single is None or hist_single.empty or "Close" not in hist_single.columns:
            return None, None
        closes = hist_single["Close"].dropna().tolist()
        if not closes:
            return None, None
        last = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else last
        return last, prev
    except Exception:
        return None, None


def render_market_snapshot() -> None:
    """Render a compact 4-metric market snapshot row.

    Shows SPY, QQQ, Top Gainer (from latest scan), and Most Active
    (by dollar volume or volume) in a single, shallow row to avoid
    eating vertical space.
    """

    st.markdown("### 🔎 Today's Market Snapshot")

    show_diag = bool(st.session_state.get("show_diagnostics_ui", False))

    # Try to pull latest scan results once, reuse for Top Gainer / Most Active
    df: pd.DataFrame | None = None
    try:
        df = get_results_df()
        if df is not None and df.empty:
            df = None
    except Exception as e:
        if show_diag:
            st.caption(f"Results snapshot error: {e}")
        df = None

    c1, c2, c3, c4 = st.columns(4)

    # --- Helper: index/ETF metrics ---
    def _render_index(col, label: str, symbol: str) -> None:
        with col:
            try:
                last, prev = _fetch_index_snapshot(symbol)
            except Exception as e:
                if show_diag:
                    st.caption(f"{label} error: {e}")
                last, prev = None, None

            if last is None or prev is None or not prev:
                st.metric(label, "—", "—")
                return

            pct = ((last - prev) / prev) * 100.0
            st.metric(label, f"{last:.2f}", f"{pct:+.2f}%")

    # --- SPY / QQQ ---
    _render_index(c1, "S&P 500 (SPY)", "SPY")
    _render_index(c2, "NASDAQ 100 (QQQ)", "QQQ")

    # --- Top Gainer ---
    with c3:
        try:
            if df is None:
                st.metric("Top Gainer", "—", "—")
            else:
                # Detect a reasonable change/gain column
                lower_map = {col: col.lower() for col in df.columns}
                metric_col: str | None = None

                # Prefer explicit change/gain columns
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

                # Fallback: any column with "gap" in the name
                if metric_col is None:
                    for col, lower in lower_map.items():
                        if "gap" in lower:
                            metric_col = col
                            break

                # Last resort: first reasonable numeric column
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
                    st.metric("Top Gainer", "—", "—")
                    if show_diag:
                        st.caption("Top Gainer: no suitable change/gain metric found in results.")
                else:
                    numeric_series = pd.to_numeric(df[metric_col], errors="coerce")
                    idx = numeric_series.idxmax()

                    if pd.isna(numeric_series.loc[idx]):
                        st.metric("Top Gainer", "—", "—")
                    else:
                        top = df.loc[idx]
                        ticker = (
                            top.get("Ticker")
                            or top.get("symbol")
                            or top.get("Symbol")
                            or str(top.name)
                        )
                        raw_val = float(numeric_series.loc[idx])
                        change_str = f"{raw_val:+.2f}%"
                        st.metric("Top Gainer", ticker, change_str)
        except Exception as e:
            st.metric("Top Gainer", "—", "—")
            if show_diag:
                st.caption(f"Top Gainer error: {e}")

    # --- Most Active ---
    with c4:
        try:
            if df is None:
                st.metric("Most Active", "—", "—")
            else:
                vol_col: str | None = None
                if "DollarVol20" in df.columns:
                    vol_col = "DollarVol20"
                elif "Volume" in df.columns:
                    vol_col = "Volume"

                if not vol_col:
                    st.metric("Most Active", "—", "—")
                else:
                    numeric_vol = pd.to_numeric(df[vol_col], errors="coerce")
                    idx = numeric_vol.idxmax()
                    if pd.isna(numeric_vol.loc[idx]):
                        st.metric("Most Active", "—", "—")
                    else:
                        row = df.loc[idx]
                        val_millions = float(numeric_vol.loc[idx]) / 1_000_000
                        suffix = "M" if vol_col == "DollarVol20" else "M sh"
                        ticker = row.get("Ticker", "—")
                        st.metric("Most Active", ticker, f"{val_millions:.1f}{suffix}")
        except Exception as e:
            st.metric("Most Active", "—", "—")
            if show_diag:
                st.caption(f"Most Active error: {e}")


# --------------- Price Ticker ----------------
TICKER_STRIP = ["SPY", "QQQ", "IWM", "DIA", "VIX", "AAPL", "MSFT", "NVDA", "TSLA"]



@st.cache_data(ttl=180, show_spinner=False)
def _fetch_ticker_quotes(symbols: list[str]) -> list[dict[str, float]]:
    """Return list of dicts: [{'symbol', 'last', 'change_pct'}, ...].

    Tries Alpaca snapshots first; if that fails or returns no data,
    falls back to a batched yfinance download and then to per-symbol history.
    """
    if not symbols:
        return []

    # Ensure unique, uppercase symbols for consistency
    symbols = [s.upper() for s in dict.fromkeys(symbols).keys()]

    results: list[dict[str, float]] = []

    # --- First attempt: Alpaca via market_data.get_latest_quotes ---
    try:
        quotes = get_latest_quotes(symbols)
    except Exception:
        quotes = {}

    if quotes:
        for sym in symbols:
            q = quotes.get(sym)
            if not isinstance(q, dict):
                continue
            last = q.get("last")
            prev = q.get("prev_close")
            try:
                if last is None:
                    continue
                last_f = float(last)
                prev_f = float(prev) if prev is not None else last_f
                if prev_f in (0, None):
                    change_pct = 0.0
                else:
                    change_pct = ((last_f - prev_f) / prev_f) * 100.0
                results.append(
                    {
                        "symbol": sym,
                        "last": last_f,
                        "change_pct": change_pct,
                    }
                )
            except Exception:
                continue

    if results:
        return results

    # --- Fallback: batched yfinance download ---
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []

    hist = None
    try:
        hist = yf.download(
            symbols,
            period="2d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        hist = None

    if hist is not None and not hist.empty:
        for sym in symbols:
            try:
                if isinstance(hist.columns, pd.MultiIndex):
                    if ("Close", sym) not in hist.columns:
                        continue
                    closes = hist[("Close", sym)].dropna()
                else:
                    if "Close" not in hist.columns:
                        continue
                    closes = hist["Close"].dropna()

                if closes.empty:
                    continue

                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) > 1 else last
                if prev in (0, None):
                    change_pct = 0.0
                else:
                    change_pct = ((last - prev) / prev) * 100.0

                results.append(
                    {
                        "symbol": sym,
                        "last": last,
                        "change_pct": change_pct,
                    }
                )
            except Exception:
                continue

        if results:
            return results

    # --- Final fallback: legacy per-symbol calls ---
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist_single = t.history(period="2d")
            if hist_single is None or hist_single.empty or "Close" not in hist_single.columns:
                continue
            closes = hist_single["Close"].dropna().tolist()
            if not closes:
                continue
            last = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else last
            if prev in (0, None):
                change_pct = 0.0
            else:
                change_pct = ((last - prev) / prev) * 100.0

            results.append(
                {
                    "symbol": sym,
                    "last": last,
                    "change_pct": change_pct,
                }
            )
        except Exception:
            continue

    return results


def render_price_ticker():
    """Render a scrolling ticker just under the main header."""
    data = _fetch_ticker_quotes(TICKER_STRIP)
    if not data:
        # Optional diagnostics: surface a hint when no ticker data is available.
        try:
            if bool(st.session_state.get("show_diagnostics_ui", False)):
                st.sidebar.warning(
                    "Price ticker: no data returned from yfinance. "
                    "This may be a network issue, rate limiting, or a weekend/holiday."
                )
        except Exception:
            pass
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

        # 🔄 Reset filters back to the saved profile from Neon
        if callable(get_user_settings) and st.sidebar.button("🔄 Reset to saved profile"):
            # Clear the loaded-profile marker so main() will reload from Neon
            st.session_state["profile_loaded_for_user"] = None
            st.sidebar.info("Reloading your saved profile...")
            st.rerun()

    elif session_username:
        ...

# --------------- UX Helpers ----------------

def render_active_filters_summary(
    *,
    tier,
    universe,
    min_price: float,
    max_price: float,
    min_dollar_vol: float,
    top_n: int,
    premarket: bool,
    afterhours: bool,
    include_ta: bool,
    unusual_vol: bool,
    apply_gap_filter: bool,
    min_gap: float,
    max_nasdaq_scan: int,
    max_combo_scan: int,
) -> None:
    """Compact summary of active filters shown above results."""
    chips: list[str] = []

    if universe:
        chips.append(f"Universe: {universe}")

    chips.append(f"Price: ${min_price:g}–${max_price:g}")

    if min_dollar_vol and min_dollar_vol > 0:
        chips.append(f"Min $Vol: {int(min_dollar_vol):,}")

    chips.append(f"Top N: {int(top_n)}")
    chips.append(f"NASDAQ cap: {int(max_nasdaq_scan):,}")
    chips.append(f"Combo cap: {int(max_combo_scan):,}")

    if premarket:
        chips.append("Session: Premarket")
    elif afterhours:
        chips.append("Session: After-hours")
    else:
        chips.append("Session: Regular")

    if include_ta:
        chips.append("TA: ON")
    if unusual_vol:
        chips.append("Unusual Vol: ON")

    if apply_gap_filter:
        chips.append(f"Gap Filter: ON (≥ {float(min_gap):g}%)")

    st.markdown("#### ✅ Active Filters")
    st.caption(" • ".join(chips))


def render_onboarding_hint(username: str, *, tier_name: str) -> None:
    """One-time onboarding hint per session per user."""
    key = f"onboarding_dismissed::{(username or '').strip().lower()}"
    if st.session_state.get(key):
        return

    with st.expander("👋 Quick start (click once)", expanded=True):
        st.markdown(
            f"""
**Welcome!** You’re signed in on **{tier_name}**.

**Fast workflow:**
1) Set filters in the sidebar  
2) Click **Run Scan** (SP500 / NASDAQ / Combo)  
3) Use **Save as my default settings** once you like your setup  
4) Use **Reset to saved profile** anytime to revert

Tip: turn on **Apply Gap Filter** to enforce **Min Gap %**.
"""
        )
        if st.button("✅ Got it", key=f"onboarding_got_it::{username}"):
            st.session_state[key] = True
            st.rerun()

# ============================================================
#                       MAIN UI
# ============================================================

def main():
    # -------- AUTH FIRST (NOW FIRST) --------
    authed, username, display_name = auth_ui()
    if not authed:
        # Not logged in: show only the login card (auth_ui handles it)
        st.stop()

    # Normalize and persist username for downstream modules (billing/settings rely on this)
    username = (username or "").strip().lower()
    if username:
        st.session_state["username"] = username

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


    # -------- ONLY NOW RENDER HEADER + TICKER --------
    # Show ticker above the header (layout option B)
    render_price_ticker()
    render_header()
    # -------- Load Users + Tier --------
    users_map = load_users()

    # DB is the source of truth for tier (Stripe webhook updates DB; users_map can be stale)
    forced_tier_key: str | None = None
    db_tier_err: str | None = None
    db_user_debug: dict | None = None

    # Re-sync tier from DB (Stripe webhook updates DB; session may be stale until this runs)
    # app.py-only DB connector (no db/core.py dependency)
    try:
        def _lookup_user_from_db(identifier: str) -> dict | None:
            # Build a SELECT that matches the *actual* users table schema.
            # Some deployments don't have an `email` column, so we must adapt.
            conn = _get_db_conn_for_app()
            try:
                # Discover available columns on `users`
                cols: set[str] = set()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = 'users'
                            """
                        )
                        rows = cur.fetchall() or []
                        for r in rows:
                            try:
                                cols.add(str(r[0]))
                            except Exception:
                                pass
                except Exception:
                    cols = set()

                # Columns we *wish* to fetch (only include those that exist)
                wanted = [
                    "username",
                    "email",
                    "tier",
                    "stripe_customer_id",
                    "stripe_subscription_id",
                ]
                select_cols = [c for c in wanted if (not cols) or (c in cols)]
                if not select_cols:
                    # Absolute minimum
                    select_cols = ["username", "tier"]

                # WHERE clause: prefer matching username; also match email if it exists
                where = "lower(username) = lower(%s)"
                params = [identifier]
                if (not cols) or ("email" in cols):
                    where = f"({where} OR lower(email) = lower(%s))"
                    params.append(identifier)

                sql = (
                    f"SELECT {', '.join(select_cols)} "
                    f"FROM users "
                    f"WHERE {where} "
                    f"LIMIT 1"
                )

                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    row = cur.fetchone()
                    if not row:
                        return None

                    # Build dict from whatever columns we selected
                    try:
                        desc = cur.description or []
                        out = {str(desc[i][0]): row[i] for i in range(len(desc))}
                    except Exception:
                        out = {}

                    # Ensure missing keys still exist for callers (debug/UI)
                    for k in wanted:
                        out.setdefault(k, None)

                    return out
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        u = _lookup_user_from_db(username)

        # Keep a small debug payload (only shown to admins)
        if isinstance(u, dict) and u:
            db_user_debug = {
                "username": u.get("username"),
                "email": u.get("email"),
                "tier": u.get("tier"),
                "stripe_customer_id": u.get("stripe_customer_id"),
                "stripe_subscription_id": u.get("stripe_subscription_id"),
            }
        else:
            db_user_debug = None

        db_tier = (u.get("tier") if isinstance(u, dict) else None)
        if db_tier:
            forced_tier_key = str(db_tier).strip().lower() or None

            # Keep users_map in sync for any legacy code paths that still read it
            if forced_tier_key and isinstance(users_map, dict):
                if username not in users_map:
                    users_map[username] = {}
                if isinstance(users_map.get(username), dict):
                    users_map[username]["tier"] = forced_tier_key

            # If the tier changed since last render, invalidate cached entitlements so UI unlocks immediately.
            prev = (st.session_state.get("tier_key") or "").strip().lower()
            if forced_tier_key and prev and prev != forced_tier_key:
                st.session_state.pop("entitlements", None)

    except Exception as e:
        forced_tier_key = None
        db_tier_err = str(e)
        db_user_debug = None

    tier = get_user_tier(username, users_map)

    # IMPORTANT: Override tier object if DB says we are Pro/Premium/etc.
    # This prevents stale users_map/tiering cache from requiring logout/login.
    if forced_tier_key:
        try:
            # Prefer Enum member lookup (Tier.PRO, Tier.PREMIUM, etc.)
            if hasattr(Tier, "__members__") and forced_tier_key.upper() in getattr(Tier, "__members__"):
                tier = Tier[forced_tier_key.upper()]  # type: ignore[index]
            else:
                # Fallback: Enum values may be strings
                tier = Tier(forced_tier_key)  # type: ignore[call-arg]
        except Exception:
            # Last resort: keep computed tier
            pass

    is_admin = _is_admin_user(username, tier)

    # DB is source of truth for the tier key. If Enum conversion failed, still honor DB.
    tier_key = (forced_tier_key or _tier_key(tier) or "basic").strip().lower()

    # If DB-forced tier differs from what was previously stored in session, drop cached flags.
    if forced_tier_key:
        prev_key = (st.session_state.get("tier_key") or "").strip().lower()
        if prev_key and prev_key != tier_key:
            st.session_state.pop("entitlements", None)

    # If the tier object doesn't reflect the DB tier key, build a tiny proxy for gating.
    # This guarantees entitlements update immediately without requiring logout/login.
    tier_for_flags = tier
    try:
        if forced_tier_key and _tier_key(tier) != tier_key:
            tier_for_flags = SimpleNamespace(key=tier_key, name=tier_key.upper())
    except Exception:
        tier_for_flags = SimpleNamespace(key=tier_key, name=tier_key.upper())

    # Day 6 – Item 2: centralized entitlements
    flags = compute_entitlements(tier_obj=tier_for_flags, is_admin=is_admin)
    tier_name = "Admin" if is_admin else tier_key.upper()

    # Persist tier + flags in session for downstream UI modules
    st.session_state["tier"] = tier_for_flags
    st.session_state["tier_key"] = tier_key
    st.session_state["is_admin"] = bool(is_admin)
    st.session_state["entitlements"] = dict(flags)

    # Safety: if AI Notes are not allowed for this user, purge any cached notes
    # so Basic/Pro accounts never see previously-generated Premium content.
    if not bool(flags.get("can_ai_notes")):
        for k in (
            "ai_notes",
            "ai_notes_text",
            "ai_notes_cache",
            "ai_notes_last",
            "ai_notes_last_text",
            "last_ai_notes",
        ):
            st.session_state.pop(k, None)

    render_onboarding_hint(username, tier_name=tier_name)

    # ---------------- Admin-only: Build stamp + One-click Earnings Debug ----------------
    # If you don't see this section after deploying, you're almost certainly not running the updated app.py.
    if bool(st.session_state.get("is_admin")):
        try:
            build_mtime = datetime.utcfromtimestamp(Path(__file__).stat().st_mtime).isoformat() + "Z"
        except Exception:
            build_mtime = "unknown"

        st.sidebar.caption(f"build: {build_mtime} | user: {username} | tier_key: {tier_key}")

        with st.sidebar.expander("🧪 Earnings Calendar Debug (One-click)", expanded=False):
            st.caption("Runs a forced earnings fetch for known symbols and verifies DB rows immediately.")

            # --- One-click: check a single ticker (DB-only or fetch+write+verify) ---
            st.markdown("**Quick single-ticker check**")
            _sym_in = st.text_input("Ticker", value="AAPL", key="earn_dbg_single_symbol")
            _sym = (_sym_in or "").strip().upper()

            c_a, c_b = st.columns(2)
            with c_a:
                run_db_only = st.button(
                    "DB lookup",
                    key="btn_earnings_db_lookup",
                    use_container_width=True,
                    disabled=not bool(_sym),
                )
            with c_b:
                run_fetch_write = st.button(
                    "Fetch + write + verify",
                    key="btn_earnings_fetch_write_verify",
                    use_container_width=True,
                    disabled=not bool(_sym),
                )

            def _db_lookup_symbol(sym: str) -> dict | None:
                conn = _get_db_conn_for_app()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT symbol, earnings_date, earnings_time, updated_at
                            FROM earnings_calendar
                            WHERE upper(symbol) = upper(%s)
                            LIMIT 1
                            """,
                            (sym,),
                        )
                        row = cur.fetchone()
                        if not row:
                            return None
                        return {
                            "symbol": row[0],
                            "earnings_date": row[1],
                            "earnings_time": row[2],
                            "updated_at": row[3],
                        }
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

            if run_fetch_write:
                try:
                    try:
                        from earnings import populate_earnings_calendar  # type: ignore
                    except Exception:
                        from db.earnings import populate_earnings_calendar  # type: ignore

                    with st.spinner(f"Fetching earnings for {_sym} + writing to DB..."):
                        import inspect

                        sig = None
                        try:
                            sig = inspect.signature(populate_earnings_calendar)
                        except Exception:
                            sig = None

                        if sig is not None and "conn" in sig.parameters:
                            conn = _get_db_conn_for_app()
                            try:
                                populate_earnings_calendar([_sym], conn=conn)
                            finally:
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                        else:
                            populate_earnings_calendar([_sym])

                    st.success(f"Fetch complete for {_sym}")
                except Exception as e:
                    st.error(f"Fetch/write failed for {_sym}: {e}")

            if run_db_only or run_fetch_write:
                try:
                    row = _db_lookup_symbol(_sym)
                    if not row:
                        st.warning(f"No DB row found for {_sym} (yet).")
                    else:
                        # Compute days-until (same logic as results column)
                        _today = date.today()
                        _ed = row.get("earnings_date")
                        _days = None
                        try:
                            if _ed:
                                _days = int((_ed - _today).days)
                        except Exception:
                            _days = None

                        st.write(
                            {
                                "symbol": row.get("symbol"),
                                "earnings_date": row.get("earnings_date"),
                                "earnings_time": row.get("earnings_time"),
                                "days_until_earnings": _days,
                                "updated_at": row.get("updated_at"),
                            }
                        )
                        if row.get("earnings_date") is None:
                            st.caption(
                                "Earnings date is NULL = provider returned TBA / not announced (still a valid row)."
                            )
                except Exception as e:
                    st.error(f"DB lookup failed for {_sym}: {e}")

            st.markdown("---")

            test_syms = ["AAPL", "MSFT", "TSLA", "META", "AMD", "INTC"]

            if st.button(
                "Fetch earnings for AAPL / MSFT / TSLA / META / AMD / INTC",
                key="btn_earnings_debug_sidebar",
                use_container_width=True,
            ):
                # 1) Fetch + upsert
                try:
                    try:
                        from earnings import populate_earnings_calendar  # type: ignore
                    except Exception:
                        from db.earnings import populate_earnings_calendar  # type: ignore

                    with st.spinner("Fetching earnings via Yahoo Finance + writing to DB..."):
                        import inspect

                        sig = None
                        try:
                            sig = inspect.signature(populate_earnings_calendar)
                        except Exception:
                            sig = None

                        # Prefer passing an explicit DB connection when supported
                        if sig is not None and "conn" in sig.parameters:
                            conn = _get_db_conn_for_app()
                            try:
                                _ = populate_earnings_calendar(test_syms, conn=conn)
                            finally:
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                        else:
                            _ = populate_earnings_calendar(test_syms)

                    st.success("Fetched earnings for AAPL, MSFT, TSLA, META, AMD, INTC")
                except Exception as e:
                    st.error(f"Earnings fetch failed: {e}")

                # 2) Verify DB writes
                try:
                    conn = _get_db_conn_for_app()
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT to_regclass('public.earnings_calendar')")
                            reg = cur.fetchone()
                            table_exists = bool(reg and reg[0])

                        if not table_exists:
                            st.error("DB verify: public.earnings_calendar not found")
                        else:
                            with conn.cursor() as cur:
                                cur.execute("SELECT COUNT(*) FROM earnings_calendar")
                                total = int((cur.fetchone() or [0])[0])

                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    SELECT COUNT(*)
                                    FROM earnings_calendar
                                    WHERE upper(symbol) = ANY(%s)
                                    """,
                                    (test_syms,),
                                )
                                n_syms = int((cur.fetchone() or [0])[0])

                            st.caption(
                                f"DB verify: earnings_calendar total_rows={total}, rows_for_{','.join(test_syms)}={n_syms}"
                            )

                            try:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        """
                                        SELECT *
                                        FROM earnings_calendar
                                        WHERE upper(symbol) = ANY(%s)
                                        ORDER BY updated_at DESC NULLS LAST
                                        LIMIT 20
                                        """,
                                        (test_syms,),
                                    )
                                    rows = cur.fetchall() or []
                                    cols = [d[0] for d in (cur.description or [])]

                                if rows and cols:
                                    st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
                                else:
                                    st.warning(
                                        "DB verify: table exists but 0 rows for test symbols. "
                                        "That means the upsert/insert did not run or wrote to a different DB."
                                    )
                            except Exception as e:
                                st.caption(f"DB verify: sample row fetch failed: {e}")
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                except Exception as e:
                    st.error(
                        "DB verify failed: "
                        + str(e)
                        + "\n\nTip: ensure Streamlit secrets has DATABASE_URL or database_url, and psycopg/psycopg2 is installed."
                    )


            st.caption("If you still see 0 rows, check Render logs for earnings upsert/insert errors.")

        # ---------------- Admin-only: Run daily earnings refresh (batch) ----------------
        with st.sidebar.expander("📅 Run daily earnings refresh (admin)", expanded=False):
            st.caption(
                "Runs a batch refresh and writes to `earnings_calendar`. "
                "Use this to populate ‘Earnings in X days’ quickly without waiting for scans."
            )

            # Choose which universe to refresh
            refresh_target = st.radio(
                "Universe",
                ["SP500", "NASDAQ (capped)", "Combo (SP500+NASDAQ capped)", "Active watchlist"],
                horizontal=False,
                key="earnings_refresh_target",
            )

            # Chunk size keeps requests + DB writes predictable
            chunk_size = st.number_input(
                "Chunk size",
                min_value=50,
                max_value=1000,
                value=250,
                step=50,
                help="Smaller chunks are safer on cold starts / rate limits; larger chunks are faster.",
                key="earnings_refresh_chunk_size",
            )

            max_symbols = st.number_input(
                "Max symbols",
                min_value=50,
                max_value=5000,
                value=1000,
                step=50,
                help="Safety cap so you can test without refreshing the entire market.",
                key="earnings_refresh_max_symbols",
            )

            def _pick_refresh_symbols() -> list[str]:
                try:
                    if refresh_target == "SP500":
                        syms = safe_call(load_sp500_universe, label="SP500 universe (earnings refresh)")
                        syms = filter_universe(syms)
                        return [str(s).strip().upper() for s in (syms or []) if str(s).strip()]

                    if refresh_target == "NASDAQ (capped)":
                        syms = safe_call(load_nasdaq_universe, label="NASDAQ universe (earnings refresh)")
                        syms = filter_universe(syms)
                        syms = (syms or [])[: int(st.session_state.get("max_nasdaq_scan", 1500))]
                        return [str(s).strip().upper() for s in syms if str(s).strip()]

                    if refresh_target == "Combo (SP500+NASDAQ capped)":
                        sp500 = safe_call(load_sp500_universe, label="SP500 universe (earnings refresh)")
                        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe (earnings refresh)")
                        sp500 = filter_universe(sp500)
                        nasdaq = filter_universe(nasdaq)
                        nasdaq = (nasdaq or [])[: int(st.session_state.get("max_nasdaq_scan", 1500))]
                        combo = list(sp500 or []) + list(nasdaq or [])
                        combo = filter_universe(combo)
                        return [str(s).strip().upper() for s in combo if str(s).strip()]

                    # Active watchlist
                    watch = st.session_state.get("active_watchlist_tickers", []) or []
                    return [str(s).strip().upper() for s in watch if str(s).strip()]
                except Exception:
                    return []

            syms = _pick_refresh_symbols()
            if max_symbols and syms:
                syms = syms[: int(max_symbols)]

            st.caption(f"Selected symbols: **{len(syms)}**")

            run_refresh = st.button(
                "Run refresh now",
                key="btn_admin_run_earnings_refresh",
                use_container_width=True,
                disabled=(len(syms) == 0),
            )

            if run_refresh:
                try:
                    # Import refresh function
                    try:
                        from earnings import populate_earnings_calendar  # type: ignore
                    except Exception:
                        from db.earnings import populate_earnings_calendar  # type: ignore

                    import inspect

                    sig = None
                    try:
                        sig = inspect.signature(populate_earnings_calendar)
                    except Exception:
                        sig = None

                    total = len(syms)
                    wrote = 0
                    progress = st.progress(0)
                    status = st.empty()

                    # Work in deterministic chunks
                    for i in range(0, total, int(chunk_size)):
                        batch = syms[i : i + int(chunk_size)]
                        status.write(f"Refreshing earnings: {i + 1}–{min(i + len(batch), total)} / {total}…")

                        if sig is not None and "conn" in getattr(sig, "parameters", {}):
                            conn = _get_db_conn_for_app()
                            try:
                                populate_earnings_calendar(batch, conn=conn)
                            finally:
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                        else:
                            populate_earnings_calendar(batch)

                        wrote = min(i + len(batch), total)
                        progress.progress(int((wrote / max(total, 1)) * 100))

                    # Quick DB count verify
                    conn = _get_db_conn_for_app()
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT COUNT(*) FROM earnings_calendar")
                            n = int((cur.fetchone() or [0])[0])
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass

                    st.session_state["earnings_last_admin_refresh"] = datetime.utcnow().isoformat() + "Z"
                    st.success(f"✅ Earnings refresh complete. Symbols processed: {total}. earnings_calendar rows: {n}.")
                except Exception as e:
                    st.error(f"Earnings refresh failed: {e}")

            last = st.session_state.get("earnings_last_admin_refresh")
            if last:
                st.caption(f"Last admin refresh: {last}")

    # -------- Load Saved User Settings (if available) --------
    if username and callable(get_user_settings):
        safe_username = (username or "").strip()
        last_profile_user = st.session_state.get("profile_loaded_for_user")

        # Only load from Neon when we haven't yet loaded for this user
        if last_profile_user != safe_username:
            try:
                saved = get_user_settings(safe_username)
            except Exception:
                saved = None


            if saved:
                # Seed session_state from saved settings when available.
                if saved.get("universe") is not None:
                    st.session_state["universe"] = saved["universe"]

                if saved.get("min_price") is not None:
                    st.session_state["min_price"] = float(saved["min_price"])
                if saved.get("max_price") is not None:
                    st.session_state["max_price"] = float(saved["max_price"])

                if saved.get("min_dollar_vol") is not None:
                    st.session_state["min_dollar_vol"] = float(saved["min_dollar_vol"])

                if saved.get("include_ta") is not None:
                    st.session_state["include_ta"] = bool(saved["include_ta"])
                if saved.get("apply_gap_filter") is not None:
                    st.session_state["apply_gap_filter"] = bool(saved["apply_gap_filter"])

                # Diagnostics UI is admin-only, even if a non-admin saved it historically.
                if saved.get("show_diagnostics_ui") is not None:
                    st.session_state["show_diagnostics_ui"] = bool(saved["show_diagnostics_ui"]) and bool(st.session_state.get("is_admin"))

                if saved.get("min_gap") is not None:
                    st.session_state["min_gap"] = float(saved["min_gap"])
                if saved.get("top_n") is not None:
                    st.session_state["top_n"] = int(saved["top_n"])
                if saved.get("max_nasdaq_scan") is not None:
                    st.session_state["max_nasdaq_scan"] = int(saved["max_nasdaq_scan"])
                if saved.get("max_combo_scan") is not None:
                    st.session_state["max_combo_scan"] = int(saved["max_combo_scan"])

                if saved.get("premarket") is not None:
                    st.session_state["premarket"] = bool(saved["premarket"])
                if saved.get("afterhours") is not None:
                    st.session_state["afterhours"] = bool(saved["afterhours"])
                if saved.get("unusual_vol") is not None:
                    st.session_state["unusual_vol"] = bool(saved["unusual_vol"])

                # Mark that we've applied the profile for this specific user in this session
                st.session_state["profile_loaded_for_user"] = safe_username

    # -------- Sidebar Account Info --------
    # Prefer display name; fall back to email prefix if missing
    raw_display = (display_name or "").strip()
    raw_username = (username or "").strip()

    if raw_display:
        name_label = raw_display
    elif "@" in raw_username:
        name_label = raw_username.split("@")[0]
    else:
        name_label = raw_username or "Account"

    st.sidebar.markdown(f"### 👤 {name_label}")
    st.sidebar.markdown(
        f"**Plan:** `{ 'Admin' if bool(st.session_state.get('is_admin')) else getattr(tier, 'name', st.session_state.get('tier_key', 'basic')) }`"
    )
    if bool(st.session_state.get("is_admin")):
        st.sidebar.caption(
            f"debug: db_tier={forced_tier_key or '-'} session_tier_key={st.session_state.get('tier_key')}"
        )
        if db_tier_err:
            st.sidebar.error(f"DB tier lookup error: {db_tier_err}")
        elif db_user_debug is not None:
            st.sidebar.caption(f"DB user: {db_user_debug}")
    # Day 6: Upgrade CTA card (Basic users only)
    render_sidebar_upgrade_card(tier)

    if st.sidebar.button("Log out", key="logout_button"):
        logout_and_reset_session()
    #st.markdown("---")

    # -------- DB Status --------
    db_status = render_db_status_badge()

    # Pre-clamp diagnostics BEFORE filters render widgets.
    # Streamlit forbids mutating widget-bound session_state keys after widget creation.
    if not flags.get("can_diagnostics"):
        st.session_state["show_diagnostics_ui"] = False

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
    # Enforce admin-only diagnostics (even if UI/modules accidentally expose it)
    if not flags.get("can_diagnostics"):
        diagnostics = False
    render_active_filters_summary(
        tier=tier,
        universe=st.session_state.get("universe"),
        min_price=float(min_price),
        max_price=float(max_price),
        min_dollar_vol=float(min_dollar_vol),
        top_n=int(top_n),
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        include_ta=bool(include_ta),
        unusual_vol=bool(unusual_vol),
        apply_gap_filter=bool(apply_gap_filter),
        min_gap=float(min_gap),
        max_nasdaq_scan=int(max_nasdaq_scan),
        max_combo_scan=int(max_combo_scan),
    )

    # -------- Market session gating for extended-hours toggles --------
    session = get_market_session()
    st.sidebar.caption(f"Market session (US/Eastern): {session.capitalize()}")

    # Premarket toggle only takes effect during premarket session
    if premarket and session != "premarket":
        # Clamp to regular mode for this run; avoid mutating widget state directly.
        premarket = False
        st.sidebar.info(
            "Premarket scans only run between 4:00–9:30am ET on trading days. "
            "The toggle has been reset to Regular mode for this scan."
        )

    # After-hours toggle only takes effect during after-hours session
    if afterhours and session != "afterhours":
        # Clamp to regular mode for this run; avoid mutating widget state directly.
        afterhours = False
        st.sidebar.info(
            "After-hours scans only run between 4:00–8:00pm ET on trading days. "
            "The toggle has been reset to Regular mode for this scan."
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
    render_market_snapshot()

    # -------- Scan Controls --------
    render_scan_controls(
        can_scan_sp500=flags["can_scan_sp500"],
        can_scan_nasdaq=flags["can_scan_nasdaq"],
        max_nasdaq_scan=int(max_nasdaq_scan),
        max_combo_scan=int(max_combo_scan),
        min_gap=float(min_gap),
        apply_gap_filter=bool(apply_gap_filter),
        min_price=float(min_price),
        max_price=float(max_price),
        top_n=int(top_n),
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        unusual_vol=bool(unusual_vol),
        diagnostics=bool(diagnostics),
        username=username,
    )

    st.markdown("## 🚀 EZ 3-Step AI Scanner")
    render_three_step_scanner()
    st.markdown("---")

    # -------- Results + Early Breakout Candidates + Scan History --------
    df = get_results_df()

    # -------- Scan timestamp (UTC) --------
    # Record a timestamp when a *new* results dataframe appears so users know freshness.
    # We detect "new" results via a lightweight signature stored in session_state.
    try:
        if df is not None and not df.empty:
            _rows = int(len(df))
            _first = str(df.iloc[0].get("Ticker", "")).strip().upper() if _rows > 0 else ""
            _last = str(df.iloc[-1].get("Ticker", "")).strip().upper() if _rows > 0 else ""
            _sig = f"{_rows}|{_first}|{_last}"

            prev_sig = str(st.session_state.get("results_signature") or "")
            if _sig and _sig != prev_sig:
                st.session_state["results_signature"] = _sig
                st.session_state["scan_ran_at_utc"] = datetime.now(timezone.utc)

        # If results cleared, clear the timestamp too (keeps UI honest)
        if df is None or df.empty:
            st.session_state.pop("results_signature", None)
            st.session_state.pop("scan_ran_at_utc", None)
    except Exception:
        pass

    scan_ran_at = st.session_state.get("scan_ran_at_utc")

    # -------- Optional: Earnings enrichment toggle --------
    # Earnings enrichment does a DB batch lookup; let users disable it for faster results.
    show_earnings = bool(st.session_state.get("enable_earnings_enrichment", False))
    with st.sidebar.expander("📅 Earnings", expanded=False):
        show_earnings = st.checkbox(
            "Enable earnings enrichment (adds 📅 Earnings in X days)",
            value=show_earnings,
            key="enable_earnings_enrichment",
            help="If enabled, the app will query the DB to add earnings timing to results and enable earnings filters.",
        )

    # Enrich results with earnings-days column (ONE DB query) before display
    if show_earnings and df is not None and not df.empty:
        try:
            df = add_earnings_days_column(df)
        except Exception as _e:
            # Admin-only hint if enrichment fails
            if bool(st.session_state.get("is_admin")):
                st.sidebar.caption(f"earnings enrich failed: {_e}")

    # Earnings filters (apply before we build tabs / counts)
    # - Exclude earnings in next 3 days
    # - Only earnings within 7 days
    if show_earnings and df is not None and not df.empty and EARN_COL_DAYS in df.columns:
        with st.sidebar.expander("📅 Earnings Filters", expanded=False):
            excl_3 = st.checkbox("Exclude earnings in next 3 days", value=False, key="earn_excl_3")
            within_7 = st.checkbox("Only earnings within 7 days", value=False, key="earn_within_7")

        s = pd.to_numeric(df[EARN_COL_DAYS], errors="coerce")
        if excl_3:
            df = df[s.isna() | (s > 3)]
            s = pd.to_numeric(df[EARN_COL_DAYS], errors="coerce")
        if within_7:
            df = df[(s >= 0) & (s <= 7)]

    rows = 0 if df is None else len(df)

    # Build tabs dynamically based on entitlements:
    # - Basic: only Latest Results
    # - Pro  : Latest Results + Scan History
    # - Premium/Admin: Latest Results + Early Breakout Candidates + Scan History
    if flags.get("can_scan_history") or flags.get("can_early_breakout"):
        tab_names = [f"📊 Latest scan results ({rows} rows)"]
        if flags.get("can_early_breakout"):
            tab_names.append("🔮 Early Breakout Candidates")
        if flags.get("can_scan_history"):
            tab_names.append("📚 Scan History")

        tabs = st.tabs(tab_names)
        tab1 = tabs[0]
        tab2 = tabs[1] if flags.get("can_early_breakout") else None
        tab3 = tabs[-1] if flags.get("can_scan_history") else None
    else:
        (tab1,) = st.tabs([f"📊 Latest scan results ({rows} rows)"])
        tab2 = None
        tab3 = None

    with tab1:
        if scan_ran_at:
            try:
                st.caption(f"🕒 Scan run at {scan_ran_at.strftime('%Y-%m-%d %H:%M UTC')}")
            except Exception:
                st.caption("🕒 Scan run time available")
        # Preserve existing latest results behavior
        if rows == 0:
            with st.expander(f"📊 Latest scan results ({rows} rows)", expanded=False):
                render_results(
                    df,
                    flags["can_export_csv"],
                    flags["can_ai_notes"],
                    render_chart_for_ticker,
                    generate_ai_note,
                )
        else:
            render_results(
                df,
                flags["can_export_csv"],
                flags["can_ai_notes"],
                render_chart_for_ticker,
                generate_ai_note,
            )

    if tab2 is not None:
        with tab2:
            # Premium-only: Basic/Pro users cannot access Early Breakout Candidates
            if require_min_tier(tier, "premium", "Early Breakout Candidates"):
                render_prebreakout_tab()

    if tab3 is not None:
        with tab3:
            # Pro+ only: Basic users cannot access Scan History
            if require_min_tier(tier, "pro", "Scan History"):
                render_history_expander(db_status)

    st.markdown("---")

    # -------- Universe State --------
    init_universe_state()
    render_universe_panel()

    # -------- Admin Panel (Admin only) --------
    if flags.get("can_admin_panel"):
        render_admin_users_panel(username, ADMIN_USERS, db_status)

    # -------- Pricing (Load Last) --------
    pricing_sidebar(username, users_map)

    # --- Debug: yfinance status (Admin only) ---
    if bool(st.session_state.get("is_admin")):
        with st.expander("🔧 Debug: Data Status", expanded=False):
            try:
                from data.prices import debug_yfinance_status  # type: ignore

                if st.button("Run yfinance self-test", key="debug_yf_status"):
                    status = debug_yfinance_status("AAPL")
                    st.write(status)
                    if not status.get("available"):
                        st.warning(
                            "yfinance is not importable. Make sure 'yfinance' is in your "
                            "requirements.txt or installed in this environment."
                        )
                    elif status.get("test_error"):
                        st.error(
                            "yfinance was imported but the test download failed. This is "
                            "usually a network or rate-limit issue on the hosting platform."
                        )
                    elif status.get("test_rows", 0) == 0:
                        st.warning(
                            "yfinance returned 0 rows for AAPL (60d). Yahoo may be "
                            "throttling or this environment may be blocking outbound calls."
                        )
                    else:
                        st.success(
                            f"yfinance looks healthy. Rows returned: {status.get('test_rows')}"
                        )

                if st.button("Test price data batch (AAPL, MSFT, NVDA)", key="debug_price_batch"):
                    try:
                        from data.prices import fetch_price_data_batch  # type: ignore

                        price_data, skipped = fetch_price_data_batch(["AAPL", "MSFT", "NVDA"])
                        st.write({
                            "price_data_len": len(price_data),
                            "skipped": skipped[:10],
                        })
                        if not price_data:
                            st.warning("fetch_price_data_batch returned no data for the test symbols.")
                        else:
                            for sym, df in list(price_data.items()):
                                st.write(f"Symbol: {sym}")
                                st.dataframe(df.tail())
                    except Exception as e2:
                        st.error(f"price data batch debug failed: {e2}")
            except Exception as e:
                st.caption(f"(yfinance debug not available: {e})")

    # -------- Footer --------
    render_footer()

if __name__ == "__main__":
    main()