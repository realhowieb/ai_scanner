from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import streamlit as st

from ui.app_boot import (
    configure_page,
    install_streamlit_compat,
    quiet_external_calls as _quiet_external_calls,
)
from ui.app_session import (
    compute_entitlements,
    is_admin_user,
    normalize_admin_users,
    tier_key as _tier_key,
)

install_streamlit_compat()

from types import SimpleNamespace

# Ensure project base directory importable
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from db.core import get_conn as _get_db_conn_for_app
except Exception:
    _get_db_conn_for_app = None  # type: ignore[assignment]

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
configure_page()


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
ADMIN_USERS = normalize_admin_users(ADMIN_USERS)

# --------------- AUTH (must load even if other modules fail) ----------------
_AUTH_IMPORT_ERROR: str | None = None
try:
    from ui.auth import auth_ui, logout_and_reset_session  # type: ignore
except Exception as _e:
    _AUTH_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

    def auth_ui():
        st.error("Auth module failed to import. Cannot render login.")
        st.code(_AUTH_IMPORT_ERROR or "unknown auth import error")
        return (False, None, None)

    def logout_and_reset_session():
        # Safe fallback so logout actions do not crash the app
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.rerun()
# --------------- DB modules & UI modules ----------------
# IMPORTANT: Keep auth import reliable so the login UI can render even if other modules break.
_IMPORT_ERROR: str | None = None

try:
    from db.users import seed_neon_users_from_local, load_users
    from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

    # User settings (per-user defaults) – optional Neon-backed feature
    try:
        from db.user_settings import get_user_settings, upsert_user_settings
    except Exception:
        get_user_settings = None
        upsert_user_settings = None

    from ui.admin_users import render_admin_users_panel
    from ui.history import render_history_expander
    from ui.results import render_results, get_results_df
    from ui.scans import render_scan_controls, render_three_step_scanner
    from ui.universe_panel import render_universe_panel, init_universe_state
    from ui.filters import render_filters
    from ui.db_status import render_db_status_badge
    from ui.header import render_header, render_price_ticker, render_market_snapshot
    from ui.prebreakout_tab import render_prebreakout_tab
    from ui.results_tabs import render_results_tabs
    from ui.earnings_results import prepare_results_with_earnings, render_earnings_controls
    from ui.footer import render_footer
    from ui.watchlists import render_watchlists_panel
    from ui.user_settings import render_user_settings_footer

    from market_data import get_latest_quotes

except Exception as _e:
    # Capture the error and provide minimal placeholders so the module loads.
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

    seed_neon_users_from_local = None  # type: ignore
    load_users = lambda: {}  # type: ignore
    save_run = save_daily_snapshot = list_runs = load_run_results = None  # type: ignore

    get_user_settings = None
    upsert_user_settings = None

    def _missing(*args, **kwargs):
        st.error(
            "A required module failed to import. "
            "Login is available, but the app cannot run until imports are fixed.\n\n"
            f"Import error: {_IMPORT_ERROR}"
        )
        st.stop()

    render_admin_users_panel = _missing  # type: ignore
    render_history_expander = _missing  # type: ignore
    render_results = _missing  # type: ignore
    get_results_df = lambda: None  # type: ignore
    render_scan_controls = _missing  # type: ignore
    render_three_step_scanner = _missing  # type: ignore
    render_universe_panel = _missing  # type: ignore
    init_universe_state = _missing  # type: ignore
    render_filters = _missing  # type: ignore
    render_db_status_badge = lambda *a, **k: None  # type: ignore
    render_header = _missing  # type: ignore
    render_price_ticker = lambda *a, **k: None  # type: ignore
    render_market_snapshot = _missing  # type: ignore
    render_prebreakout_tab = _missing  # type: ignore
    render_results_tabs = _missing  # type: ignore
    prepare_results_with_earnings = _missing  # type: ignore
    render_earnings_controls = _missing  # type: ignore
    render_footer = lambda *a, **k: None  # type: ignore
    render_watchlists_panel = _missing  # type: ignore
    render_user_settings_footer = _missing  # type: ignore

    def get_latest_quotes(*args, **kwargs):
        return {}
# --------------- Earnings (shared implementation) ----------------
# Prefer UI-layer earnings helpers; fall back to repo-only DB helpers.
# NOTE: db.earnings is DB/repo logic and may not include UI render functions.
try:
    # Primary (recommended): UI module provides render + helpers
    from ui.earnings import (
        EARN_COL_DAYS,
        add_earnings_days_column,
        fetch_earnings_this_week,
        render_earnings_this_week_panel,
    )
except Exception:
    try:
        # Legacy single-module implementation (older builds)
        from earnings import (
            EARN_COL_DAYS,
            add_earnings_days_column,
            fetch_earnings_this_week,
            render_earnings_this_week_panel,
        )
    except Exception:
        # DB-only fallback: keep the app alive even if UI earnings module is missing.
        # IMPORTANT: import lazily to avoid app startup failures when db deps are missing.
        EARN_COL_DAYS = "earnings_in_days"

        def add_earnings_days_column(df: pd.DataFrame) -> pd.DataFrame:
            try:
                from db.earnings import add_earnings_days_column as _impl  # type: ignore
                return _impl(df)
            except Exception:
                # No-op fallback: return df unchanged
                return df

        def fetch_earnings_this_week(*args, **kwargs):
            try:
                from db.earnings import fetch_earnings_this_week as _impl  # type: ignore
                return _impl(*args, **kwargs)
            except Exception:
                return []

        def render_earnings_this_week_panel(*args, **kwargs):
            st.info(
                "Earnings panel not available (earnings module unavailable or DB not configured)."
            )

# --------------- Tier Sync (DB-first tier resolver) ----------------
# Uses DB as source-of-truth (Stripe webhooks write to DB), with safe fallback to legacy behavior.
try:
    from tier_sync import resolve_user_tier  # type: ignore
except Exception:
    resolve_user_tier = None  # type: ignore


def _resolve_tier_state(username: str, users_map: dict) -> dict:
    """Resolve tier + debug info.

    Priority:
      1) tier_sync.resolve_user_tier (DB-first)
      2) legacy get_user_tier(username, users_map)

    Returns dict keys:
      tier_obj, tier_key, forced_tier_key, db_user_debug, db_tier_err
    """
    # Legacy baseline
    tier_obj = get_user_tier(username, users_map)
    tier_key = (_tier_key(tier_obj) or "basic").strip().lower()

    state = {
        "tier_obj": tier_obj,
        "tier_key": tier_key,
        "forced_tier_key": None,
        "db_user_debug": None,
        "db_tier_err": None,
    }

    # If Tier Sync isn't available, keep legacy behavior
    if not callable(resolve_user_tier):
        return state

    try:
        # Tier Sync should support this call contract
        res = resolve_user_tier(
            username=username,
            users_map=users_map,
            Tier=Tier,
            get_user_tier=get_user_tier,
            get_db_conn=_get_db_conn_for_app,
            admin_users=ADMIN_USERS,
        )

        # Normalize outputs (dict preferred)
        if isinstance(res, dict):
            forced = res.get("forced_tier_key") or res.get("forced_tier") or res.get("db_tier")
            if forced:
                forced = str(forced).strip().lower()

            tier_obj2 = res.get("tier_obj") or res.get("tier") or tier_obj
            tier_key2 = res.get("tier_key") or (_tier_key(tier_obj2) if tier_obj2 is not None else None)
            tier_key2 = str(tier_key2 or tier_key).strip().lower()

            state["tier_obj"] = tier_obj2
            state["tier_key"] = tier_key2
            state["forced_tier_key"] = forced
            state["db_user_debug"] = res.get("db_user_debug") or res.get("db_user")
            state["db_tier_err"] = res.get("db_tier_err") or res.get("error")
            return state

        # Tuple/list fallback: (tier_obj, tier_key, optional_debug_dict)
        if isinstance(res, (tuple, list)):
            if len(res) >= 1 and res[0] is not None:
                state["tier_obj"] = res[0]
            if len(res) >= 2 and res[1]:
                state["tier_key"] = str(res[1]).strip().lower()
            if len(res) >= 3 and isinstance(res[2], dict):
                dbg = res[2]
                forced = dbg.get("forced_tier_key") or dbg.get("forced_tier") or dbg.get("db_tier")
                if forced:
                    state["forced_tier_key"] = str(forced).strip().lower()
                state["db_user_debug"] = dbg.get("db_user_debug") or dbg.get("db_user")
                state["db_tier_err"] = dbg.get("db_tier_err") or dbg.get("error")
            return state

        return state

    except Exception as e:
        state["db_tier_err"] = str(e)
        return state


def _is_admin_user(username: str | None, tier_obj: object | None) -> bool:
    return is_admin_user(username, tier_obj, admin_users=ADMIN_USERS)


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
            if st.button("🚀 Upgrade to Pro", key="upgrade_to_pro", width="stretch"):
                st.session_state["pricing_focus"] = "pro"
                st.switch_page("pages/billing.py")
        with c2:
            if st.button("⭐ Upgrade to Premium", key="upgrade_to_premium", width="stretch"):
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
        with _quiet_external_calls():
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
        with _quiet_external_calls():
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


def _render_market_snapshot_legacy() -> None:
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
        with _quiet_external_calls():
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
            with _quiet_external_calls():
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


def _render_price_ticker_legacy():
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

def _normalize_results_to_df(obj: object) -> pd.DataFrame | None:
    """Normalize load_run_results output to a DataFrame (or None).

    Historical rows may store results as:
      - pandas.DataFrame
      - list[dict]
      - dict
      - JSON-serialized string
    """
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        return None if obj.empty else obj

    if isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
            return None if df.empty else df
        except Exception:
            return None

    if isinstance(obj, dict):
        try:
            df = pd.DataFrame([obj])
            return None if df.empty else df
        except Exception:
            return None

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        try:
            import json
            parsed = json.loads(s)
        except Exception:
            return None
        return _normalize_results_to_df(parsed)

    return None

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
    # If auth import failed, show a clear error instead of a blank screen.
    if _AUTH_IMPORT_ERROR:
        st.error("Auth failed to load; cannot continue.")
        st.code(_AUTH_IMPORT_ERROR)
        st.stop()

    try:
        authed, username, display_name = auth_ui()
    except Exception as e:
        st.error("Login failed to render due to an auth error.")
        try:
            st.exception(e)
        except Exception:
            st.caption(f"Auth error: {type(e).__name__}: {e}")
        st.stop()

    if not authed:
        # Not logged in: show only the login card (auth_ui handles it)
        st.stop()

    # If non-auth modules failed to import, surface the error after login.
    # This ensures users can still log in and we get a visible failure reason.
    if _IMPORT_ERROR:
        st.error(
            "Login succeeded, but the app failed to initialize due to an import error.\n\n"
            f"Import error: {_IMPORT_ERROR}"
        )
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
    # -------- Load Users + Tier (DB-first via Tier Sync) --------
    users_map = load_users()

    # Resolve tier using Tier Sync (DB-first), with legacy fallback
    tier_state = _resolve_tier_state(username, users_map)
    tier = tier_state["tier_obj"]
    forced_tier_key = tier_state.get("forced_tier_key")
    db_tier_err = tier_state.get("db_tier_err")
    db_user_debug = tier_state.get("db_user_debug")

    # Admin + tier key
    is_admin = _is_admin_user(username, tier)
    tier_key = (tier_state.get("tier_key") or "basic").strip().lower()

    # If the tier changed since last render, invalidate cached entitlements so UI unlocks immediately.
    prev_key = (st.session_state.get("tier_key") or "").strip().lower()
    if prev_key and prev_key != tier_key:
        st.session_state.pop("entitlements", None)

    # If the tier object doesn't reflect the resolved key, build a tiny proxy for gating.
    tier_for_flags = tier
    try:
        if _tier_key(tier) != tier_key:
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

    # ---------------- Admin-only: Build stamp (lightweight) ----------------
    # Keep this tiny so admin diagnostics never interfere with scan execution.
    if bool(st.session_state.get("is_admin")):
        try:
            build_mtime = datetime.fromtimestamp(Path(__file__).stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            build_mtime = "unknown"

        st.sidebar.caption(f"build: {build_mtime} | user: {username} | tier_key: {tier_key}")

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
        get_user_settings=get_user_settings,
        upsert_user_settings=upsert_user_settings,
    )

    # -------- Admin: allow larger scans / full universe --------
    # Keep this override in app.py so admin can test at scale even if UI defaults are capped.
    if bool(st.session_state.get("is_admin")):
        # Use very large caps instead of None so downstream scan code that expects ints doesn't break.
        # (We can later switch to None-unlimited once ui/scans.py fully supports it.)
        _ADMIN_SCAN_CAP = 100_000
        _ADMIN_TOP_N = 10_000
        try:
            max_nasdaq_scan = _ADMIN_SCAN_CAP
            max_combo_scan = _ADMIN_SCAN_CAP
            top_n = _ADMIN_TOP_N
        except Exception:
            pass

    # -------- Market Snapshot (moved back up near the top) --------
    # Render early so it appears above scans/results like before.
    # It can render with or without a recent results_df.
    _snapshot_df: pd.DataFrame | None = None
    try:
        _snapshot_df = get_results_df()
        if _snapshot_df is not None and getattr(_snapshot_df, "empty", False):
            _snapshot_df = None
    except Exception:
        _snapshot_df = None

    try:
        st.session_state["latest_results_df"] = _snapshot_df
    except Exception:
        pass

    # 🔄 Clear stale selection state after a new scan
    for k in (
        "results_selected_ticker",
        "results_chart_picker",
        "results_chart_picker_fast",
    ):
        st.session_state.pop(k, None)

    try:
        render_market_snapshot(results_df=_snapshot_df)
    except TypeError:
        _render_market_snapshot_legacy()

    st.markdown("---")

    # -------- Watchlists --------
    watch_id, watch_tickers = render_watchlists_panel(username)
    st.session_state["active_watchlist_id"] = watch_id
    st.session_state["active_watchlist_tickers"] = watch_tickers

    render_earnings_controls(
        flags=flags,
        render_earnings_this_week_panel=render_earnings_this_week_panel,
    )

    # -------- Scan Controls --------
    render_scan_controls(
        can_scan_sp500=flags["can_scan_sp500"],
        can_scan_nasdaq=flags["can_scan_nasdaq"],
        max_nasdaq_scan=int(max_nasdaq_scan) if max_nasdaq_scan is not None else 0,
        max_combo_scan=int(max_combo_scan) if max_combo_scan is not None else 0,
        min_gap=float(min_gap),
        apply_gap_filter=bool(apply_gap_filter),
        min_price=float(min_price),
        max_price=float(max_price),
        top_n=int(top_n) if top_n is not None else 0,
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        unusual_vol=bool(unusual_vol),
        diagnostics=bool(diagnostics),
        username=username,
    )

    # ✅ Force results refresh after a scan completes (prevents blank / stale results)
    if st.session_state.pop("force_results_refresh", False):
        # Best effort: clear only results cache if available
        try:
            get_results_df.clear()  # works if get_results_df is @st.cache_data
        except Exception:
            # Fallback: clear all cache_data (safe but broader)
            try:
                st.cache_data.clear()
            except Exception:
                pass
        st.rerun()

    st.markdown("## 🚀 EZ 3-Step AI Scanner")
    render_three_step_scanner()
    st.markdown("---")

    df = get_results_df()
    df, scan_ran_at = prepare_results_with_earnings(
        df,
        flags=flags,
        earn_col_days=EARN_COL_DAYS,
        add_earnings_days_column=add_earnings_days_column,
        quiet_external_calls=_quiet_external_calls,
    )

    render_results_tabs(
        df=df,
        flags=flags,
        scan_ran_at=scan_ran_at,
        username=username,
        db_status=db_status,
        admin_users=ADMIN_USERS,
        list_runs=list_runs,
        load_run_results=load_run_results,
        render_results=render_results,
        render_prebreakout_tab=render_prebreakout_tab,
        render_admin_users_panel=render_admin_users_panel,
        render_chart_for_ticker=render_chart_for_ticker,
        generate_ai_note=generate_ai_note,
        fetch_earnings_this_week=fetch_earnings_this_week,
        get_db_conn=_get_db_conn_for_app,
        normalize_results_to_df=_normalize_results_to_df,
    )

# ============================================================
#                     APP ENTRYPOINT
# ============================================================

# Streamlit executes this script top-to-bottom.
# Ensure main() is always called and errors are surfaced in the UI.
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ App failed during startup.")
        try:
            st.exception(e)
        except Exception:
            st.write(f"{type(e).__name__}: {e}")
        st.stop()
