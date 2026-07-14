from __future__ import annotations

# Print the Python stack on hard crashes (SIGSEGV/SIGABRT): the recurring
# Streamlit Cloud segfault dies without a traceback, so this is the tool that
# finally names the crashing line in the deploy logs. Must run before any
# native-heavy imports.
import faulthandler

faulthandler.enable()

import sys
from pathlib import Path

# Ensure project base directory is importable before local package imports.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st

try:
    from ui.app_boot import (
        configure_page,
        install_streamlit_compat,
    )
    from ui.app_boot import (
        quiet_external_calls as _quiet_external_calls,
    )
except (ImportError, KeyError):
    import importlib.util

    _APP_BOOT_PATH = BASE_DIR / "ui" / "app_boot.py"
    _APP_BOOT_SPEC = importlib.util.spec_from_file_location(
        "_ai_scanner_ui_app_boot",
        _APP_BOOT_PATH,
    )
    if _APP_BOOT_SPEC is None or _APP_BOOT_SPEC.loader is None:
        raise
    _app_boot = importlib.util.module_from_spec(_APP_BOOT_SPEC)
    _APP_BOOT_SPEC.loader.exec_module(_app_boot)
    configure_page = _app_boot.configure_page
    install_streamlit_compat = _app_boot.install_streamlit_compat
    _quiet_external_calls = _app_boot.quiet_external_calls
try:
    from ui.app_runtime import (
        get_market_session,
        render_active_filters_summary,
        render_onboarding_hint,
        render_sidebar_upgrade_card,
    )
    from ui.app_runtime import (
        normalize_results_to_df as _normalize_results_to_df,
    )
    from ui.app_session import (
        compute_entitlements,
        is_admin_user,
        normalize_admin_users,
    )
    from ui.app_session import (
        tier_key as _tier_key,
    )
    from ui.app_user_profile import (
        apply_admin_scan_caps,
        load_latest_results_snapshot,
        load_saved_user_settings,
        render_account_sidebar,
        render_admin_build_stamp,
        set_latest_results_snapshot,
    )
except KeyError:
    # Streamlit Cloud hot-redeploy race: the module table is mid-swap when the
    # watcher re-executes this script, so imports raise KeyError instead of
    # loading (same failure app_boot guards above). Retry a few reruns, then
    # surface the real error instead of looping forever.
    import time as _boot_time

    _boot_retries = int(st.session_state.get("_boot_import_retries", 0))
    st.session_state["_boot_import_retries"] = _boot_retries + 1
    if _boot_retries < 3:
        st.caption("⏳ App is updating — one moment…")
        _boot_time.sleep(2)
        st.rerun()
    raise
st.session_state.pop("_boot_import_retries", None)

install_streamlit_compat()

# Optional error monitoring (no-op unless SENTRY_DSN is set).
try:
    from ui.monitoring import init_sentry

    init_sentry("streamlit")
except Exception:
    pass

from types import SimpleNamespace

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


# --------------- Tiering ----------------
# --------------- Tiering ----------------
try:
    from auth.tiering import (
        ADMIN_USERS,
        USERS_DB,
        Tier,
        get_user_tier,
        has_min_tier,
        require_min_tier,
    )
except Exception:
    from auth.tiering_fallback import ADMIN_USERS, USERS_DB, Tier, get_user_tier

    def has_min_tier(tier_or_key, required: str) -> bool:
        order = {"basic": 0, "pro": 1, "premium": 2, "admin": 3}
        key = _tier_key(tier_or_key) or str(tier_or_key or "basic")
        return order.get(str(key).strip().lower(), 0) >= order.get(str(required).strip().lower(), 0)

    def require_min_tier(tier_or_key, required: str, feature_name: str) -> bool:
        allowed = has_min_tier(tier_or_key, required)
        if not allowed:
            st.info(f"Upgrade to {required.title()} to use {feature_name}.")
        return allowed

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
    from db.runs import list_runs, load_run_results, save_daily_snapshot, save_run
    from db.users import load_users, seed_neon_users_from_local

    # User settings (per-user defaults) – optional Neon-backed feature
    try:
        from db.user_settings import get_user_settings, upsert_user_settings
    except Exception:
        get_user_settings = None
        upsert_user_settings = None

    from ui.admin_users import render_admin_users_panel
    from ui.alerts import render_alerts_panel
    from ui.day_trader import render_day_trader_panel
    from ui.db_status import render_db_status_badge
    from ui.earnings_results import prepare_results_with_earnings, render_earnings_controls
    from ui.filters import render_filters
    from ui.footer import render_footer
    from ui.header import render_header, render_market_snapshot, render_price_ticker
    from ui.history import render_history_expander
    from ui.journal import render_journal_panel
    from ui.prebreakout_tab import render_prebreakout_tab
    from ui.result_explain import add_why_column
    from ui.results import get_results_df, render_results
    from ui.results_tabs import render_results_tabs
    from ui.scans import render_scan_controls, render_three_step_scanner
    from ui.universe_panel import init_universe_state, render_universe_panel
    from ui.user_settings import render_user_settings_footer
    from ui.watchlists import render_watchlists_panel

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
    render_alerts_panel = lambda *a, **k: None  # type: ignore
    render_day_trader_panel = lambda *a, **k: None  # type: ignore
    render_journal_panel = lambda *a, **k: None  # type: ignore
    add_why_column = lambda df: df  # type: ignore
    render_user_settings_footer = _missing  # type: ignore

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
    from auth.tier_sync import resolve_user_tier  # type: ignore
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
    _ = st.session_state.get("authentication_status") is True


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

    # Detect tier changes since last render.
    prev_key = (st.session_state.get("tier_key") or "").strip().lower()
    _tier_rank = {"basic": 0, "pro": 1, "premium": 2, "admin": 3}
    if prev_key and prev_key != tier_key:
        st.session_state.pop("entitlements", None)
        # Downgrade: DB resolved a lower tier than what was in session.
        if _tier_rank.get(tier_key, 0) < _tier_rank.get(prev_key, 0):
            # Clear Premium/Pro-only state that would now be inaccessible.
            for _k in (
                "price_snapshot_id", "snapshot_id",
                "ai_notes", "ai_notes_text", "ai_notes_cache",
                "ai_notes_last", "ai_notes_last_text", "last_ai_notes",
            ):
                st.session_state.pop(_k, None)
            st.warning(
                f"Your plan has changed from **{prev_key.upper()}** to **{tier_key.upper()}**. "
                "Some features have been locked. Visit the Billing page to upgrade."
            )

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

    render_admin_build_stamp(app_file=__file__, username=username, tier_key=tier_key)

    # -------- Load Saved User Settings (if available) --------
    load_saved_user_settings(
        username=username,
        get_user_settings=get_user_settings,
        is_admin=bool(st.session_state.get("is_admin")),
    )

    # -------- Sidebar Account Info --------
    render_account_sidebar(
        display_name=display_name,
        username=username,
        tier=tier,
        is_admin=bool(st.session_state.get("is_admin")),
        forced_tier_key=forced_tier_key,
        db_tier_err=db_tier_err,
        db_user_debug=db_user_debug,
        render_sidebar_upgrade_card=render_sidebar_upgrade_card,
        has_min_tier=has_min_tier,
        logout_and_reset_session=logout_and_reset_session,
    )
    #st.markdown("---")

    # -------- DB Status --------
    db_status = render_db_status_badge()

    # -------- Provider Health (admin diagnostics) --------
    if flags.get("can_diagnostics"):
        try:
            from ui.provider_health import render_provider_health

            with st.expander("🩺 Provider Health", expanded=False):
                render_provider_health()
        except Exception:
            pass

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
    max_nasdaq_scan, max_combo_scan, top_n = apply_admin_scan_caps(
        max_nasdaq_scan=max_nasdaq_scan,
        max_combo_scan=max_combo_scan,
        top_n=top_n,
        is_admin=bool(st.session_state.get("is_admin")),
    )

    # -------- Market Snapshot (moved back up near the top) --------
    # Render early so it appears above scans/results like before.
    # It can render with or without a recent results_df.
    _snapshot_df = load_latest_results_snapshot(get_results_df)
    set_latest_results_snapshot(_snapshot_df)

    # 🔄 Clear stale selection state after a new scan
    for k in (
        "results_selected_ticker",
        "results_chart_picker",
        "results_chart_picker_fast",
    ):
        st.session_state.pop(k, None)

    render_market_snapshot(results_df=_snapshot_df)

    st.markdown("---")

    # -------- Watchlists --------
    watch_id, watch_tickers = render_watchlists_panel(username)

    # -------- Morning pulse: alerts fired · since-yesterday · Day Trader ----
    # One compact block right under the watchlist card wall (the heat strip was
    # retired — the card wall shows the same per-name day read).
    try:
        from ui.notifications import render_alert_bell

        render_alert_bell(username)
    except Exception:
        pass
    try:
        from ui.whats_new import render_whats_new_strip

        render_whats_new_strip()
    except Exception:
        pass
    try:
        st.page_link(
            "pages/day_trader.py",
            label="⚡ Day Trader — live (gappers · VWAP · RVOL, real-time)",
            icon="📈",
        )
    except Exception:
        pass
    st.session_state["active_watchlist_id"] = watch_id
    st.session_state["active_watchlist_tickers"] = watch_tickers
    st.markdown("---")

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

    st.markdown("## 🚀 AI Scanner")
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
    df = add_why_column(df)  # plain-English "why this passed" per row

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
        get_db_conn=_get_db_conn_for_app,
        normalize_results_to_df=_normalize_results_to_df,
    )

    # -------- Alerts: moved to their own page --------
    # The bell at the top of the page surfaces fired alerts; management
    # (create/scorecards/feed) lives on pages/alerts.py. In-context creation
    # survives via 'Alert me on this' in the ticker details.
    st.markdown("---")
    try:
        st.page_link(
            "pages/alerts.py",
            label="🔔 Alerts — create & manage (breakout · watchlist · price · live % move · RVOL)",
            icon="⚙️",
        )
    except Exception:
        pass

    # Trade journal (positions logged from trade plans; hidden until first log).
    try:
        render_journal_panel(username)
    except Exception:
        pass

    # Legal disclaimer footer (financial product) — rendered app-wide.
    try:
        render_footer()
    except Exception:
        pass

# ============================================================
#                     APP ENTRYPOINT
# ============================================================

# Streamlit executes this script top-to-bottom.
# Ensure main() is always called and errors are surfaced in the UI.
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            from ui.monitoring import capture

            capture(e)
        except Exception:
            pass
        st.error("❌ App failed during startup.")
        try:
            st.exception(e)
        except Exception:
            st.write(f"{type(e).__name__}: {e}")
        st.stop()
