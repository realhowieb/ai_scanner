"""Session, tier, and entitlement helpers for the Streamlit app."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

FEATURE_MIN_TIER: dict[str, str] = {
    "can_scan_sp500": "basic",
    "can_scan_nasdaq": "pro",
    "can_premarket": "pro",
    "can_afterhours": "pro",
    "can_unusual_volume": "pro",
    "can_export_csv": "pro",
    "can_earnings": "pro",
    "can_ai_notes": "premium",
    "can_scan_history": "pro",
    "can_track_record": "pro",
    "can_early_breakout": "premium",
    "can_full_universe": "premium",
    "can_paper_trade": "premium",
    "can_email_alerts": "pro",  # in-app alerts are open to all; email is Pro+
    "can_day_trader": "pro",  # live intraday monitor (pages/day_trader.py)
    "can_diagnostics": "admin",
    "can_admin_panel": "admin",
}

# Max number of alerts a user may create, by tier. Basic gets a taste; upgrade
# for more. Admin shares the Premium cap.
ALERT_LIMIT_BY_TIER: dict[str, int] = {
    "basic": 1,
    "pro": 5,
    "premium": 25,
    "admin": 25,
}
TODAY_LANDING_KEY = "hsf_today_landed_for"

# Account-derived values must never survive logout into another user's session.
# Browser convenience preferences (tour, Scanner view/lenses, saved screens) are
# intentionally absent: they belong to the browser, not to an account.
ACCOUNT_SESSION_KEYS = (
    "user_id", "username", "display_name", "plan", "tier", "tier_key",
    "entitlements", "is_admin", "authentication_status",
    "active_watchlist_id", "active_watchlist_tickers",
    "results_df", "last_scan_at", "last_scan_universe", "scan_settings",
    "user_settings", "profile_loaded_for_user", "user_profile_loaded",
    TODAY_LANDING_KEY, "hsf_after_login_page",
    "hsf_stock_opp", "hsf_stock_row", "hsf_stock_ticker",
    "hsf_stock_ticker_input", "hsf_stock_view_selected",
    "hsf_alert_prefill_ticker", "hsf_alert_prefill_event",
    # Run 83 (B2): account-derived state that used to survive logout.
    "active_watchlist_quote_rows", "_watchlist_prior_rows", "_loaded_user_settings",
    "_portal_url", "post_checkout_refreshed", "_tier_poll_attempt",
    "_wl_pending_scan", "_wl_pending_scan_all", "_wl_tools_state",
    "dt_rows", "dt_watch_symbols", "dt_watch_baseline",
    "latest_results_df", "results_signature", "scan_ran_at_utc", "force_results_refresh",
    "earnings_enriched_df", "earnings_enriched_signature",
    "_brief_prior_state", "_brief_prior_opps", "_opp_compare_cache",
    "_nl_pending_explanation", "_nl_pending_filters", "_ai_pending_settings", "_ai_run_pending",
    "hsf_my_watchlist_viewed", "hsf_stock_intelligence_viewed",
    "alert_price_tk", "alert_price_val", "alert_break_thr", "alert_break_wl",
    "alert_ema_tk", "alert_ema_dir", "alert_rvol_tk", "alert_rvol_thr",
    "pt_key", "pt_secret", "_three_step_flash",
    "hsf_start_scanner_after_auth", "hsf_new_signup_scanner_hint", "hsf_restored_session",
)

# Premium AI output keys include ticker/snapshot identifiers, so enumerating
# every possible key is neither complete nor durable. These prefixes are
# account- and entitlement-sensitive; browser convenience preferences are not.
ENTITLEMENT_SENSITIVE_SESSION_PREFIXES = (
    "_ai_summary_",
    "_ai_ticker_",
    "_ai_chat_",
    "brief_narrative_",
    "opp_ai_",
    "aic_explain_",
)
ENTITLEMENT_SENSITIVE_SESSION_KEYS = (
    "ai_notes",
    "ai_notes_text",
    "ai_notes_cache",
    "ai_notes_last",
    "ai_notes_last_text",
    "last_ai_notes",
)

# Identity keys every sign-in path sets for the NEW account in the same run, so
# the identity boundary below keeps them and clears everything else.
IDENTITY_KEYS = ("user_id", "username", "display_name", "tier", "plan", "is_admin",
                 "authentication_status")
# Who the account-scoped session state belongs to. Deliberately NOT cleared on
# logout: it is how a later sign-in as someone else is detected. Stores a hash.
ACCOUNT_OWNER_KEY = "_hsf_account_owner"


def clear_entitlement_sensitive_state(session_state: Any) -> None:
    """Remove cached Premium output without clearing browser preferences."""
    for key in list(session_state.keys()):
        if key in ENTITLEMENT_SENSITIVE_SESSION_KEYS or str(key).startswith(
            ENTITLEMENT_SENSITIVE_SESSION_PREFIXES
        ):
            try:
                session_state.pop(key, None)
            except Exception:
                continue


def clear_account_session_state(session_state: Any, extra_keys: tuple[str, ...] = ()) -> None:
    """Remove account-specific state while preserving browser UI preferences."""
    for key in (*ACCOUNT_SESSION_KEYS, *extra_keys):
        try:
            session_state.pop(key, None)
        except Exception:
            continue
    clear_entitlement_sensitive_state(session_state)


def _owner_tag(username: object) -> str:
    import hashlib

    return hashlib.sha256(str(username or "").strip().lower().encode("utf-8")).hexdigest()[:24]


def enforce_account_boundary(session_state: Any, username: object) -> bool:
    """Run 83 (B2): account isolation by identity, not only by the logout button.

    Call after authentication on every run. When the signed-in account differs
    from the one the session's account state belongs to (logout → another
    sign-in, session restore, any future auth path), clear all account-scoped
    state except the identity the sign-in just established. Returns True when
    state was cleared.
    """
    user = str(username or "").strip().lower()
    if not user:
        return False
    tag = _owner_tag(user)
    owner = session_state.get(ACCOUNT_OWNER_KEY)
    cleared = owner is not None and owner != tag
    if cleared:
        # The shared-link destination was chosen by whoever is signing in now.
        drop = tuple(k for k in ACCOUNT_SESSION_KEYS
                     if k not in IDENTITY_KEYS and k != "hsf_after_login_page")
        for key in drop:
            try:
                session_state.pop(key, None)
            except Exception:
                continue
        clear_entitlement_sensitive_state(session_state)
    session_state[ACCOUNT_OWNER_KEY] = tag
    return cleared


def should_land_on_today(session_state: Any, username: object) -> bool:
    """Return true once per authenticated user session, then remember it."""
    user = str(username or "").strip().lower()
    if not user or session_state.get(TODAY_LANDING_KEY) == user:
        return False
    session_state[TODAY_LANDING_KEY] = user
    return True


def alert_limit_for_tier(tier_key: object | None) -> int:
    """Resolve the max-alerts cap for a tier key (defaults to Basic = 1)."""
    key = str(tier_key or "basic").strip().lower()
    return ALERT_LIMIT_BY_TIER.get(key, 1)

TIER_ORDER = {
    "basic": 0,
    "pro": 1,
    "premium": 2,
    "admin": 3,
}


def norm_str(value: object | None) -> str:
    """Normalize user-provided or DB-provided strings to a safe canonical form."""
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def norm_lower(value: object | None) -> str:
    return norm_str(value).lower()


def normalize_admin_users(admin_users: object) -> set[str]:
    """Normalize configured admin users to lowercase usernames."""
    try:
        if isinstance(admin_users, (list, set, tuple)):
            return {str(user).strip().lower() for user in admin_users}
        if isinstance(admin_users, dict):
            return {str(user).strip().lower() for user in admin_users.keys()}
    except Exception:
        pass
    return set()


def is_admin_user(
    username: str | None,
    tier_obj: object | None,
    *,
    admin_users: object,
) -> bool:
    """Admin check that is resilient to whitespace, case, and tier-object shape."""
    username_norm = norm_lower(username)
    if username_norm and username_norm in normalize_admin_users(admin_users):
        return True

    try:
        if norm_lower(getattr(tier_obj, "key", None)) == "admin":
            return True
    except Exception:
        pass

    try:
        if norm_lower(getattr(tier_obj, "name", None)) == "admin":
            return True
    except Exception:
        pass

    return norm_lower(tier_obj) == "admin"


def tier_key(tier_obj: object | None) -> str:
    """Return a stable tier key string for logging, comparisons, and UI state."""
    try:
        key = getattr(tier_obj, "key", None)
        if key is not None:
            return norm_lower(key)
    except Exception:
        pass

    try:
        name = getattr(tier_obj, "name", None)
        if name is not None:
            return norm_lower(name)
    except Exception:
        pass

    return norm_lower(tier_obj) or "basic"


def _fallback_has_min_tier(tier_obj: object | None, required: str) -> bool:
    current_rank = TIER_ORDER.get(tier_key(tier_obj), 0)
    required_rank = TIER_ORDER.get(norm_lower(required) or "basic", 0)
    return current_rank >= required_rank


def compute_entitlements(
    *,
    tier_obj: object | None,
    is_admin: bool,
    has_min_tier_fn: Callable[[Any, str], bool] | None = None,
) -> dict[str, bool]:
    """Compute deterministic feature flags from tier state."""
    if bool(is_admin):
        return {feature: True for feature in FEATURE_MIN_TIER}

    if has_min_tier_fn is None:
        has_min_tier_fn = _fallback_has_min_tier

    flags: dict[str, bool] = {}
    for feature, min_tier in FEATURE_MIN_TIER.items():
        if min_tier == "admin":
            flags[feature] = False
            continue
        try:
            flags[feature] = bool(has_min_tier_fn(tier_obj, min_tier))
        except Exception:
            flags[feature] = False
    return flags
