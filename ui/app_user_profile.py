from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

APP_PROFILE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
)

ADMIN_SCAN_CAP = 100_000
ADMIN_TOP_N = 10_000


def render_admin_build_stamp(*, app_file: str, username: str, tier_key: str) -> None:
    """Render a tiny admin-only build stamp."""
    if not bool(st.session_state.get("is_admin")):
        return
    try:
        build_mtime = datetime.fromtimestamp(Path(app_file).stat().st_mtime, tz=timezone.utc).isoformat()
    except APP_PROFILE_ERRORS:
        build_mtime = "unknown"
    st.sidebar.caption(f"build: {build_mtime} | user: {username} | tier_key: {tier_key}")


def load_saved_user_settings(
    *,
    username: str,
    get_user_settings: Callable[[str], dict[str, Any] | None] | None,
    is_admin: bool,
) -> None:
    """Seed Streamlit session state from persisted per-user settings."""
    if not username or not callable(get_user_settings):
        return

    safe_username = (username or "").strip()
    if st.session_state.get("profile_loaded_for_user") == safe_username:
        return

    try:
        saved = get_user_settings(safe_username)
    except APP_PROFILE_ERRORS:
        saved = None

    if not saved:
        return

    direct_keys = (
        "universe",
        "min_price",
        "max_price",
        "min_dollar_vol",
        "include_ta",
        "apply_gap_filter",
        "min_gap",
        "top_n",
        "max_nasdaq_scan",
        "max_combo_scan",
        "premarket",
        "afterhours",
        "unusual_vol",
    )
    for key in direct_keys:
        value = saved.get(key)
        if value is not None:
            st.session_state[key] = value

    if saved.get("show_diagnostics_ui") is not None:
        st.session_state["show_diagnostics_ui"] = bool(saved["show_diagnostics_ui"]) and bool(is_admin)

    st.session_state["profile_loaded_for_user"] = safe_username


def render_account_sidebar(
    *,
    display_name: str | None,
    username: str,
    tier: object,
    is_admin: bool,
    forced_tier_key: object,
    db_tier_err: object,
    db_user_debug: object,
    render_sidebar_upgrade_card: Callable[..., Any],
    has_min_tier: Callable[..., bool],
    logout_and_reset_session: Callable[[], Any],
) -> None:
    """Render account identity, plan, tier debug, upgrade CTA, and logout."""
    name_label = _account_label(display_name, username)
    st.sidebar.markdown(f"### 👤 {name_label}")
    st.sidebar.markdown(
        f"**Plan:** `{ 'Admin' if is_admin else getattr(tier, 'name', st.session_state.get('tier_key', 'basic')) }`"
    )
    if is_admin:
        st.sidebar.caption(
            f"debug: db_tier={forced_tier_key or '-'} session_tier_key={st.session_state.get('tier_key')}"
        )
        if db_tier_err:
            st.sidebar.error(f"DB tier lookup error: {db_tier_err}")
        elif db_user_debug is not None:
            st.sidebar.caption(f"DB user: {db_user_debug}")

    render_sidebar_upgrade_card(tier, has_min_tier=has_min_tier)

    if st.sidebar.button("Log out", key="logout_button"):
        logout_and_reset_session()


def _account_label(display_name: str | None, username: str) -> str:
    raw_display = (display_name or "").strip()
    raw_username = (username or "").strip()
    if raw_display:
        return raw_display
    if "@" in raw_username:
        return raw_username.split("@")[0]
    return raw_username or "Account"


def apply_admin_scan_caps(
    *,
    max_nasdaq_scan: int,
    max_combo_scan: int,
    top_n: int,
    is_admin: bool,
) -> tuple[int, int, int]:
    """Expand scan caps for admins while leaving non-admin values untouched."""
    if not is_admin:
        return max_nasdaq_scan, max_combo_scan, top_n
    return ADMIN_SCAN_CAP, ADMIN_SCAN_CAP, ADMIN_TOP_N


def load_latest_results_snapshot(get_results_df: Callable[[], pd.DataFrame | None]) -> pd.DataFrame | None:
    """Best-effort latest results snapshot for market header rendering."""
    try:
        snapshot_df = get_results_df()
        if snapshot_df is not None and getattr(snapshot_df, "empty", False):
            return None
        return snapshot_df
    except APP_PROFILE_ERRORS:
        return None


def set_latest_results_snapshot(snapshot_df: pd.DataFrame | None) -> None:
    try:
        st.session_state["latest_results_df"] = snapshot_df
    except APP_PROFILE_ERRORS:
        return
