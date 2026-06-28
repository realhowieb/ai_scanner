from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def render_user_settings_footer(
    username: str | None,
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    diagnostics: bool | None = None,
    get_user_settings: Callable[..., Any] | None = None,
    upsert_user_settings: Callable[..., Any] | None = None,
) -> None:
    """Render sidebar controls for saving and reloading per-user scan defaults."""
    session_username = username or st.session_state.get("username") or st.session_state.get("user")
    if session_username:
        st.session_state["username"] = session_username

    # Debug status line only when diagnostics are on (kept out of the default UI).
    if st.session_state.get("show_diagnostics_ui"):
        st.sidebar.caption(
            f"User settings status — user: {session_username or 'not set'}, "
            f"storage: {'available' if callable(upsert_user_settings) else 'unavailable'}"
        )

    if not session_username:
        return

    if callable(upsert_user_settings):
        if st.sidebar.button("💾 Save as my default settings"):
            _save_user_settings(
                session_username=session_username,
                min_price=min_price,
                max_price=max_price,
                diagnostics=diagnostics,
                upsert_user_settings=upsert_user_settings,
            )

    if callable(get_user_settings) and st.sidebar.button("🔄 Reset to saved profile"):
        st.session_state["profile_loaded_for_user"] = None
        st.sidebar.info("Reloading your saved profile...")
        st.rerun()


def _save_user_settings(
    *,
    session_username: str,
    min_price: float | None,
    max_price: float | None,
    diagnostics: bool | None,
    upsert_user_settings: Callable[..., Any],
) -> None:
    min_price_val = min_price if min_price is not None else st.session_state.get("min_price")
    max_price_val = max_price if max_price is not None else st.session_state.get("max_price")
    show_diag_val = diagnostics if diagnostics is not None else st.session_state.get("show_diagnostics_ui")

    try:
        upsert_user_settings(
            user_id=session_username,
            universe=st.session_state.get("universe"),
            min_price=min_price_val,
            max_price=max_price_val,
            min_dollar_vol=st.session_state.get("min_dollar_vol"),
            include_ta=st.session_state.get("include_ta"),
            apply_gap_filter=st.session_state.get("apply_gap_filter"),
            show_diagnostics_ui=show_diag_val,
            min_gap=st.session_state.get("min_gap"),
            top_n=st.session_state.get("top_n"),
            max_nasdaq_scan=st.session_state.get("max_nasdaq_scan"),
            max_combo_scan=st.session_state.get("max_combo_scan"),
            premarket=st.session_state.get("premarket"),
            afterhours=st.session_state.get("afterhours"),
            unusual_vol=st.session_state.get("unusual_vol"),
            enable_earnings_enrichment=st.session_state.get("enable_earnings_enrichment"),
        )
        st.sidebar.success("Default scan settings saved for your account.")
    except Exception as e:
        st.sidebar.error(f"Failed to save default settings: {e}")
