"""DB status badge UI helper."""

import streamlit as st

from db.engine import get_db_status


def render_db_status_badge(show_badge: bool = True) -> str:
    """Compute the DB status and (optionally) show the sidebar badge.

    The badge is internal infrastructure detail, so callers should pass
    show_badge=True only for admins. The status string is always returned for
    downstream logic regardless.

    Returns:
        "neon", "sqlite", or "none" depending on availability.
    """
    try:
        db_status = get_db_status()
    except Exception:
        db_status = "none"

    if show_badge:
        if db_status == "neon":
            st.sidebar.markdown("🟢 **DB:** Neon (cloud)")
        elif db_status == "sqlite":
            st.sidebar.markdown("🟡 **DB:** Local SQLite")
        else:
            st.sidebar.markdown("🔴 **DB:** Unavailable")

    return db_status