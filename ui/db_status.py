"""DB status badge UI helper."""

import streamlit as st

from db.engine import get_db_status


def render_db_status_badge() -> str:
    """Render the DB status badge in the sidebar and return the status string.

    Returns:
        "neon", "sqlite", or "none" depending on availability.
    """
    try:
        db_status = get_db_status()
    except Exception:
        db_status = "none"

    if db_status == "neon":
        st.sidebar.markdown("🟢 **DB:** Neon (cloud)")
    elif db_status == "sqlite":
        st.sidebar.markdown("🟡 **DB:** Local SQLite")
    else:
        st.sidebar.markdown("🔴 **DB:** Unavailable")

    return db_status