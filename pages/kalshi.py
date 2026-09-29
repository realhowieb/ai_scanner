"""🪙 Kalshi BTC Monitor — dedicated page.

Read-only directional scanner for Kalshi's up/down BTC event contracts. Its own
page so it can auto-refresh independently of the stock scanner.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Kalshi BTC Monitor", page_icon="🪙", layout="wide")
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # Run 62/P2: before any sign-in gate

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to use the Kalshi BTC scanner.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()


def _is_admin(username: str) -> bool:
    """P2-25: Labs is admin-only. The database is the authority (not session
    state); the answer is cached per session so auto-refresh doesn't re-query."""
    cached = st.session_state.get("_kalshi_admin_check")
    if isinstance(cached, tuple) and cached[0] == username:
        return bool(cached[1])
    try:
        from db.users import is_admin_from_db

        ok = bool(is_admin_from_db(username))
    except Exception:
        ok = False
    st.session_state["_kalshi_admin_check"] = (username, ok)
    return ok


try:
    from ui.header import render_page_logo

    render_page_logo()
except Exception:
    pass

if not _is_admin(_username):
    st.info("Kalshi BTC is an admin-only Labs tool.")
    st.page_link("app.py", label="← Back to scanner", icon="🏠")
    st.stop()

try:
    from ui.kalshi_scanner import render_kalshi_scanner

    render_kalshi_scanner()
except Exception as e:
    from ui.safe_errors import show_error

    show_error("the Kalshi BTC monitor", e)

st.page_link("app.py", label="← Back to scanner", icon="🏠")
