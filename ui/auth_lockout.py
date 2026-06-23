"""Login lockout helpers for auth UI."""
from __future__ import annotations

import time

import streamlit as st


def is_login_locked() -> bool:
    """Return True if the user is currently locked out due to failed logins."""
    try:
        locked_until = st.session_state.get("login_locked_until")
        if not locked_until:
            return False
        return time.time() < float(locked_until)
    except (TypeError, ValueError):
        return False


def lockout_remaining_seconds() -> int:
    try:
        locked_until = st.session_state.get("login_locked_until")
        if not locked_until:
            return 0
        remaining = int(float(locked_until) - time.time())
        return max(0, remaining)
    except (TypeError, ValueError):
        return 0


def register_failed_login_attempt(max_attempts: int = 5, lockout_seconds: int = 300) -> None:
    """Increment failed attempts and apply lockout if threshold exceeded."""
    try:
        failed = int(st.session_state.get("failed_login_attempts") or 0) + 1
    except (TypeError, ValueError):
        failed = 1

    st.session_state["failed_login_attempts"] = failed
    if failed >= int(max_attempts):
        st.session_state["login_locked_until"] = time.time() + int(lockout_seconds)


def clear_failed_login_attempts() -> None:
    st.session_state.pop("failed_login_attempts", None)
    st.session_state.pop("login_locked_until", None)
