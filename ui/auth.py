"""Authentication UI module.

Provides:
    - auth_ui(): returns (authenticated: bool, username: Optional[str], display_name: Optional[str])
"""

from typing import Tuple, Optional, Dict

import streamlit as st

try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None

from db.users import load_users


def banner(msg: str, level: str = "info") -> None:
    """Lightweight banner helper for showing auth-related messages."""
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def auth_ui() -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (authenticated, username, display_name)."""
    # Demo mode when streamlit-authenticator isn't available
    if stauth is None:
        banner("streamlit-authenticator not installed. Running in DEMO mode.", "warning")
        return True, "howard", "Howard"

    users_map: Dict[str, Dict[str, str]] = load_users()
    usernames = list(users_map.keys())

    authenticator = stauth.Authenticate(
        {"usernames": {u: {"name": users_map[u]["name"], "password": users_map[u]["password"]} for u in usernames}},
        "breakout_scanner_cookie",
        "breakout_scanner_signature",
        cookie_expiry_days=7,
    )

    # New API (v0.3+): login() returns None for rendered locations; values are in st.session_state
    try:
        authenticator.login(
            "main",
            fields={
                "Form name": "Login",
                "Username": "Username",
                "Password": "Password",
                "Login": "Login",
            },
        )
    except Exception as e:
        banner(f"Auth error: {e}", "error")
        return False, None, None

    auth_status = st.session_state.get("authentication_status", None)
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        banner("Username/password incorrect", "error")
        return False, None, None
    if auth_status is None:
        banner("Please enter your credentials.", "info")
        return False, None, None

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    return True, username, name