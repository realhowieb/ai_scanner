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
    if not users_map:
        banner("No users found in auth backend. Falling back to DEMO login.", "warning")
        return True, "howard", "Howard"

    # Build credentials dict for streamlit-authenticator.
    # Handle both plain-text and already-hashed passwords.
    credentials: Dict[str, Dict[str, Dict[str, str]]] = {"usernames": {}}
    for username, data in users_map.items():
        display_name = data.get("name") or data.get("full_name") or username
        stored_password = data.get("password", "")

        # If the password already looks like a bcrypt hash (starts with "$2"),
        # use it as-is. Otherwise, hash it on the fly so authenticator can
        # validate against the plain-text password the user types.
        if isinstance(stored_password, str) and stored_password.startswith("$2"):
            hashed_password = stored_password
        else:
            try:
                hashed_password = stauth.Hasher([stored_password]).generate()[0]
            except Exception:
                hashed_password = stored_password  # last-resort fallback

        credentials["usernames"][username] = {
            "name": display_name,
            "password": hashed_password,
        }

    authenticator = stauth.Authenticate(
        credentials,
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