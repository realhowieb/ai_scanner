"""Authentication UI module.

Provides:
    - auth_ui(): returns (authenticated: bool, username: Optional[str], display_name: Optional[str])
"""

from typing import Tuple, Optional, Dict

import streamlit as st

# streamlit-authenticator is optional so the app can still run in demo mode
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
    """Render the auth UI and return (authenticated, username, display_name).

    Uses streamlit-authenticator when available; otherwise falls back to a demo login.
    """

    # --- DEMO MODE: streamlit-authenticator not installed ---
    if stauth is None:
        banner("streamlit-authenticator not installed. Running in DEMO mode.", "warning")
        return True, "howard", "Howard"

    # --- Load users from Neon / SQLite (or local USERS_DB fallback) ---
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

    # --- Already authenticated? Short-circuit and skip rendering the login form ---
    existing_status = st.session_state.get("authentication_status", None)
    if existing_status is True:
        name = st.session_state.get("name") or st.session_state.get("username")
        username = st.session_state.get("username")
        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
        if username:
            return True, username, name

    # --- First-time / not yet authenticated: render login form ---

    # Tighten top padding and define the login card style
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem !important;
            }
            .auth-card {
                background-color: #111319;
                border-radius: 14px;
                padding: 0rem 2rem 2rem;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
                border: 1px solid rgba(255, 255, 255, 0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Perfectly centered full-width login logo + tagline
    logo_col_left, logo_col_center, logo_col_right = st.columns([1, 2, 1])
    with logo_col_center:
        st.image(
            "assets/market_ai_logo_tighter.png",
            width=380,
            use_container_width=False,
        )

    st.markdown(
        """
        <p style="margin-top:0.3rem; font-size:1rem; color:gray; text-align:center;">
            Sign in to MarketPulse AI
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Center the login form inside a card
    card_left, card_center, card_right = st.columns([1, 2, 1])
    with card_center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

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
            # close the card div so layout isn't broken
            st.markdown("</div>", unsafe_allow_html=True)
            return False, None, None

        st.markdown("</div>", unsafe_allow_html=True)

    auth_status = st.session_state.get("authentication_status", None)
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        banner("Username/password incorrect", "error")
        return False, None, None
    if auth_status is None:
        banner("Please enter your credentials.", "info")
        return False, None, None

    # Successful login: show logout in sidebar and return identity
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    return True, username, name