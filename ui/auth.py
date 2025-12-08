# ui/auth.py

import streamlit as st
import yaml
import streamlit_authenticator as stauth
from pathlib import Path


# assume your authenticator + USERS_DB setup is already defined above:
# authenticator = stauth.Authenticate(...)


LOGO_PATH = Path("assets/marketpulse_ai_logo.png")  # adjust path if needed

# Load authentication configuration and create a global authenticator instance
# Try a few likely locations for config.yaml so imports don't crash if it's missing.
_config = None
CONFIG_PATH = None

_candidate_paths = [
    Path(__file__).resolve().parent.parent / "config.yaml",  # ai_scanner/config.yaml
    Path(__file__).resolve().parents[2] / "config.yaml",      # project root/config.yaml
    Path("config.yaml"),                                      # current working dir
]

for candidate in _candidate_paths:
    if candidate.is_file():
        CONFIG_PATH = candidate
        break

if CONFIG_PATH is not None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

if _config:
    authenticator = stauth.Authenticate(
        _config.get("credentials", {}),
        _config.get("cookie", {}).get("name", "mpai_auth"),
        _config.get("cookie", {}).get("key", "mpai_secret"),
        _config.get("cookie", {}).get("expiry_days", 30),
        _config.get("preauthorized", {}),
    )
else:
    # Fall back to no authenticator; the UI will show a friendly message instead of crashing.
    authenticator = None


def auth_ui():
    """
    Render the branded MarketPulse AI login screen.

    Returns:
        (authed: bool, username: str | None, display_name: str)
    """
    # ---------- Global page padding tweaks just for the login view ----------
    st.markdown(
        """
        <style>
        /* tighten top padding on the login page */
        .block-container {
            padding-top: 4rem !important;
        }
        .mp-login-card {
            background: #101218;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            border-radius: 18px;
            border: 1px solid #2b2f3a;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55);
        }
        .mp-login-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Main horizontal layout: logo | login card ----------
    # Using a little extra width for the form to balance visual weight
    col_logo, col_form = st.columns([1, 1.15], gap="large")

    with col_logo:
        st.markdown(" ")  # small spacer
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_column_width=False, width=360)
        else:
            st.markdown("### MarketPulse AI")

        st.markdown(
            "<p style='margin-top: 1.5rem; font-size: 1.1rem; color:#d0d4e4;'>"
            "Welcome back to <strong>MarketPulse AI</strong>."
            "</p>",
            unsafe_allow_html=True,
        )

    with col_form:
        st.markdown('<div class="mp-login-card">', unsafe_allow_html=True)

        st.markdown('<div class="mp-login-title">Login</div>', unsafe_allow_html=True)

        # ----- Streamlit-authenticator form -----
        if authenticator is None:
            st.error("Authentication is not configured. Please contact support.")
            name, auth_status, username = None, None, None
        else:
            name, auth_status, username = authenticator.login(
                "Login",
                "main",
                # You can keep / remove kwargs you were already using (e.g. location)
            )

        # Inline status message *inside* the card instead of full-width bar
        if auth_status is False:
            st.error("Username or password is incorrect.")
        elif auth_status is None:
            st.info("Please enter your credentials.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Return values for app.py ----------
    if auth_status:
        # Normalise values for rest of the app
        user_id = username or name
        display_name = name or (username or "")
        st.session_state["username"] = user_id
        return True, user_id, display_name

    return False, None, ""
# ui/auth.py

from pathlib import Path
import typing as t

import streamlit as st

# Optional imports – we fall back gracefully if they are missing or misconfigured.
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

try:
    import streamlit_authenticator as stauth  # type: ignore
except ImportError:
    stauth = None  # type: ignore


LOGO_PATH = Path("assets/marketpulse_ai_logo.png")  # adjust path if needed


# -------------------------------------------------------------------
# Config loading (optional)
# -------------------------------------------------------------------

def _load_config() -> dict | None:
    """Best-effort load of config.yaml.

    Returns None if:
      - file does not exist
      - yaml is not installed
      - yaml parsing fails
    """
    if yaml is None:
        return None

    candidate_paths = [
        Path(__file__).resolve().parent.parent / "config.yaml",  # ai_scanner/config.yaml
        Path(__file__).resolve().parents[2] / "config.yaml",     # project-root/config.yaml
        Path("config.yaml"),                                     # CWD/config.yaml
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
            except Exception:
                # Bad YAML or unreadable file – just treat as no config.
                return None

    return None


_config = _load_config()


def _build_streamlit_authenticator() -> t.Any | None:
    """Construct a streamlit_authenticator.Authenticate instance if possible."""
    if stauth is None or _config is None:
        return None

    creds = _config.get("credentials")
    cookie = _config.get("cookie", {})
    preauth = _config.get("preauthorized", {})

    if not isinstance(creds, dict) or not isinstance(cookie, dict):
        return None

    try:
        authenticator = stauth.Authenticate(
            credentials=creds,
            cookie_name=cookie.get("name", "mp_ai_cookie"),
            key=cookie.get("key", "mp_ai_key"),
            cookie_expiry_days=cookie.get("expiry_days", 30),
            preauthorized=preauth,
        )
        return authenticator
    except Exception:
        # Mis-configured authenticator; fall back to simple auth.
        return None


# This may be None; auth_ui() will handle that.
_authenticator = _build_streamlit_authenticator()


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def auth_ui(authenticator: t.Any | None = None) -> tuple[bool, t.Optional[str], t.Optional[str]]:
    """
    Render the branded MarketPulse AI login screen.

    Returns:
        (authed: bool, username: str | None, display_name: str | None)
    """
    auth = authenticator or _authenticator

    # ---------- Global page padding tweaks just for the login view ----------
    st.markdown(
        """
        <style>
        /* tighten top padding on the login page */
        .block-container {
            padding-top: 4rem !important;
        }
        .mp-login-card {
            background: #101218;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            border-radius: 18px;
            border: 1px solid #2b2f3a;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55);
        }
        .mp-login-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Main horizontal layout: logo | login card ----------
    col_logo, col_form = st.columns([1, 1.15], gap="large")

    with col_logo:
        st.markdown(" ")  # small spacer
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_column_width=False, width=360)
        else:
            st.markdown("### MarketPulse AI")

        st.markdown(
            "<p style='margin-top: 1.5rem; font-size: 1.1rem; color:#d0d4e4;'>"
            "Welcome back to <strong>MarketPulse AI</strong>."
            "</p>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------
    # Right-hand side: login card
    # --------------------------------------------------------------
    with col_form:
        st.markdown('<div class="mp-login-card">', unsafe_allow_html=True)
        st.markdown('<div class="mp-login-title">Login</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Path 1: Use streamlit_authenticator if properly configured
        # ------------------------------------------------------------------
        name = None
        username = None
        auth_status = None

        if auth is not None:
            try:
                name, auth_status, username = auth.login("Login", "main")
                # Force hide authenticator warnings so fallback form displays
                if auth_status is None:
                    raise Exception("skip to fallback")
            except Exception as e:
                # If authenticator blows up, show warning and fall back to simple auth.
                st.warning(f"Authenticator error: {e}. Falling back to basic login…")
                auth = None
                auth_status = None

        # ------------------------------------------------------------------
        # Path 2: Simple built-in login (always available)
        # ------------------------------------------------------------------
        if auth is None:
            # One-time session state init
            if "simple_auth_ok" not in st.session_state:
                st.session_state.simple_auth_ok = False
                st.session_state.simple_username = None

            # Already logged in?
            if st.session_state.simple_auth_ok:
                username = st.session_state.simple_username or "demo"
                name = username
                auth_status = True
            else:
                with st.form("simple_login_form"):
                    username_input = st.text_input("Username")
                    password_input = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Login")

                if submitted:
                    # DEV MODE: allow any username, fixed password "demo"
                    if password_input == "demo" and username_input.strip():
                        st.session_state.simple_auth_ok = True
                        st.session_state.simple_username = username_input.strip()
                        username = username_input.strip()
                        name = username
                        auth_status = True
                        st.success(f"Logged in as {username} ✅")
                    else:
                        auth_status = False
                        st.error("Invalid username or password.")
                else:
                    auth_status = None

        # Inline status message *inside* the card instead of full-width bar
        if auth_status is False:
            if auth is not None:
                # streamlit_authenticator handled this path
                st.error("Username or password is incorrect.")
            # simple auth error already shows its own message
        elif auth_status is None:
            st.info("Please enter your credentials.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Return values for app.py ----------
    if auth_status:
        user_id = username or name or "user"
        display_name = name or user_id
        st.session_state["username"] = user_id
        return True, user_id, display_name

    return False, None, None