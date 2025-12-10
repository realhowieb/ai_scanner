import streamlit as st
import bcrypt
import time
from db.users import load_users, seed_neon_users_from_local, update_neon_user_password


def auth_ui():
    if "username" in st.session_state:
        username = st.session_state["username"]
        display_name = st.session_state.get("display_name", username)
        return True, username, display_name

    # Basic login rate limiting to prevent brute-force attempts
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300  # 5 minutes

    # If the user is currently locked out, show a message and block login
    locked_until = st.session_state.get("login_locked_until")
    if locked_until:
        now = time.time()
        if now < locked_until:
            remaining = int(locked_until - now)
            # Round up to at least 1 second for UX clarity
            if remaining < 1:
                remaining = 1
            st.error(f"Too many failed login attempts. Please try again in {remaining} seconds.")
            return False, None, None
        else:
            # Lockout expired; reset counters
            st.session_state.pop("login_locked_until", None)
            st.session_state.pop("failed_login_attempts", None)

    def _register_failed_login_attempt():
        """Increment failed attempts and apply lockout if threshold exceeded."""
        failed = st.session_state.get("failed_login_attempts", 0) + 1
        st.session_state["failed_login_attempts"] = failed
        if failed >= MAX_FAILED_ATTEMPTS:
            st.session_state["login_locked_until"] = time.time() + LOCKOUT_SECONDS

    login_placeholder = st.empty()
    with login_placeholder.container():
        st.markdown("### 🔐 Login")
        card = st.container(border=True)
        with card:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_clicked = st.button("Login", key="login_button")

    if login_clicked:
        if not username or not password:
            st.error("Please enter username and password.")
            # Do not count this as a brute-force attempt; user simply forgot fields.
            return False, None, None

        # Try to ensure Neon users are seeded from local demo users.
        # If this fails, we still fall back to local USERS_DB via load_users().
        try:
            seed_neon_users_from_local()
        except Exception:
            # Don't hard-fail login just because seeding didn't work.
            pass

        try:
            users = load_users() or {}
        except Exception as e:
            st.error(f"Login failed while loading users: {e}")
            return False, None, None

        user = users.get(username)
        if user is None:
            st.error("User not found.")
            _register_failed_login_attempt()
            return False, None, None

        # Expect user dict to contain a 'password' field.
        # Supports legacy plain-text passwords and new bcrypt hashes, with auto-migration.
        stored_password = user.get("password")
        if stored_password is None:
            st.error("User record is missing a password field.")
            _register_failed_login_attempt()
            return False, None, None

        pwd_bytes = password.encode("utf-8")
        # Normalize stored password to a clean string (handles bytes from DB as well)
        if isinstance(stored_password, (bytes, bytearray)):
            stored_str = stored_password.decode("utf-8", errors="ignore")
        else:
            stored_str = str(stored_password)

        # Helper: does this string look like a bcrypt hash?
        def _looks_bcrypt(s: str) -> bool:
            return s.startswith("$2a$") or s.startswith("$2b$") or s.startswith("$2y$")

        if _looks_bcrypt(stored_str):
            # New-style bcrypt hash
            if not bcrypt.checkpw(pwd_bytes, stored_str.encode("utf-8")):
                st.error("Incorrect password.")
                _register_failed_login_attempt()
                return False, None, None
        else:
            # Legacy plain-text password in DB
            if stored_str != password:
                st.error("Incorrect password.")
                _register_failed_login_attempt()
                return False, None, None

            # Auto-migrate: convert legacy plain-text to bcrypt hash in Neon
            try:
                new_hash = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")
                update_neon_user_password(username, new_hash)
                # Optionally update the in-memory user dict so future checks see the hash
                user["password"] = new_hash
            except Exception:
                # Migration failure should not block a valid login
                pass

        # Success → create session keys based on user record.
        # Use whatever fields exist in the user dict; fall back where needed.
        st.session_state["user_id"] = user.get("id") or user.get("user_id") or username
        st.session_state["username"] = username

        display_name = user.get("display_name", username)
        st.session_state["display_name"] = display_name

        if "plan" in user:
            st.session_state["plan"] = user.get("plan")
        if "tier" in user:
            st.session_state["tier"] = user.get("tier")
        if "is_admin" in user:
            st.session_state["is_admin"] = bool(user.get("is_admin"))

        # Reset failed-attempt tracking on successful login
        st.session_state.pop("failed_login_attempts", None)
        st.session_state.pop("login_locked_until", None)

        # Remove the login form from the screen after successful login.
        login_placeholder.empty()
        # No success banner to keep the UI clean after login.
        return True, username, display_name

    return False, None, None

def logout_and_reset_session() -> None:
    """Clear all app-specific session state and rerun so the login screen is shown again."""
    try:
        # Remove all keys from Streamlit session_state
        keys = list(st.session_state.keys())
        for k in keys:
            del st.session_state[k]
    except Exception:
        # Best-effort cleanup; ignore issues with internal keys
        pass

    # Trigger a full rerun so the app re-evaluates from the top and shows login
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()