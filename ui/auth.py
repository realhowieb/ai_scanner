import streamlit as st
import bcrypt
from db.users import load_users, seed_neon_users_from_local, update_neon_user_password


def auth_ui():
    if "username" in st.session_state:
        username = st.session_state["username"]
        display_name = st.session_state.get("display_name", username)
        return True, username, display_name

    login_placeholder = st.empty()
    with login_placeholder.container():
        st.markdown("### Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_clicked = st.button("Login", key="login_button")

    if login_clicked:
        if not username or not password:
            st.error("Please enter username and password.")
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
            return False, None, None

        # Expect user dict to contain a 'password' field.
        # Supports legacy plain-text passwords and new bcrypt hashes, with auto-migration.
        stored_password = user.get("password")
        if stored_password is None:
            st.error("User record is missing a password field.")
            return False, None, None

        pwd_bytes = password.encode("utf-8")
        stored_str = str(stored_password)

        # Helper: does this string look like a bcrypt hash?
        def _looks_bcrypt(s: str) -> bool:
            return s.startswith("$2a$") or s.startswith("$2b$") or s.startswith("$2y$")

        if _looks_bcrypt(stored_str):
            # New-style bcrypt hash
            if not bcrypt.checkpw(pwd_bytes, stored_str.encode("utf-8")):
                st.error("Incorrect password.")
                return False, None, None
        else:
            # Legacy plain-text password in DB
            if stored_str != password:
                st.error("Incorrect password.")
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
        # Remove the login form from the screen after successful login.
        login_placeholder.empty()
        # No success banner to keep the UI clean after login.
        return True, username, display_name

    return False, None, None