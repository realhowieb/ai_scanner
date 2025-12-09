import streamlit as st
from db.users import load_users, seed_neon_users_from_local


def auth_ui():
    st.markdown("### Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        if not username or not password:
            st.error("Please enter username and password.")
            return None

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
            return None

        user = users.get(username)
        if user is None:
            st.error("User not found.")
            return None

        # Expect user dict to contain a 'password' field.
        # Very basic auth: compare raw text (same as your previous system).
        # TODO: replace with hashed passwords later.
        stored_password = user.get("password")
        if stored_password is None:
            st.error("User record is missing a password field.")
            return None

        if password != stored_password:
            st.error("Incorrect password.")
            return None

        # Success → create session keys based on user record.
        # Use whatever fields exist in the user dict; fall back where needed.
        st.session_state["user_id"] = user.get("id") or user.get("user_id") or username
        st.session_state["username"] = username
        if "plan" in user:
            st.session_state["plan"] = user.get("plan")
        if "tier" in user:
            st.session_state["tier"] = user.get("tier")
        if "is_admin" in user:
            st.session_state["is_admin"] = bool(user.get("is_admin"))

        st.success("Logged in successfully. Reloading...")
        st.rerun()

    return None