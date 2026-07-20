"""Password reset flow: request a reset link and set a new password via token."""
from __future__ import annotations

import re

import streamlit as st

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PW_MIN_LEN = 8


def _request_reset_form() -> None:
    st.markdown("### Forgot your password?")
    st.caption("Enter your account email and we'll send you a reset link.")

    with st.form("pw_reset_request"):
        email = st.text_input("Email address", placeholder="you@example.com").strip().lower()
        submitted = st.form_submit_button("Send reset link")

    if not submitted:
        return

    if not EMAIL_RE.match(email):
        st.error("Please enter a valid email address.")
        return

    # Look up user by email (username == email in this app)
    try:
        from db.users import get_user_by_username
        user = get_user_by_username(email)
    except Exception:
        user = None

    # Always show success — don't reveal whether the email exists.
    if user:
        _issue_and_send_token(email)
    else:
        # Log silently; show the same message to avoid enumeration.
        pass

    st.success(
        "If that email is registered, a reset link has been sent. "
        "Check your inbox (and spam folder). The link expires in 30 minutes."
    )


def _issue_and_send_token(email: str) -> None:
    try:
        from config import APP_BASE_URL, RESET_TOKEN_TTL_MINUTES
        from db.password_reset import create_reset_token
        from ui.email_utils import send_password_reset_email

        token = create_reset_token(email, ttl_minutes=RESET_TOKEN_TTL_MINUTES)
        if not token:
            return

        reset_url = f"{APP_BASE_URL.rstrip('/')}/reset_password?token={token}"
        send_password_reset_email(email, reset_url)
    except Exception:
        pass


def _set_new_password_form(token: str) -> None:
    st.markdown("### Set a new password")

    with st.form("pw_reset_set"):
        new_pw = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Update password")

    if not submitted:
        return

    if len(new_pw) < _PW_MIN_LEN:
        st.error(f"Password must be at least {_PW_MIN_LEN} characters.")
        return

    if new_pw != confirm:
        st.error("Passwords do not match.")
        return

    try:
        from db.password_reset import consume_reset_token
        username = consume_reset_token(token)
    except Exception:
        username = None

    if not username:
        st.error("This link is invalid or has expired. Please request a new one.")
        return

    try:
        import bcrypt

        from db.users import update_neon_user_password
        hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
        update_neon_user_password(username, hashed)
        st.success("Password updated! You can now log in with your new password.")
        st.page_link("app.py", label="Go to login →")
    except Exception as exc:
        st.error(f"Password update failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Reset Password | HSFinest.AI", page_icon="🔑")
    try:
        from ui.header import render_page_logo

        render_page_logo()
    except Exception:
        pass
    st.title("🔑 Password Reset")

    raw_token = st.query_params.get("token", "")
    # Guard against oversized or malformed tokens before hitting the DB.
    token = (raw_token or "").strip()
    if token and (len(token) > 128 or not token.replace("-", "").replace("_", "").isalnum()):
        st.error("Invalid reset link. Please request a new one.")
        token = ""

    if token:
        _set_new_password_form(token)
    else:
        _request_reset_form()

    st.markdown("---")
    st.page_link("app.py", label="← Back to login")


main()
