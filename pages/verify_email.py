"""Email address verification page — handles ?token=<token> from signup email."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Verify Email | AI Scanner", page_icon="✉️")
    st.title("✉️ Email Verification")

    raw_token = (st.query_params.get("token", "") or "").strip()
    token = raw_token if raw_token and len(raw_token) <= 128 else ""

    if not token:
        st.info(
            "No verification token found in the link. "
            "Check your email for the verification link, or sign up again."
        )
        st.page_link("app.py", label="← Back to login")
        return

    try:
        from db.email_verification import consume_verification_token
        username = consume_verification_token(token)
    except Exception:
        username = None

    if username:
        st.success(
            f"✅ Email verified for **{username}**. "
            "You now have full access to AI Scanner."
        )
    else:
        st.error(
            "This verification link is invalid or has already been used. "
            "Links expire after 24 hours."
        )
        st.caption("Sign up again or contact support if you need help.")

    st.page_link("app.py", label="← Go to login")


main()
