"""Email address verification page — handles ?token=<token> from signup email."""
from __future__ import annotations

import streamlit as st


def _render_resend_box() -> None:
    """Let the user re-send the verification email to their address."""
    st.markdown("---")
    st.subheader("Resend verification email")
    st.caption("Enter your account email and we'll send a fresh verification link.")
    email = st.text_input("Email address", key="resend_verify_email", placeholder="you@example.com")
    if st.button("✉️ Resend verification email", key="resend_verify_btn"):
        addr = (email or "").strip().lower()
        if "@" not in addr:
            st.warning("Enter a valid email address.")
            return
        try:
            from ui.email_verification_gate import _resend_verification

            sent = _resend_verification(addr)
        except Exception:
            sent = False
        if sent:
            st.success("✅ Verification email sent. Check your inbox (and spam).")
        else:
            st.warning(
                "Could not send the verification email. Make sure the address has "
                "an account, and that email is configured."
            )


def main() -> None:
    st.set_page_config(page_title="Verify Email | HSFinest.AI", page_icon="✉️")
    try:
        from ui.header import render_page_logo

        render_page_logo()
    except Exception:
        pass
    st.title("✉️ Email Verification")

    raw_token = (st.query_params.get("token", "") or "").strip()
    token = raw_token if raw_token and len(raw_token) <= 128 else ""

    if not token:
        st.info(
            "No verification token found in the link. "
            "Use the box below to resend it, or check your email for an existing link."
        )
        _render_resend_box()
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
            "You now have full access to HSFinest.AI."
        )
    else:
        st.error(
            "This verification link is invalid or has already been used. "
            "Links expire after 24 hours."
        )
        _render_resend_box()

    st.page_link("app.py", label="← Go to login")


main()
