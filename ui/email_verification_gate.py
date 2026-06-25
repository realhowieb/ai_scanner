"""Soft email-verification gate: block upgrades until the email is verified.

Basic browsing stays open; only paid-upgrade and outbound email are gated.
Fail-open: if the DB or verification module is unavailable, do not block.
"""
from __future__ import annotations


def email_is_verified(username: str) -> bool:
    """True if verified, or if we can't tell (fail-open)."""
    try:
        from db.email_verification import is_email_verified
        return bool(is_email_verified(username))
    except Exception:
        return True


def _resend_verification(email: str) -> bool:
    """Mint a fresh token and email it. Returns True on send."""
    try:
        from config import APP_BASE_URL
        from db.email_verification import create_verification_token
        from ui.email_utils import send_verification_email
        token = create_verification_token(email)
        if not token:
            return False
        url = f"{APP_BASE_URL.rstrip('/')}/verify_email?token={token}"
        return bool(send_verification_email(to_address=email, verify_url=url))
    except Exception:
        return False


def require_verified_for_upgrade(email: str, *, key_suffix: str = "") -> bool:
    """Return True if the user may proceed to checkout.

    When unverified, renders a notice + "resend verification" button and
    returns False so the caller blocks the upgrade.
    """
    import streamlit as st

    if not email or email_is_verified(email):
        return True

    st.warning(
        "📧 Please verify your email before upgrading. "
        "Check your inbox for the verification link."
    )
    if st.button("✉️ Resend verification email", key=f"resend_verify_{key_suffix}"):
        if _resend_verification(email):
            st.success("Verification email sent. Check your inbox (and spam).")
        else:
            st.warning("Could not send the verification email. Try again shortly.")
    return False
