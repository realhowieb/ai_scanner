"""Soft email-verification gate: block upgrades until the email is verified.

Basic browsing stays open; only paid-upgrade and outbound email are gated.
Fail-closed: if the verification status can't be read, the upgrade waits and the
user is told verification is temporarily unavailable (not that they're unverified).
"""
from __future__ import annotations


class VerificationUnavailable(RuntimeError):
    """The verification status couldn't be read right now."""


def verification_status(username: str) -> bool:
    """True if verified, False if not; VerificationUnavailable if we can't tell."""
    try:
        from db.email_verification import email_verification_status
        return bool(email_verification_status(username))
    except Exception as e:
        raise VerificationUnavailable("email verification lookup failed") from e


def email_is_verified(username: str) -> bool:
    """True only when confirmed verified (fail-closed)."""
    try:
        return verification_status(username)
    except VerificationUnavailable:
        return False


def _note(reason: str, detail: str = "") -> None:
    try:
        from ui import email_failure

        email_failure.note(reason, detail)
    except Exception:
        pass


def _clear_note() -> None:
    try:
        from ui import email_failure

        email_failure.clear()
    except Exception:
        pass


def last_failure_for_admin() -> str | None:
    """P2-32: why the last resend failed — only for admins, else None."""
    try:
        import streamlit as st

        if not st.session_state.get("is_admin"):
            return None
        from ui import email_failure

        return email_failure.describe(email_failure.last())
    except Exception:
        return None


def _resend_verification(email: str) -> bool:
    """Mint a fresh token and email it. Returns True on send."""
    _clear_note()
    try:
        from config import APP_BASE_URL
        from db.email_verification import create_verification_token
        from ui.email_utils import send_verification_email
        token = create_verification_token(email)
        if not token:
            _note("no_account")
            return False
        url = f"{APP_BASE_URL.rstrip('/')}/verify_email?token={token}"
        return bool(send_verification_email(to_address=email, verify_url=url))
    except Exception as exc:
        _note("provider_error", type(exc).__name__)
        return False


def require_verified_for_upgrade(email: str, *, key_suffix: str = "") -> bool:
    """Return True if the user may proceed to checkout.

    When unverified, renders a notice + "resend verification" button and
    returns False so the caller blocks the upgrade.
    """
    import streamlit as st

    if not email:
        return True
    try:
        if verification_status(email):
            return True
    except VerificationUnavailable:
        st.warning(
            "We can't confirm your email verification right now, so upgrades are paused. "
            "Please try again in a few minutes."
        )
        return False

    st.warning(
        "📧 Please verify your email before upgrading. "
        "Check your inbox for the verification link."
    )
    if st.button("✉️ Resend verification email", key=f"resend_verify_{key_suffix}"):
        if _resend_verification(email):
            st.success("Verification email sent. Check your inbox (and spam).")
        else:
            st.warning("Could not send the verification email. Try again shortly.")
            why = last_failure_for_admin()
            if why:
                st.caption(f"Admin: {why}")
    return False
