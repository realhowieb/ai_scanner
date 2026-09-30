"""P2-32 — why the last email send failed, for admins.

_send_smtp and the resend-verification helper note a short reason here when a
send fails. The value is per thread (each Streamlit run has its own), so one
user's failure never shows up on another user's page. Reasons never include an
address or a secret: only a category, config names and an SMTP code/class.
"""
from __future__ import annotations

import smtplib
import threading
from typing import Optional, Tuple

_state = threading.local()

MESSAGES = {
    "no_account": "No account uses this address, or a verification link couldn't be created.",
    "not_an_email": "The recipient isn't an email address.",
    "not_configured": "Email isn't configured in this environment (missing {detail}).",
    "auth_failed": "The email provider rejected the SMTP login ({detail}). Check SMTP_USER and SMTP_PASS.",
    "sender_rejected": "The email provider rejected the sender ({detail}). Check SMTP_FROM uses the verified domain.",
    "recipient_rejected": "The email provider refused this recipient ({detail}). In Resend test mode only the account owner can receive email.",
    "network": "Couldn't reach the email server ({detail}).",
    "provider_error": "The email provider returned an error ({detail}).",
}


def classify(exc: BaseException) -> Tuple[str, str]:
    """(reason, detail) for an exception raised while sending."""
    name = type(exc).__name__
    code = getattr(exc, "smtp_code", None)
    detail = f"{name} {code}" if code else name
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "auth_failed", detail
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return "sender_rejected", detail
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        codes = sorted({str(v[0]) for v in (getattr(exc, "recipients", None) or {}).values() if v})
        return "recipient_rejected", f"{name} {', '.join(codes)}".strip()
    if isinstance(exc, (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected)):
        return "network", name
    if isinstance(exc, smtplib.SMTPException):
        return "provider_error", detail
    if isinstance(exc, OSError):
        return "network", name
    return "provider_error", detail


def note(reason: str, detail: str = "") -> None:
    _state.value = (reason, detail)


def clear() -> None:
    _state.value = None


def last() -> Optional[Tuple[str, str]]:
    return getattr(_state, "value", None)


def describe(failure: Optional[Tuple[str, str]]) -> str:
    """Admin-facing sentence for a (reason, detail) pair."""
    if not failure:
        return "No reason was recorded (the send may have failed before reaching the email code)."
    reason, detail = failure
    return MESSAGES.get(reason, "Unknown failure ({detail}).").format(detail=detail or "no details")
