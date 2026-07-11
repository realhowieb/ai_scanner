"""Thin SMTP email helper. Requires SMTP_HOST/SMTP_USER/SMTP_PASS in secrets or env."""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Report silent failures to Sentry when configured (no-op otherwise). Guarded so
# this module keeps working in any context, headless or app.
try:
    from ui.monitoring import capture as _capture
except Exception:  # pragma: no cover - fallback when monitoring is unavailable
    def _capture(exc: BaseException) -> None:
        pass


def send_password_reset_email(to_address: str, reset_url: str) -> bool:
    """Send a password reset email. Returns True on success, False on any failure."""
    try:
        from config import SMTP_FROM, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
    except Exception:
        return False

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        return False

    subject = "Reset your HSFinest.AI password"
    body_text = (
        f"You requested a password reset for your HSFinest.AI account.\n\n"
        f"Click the link below to set a new password (valid for 30 minutes):\n\n"
        f"{reset_url}\n\n"
        f"If you did not request this, you can ignore this email.\n"
    )
    body_html = f"""
<p>You requested a password reset for your <strong>HSFinest.AI</strong> account.</p>
<p><a href="{reset_url}">Reset my password</a></p>
<p>This link expires in 30 minutes. If you did not request this, ignore this email.</p>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_address
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_address], msg.as_string())
        return True
    except Exception as e:
        # A silently-failing password reset locks the user out with no trace.
        print(f"[email] password reset SEND FAILED to {to_address}: {type(e).__name__}: {e}")
        _capture(e)
        return False


def _send_smtp(to_address: str, subject: str, body_text: str, body_html: str) -> bool:
    """Internal shared SMTP sender. Logs the failure reason instead of failing silently."""
    try:
        from config import SMTP_FROM, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
    except Exception as e:
        print(f"[email] config import failed: {e}")
        return False
    missing = [
        name
        for name, val in (("SMTP_HOST", SMTP_HOST), ("SMTP_USER", SMTP_USER), ("SMTP_PASS", SMTP_PASS))
        if not val
    ]
    if missing:
        print(f"[email] not sending — missing SMTP config: {', '.join(missing)}")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_address
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_address], msg.as_string())
        print(f"[email] sent '{subject}' to {to_address} from {SMTP_FROM} via {SMTP_HOST}")
        return True
    except Exception as e:
        print(f"[email] SEND FAILED to {to_address} via {SMTP_HOST}:{SMTP_PORT} from {SMTP_FROM} — {type(e).__name__}: {e}")
        _capture(e)
        return False


def send_verification_email(to_address: str, verify_url: str) -> bool:
    """Send an email address verification email."""
    return _send_smtp(
        to_address=to_address,
        subject="Verify your HSFinest.AI email address",
        body_text=(
            f"Welcome to HSFinest.AI — Scan. Analyze. Trade. Win.\n\n"
            f"Please verify your email address by clicking the link below "
            f"(valid for 24 hours):\n\n{verify_url}\n\n"
            f"If you did not sign up, ignore this email.\n"
        ),
        body_html=(
            f"<p>Welcome to <strong>HSFinest.AI</strong> — Scan. Analyze. Trade. Win.</p>"
            f"<p><a href='{verify_url}'>Verify my email address</a></p>"
            f"<p>This link expires in 24 hours. If you didn't sign up, ignore this email.</p>"
        ),
    )


def send_digest_email(to_address: str, subject: str, html_inner: str, text_inner: str) -> bool:
    """Send a branded rich-HTML digest (e.g. the pre-open morning digest).

    `html_inner` is an HTML fragment (tables/headings) placed inside the branded
    shell; `text_inner` is the plain-text fallback for non-HTML clients.
    """
    disclaimer = (
        "Informational and educational purposes only — not financial, investment, "
        "or trading advice. Trading involves risk of loss; do your own research."
    )
    return _send_smtp(
        to_address=to_address,
        subject=f"HSFinest.AI — {subject}",
        body_text=(f"HSFinest.AI\n\n{text_inner}\n\n— Scan. Analyze. Trade. Win.\n\n{disclaimer}"),
        body_html=(
            "<div style='font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
            "max-width:640px;margin:0 auto;color:#111'>"
            "<p style='font-size:18px;margin:0 0 4px'><strong>⚡ HSFinest.AI</strong></p>"
            f"{html_inner}"
            "<p style='color:#888;margin-top:20px'>— Scan. Analyze. Trade. Win.</p>"
            f"<p style='color:#aaa;font-size:11px'>{disclaimer}</p>"
            "</div>"
        ),
    )


def send_alert_email(to_address: str, subject: str, body: str) -> bool:
    """Send a branded alert email."""
    return _send_smtp(
        to_address=to_address,
        subject=f"HSFinest.AI — {subject}",
        body_text=(
            f"HSFinest.AI alert\n\n{body}\n\n— Scan. Analyze. Trade. Win.\n\n"
            "Informational and educational purposes only — not financial, investment, "
            "or trading advice. Trading involves risk of loss; do your own research."
        ),
        body_html=(
            f"<p><strong>HSFinest.AI</strong> alert</p>"
            f"<pre>{body}</pre>"
            f"<p style='color:#888'>— Scan. Analyze. Trade. Win.</p>"
            f"<p style='color:#aaa;font-size:11px'>Informational and educational "
            "purposes only — not financial, investment, or trading advice. Trading "
            "involves risk of loss; do your own research.</p>"
        ),
    )
