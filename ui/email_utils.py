"""Thin SMTP email helper. Requires SMTP_HOST/SMTP_USER/SMTP_PASS in secrets or env."""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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
    except Exception:
        return False


def _send_smtp(to_address: str, subject: str, body_text: str, body_html: str) -> bool:
    """Internal shared SMTP sender."""
    try:
        from config import SMTP_FROM, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
    except Exception:
        return False
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
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
        return True
    except Exception:
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


def send_alert_email(to_address: str, subject: str, body: str) -> bool:
    """Send a branded alert email."""
    return _send_smtp(
        to_address=to_address,
        subject=f"HSFinest.AI — {subject}",
        body_text=f"HSFinest.AI alert\n\n{body}\n\n— Scan. Analyze. Trade. Win.",
        body_html=(
            f"<p><strong>HSFinest.AI</strong> alert</p>"
            f"<pre>{body}</pre>"
            f"<p style='color:#888'>— Scan. Analyze. Trade. Win.</p>"
        ),
    )
