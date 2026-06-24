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

    subject = "Reset your AI Scanner password"
    body_text = (
        f"You requested a password reset for your AI Scanner account.\n\n"
        f"Click the link below to set a new password (valid for 30 minutes):\n\n"
        f"{reset_url}\n\n"
        f"If you did not request this, you can ignore this email.\n"
    )
    body_html = f"""
<p>You requested a password reset for your <strong>AI Scanner</strong> account.</p>
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
