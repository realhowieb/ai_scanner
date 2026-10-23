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

# Customer addresses never reach logs unmasked (GitHub Actions logs are public).
try:
    from ui.log_privacy import log_id, redact
except Exception:  # pragma: no cover - never print an address if the helper is missing
    def log_id(value) -> str:  # type: ignore[no-redef]
        return "***"

    def redact(value) -> str:  # type: ignore[no-redef]
        return "(details hidden)"


# P2-32: remember why a send failed so admins can see it. Guarded: a missing
# module must never break sending.
try:
    from ui import email_failure as _failure
except Exception:  # pragma: no cover
    _failure = None


def _note_failure(reason: str, detail: str = "", exc: BaseException | None = None) -> None:
    if _failure is None:
        return
    try:
        if exc is not None:
            reason, detail = _failure.classify(exc)
        _failure.note(reason, detail)
    except Exception:
        pass


def _sender(smtp_from: str) -> tuple[str, str]:
    """(From header, envelope address). The inbox shows a display name —
    "HSF Alerts" unless SMTP_FROM already carries one ("Name <addr>") or
    SMTP_FROM_NAME overrides it. The envelope sender is always the bare address."""
    from email.utils import formataddr, parseaddr

    name, addr = parseaddr(str(smtp_from or ""))
    addr = addr or str(smtp_from or "")
    try:
        import config as _config

        default_name = getattr(_config, "SMTP_FROM_NAME", "") or "HSF Alerts"
    except Exception:
        default_name = "HSF Alerts"
    return formataddr((name or default_name, addr)), addr


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
    body_html = _branded_card(
        "Reset your password",
        "<p style='margin:0 0 4px'>You asked to reset the password for your <strong>HSFinest.AI</strong> account.</p>",
        reset_url, "Reset my password",
        "This link expires in 30 minutes. If you didn't ask for this, you can ignore this email.",
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    from_header, envelope_from = _sender(SMTP_FROM)
    msg["From"] = from_header
    msg["To"] = to_address
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(envelope_from, [to_address], msg.as_string())
        return True
    except Exception as e:
        # A silently-failing password reset locks the user out with no trace.
        print(f"[email] password reset SEND FAILED to {log_id(to_address)}: {type(e).__name__}: {redact(e)}")
        _capture(e)
        return False


def _send_smtp(to_address: str, subject: str, body_text: str, body_html: str,
               headers: dict | None = None) -> bool:
    """Internal shared SMTP sender. Logs the failure reason instead of failing silently."""
    # Usernames double as email addresses; an account like "admin" has none.
    if "@" not in str(to_address or ""):
        print("[email] not sending — recipient is not an email address")
        _note_failure("not_an_email")
        return False
    try:
        from config import SMTP_FROM, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
    except Exception as e:
        print(f"[email] config import failed: {e}")
        _note_failure("not_configured", "email settings")
        return False
    missing = [
        name
        for name, val in (("SMTP_HOST", SMTP_HOST), ("SMTP_USER", SMTP_USER), ("SMTP_PASS", SMTP_PASS))
        if not val
    ]
    if missing:
        print(f"[email] not sending — missing SMTP config: {', '.join(missing)}")
        _note_failure("not_configured", ", ".join(missing))
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    from_header, envelope_from = _sender(SMTP_FROM)
    msg["From"] = from_header
    msg["To"] = to_address
    for name, value in (headers or {}).items():
        msg[name] = value
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(envelope_from, [to_address], msg.as_string())
        print(f"[email] sent '{subject}' to {log_id(to_address)} from {SMTP_FROM} via {SMTP_HOST}")
        return True
    except Exception as e:
        print(f"[email] SEND FAILED to {log_id(to_address)} via {SMTP_HOST}:{SMTP_PORT} from {SMTP_FROM} — {type(e).__name__}: {redact(e)}")
        _note_failure("", exc=e)
        _capture(e)
        return False


def send_verification_email(to_address: str, verify_url: str) -> bool:
    """Send an email address verification email."""
    return _send_smtp(
        to_address=to_address,
        subject="Verify your HSFinest.AI email address",
        body_text=(
            f"Welcome to HSFinest.AI — Know what matters in the market right now.\n\n"
            f"Please verify your email address by clicking the link below "
            f"(valid for 24 hours):\n\n{verify_url}\n\n"
            f"If you did not sign up, ignore this email.\n"
        ),
        body_html=_branded_card(
            "Verify your email",
            "<p style='margin:0 0 4px'>Welcome to <strong>HSFinest.AI</strong>. "
            "Know what matters in the market right now.</p>",
            verify_url, "Verify my email address",
            "This link expires in 24 hours. If you didn't sign up, ignore this email.",
        ),
    )


DISCLAIMER = (
    "Informational and educational purposes only — not financial, investment, "
    "or trading advice. Trading involves risk of loss; do your own research."
)


def _footer(unsubscribe_url: str | None = None) -> str:
    from ui.email_kit import FAINT

    return ("Know what matters in the market right now.<br>"
            f"{DISCLAIMER}"
            + (f"<br>Don't want these emails? <a href='{unsubscribe_url}' style='color:{FAINT}'>Unsubscribe</a> "
               "or change your email settings in HSF Settings." if unsubscribe_url else ""))


def _branded_card(title: str, intro_html: str, url: str, label: str, fine_print: str) -> str:
    """Account emails (verify, reset): one card with a button, in the shared shell."""
    from ui import email_kit as k

    body = intro_html + k.button(url, label) + k.note(
        f"{k.esc(fine_print)}<br>Button not working? Paste this link into your browser:<br>"
        f"<a href='{k.esc(url)}' style='color:{k.LINK};word-break:break-all'>{k.esc(url)}</a>")
    return k.shell(k.card(k.esc(title), body), footer_html="Know what matters in the market right now.")


def alert_html(subject: str, body: str, unsubscribe_url: str | None = None) -> str:
    """A plain-text alert body as one card in the shared shell (one line per row)."""
    from ui import email_kit as k

    lines = [ln for ln in str(body or "").split("\n")]
    inner = "".join(f"<p style='margin:0 0 6px;font-size:14px'>{k.esc(ln)}</p>" if ln.strip()
                    else "<div style='height:6px'></div>" for ln in lines)
    return k.shell(k.card(k.esc(subject), inner), title_line="Alert", footer_html=_footer(unsubscribe_url))


def _unsubscribe_parts(unsubscribe_url: str | None) -> tuple[str, str, dict]:
    """(text footer, html footer, headers) for an optional P1-41 unsubscribe link."""
    if not unsubscribe_url:
        return "", "", {}
    return (
        f"\n\nDon't want these emails? Unsubscribe: {unsubscribe_url}",
        f"<p style='color:#aaa;font-size:11px'>Don't want these emails? "
        f"<a href='{unsubscribe_url}' style='color:#888'>Unsubscribe</a> "
        "or change your email settings in HSF Settings.</p>",
        {"List-Unsubscribe": f"<{unsubscribe_url}>"},
    )


def send_digest_email(to_address: str, subject: str, html_inner: str, text_inner: str,
                      unsubscribe_url: str | None = None, *, title_line: str = "",
                      glance_html: str = "") -> bool:
    """Send a branded rich-HTML digest (e.g. the pre-open morning digest).

    `html_inner` is an HTML fragment placed inside the branded shell (ui.email_kit):
    a dark header with `title_line`, an optional "At a glance" band, then the body.
    A fragment built from email_kit cards is placed as is; anything else is wrapped
    in one white card. `text_inner` is the plain-text fallback for non-HTML clients.
    """
    from ui.email_kit import shell

    disclaimer = DISCLAIMER
    unsub_text, unsub_html, headers = _unsubscribe_parts(unsubscribe_url)
    footer = _footer(unsubscribe_url)
    return _send_smtp(
        to_address=to_address,
        subject=f"HSFinest.AI — {subject}",
        body_text=(f"HSFinest.AI\n\n{text_inner}\n\n— Know what matters in the market right now.\n\n{disclaimer}"
                   f"{unsub_text}"),
        body_html=shell(html_inner, title_line=title_line, glance=glance_html, footer_html=footer),
        headers=headers,
    )


def send_alert_email(to_address: str, subject: str, body: str,
                     unsubscribe_url: str | None = None) -> bool:
    """Send a branded alert email."""
    unsub_text, unsub_html, headers = _unsubscribe_parts(unsubscribe_url)
    return _send_smtp(
        to_address=to_address,
        subject=f"HSFinest.AI — {subject}",
        body_text=(
            f"HSFinest.AI alert\n\n{body}\n\n— Know what matters in the market right now.\n\n"
            "Informational and educational purposes only — not financial, investment, "
            "or trading advice. Trading involves risk of loss; do your own research."
            f"{unsub_text}"
        ),
        body_html=alert_html(subject, body, unsubscribe_url),
        headers=headers,
    )
