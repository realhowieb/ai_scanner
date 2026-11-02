"""The "weekly setups" email list: confirm emails and the optional newsletter-tool sync.

Flow: a visitor enters an email on a public page (POST /v1/public/subscribe) -> one
confirm email with a link to /subscribe/confirm?t=<token> -> the address is confirmed
and, when configured, copied to Resend as a contact in a segment so it can receive
Resend Broadcasts. Unsubscribing on HSF's page marks the Resend contact unsubscribed too.

HSF never sends the newsletter itself: broadcasts are written and sent from Resend (or any
tool fed by the admin CSV export).

Env:
  HSF_MARKETING_LIST_ENABLED=0   hides the form and refuses sign-ups (default on)
  HSF_RESEND_SEGMENT_ID          Resend segment that confirmed addresses join (unset = no sync)
  RESEND_API_KEY                 a Resend key with full access (contacts need more than a
                                 sending-only key); falls back to SMTP_PASS when SMTP_HOST is
                                 smtp.resend.com, since that password is a Resend key
Best effort: nothing here raises.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import quote

log = logging.getLogger("hsf.marketing_list")

RESEND_API = "https://api.resend.com"


def enabled() -> bool:
    return os.environ.get("HSF_MARKETING_LIST_ENABLED", "1").strip() != "0"


def _base_url() -> str:
    from services.lifecycle_emails import _base_url as base

    return base()


def confirm_url(token: str) -> str:
    return f"{_base_url()}/subscribe/confirm?t={quote(token)}"


def unsubscribe_url(token: str) -> str:
    return f"{_base_url()}/subscribe/confirm?t={quote(token)}&action=unsubscribe"


def compose_confirm(token: str) -> tuple:
    """(subject, text, html) of the confirm email."""
    from ui import email_kit as k
    from ui.email_utils import DISCLAIMER

    url = confirm_url(token)
    subject = "Confirm your HSF weekly setups email"
    text = ("Thanks for signing up for the HSF weekly setups email.\n\n"
            f"Confirm your address to start getting it:\n{url}\n\n"
            "Didn't sign up? Ignore this email and you won't hear from us.\n\n" + DISCLAIMER)
    body = ("<p style='margin:0 0 4px;font-size:14px'>Thanks for signing up. Once a week you'll get the setups "
            "HSF's scanner ranked most, plus the biggest earnings coming up. One click confirms it.</p>"
            + k.button(url, "Confirm my email")
            + k.note("Didn't sign up? Ignore this email and you won't hear from us.<br>"
                     f"Button not working? Paste this link into your browser:<br>"
                     f"<a href='{k.esc(url)}' style='color:{k.LINK};word-break:break-all'>{k.esc(url)}</a>"))
    html = k.shell(k.card("Confirm your email", body), title_line="Weekly setups",
                   footer_html=f"Know what matters in the market right now.<br>{DISCLAIMER}")
    return subject, text, html


def send_confirm(email: str, token: str) -> bool:
    try:
        from ui.email_utils import _send_smtp

        subject, text, html = compose_confirm(token)
        return _send_smtp(email, subject, text, html)
    except Exception:
        log.warning(json.dumps({"event": "marketing_confirm_send_failed"}))
        return False


# ---- Resend sync -----------------------------------------------------------------------------
def resend_key() -> Optional[str]:
    key = (os.environ.get("RESEND_API_KEY") or "").strip()
    if key:
        return key
    host = (os.environ.get("SMTP_HOST") or "").strip().lower()
    pw = (os.environ.get("SMTP_PASS") or "").strip()
    return pw if host == "smtp.resend.com" and pw.startswith("re_") else None


def resend_segment() -> Optional[str]:
    return (os.environ.get("HSF_RESEND_SEGMENT_ID") or "").strip() or None


def sync_configured() -> bool:
    return bool(resend_key() and resend_segment())


def _resend(method: str, path: str, body: Optional[Dict[str, Any]] = None, *, session=None) -> int:
    import requests

    http = session or requests
    res = http.request(method, f"{RESEND_API}{path}", json=body, timeout=10,
                       headers={"Authorization": f"Bearer {resend_key()}", "Content-Type": "application/json"})
    return int(res.status_code)


def sync_confirmed(email: str, *, session=None) -> bool:
    """Create (or update) the Resend contact as subscribed and add it to the segment."""
    if not sync_configured():
        return False
    try:
        status = _resend("POST", "/contacts", {"email": email, "unsubscribed": False}, session=session)
        if status == 409 or status == 422:  # already a contact: make sure it is subscribed
            _resend("PATCH", f"/contacts/{quote(email)}", {"unsubscribed": False}, session=session)
        elif status >= 300:
            log.warning(json.dumps({"event": "marketing_resend_contact_failed", "status": status}))
            return False
        status = _resend("POST", f"/contacts/{quote(email)}/segments/{quote(resend_segment() or '')}", session=session)
        if status >= 300:
            log.warning(json.dumps({"event": "marketing_resend_segment_failed", "status": status}))
            return False
        return True
    except Exception as e:
        log.warning(json.dumps({"event": "marketing_resend_sync_failed", "error": type(e).__name__}))
        return False


def sync_unsubscribed(email: str, *, session=None) -> bool:
    if not resend_key():
        return False
    try:
        return _resend("PATCH", f"/contacts/{quote(email)}", {"unsubscribed": True}, session=session) < 300
    except Exception:
        return False
