"""Messages for the people who run HSF (the admins), not for customers.

notify_ops(subject, text) sends `text` to every linked Telegram/Discord chat of the
admin accounts (the same chats their alerts go to). When no chat took it, or with
email=True, it is also emailed to HSF_OPS_ALERT_EMAIL (comma-separated) or the active
admin accounts, like the backup scan check's email.

    HSF_OPS_NOTIFY_ENABLED=0   turns every ops message off (live check, nightly note,
                               weekly business email)

once(job, key) records that a scheduled ops job ran for `key` (a date or a week), in
the lifecycle email log under the user "__ops__", so a restart or a second API
instance never runs it twice.

Best effort: nothing here raises.
"""
from __future__ import annotations

import html
import json
import logging
import os
from typing import List

log = logging.getLogger("hsf.ops_notify")

OPS_USER = "__ops__"


def enabled() -> bool:
    return os.environ.get("HSF_OPS_NOTIFY_ENABLED", "1").strip() != "0"


def admin_accounts() -> List[str]:
    try:
        from db.scan_watchdog import admin_emails

        return admin_emails()
    except Exception:
        return []


def email_recipients() -> List[str]:
    raw = os.environ.get("HSF_OPS_ALERT_EMAIL", "")
    listed = sorted({p.strip().lower() for p in raw.split(",") if "@" in p})
    return listed or admin_accounts()


def send_chats(text: str) -> int:
    """Send to each admin's linked chats. Returns how many chats accepted it."""
    sent = 0
    try:
        from api import chat_delivery
    except Exception:
        return 0
    for user in admin_accounts():
        try:
            sent += chat_delivery.send_all(user, text)
        except Exception as e:
            log.warning(json.dumps({"event": "ops_chat_failed", "error": type(e).__name__}))
    return sent


def send_email(subject: str, text: str, inner_html: str = "") -> int:
    try:
        from ui.email_utils import send_digest_email
    except Exception:
        return 0
    inner = inner_html or ("<pre style='font-family:inherit;white-space:pre-wrap'>" + html.escape(text) + "</pre>")
    sent = 0
    for to in email_recipients():
        try:
            if send_digest_email(to, subject, inner, text):
                sent += 1
        except Exception as e:
            log.warning(json.dumps({"event": "ops_email_failed", "error": type(e).__name__}))
    return sent


def notify_ops(subject: str, text: str, *, inner_html: str = "", email: bool = False) -> dict:
    """Chat first; email too when asked or when no chat took it."""
    if not enabled():
        return {"chats": 0, "emails": 0, "disabled": True}
    chats = send_chats(f"{subject}\n\n{text}".strip())
    emails = send_email(subject, text, inner_html) if (email or chats == 0) else 0
    log.info(json.dumps({"event": "ops_notified", "subject": subject[:80], "chats": chats, "emails": emails}))
    return {"chats": chats, "emails": emails}


def once(job: str, key: str) -> bool:
    """Claim `job` for `key`. True the first time only (False when storage is down too,
    so a job never runs unrecorded)."""
    try:
        from db import referrals as store

        return store.claim_email(OPS_USER, job, key)
    except Exception:
        return False


def release(job: str, key: str) -> None:
    try:
        from db import referrals as store

        store.release_email(OPS_USER, job, key)
    except Exception:
        pass
