"""Lifecycle emails: a welcome note when an account verifies its email, a heads-up
a few days before a free Pro trial (from an invite) runs out, and a note when it has.

    welcome        sent once per account, right after its first email verification
    trial_ending   sent once per trial end, when free Pro ends within REMINDER_DAYS;
                   skipped for accounts that now pay (or are admins) and inactive ones
    trial_ended    sent once per trial end, within 3 days after it ended (api/ops_jobs.py
                   runs it daily); same skips as trial_ending

Each email is claimed in db.referrals' hsf_lifecycle_emails table before it is sent and
the claim is released if the send fails, so a job that runs every scan slot sends each
email once and retries a failed one on the next run. These are account emails about the
person's own plan, so they go out regardless of the digest/evening/alerts preferences.
Everything here is best effort and never raises.

Env: HSF_LIFECYCLE_EMAILS_ENABLED=0 turns both off; HSF_TRIAL_REMINDER_DAYS (default 3)
sets how early the trial reminder goes out.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import math
import os
from typing import Any, Dict, Optional

log = logging.getLogger("hsf.lifecycle_emails")

WELCOME = "welcome"
TRIAL_ENDING = "trial_ending"
TRIAL_ENDED = "trial_ended"
MAX_PER_RUN = 200
# Pro-only features (domain/plans.py FEATURE_MIN_TIER and ALERT_LIMIT_BY_TIER).
PRO_PERKS = ("the Day Trader page", "pre-market and Nasdaq scans", "the 7-day earnings calendar",
             "the track record", "up to 5 alerts instead of 1")


def enabled() -> bool:
    return os.environ.get("HSF_LIFECYCLE_EMAILS_ENABLED", "1").strip() != "0"


def reminder_days() -> int:
    try:
        return max(1, min(14, int(os.environ.get("HSF_TRIAL_REMINDER_DAYS", "3"))))
    except ValueError:
        return 3


def _base_url() -> str:
    try:
        from config import APP_BASE_URL
    except Exception:
        APP_BASE_URL = "https://app.hsfinest.com"
    return (APP_BASE_URL or "https://app.hsfinest.com").rstrip("/")


def _day(t: dt.datetime) -> str:
    """'Friday, October 23' in New York time (what US traders read dates in)."""
    try:
        from zoneinfo import ZoneInfo

        t = t.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        pass
    return f"{t:%A, %B} {t.day}"


def _button(url: str, label: str) -> str:
    return (f"<p style='margin:20px 0'><a href='{html.escape(url, quote=True)}' "
            "style='background:#16202E;color:#fff;padding:10px 18px;border-radius:6px;"
            f"text-decoration:none;font-weight:600'>{html.escape(label)}</a></p>")


# ---- welcome ------------------------------------------------------------------------------------
def welcome_content(trial_until: Optional[dt.datetime] = None) -> tuple:
    """(subject, text, html inner) for the welcome email."""
    base = _base_url()
    steps = [
        ("Today", f"{base}/today", "the market at a glance, with the stocks worth a look first"),
        ("Scanner", f"{base}/scanner", "every stock scored, refreshed through the trading day"),
        ("Watchlists", f"{base}/watchlists", "save the names you follow and see how they did"),
        ("Alerts", f"{base}/alerts", "get told when a stock you care about moves"),
    ]
    trial_text = trial_html = ""
    if trial_until:
        day = _day(trial_until)
        trial_text = (f"Your invite gave you free Pro until {day}, so every feature is open to you now. "
                      "Nothing is charged when it ends; your account just goes back to Starter.\n\n")
        trial_html = (f"<p>Your invite gave you <strong>free Pro until {html.escape(day)}</strong>, so every "
                      "feature is open to you now. Nothing is charged when it ends; your account just goes "
                      "back to Starter.</p>")
    text = ("Your email is verified and your account is ready.\n\n" + trial_text +
            "Four places to start:\n" +
            "".join(f"- {name}: {what}\n  {url}\n" for name, url, what in steps) +
            f"\nNew to HSF? This page explains how the scores work: {base}/how-hsf-works\n")
    items = "".join(f"<li style='margin:6px 0'><a href='{html.escape(url, quote=True)}'>{name}</a>: {what}</li>"
                    for name, url, what in steps)
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Welcome to HSFinest.AI</h2>"
             "<p>Your email is verified and your account is ready.</p>" + trial_html +
             f"<p>Four places to start:</p><ul style='padding-left:20px'>{items}</ul>" +
             _button(f"{base}/today", "Open Today") +
             f"<p>New to HSF? <a href='{base}/how-hsf-works'>See how the scores work</a>.</p>")
    return "Welcome, your account is ready", text, inner


def send_welcome(email: str) -> bool:
    """Send the welcome email once per account. True when it was sent now."""
    email = (email or "").strip().lower()
    if not enabled() or "@" not in email:
        return False
    try:
        from db import referrals as store
        from ui.email_utils import send_digest_email

        if not store.claim_email(email, WELCOME):
            return False
        try:
            trial_until = store.grant_until(email)
        except Exception:
            trial_until = None
        subject, text, inner = welcome_content(trial_until)
        if send_digest_email(email, subject, inner, text):
            log.info(json.dumps({"event": "lifecycle_email_sent", "kind": WELCOME}))
            return True
        store.release_email(email, WELCOME)
    except Exception as e:
        log.warning(json.dumps({"event": "lifecycle_email_failed", "kind": WELCOME, "error": type(e).__name__}))
    return False


# ---- trial ending -------------------------------------------------------------------------------
def trial_ending_content(until: dt.datetime, now: Optional[dt.datetime] = None) -> tuple:
    """(subject, text, html inner) for the 'free Pro ends soon' email."""
    now = now or dt.datetime.now(dt.timezone.utc)
    days = max(1, math.ceil((until - now).total_seconds() / 86400))
    left = "1 day" if days == 1 else f"{days} days"
    day = _day(until)
    pricing = f"{_base_url()}/pricing"
    perks = ", ".join(PRO_PERKS[:-1]) + f" and {PRO_PERKS[-1]}"
    subject = f"Your free Pro ends in {left}"
    text = (f"Your free Pro from an invite ends on {day} ({left} from now).\n\n"
            f"After that your account goes back to Starter, which stays free. You'd lose {perks}.\n\n"
            f"To keep Pro, upgrade here: {pricing}\n\n"
            "If Starter is all you need, there's nothing to do. You won't be charged.\n")
    inner = (f"<h2 style='font-size:20px;margin:16px 0 8px'>Your free Pro ends in {left}</h2>"
             f"<p>Your free Pro from an invite ends on <strong>{html.escape(day)}</strong>.</p>"
             f"<p>After that your account goes back to Starter, which stays free. You'd lose {html.escape(perks)}.</p>"
             + _button(pricing, "Keep Pro") +
             "<p style='color:#555'>If Starter is all you need, there's nothing to do. You won't be charged.</p>")
    return subject, text, inner


def _still_on_trial(email: str, users_lookup) -> bool:
    """Only Starter accounts that are active get the reminder; a paying account or an
    admin isn't losing anything."""
    try:
        user = users_lookup(email)
    except Exception:
        return False
    if not user or not user.get("is_active", True) or user.get("is_admin"):
        return False
    return str(user.get("tier") or "basic").strip().lower() in ("", "basic")


def run_trial_reminders(now: Optional[dt.datetime] = None, users_lookup=None) -> Dict[str, Any]:
    """Email every account whose free Pro ends within reminder_days(). Safe to call every
    scan slot: each trial end is emailed once. Returns counts for the job log."""
    counts: Dict[str, Any] = {"sent": 0, "skipped": 0, "failed": 0}
    if not enabled():
        counts["disabled"] = True
        return counts
    try:
        from db import referrals as store
        from ui.email_utils import send_digest_email

        if users_lookup is None:
            from db.users import get_user_by_username as users_lookup
        ending = store.grants_ending(dt.timedelta(days=reminder_days()))
    except Exception as e:
        log.warning(json.dumps({"event": "trial_reminders_failed", "error": type(e).__name__}))
        counts["error"] = type(e).__name__
        return counts
    for user_id, until_text in ending[:MAX_PER_RUN]:
        until = store.parse_ts(until_text)
        if until is None or "@" not in user_id or not _still_on_trial(user_id, users_lookup):
            counts["skipped"] += 1
            continue
        try:
            if not store.claim_email(user_id, TRIAL_ENDING, until_text):
                continue  # already sent for this trial end
            subject, text, inner = trial_ending_content(until, now)
            if send_digest_email(user_id, subject, inner, text):
                counts["sent"] += 1
            else:
                store.release_email(user_id, TRIAL_ENDING, until_text)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "trial_reminder_failed", "error": type(e).__name__}))
    return counts


# ---- trial ended --------------------------------------------------------------------------------
def trial_ended_content() -> tuple:
    """(subject, text, html inner) for the 'free Pro has ended' email."""
    pricing = f"{_base_url()}/pricing"
    perks = ", ".join(PRO_PERKS[:-1]) + f" and {PRO_PERKS[-1]}"
    subject = "Your free Pro has ended"
    text = ("Your free Pro from an invite has ended, so your account is on Starter now. Nothing was charged.\n\n"
            "You keep Today, the Scanner, your watchlists, 1 alert and your journal.\n"
            f"Paused until you upgrade: {perks}.\n\n"
            f"To get Pro back: {pricing}\n")
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Your free Pro has ended</h2>"
             "<p>Your account is on Starter now. Nothing was charged.</p>"
             "<p><strong>You keep</strong> Today, the Scanner, your watchlists, 1 alert and your journal.</p>"
             f"<p><strong>Paused until you upgrade:</strong> {html.escape(perks)}.</p>"
             + _button(pricing, "Get Pro back"))
    return subject, text, inner


def run_trial_ended(now: Optional[dt.datetime] = None, users_lookup=None) -> Dict[str, Any]:
    """Email accounts whose free Pro ended in the last 3 days and that didn't upgrade.
    Each trial end is emailed once."""
    counts: Dict[str, Any] = {"sent": 0, "skipped": 0, "failed": 0}
    if not enabled():
        counts["disabled"] = True
        return counts
    try:
        from db import referrals as store
        from ui.email_utils import send_digest_email

        if users_lookup is None:
            from db.users import get_user_by_username as users_lookup
        ended = store.grants_ended(dt.timedelta(days=3))
    except Exception as e:
        log.warning(json.dumps({"event": "trial_ended_failed", "error": type(e).__name__}))
        counts["error"] = type(e).__name__
        return counts
    subject, text, inner = trial_ended_content()
    for user_id, until_text in ended[:MAX_PER_RUN]:
        if "@" not in user_id or not _still_on_trial(user_id, users_lookup):
            counts["skipped"] += 1
            continue
        try:
            if not store.claim_email(user_id, TRIAL_ENDED, until_text):
                continue
            if send_digest_email(user_id, subject, inner, text):
                counts["sent"] += 1
            else:
                store.release_email(user_id, TRIAL_ENDED, until_text)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "trial_ended_email_failed", "error": type(e).__name__}))
    return counts
