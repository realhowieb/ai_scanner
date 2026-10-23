"""Billing emails: a payment that didn't go through, and a paid plan that ended.

The billing engine (billing_service/engine.py) calls notify_change(email, before, after)
each time it saves a subscription snapshot from Stripe (webhook or reconciliation):

    payment_failed   the subscription went past due (a new past_due_since): "update
                     your card", with the grace end date. Once per past-due spell.
    plan_ended       the account had Pro/Premium and is now on Starter because the
                     subscription ended or went unpaid. Once per ended subscription.

Claims use the lifecycle email log (db.referrals.claim_email), so a webhook delivered
twice, or a webhook plus a reconciliation pass, sends one email. Account emails about
the person's own plan, so they go out regardless of email preferences.

Env: HSF_BILLING_EMAILS_ENABLED=0 turns both off. Best effort: never raises.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import threading
from typing import Any, Mapping, Optional

log = logging.getLogger("hsf.billing_emails")

PAYMENT_FAILED = "payment_failed"
PLAN_ENDED = "plan_ended"
PAID = ("pro", "premium")
PLAN_NAMES = {"pro": "Pro", "premium": "Premium"}


def enabled() -> bool:
    return os.environ.get("HSF_BILLING_EMAILS_ENABLED", "1").strip() != "0"


def _parse(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def payment_failed_content(plan: str, grace_ends_at: Optional[str]) -> tuple:
    from services.lifecycle_emails import _base_url, _button, _day

    name = PLAN_NAMES.get(plan, "paid")
    account = f"{_base_url()}/account"
    until = _parse(grace_ends_at)
    keep = (f"You keep {name} until {_day(until)} while you sort it out." if until
            else f"You keep {name} for a few days while you sort it out.")
    subject = "Your payment didn't go through"
    text = (f"We couldn't charge your card for HSFinest.AI {name}.\n\n{keep} "
            "After that your account goes back to Starter.\n\n"
            f"Update your card under Account, then Manage subscription: {account}\n\n"
            "If you meant to stop paying, there's nothing to do.\n")
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Your payment didn't go through</h2>"
             f"<p>We couldn't charge your card for HSFinest.AI {html.escape(name)}.</p>"
             f"<p>{html.escape(keep)} After that your account goes back to Starter.</p>"
             + _button(account, "Update payment method") +
             "<p style='color:#555'>If you meant to stop paying, there's nothing to do.</p>")
    return subject, text, inner


def plan_ended_content(plan: str) -> tuple:
    from services.lifecycle_emails import PRO_PERKS, _base_url, _button

    name = PLAN_NAMES.get(plan, "paid")
    pricing = f"{_base_url()}/pricing"
    perks = ", ".join(PRO_PERKS[:-1]) + f" and {PRO_PERKS[-1]}"
    subject = f"Your {name} plan has ended"
    text = (f"Your HSFinest.AI {name} subscription has ended, so your account is on Starter now. "
            "Starter stays free, and your watchlists, alerts and journal are all still there.\n\n"
            f"What's paused: {perks}.\n\n"
            f"Come back any time: {pricing}\n")
    inner = (f"<h2 style='font-size:20px;margin:16px 0 8px'>Your {html.escape(name)} plan has ended</h2>"
             f"<p>Your subscription has ended, so your account is on Starter now. Starter stays free, and "
             "your watchlists, alerts and journal are all still there.</p>"
             f"<p>What's paused: {html.escape(perks)}.</p>"
             + _button(pricing, f"Get {name} back"))
    return subject, text, inner


def changes(before: Optional[Mapping[str, Any]], after: Mapping[str, Any], prior_tier: str = "") -> list:
    """[(kind, ref, plan)] emails this snapshot change calls for. Pure, for tests."""
    before = before or {}
    out = []
    since = after.get("past_due_since")
    if since and since != before.get("past_due_since"):
        plan = str(after.get("plan") or prior_tier or before.get("plan") or "")
        out.append((PAYMENT_FAILED, str(since), plan if plan in PAID else (prior_tier or "pro")))
    was = str(before.get("plan") or prior_tier or "").lower()
    if was in PAID and str(after.get("plan") or "").lower() == "basic" and \
            after.get("status") in ("canceled", "unpaid", "incomplete_expired", "paused", "past_due", "none"):
        ref = str(after.get("cancellation_at") or before.get("renewal_at") or before.get("past_due_since")
                  or dt.datetime.now(dt.timezone.utc).date().isoformat())
        out.append((PLAN_ENDED, ref, was))
    return out


def _send(email: str, kind: str, ref: str, plan: str, grace_ends_at: Optional[str]) -> bool:
    from db import referrals as store
    from ui.email_utils import send_digest_email

    if not store.claim_email(email, kind, ref):
        return False
    if kind == PAYMENT_FAILED:
        subject, text, inner = payment_failed_content(plan, grace_ends_at)
    else:
        subject, text, inner = plan_ended_content(plan)
    if send_digest_email(email, subject, inner, text):
        log.info(json.dumps({"event": "billing_email_sent", "kind": kind}))
        return True
    store.release_email(email, kind, ref)
    return False


def notify_change(email: str, before: Optional[Mapping[str, Any]], after: Mapping[str, Any],
                  prior_tier: str = "", *, background: bool = True) -> None:
    """Send what the change calls for, off the request thread by default. Never raises."""
    email = (email or "").strip().lower()
    if not enabled() or "@" not in email:
        return
    try:
        todo = changes(before, after, prior_tier)
    except Exception:
        return
    if not todo:
        return

    def run() -> None:
        for kind, ref, plan in todo:
            try:
                _send(email, kind, ref, plan, after.get("grace_ends_at"))
            except Exception as e:
                log.warning(json.dumps({"event": "billing_email_failed", "kind": kind, "error": type(e).__name__}))

    if background:
        threading.Thread(target=run, name="billing-email", daemon=True).start()
    else:
        run()
