"""One follow-up email for a Pro or Premium checkout that was started but not finished.

    record_start  /v1/billing/checkout logs ("checkout_started", "<plan>|<interval>|<time>")
                  in the lifecycle log (db.referrals) when it hands out a new Stripe Checkout
    run           api/ops_jobs.py, once a day: an account whose latest start is 20-72 hours
                  old and that still isn't on that plan gets one short note with a link back
                  to checkout (and the Founding 100 seats left, while that's on sale)

At most one follow-up per account, ever ("checkout_followup" in the same log). It respects
the "tips" email preference and carries its unsubscribe link. Admins are skipped.

Env: HSF_CHECKOUT_FOLLOWUP_ENABLED=0 turns it off. Best effort: never raises.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("hsf.checkout_followup")

STARTED = "checkout_started"
FOLLOWUP = "checkout_followup"
MIN_AGE = dt.timedelta(hours=20)
MAX_AGE = dt.timedelta(hours=72)
MAX_PER_RUN = 100
UTC = dt.timezone.utc
_RANK = {"basic": 0, "pro": 1, "premium": 2, "admin": 9}
PLAN_LINE = {
    "pro": "Pro adds Day Trader, pre-market and Nasdaq scans, email alerts, the 7-day earnings calendar and the track record.",
    "premium": "Premium adds AI research (Ask HSF, AI notes and summaries), PreBreakout, scheduled scans and paper trading.",
}


def enabled() -> bool:
    return os.environ.get("HSF_CHECKOUT_FOLLOWUP_ENABLED", "1").strip() != "0"


def _parse(value: Any) -> Optional[dt.datetime]:
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def record_start(user: str, plan: str, interval: str, mode: str, now: Optional[dt.datetime] = None) -> None:
    """Note a new Stripe Checkout (not a plan change in the portal). Never raises."""
    if mode != "checkout" or plan not in PLAN_LINE:
        return
    now = now or dt.datetime.now(UTC)
    try:
        from db import referrals as store

        store.claim_email(user, STARTED, f"{plan}|{interval}|{now.astimezone(UTC).isoformat(timespec='seconds')}")
    except Exception as e:
        log.warning(json.dumps({"event": "checkout_start_log_failed", "error": type(e).__name__}))


# ---- selection (pure) ---------------------------------------------------------------------------
def pick(rows: Iterable[Tuple[str, str, str]], *, now: dt.datetime) -> List[Dict[str, str]]:
    """Each account's latest start, when it is 20-72 hours old: [{user, plan, interval}]."""
    latest: Dict[str, Tuple[dt.datetime, str, str]] = {}
    for user, ref, sent_at in rows:
        parts = str(ref).split("|")
        t = _parse(parts[2]) if len(parts) > 2 else _parse(sent_at)
        if len(parts) < 2 or parts[0] not in PLAN_LINE or t is None:
            continue
        key = user.strip().lower()
        if key not in latest or t > latest[key][0]:
            latest[key] = (t, parts[0], parts[1] if parts[1] in ("month", "year") else "month")
    out = []
    for user, (t, plan, interval) in sorted(latest.items()):
        if MIN_AGE <= now - t <= MAX_AGE:
            out.append({"user": user, "plan": plan, "interval": interval})
    return out


def still_wanted(plan: str, account: Optional[Dict[str, Any]], current_plan: str) -> bool:
    """True while the account exists, isn't an admin and isn't on `plan` (or higher) yet."""
    if not account or account.get("is_admin") or account.get("is_active") is False:
        return False
    return _RANK.get(current_plan, 0) < _RANK[plan]


# ---- content ------------------------------------------------------------------------------------
def content(plan: str, interval: str, seats_left: Optional[int] = None) -> tuple:
    from services.lifecycle_emails import _base_url, _button

    base = _base_url()
    name = "Pro" if plan == "pro" else "Premium"
    url = f"{base}/account?upgrade={plan}&interval={interval}#plan"
    subject = f"You were one step from {name}"
    founding_t = founding_h = ""
    if seats_left:
        f_url = f"{base}/founding-100"
        founding_t = (f"Rather pay once? The Founding 100 is five years of Pro or Premium for one payment, "
                      f"and {seats_left} of 100 seats are left: {f_url}\n\n")
        founding_h = (f"<p>Rather pay once? The <a href='{html.escape(f_url, quote=True)}'>Founding 100</a> is five years "
                      f"of Pro or Premium for one payment, and {seats_left} of 100 seats are left.</p>")
    text = (f"You started upgrading to {name} on HSFinest.com but didn't finish. Nothing was charged.\n\n"
            f"{PLAN_LINE[plan]}\n\n"
            f"Pick up where you left off: {url}\n\n" + founding_t +
            "If you decided against it, there's nothing to do. This is the only reminder we send.\n")
    inner = (f"<h2 style='font-size:20px;margin:16px 0 8px'>You were one step from {name}</h2>"
             f"<p>You started upgrading to {name} but didn't finish. Nothing was charged.</p>"
             f"<p>{html.escape(PLAN_LINE[plan])}</p>"
             + _button(url, f"Finish upgrading to {name}") + founding_h +
             "<p style='color:#5b6472'>If you decided against it, there's nothing to do. "
             "This is the only reminder we send.</p>")
    return subject, text, inner


def _seats_left() -> Optional[int]:
    try:
        from api.founding import public_status

        s = public_status()
        return int(s.get("remaining") or 0) if s.get("checkout_open") else None
    except Exception:
        return None


# ---- run ----------------------------------------------------------------------------------------
def run(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now(UTC)
    counts: Dict[str, Any] = {"sent": 0, "skipped": 0, "failed": 0}
    if not enabled():
        counts["disabled"] = True
        return counts
    try:
        from api.plan_access import with_founder
        from api.store import get_account
        from db import referrals as store
        from db.email_prefs import unsubscribe_url, wants_email
        from ui.email_utils import send_digest_email

        due = pick(store.log_since(STARTED, now - MAX_AGE - dt.timedelta(hours=1)), now=now)
    except Exception as e:
        log.warning(json.dumps({"event": "checkout_followup_failed", "error": type(e).__name__}))
        counts["error"] = type(e).__name__
        return counts
    seats: Any = ...
    for d in due[:MAX_PER_RUN]:
        user, plan = d["user"], d["plan"]
        try:
            account = get_account(user)
            current = with_founder(user, (account or {}).get("tier"))
            if not still_wanted(plan, account, current) or not wants_email(user, "tips"):
                counts["skipped"] += 1
                continue
            if not store.claim_email(user, FOLLOWUP):
                continue  # one follow-up per account, ever
            if seats is ...:
                seats = _seats_left()
            subject, text, inner = content(plan, d["interval"], seats)
            if send_digest_email(user, subject, inner, text, unsubscribe_url=unsubscribe_url(user, "tips")):
                counts["sent"] += 1
            else:
                store.release_email(user, FOLLOWUP)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "checkout_followup_email_failed", "error": type(e).__name__}))
    return counts
