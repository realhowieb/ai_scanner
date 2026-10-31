"""Founding 100 emails, sent through the existing SMTP sender (ui/email_utils).

Every message is built from verified records (payments confirmed by Stripe, refund
requests and decisions in the database), never from what a browser sent, and never
includes card or payment-credential details. Each send is best effort: it returns
False instead of raising, so it can't break a webhook or an admin action.

Sent automatically: purchase confirmation (which also confirms the policy versions
accepted at checkout), refund request received, approved, denied, started (initiated)
and completed. Prepared for an admin to send: the expiry reminder and the permanent
discontinuation notice (nothing schedules them).
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from domain import founding as rules

log = logging.getLogger("hsf.founding_emails")

LABELS = {"founding_terms": "Founding Membership Terms", "founding_refunds": "Refund Policy", "privacy": "Privacy Policy"}


def _base() -> str:
    try:
        from config import APP_BASE_URL

        return APP_BASE_URL.rstrip("/")
    except Exception:
        return "https://app.hsfinest.com"


def _support() -> str:
    from domain import legal

    return legal.details()["support_email"]


def _money(cents: Optional[int]) -> str:
    return f"${(int(cents or 0)) / 100:,.2f}"


def _date(value: Any) -> str:
    from db.founding import parse_ts

    t = parse_ts(value)
    return t.strftime("%B %-d, %Y") if t else "—"


def _send(to: str, subject: str, title: str, lines: List[str], *, button: str = "Open your founder page",
          fine: str = "") -> bool:
    try:
        from ui import email_kit as k
        from ui.email_utils import _send_smtp

        url = f"{_base()}/account/founder"
        support = _support()
        fine = fine or f"Questions? Email {support}."
        intro = "".join(f"<p style='margin:0 0 8px'>{k.esc(line)}</p>" for line in lines)
        html = k.shell(k.card(k.esc(title), intro + k.button(url, button) + k.note(k.esc(fine))),
                       footer_html="HSFinest.AI. Educational research, not financial advice.")
        text = "\n\n".join([title, *lines, f"{button}: {url}", fine])
        return bool(_send_smtp(to, subject, text, html))
    except Exception as e:
        log.warning("founding email %r not sent: %s", subject, type(e).__name__)
        return False


def _membership(user: str) -> Optional[Dict[str, Any]]:
    from db import founding

    return founding.membership(user)


# ---- 1 + 2: purchase confirmation, with the policies accepted --------------------------------

def purchase_confirmation(user: str, *, upgrade: bool = False) -> bool:
    from db import founding_refunds as refunds

    m = _membership(user)
    if not m or m["status"] != "active":
        return False
    pays = refunds.payments(user)
    pay = next((p for p in reversed(pays) if p["kind"] == ("upgrade" if upgrade else "purchase")), None)
    if not pay:
        return False
    accepted = refunds.accepted_policies(pay["payment_reference"])
    label = rules.TIER_LABEL[m["tier"]]
    badge = rules.badge_title(m["tier"], m.get("membership_number"))
    lines = [
        (f"Your upgrade to {label} is confirmed." if upgrade else f"Welcome, {badge}. Your {label} membership is active."),
        f"One-time payment: {_money(pay['amount_cents'])}. Recurring payments: none. It does not renew.",
        f"Access until: {_date(m['expires_at'])}.",
        f"14-day refund guarantee: you can ask for a full refund of this payment until {_date(pay['deadline'])}.",
    ]
    if accepted:
        lines.append("You accepted: " + "; ".join(
            f"{LABELS.get(a['policy_type'], a['policy_type'])} version {a['version']}" for a in accepted
        ) + f" on {_date(accepted[0]['accepted_at'])}. Read them at {_base()}/founding-100/terms, "
            f"{_base()}/founding-100/refunds and {_base()}/privacy.")
    subject = f"Your HSF AI {label} {'upgrade' if upgrade else 'membership'} is confirmed"
    return _send(user, subject, "Founding membership confirmed" if not upgrade else "Upgrade confirmed", lines)


# ---- 3-7: refunds --------------------------------------------------------------------------

def _what(kind: Optional[str]) -> str:
    return "upgrade payment" if kind == "upgrade" else "founding membership payment"


def refund_received(user: str, request: Dict[str, Any]) -> bool:
    lines = [f"We received your refund request for your {_what(request.get('kind'))} on {_date(request['requested_at'])}."]
    lines.append("It is inside the 14-day guarantee." if int(request.get("within_guarantee") or 0) else
                 "It arrived after the 14-day guarantee, so a person will review it against the refund policy.")
    lines.append(f"We'll reply by {_date(request['due_at'])}. Your access continues while we review it.")
    return _send(user, "We received your refund request", "Refund request received", lines)


def refund_approved(user: str, request: Dict[str, Any]) -> bool:
    lines = [f"Your refund of {_money(request.get('amount_cents'))} for your {_what(request.get('kind'))} is approved.",
             "We'll send it to your original payment method through Stripe. When it appears depends on your bank."]
    return _send(user, "Your refund is approved", "Refund approved", lines)


def refund_denied(user: str, request: Dict[str, Any]) -> bool:
    lines = [f"We reviewed your refund request for your {_what(request.get('kind'))} and can't approve it.",
             f"Reason: {request.get('decision_reason') or 'See the refund policy.'}",
             "If you think this is wrong, or the law where you live gives you a refund, reply to this email."]
    return _send(user, "About your refund request", "Refund request reviewed", lines)


def refund_initiated(user: str, request: Dict[str, Any]) -> bool:
    lines = [f"We've started your refund of {_money(request.get('amount_cents'))}.",
             "Stripe returns it to your original payment method. Banks usually show it within 5 to 10 business "
             "days, but the timing is up to your bank."]
    return _send(user, "Your refund has been started", "Refund started", lines)


def refund_completed(user: str, *, kind: Optional[str], amount_cents: Optional[int], full: bool) -> bool:
    if full and kind == "upgrade":
        after = "You're back on Founder Pro, with the same founder number and end date."
    elif full:
        after = "Your founding membership has ended. Your HSF AI account stays open on the free Starter plan."
    else:
        after = "Your membership is unchanged."
    lines = [f"Your refund is complete: {_money(amount_cents)} in total for your {_what(kind)}.", after]
    return _send(user, "Your refund is complete", "Refund complete", lines)


# ---- 8 + 9: prepared, sent only by an admin ---------------------------------------------------

def expiry_reminder(user: str, *, days_left: int) -> bool:
    m = _membership(user)
    if not m or m["status"] != "active":
        return False
    lines = [f"Your {rules.TIER_LABEL[m['tier']]} membership ends on {_date(m['expires_at'])} ({days_left} days from now).",
             "It won't renew or charge you. After it ends you keep your founder badge and can stay on the free "
             "Starter plan or choose any plan."]
    return _send(user, "Your HSF AI founding membership ends soon", "Membership ending soon", lines)


def discontinuation_notice(user: str, *, stops_on: dt.date, refund_cents: Optional[int] = None) -> bool:
    m = _membership(user)
    if not m:
        return False
    lines = [f"HSF AI will permanently stop providing the founding membership service on {stops_on:%B %-d, %Y}.",
             "Under the refund policy, you'll be offered a prorated refund for your unused time."]
    if refund_cents is not None:
        lines.append(f"Based on that date, your refund would be {_money(refund_cents)}. An administrator confirms "
                     "the final amount before it is sent.")
    return _send(user, "Important: HSF AI founding membership service is ending", "Service ending", lines)


# ---- 10: the monthly Founder Briefing (an admin previews it, then sends it) ---------------------

BRIEFING_NOTE_MAX = 4000


def briefing_month(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return f"{now:%Y-%m}"


def briefing_subject(month: str) -> str:
    t = dt.datetime.strptime(month + "-01", "%Y-%m-%d")
    return f"Your Founder Briefing: {t:%B %Y}"


def value_lines(v: Optional[Dict[str, Any]]) -> List[str]:
    """The value meter in words (the founder page, the briefing and the morning digest)."""
    if not v:
        return []
    out = [f"Month {v['months_used']} of {v['term_months']}: {v['months_left']} months of access left."]
    if v["months_used"] > 0:
        out.append(f"Paying monthly would have cost {_money(v['monthly_cost_so_far_cents'])} so far, "
                   f"against your one-time {_money(v['paid_cents'])}.")
    out.append(f"You're {_money(v['ahead_cents'])} ahead of paying monthly." if v["ahead_cents"] > 0 else
               f"Your seat pays for itself in month {v['break_even_month']}.")
    if v.get("alerts_used") is not None:
        out.append(f"Alerts in use: {v['alerts_used']} of {rules.ELITE_ALERT_LIMIT}.")
    return out


def briefing_parts(note: str) -> Dict[str, Any]:
    """What every founder's briefing shares: the admin's "what shipped" note, the poll
    results and the features open early to founders."""
    from db import founding_benefits as extras

    options = rules.poll_options()
    counts = extras.tally() if options else {}
    total = sum(counts.values())
    poll = [(options[k], counts.get(k, 0)) for k in sorted(options, key=lambda k: -counts.get(k, 0))]
    early = [k.removeprefix("can_").replace("_", " ").capitalize() for k in rules.early_access_features()]
    return {"note": note.strip()[:BRIEFING_NOTE_MAX], "poll": poll, "poll_total": total, "early": early,
            "options": options}


def briefing_message(member: Dict[str, Any], parts: Dict[str, Any], month: str,
                     value: Optional[Dict[str, Any]] = None, choice: Optional[str] = None) -> Dict[str, str]:
    """(subject, text, html) for one founder."""
    from ui import email_kit as k

    badge = rules.badge_title(member["tier"], member.get("membership_number")) or "Founder"
    subject = briefing_subject(month)
    paras = [p.strip() for p in parts["note"].split("\n\n") if p.strip()]
    shipped_html = "".join(f"<p style='margin:0 0 8px;line-height:1.5'>{k.esc(p).replace(chr(10), '<br>')}</p>"
                           for p in paras)
    cards = [k.card("What shipped", shipped_html)]
    text = [f"{subject}", f"Hi {badge},", "What shipped:", *paras]
    if parts["poll"]:
        rows = "".join(f"<li>{k.esc(label)}: {n} vote{'s' if n != 1 else ''}</li>" for label, n in parts["poll"])
        mine = (f"Your vote: {parts['options'][choice]}." if choice in parts["options"]
                else "You haven't voted yet. Vote on your founder page.")
        cards.append(k.card("Founder poll", f"<ul style='margin:0 0 8px;padding-left:18px'>{rows}</ul>"
                            + k.note(k.esc(mine)), sub=f"{parts['poll_total']} founder votes so far"))
        text += ["", "Founder poll:", *[f"- {label}: {n}" for label, n in parts["poll"]], mine]
    early = (", ".join(parts["early"]) + ".") if parts["early"] else "Nothing new in early access this month."
    cards.append(k.card("Early access", f"<p style='margin:0'>{k.esc(early)}</p>"))
    text += ["", "Early access:", early]
    vl = value_lines(value)
    if vl:
        cards.append(k.card("Your founder value", "".join(f"<p style='margin:0 0 6px'>{k.esc(x)}</p>" for x in vl)))
        text += ["", "Your founder value:", *vl]
    url = f"{_base()}/account/founder"
    cards.append(k.button(url, "Open your founder page"))
    fine = f"You get this monthly briefing as a founding member. Questions or ideas? Reply in the app or email {_support()}."
    html = k.shell("".join(cards) + k.note(k.esc(fine)), title_line=f"Founder Briefing · {badge}",
                   footer_html="HSFinest.AI. Educational research, not financial advice.")
    text += ["", f"Open your founder page: {url}", fine]
    return {"subject": subject, "text": "\n".join(text), "html": html}


def send_briefing(note: str, *, month: str, now: Optional[dt.datetime] = None) -> Dict[str, int]:
    """Mail the briefing to every active founding member. Best effort per member."""
    from db import founding_benefits as extras
    from services import founder_value
    from ui.email_utils import _send_smtp

    parts = briefing_parts(note)
    sent = failed = 0
    for m in extras.active_members():
        user = str(m["user_id"])
        try:
            msg = briefing_message(m, parts, month, founder_value.for_membership(m, user, now),
                                   extras.prefs(user)["poll_choice"])
            if _send_smtp(user, msg["subject"], msg["text"], msg["html"]):
                sent += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log.warning("founder briefing not sent: %s", type(e).__name__)
    return {"sent": sent, "failed": failed}
