"""Founding 100 push: getting the word out and hearing about the first sales.

  launch / reminder email   An admin previews it in Admin > Founding, then sends it once.
                            It goes to verified, active accounts that haven't switched
                            "tips and reminders" off and aren't founders. The reminder only
                            goes to people who got the launch email. Each account is claimed
                            in the lifecycle email log (kind "founding_push", ref the
                            campaign), so nobody gets the same one twice. Never automatic.
  weekly recap card         A short Founding 100 card in the Saturday weekly recap email
                            for non-founders while seats are on sale.
  sale alert                notify_ops to the admins the moment a founder payment completes:
                            tier, founder number, amount and seats left.

Copy follows the marketing rule: no returns, win rates or edge claims.
Best effort everywhere: nothing here raises into a webhook or a scheduled job.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("hsf.founding_push")

KIND = "founding_push"
CAMPAIGNS = ("launch", "reminder")
PAGE = "/founding-100"


def _money(cents: Any) -> str:
    return f"${int(cents or 0) // 100:,}"


def _max_users() -> int:
    try:
        return max(1, int(os.environ.get("HSF_FOUNDING_PUSH_MAX_USERS", "2000")))
    except ValueError:
        return 2000


def _base() -> str:
    from services.lifecycle_emails import _base_url

    return _base_url()


def page_url(base: str, campaign: str) -> str:
    """The public Founding 100 page, tagged so Admin > Marketing counts the email's visits."""
    return f"{base.rstrip('/')}{PAGE}?utm_source=newsletter&utm_medium=email&utm_campaign=founding-{campaign}"


def offer(status: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The public status when at least one tier is on sale with seats left, else None."""
    if status is None:
        try:
            from api.founding import public_status

            status = public_status()
        except Exception:
            return None
    tiers = [t for t in status.get("tiers") or [] if t.get("on_sale") and int(t.get("remaining") or 0) > 0]
    if not status.get("open") or not tiers:
        return None
    return {**status, "tiers": tiers}


def is_founder(email: str) -> bool:
    """True for a live (active or pending) founding member. Unknown counts as a founder,
    so a database hiccup never pitches a seat to someone who already has one."""
    try:
        from db import founding

        m = founding.membership(email) or {}
    except Exception:
        return True
    return m.get("status") in ("active", "pending")


def tier_lines(o: Dict[str, Any]) -> List[str]:
    out = []
    for t in o["tiers"]:
        five = t.get("five_year") or {}
        line = f"{t['label']}: {_money(t['price_cents'])} once, {int(t['remaining'])} of {int(t['capacity'])} seats left."
        if five.get("five_years_monthly_cents"):
            line += f" Five years paying monthly would be {_money(five['five_years_monthly_cents'])}."
        out.append(line)
    return out


EXTRAS = ("Every founder also gets a numbered seat and badge, 50% off any yearly plan after the five years, "
          "3 invites that give friends 60 days of Pro, and a vote on what we build next.")


# ---- the weekly recap card ------------------------------------------------------------------

def weekly_card(base: str, status: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
    """(text, html card) for the Saturday weekly recap while seats are on sale, else None."""
    o = offer(status)
    if not o:
        return None
    from ui import email_kit as k

    url = page_url(base, "weekly-recap")
    lead = f"The Founding 100 is open: {int(o['remaining'])} of 100 seats left. One payment, five years of HSF, no renewals."
    lines = [lead, *tier_lines(o)]
    html = k.card("⭐ Founding 100", "".join(k.note(k.esc(line)) for line in lines) + k.button(url, "See the Founding 100"))
    return "\n".join([*lines, f"See the Founding 100: {url}", ""]), html


# ---- launch and reminder emails ---------------------------------------------------------------

def compose(campaign: str, o: Dict[str, Any], *, base: str, note: str = "") -> Dict[str, str]:
    """{subject, text, html inner}. Pure apart from the email kit."""
    from ui import email_kit as k

    left = int(o["remaining"])
    if campaign == "reminder":
        subject = f"{left} of 100 Founding seats left"
        title = f"{left} Founding seats left"
        lines = [f"Last week we opened the Founding 100. {left} of the 100 seats are still open.",
                 "One payment covers five calendar years of HSF. Nothing renews and you are never charged again."]
    else:
        subject = "The Founding 100 is open: one payment, five years of HSF"
        title = "The Founding 100 is open"
        lines = ["We just opened the Founding 100: 100 numbered memberships of HSF, and there will never be more.",
                 "One payment covers five calendar years. Nothing renews and you are never charged again."]
    lines += tier_lines(o)
    lines += [EXTRAS, "Not sure? Every founding payment has a 14-day full refund guarantee.",
              "Already on a monthly or yearly plan? Cancel it in Account, and you can claim a seat once it ends."]
    paras = [p.strip() for p in (note or "").split("\n\n") if p.strip()]
    url = page_url(base, campaign)
    fine = "HSF is a research tool, not financial advice. It makes no promise of returns."
    body = ("".join(f"<p style='margin:0 0 10px'>{k.esc(p)}</p>" for p in paras)
            + "".join(f"<p style='margin:0 0 8px'>{k.esc(line)}</p>" for line in lines)
            + k.button(url, "See the Founding 100") + k.note(k.esc(fine)))
    text = "\n\n".join([title, *paras, *lines, f"See the Founding 100: {url}", fine])
    return {"subject": subject, "title": title, "text": text, "html": k.card(k.esc(title), body)}


def recipients(campaign: str) -> Dict[str, Any]:
    """{"to": [emails], "skipped": {reason: n}} for one campaign."""
    from db import referrals as store
    from db.email_prefs import wants_email
    from db.email_verification import is_email_verified
    from db.users import load_users

    skipped: Dict[str, int] = {}

    def skip(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1

    already = store.emails_sent(KIND, campaign)
    got_launch = store.emails_sent(KIND, "launch") if campaign == "reminder" else {}
    to: List[str] = []
    for username, record in list((load_users() or {}).items())[:_max_users()]:
        email = str(username or "").strip().lower()
        if "@" not in email or (record or {}).get("is_active") is False:
            skip("not_eligible")
            continue
        if email in already:
            skip("already_sent")
            continue
        if campaign == "reminder" and email not in got_launch:
            skip("no_launch_email")
            continue
        try:
            if not is_email_verified(email):
                skip("unverified")
                continue
        except Exception:
            pass
        if not wants_email(email, "tips"):
            skip("unsubscribed")
            continue
        if is_founder(email):
            skip("founder")
            continue
        to.append(email)
    return {"to": to, "skipped": skipped}


def campaign_state(campaign: str) -> Dict[str, Any]:
    """How many got it and when it first went out."""
    from db import referrals as store

    sent = store.emails_sent(KIND, campaign)
    return {"sent": len(sent), "first_sent_at": min(sent.values()) if sent else None}


def send(campaign: str, note: str = "", *, status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send one campaign to everyone eligible who hasn't had it. {sent, failed, skipped}."""
    from db import referrals as store
    from db.email_prefs import unsubscribe_url
    from ui.email_utils import send_digest_email

    o = offer(status)
    if not o:
        return {"sent": 0, "failed": 0, "skipped": {"closed": 1}}
    base = _base()
    msg = compose(campaign, o, base=base, note=note)
    who = recipients(campaign)
    sent = failed = 0
    for email in who["to"]:
        if not store.claim_email(email, KIND, campaign):
            who["skipped"]["already_sent"] = who["skipped"].get("already_sent", 0) + 1
            continue
        try:
            ok = send_digest_email(email, msg["subject"], msg["html"], msg["text"],
                                   unsubscribe_url=unsubscribe_url(email, "tips"), title_line="Founding 100")
        except Exception:
            ok = False
        if ok:
            sent += 1
        else:
            store.release_email(email, KIND, campaign)
            failed += 1
    log.info(json.dumps({"event": "founding_push_sent", "campaign": campaign, "sent": sent, "failed": failed}))
    return {"sent": sent, "failed": failed, "skipped": who["skipped"]}


# ---- sale alert -----------------------------------------------------------------------------

def sale_alert(email: str, *, upgrade: bool = False) -> bool:
    """Tell the admins a founder payment completed: tier, number, amount, seats left."""
    try:
        from db import founding, founding_refunds
        from domain import founding as rules
        from services.ops_notify import notify_ops

        m = founding.membership(email) or {}
        if not m:
            return False
        kind = "upgrade" if upgrade else "purchase"
        pay = next((p for p in reversed(founding_refunds.payments(email)) if p.get("kind") == kind), None) or {}
        avail = founding.availability()
        label = rules.TIER_LABEL.get(m.get("tier"), "Founder")
        number = m.get("membership_number")
        num = f"#{int(number):03d}" if number else "#?"
        seats = " · ".join(f"{rules.TIER_LABEL[t]} {v['remaining']} of {v['capacity']} left"
                           for t, v in avail["tiers"].items())
        what = "upgrade to" if upgrade else "sale:"
        subject = f"Founding {what} {label} {num}, ${int(pay.get('amount_cents') or 0) / 100:,.2f}"
        text = "\n".join([
            f"{'Upgrade' if upgrade else 'New founder'}: {label} {num}",
            f"Amount: ${int(pay.get('amount_cents') or 0) / 100:,.2f}",
            f"Account: {email}",
            f"Seats: {seats} ({avail['sold']} of {avail['capacity']} sold)",
            f"Admin > Founding: {_base()}/admin",
        ])
        notify_ops(subject, text, email=True)
        return True
    except Exception as e:
        log.warning("founding sale alert not sent: %s", type(e).__name__)
        return False
