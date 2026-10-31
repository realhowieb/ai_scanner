"""Founding 100 for the API: the policy texts, the offer an account sees, and starting a
one-time founder checkout once the buyer has accepted the current policies.

Order of a checkout (each step refuses before anything is charged):
1. switches on (FOUNDING_100_ENABLED, FOUNDING_CHECKOUT_ENABLED, FOUNDING_LEGAL_PAGES_ENABLED,
   plus FOUNDING_ELITE_ENABLED / FOUNDING_UPGRADES_ENABLED where they apply);
2. the acknowledgment box was ticked, and the request names the current version and hash
   of the Terms, Refund Policy and Privacy Policy;
3. no live monthly or yearly subscription on the account (cancel it first);
4. a seat hold (db/founding.reserve), then a Stripe session for exactly that hold;
5. the acceptances are recorded against that session (db/legal.record_acceptance).
"""
from __future__ import annotations

import datetime as dt
import importlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from api.errors import CodedError
from domain import founding as rules
from domain import legal as policies

log = logging.getLogger("hsf_api.founding")

_state_lock = threading.Lock()
_state: Dict[str, Any] = {}
SYNC_TTL_S = 300


def _billing():
    """The embedded billing module (looked up in sys.modules, so tests can swap it)."""
    return importlib.import_module("billing_service.main")


def _flag(name: str) -> bool:
    return rules.flag(name)


def pages_open() -> bool:
    return _flag("FOUNDING_LEGAL_PAGES_ENABLED")


def program_open() -> bool:
    return _flag("FOUNDING_100_ENABLED")


def loaded() -> Dict[str, List[Dict[str, Any]]]:
    return policies.load()


def policy(policy_type: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """One policy text (the newest, or `version`), straight from legal/; None if unknown."""
    versions = loaded().get(policy_type) or []
    if version is None:
        return versions[-1] if versions else None
    return next((p for p in versions if p["version"] == version), None)


def public_policy(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: p[k] for k in ("policy_type", "version", "title", "effective_date", "status", "content_hash",
                              "path", "sections")}


def recorded() -> Dict[str, Dict[str, Any]]:
    """Newest version of each policy as recorded in the database (synced from legal/ at
    most every SYNC_TTL_S per process). Raises when the database can't be reached."""
    now = time.monotonic()
    with _state_lock:
        if _state.get("until", 0) > now:
            return _state["value"]
    from db import legal as db

    value = db.sync(loaded())
    with _state_lock:
        _state.update(value=value, until=now + SYNC_TTL_S)
    return value


def clear_cache() -> None:
    with _state_lock:
        _state.clear()
    with _public_lock:
        _public_cache.clear()


def five_year(tier: str) -> Dict[str, int]:
    from services import founder_value

    return founder_value.five_year(tier)


def _membership_out(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not m:
        return None
    number = m.get("membership_number") or m.get("former_number")
    return {"tier": m["tier"], "label": rules.TIER_LABEL.get(m["tier"], m["tier"]), "status": m["status"],
            "number": rules.format_number(m.get("membership_number")),
            "badge": rules.badge_title(m["tier"], number) if m["status"] in ("active", "expired") else None,
            "activated_at": m.get("activated_at"), "expires_at": m.get("expires_at"),
            "upgraded_at": m.get("upgraded_at")}


def offer(account: Dict[str, Any]) -> Dict[str, Any]:
    """What the founder page shows this account. 404 for everyone but admins while the
    program is off (admins see a preview; checkout itself stays refused)."""
    is_admin = bool(account.get("is_admin"))
    if not program_open() and not is_admin:
        raise CodedError(404, "The Founding 100 isn't open yet.", "founding_closed")
    from db import founding as db

    user = str(account["username"]).strip().lower()
    avail = db.availability()
    member = db.membership(user)
    checkout_on = program_open() and _flag("FOUNDING_CHECKOUT_ENABLED") and pages_open()
    tiers = []
    for tier in rules.TIERS:
        left = int(avail["tiers"][tier]["remaining"])
        tiers.append({"tier": tier, "label": rules.TIER_LABEL[tier], "plan": rules.PLAN_FOR_TIER[tier],
                      "price_cents": rules.PRICE_CENTS[tier], "capacity": rules.CAPACITY[tier], "remaining": left,
                      "on_sale": checkout_on and rules.tier_available(tier) and left > 0,
                      "disclosure": rules.disclosure(tier), "five_year": five_year(tier)})
    live = member if member and member["status"] in ("pending", "active") else None
    can_upgrade = bool(live and live["status"] == "active" and live["tier"] == rules.FOUNDER_PRO)
    upgrade = {"price_cents": rules.UPGRADE_PRICE_CENTS, "disclosure": rules.disclosure(rules.FOUNDER_ELITE, "upgrade"),
               "eligible": can_upgrade,
               "on_sale": can_upgrade and checkout_on and _flag("FOUNDING_UPGRADES_ENABLED")
               and rules.tier_available(rules.FOUNDER_ELITE) and int(avail["tiers"][rules.FOUNDER_ELITE]["remaining"]) > 0}
    current = policies.current(loaded())
    return {
        "open": program_open(), "checkout_open": checkout_on, "preview": not program_open(),
        "tiers": tiers, "upgrade": upgrade, "membership": _membership_out(member),
        "policies": [{k: current[t][k] for k in ("policy_type", "version", "title", "content_hash", "path", "status")}
                     for t in policies.CHECKOUT_POLICIES if t in current],
        "acknowledgment": policies.ACKNOWLEDGMENT,
        "missing_details": policies.missing_details() if is_admin else [],
    }


_public_lock = threading.Lock()
_public_cache: Dict[str, Any] = {}
PUBLIC_TTL_S = 30


def public_status() -> Dict[str, Any]:
    """Whether the Founding 100 is open, and seats left per tier, for the signed-out landing
    and pricing pages. Says nothing beyond "closed" while FOUNDING_100_ENABLED is off.
    Cached for PUBLIC_TTL_S so page views don't each hit the database."""
    if not program_open():
        return {"open": False, "checkout_open": False, "tiers": [], "remaining": 0}
    now = time.monotonic()
    with _public_lock:
        hit = _public_cache.get("v")
        if hit and now - hit[0] < PUBLIC_TTL_S:
            return hit[1]
    from db import founding as db

    avail = db.availability()
    checkout_on = _flag("FOUNDING_CHECKOUT_ENABLED") and pages_open()
    tiers = []
    for tier in rules.TIERS:
        left = int(avail["tiers"][tier]["remaining"])
        tiers.append({"tier": tier, "label": rules.TIER_LABEL[tier], "price_cents": rules.PRICE_CENTS[tier],
                      "capacity": rules.CAPACITY[tier], "remaining": left,
                      "on_sale": checkout_on and rules.tier_available(tier) and left > 0,
                      "five_year": five_year(tier)})
    out = {"open": True, "checkout_open": checkout_on, "tiers": tiers,
           "remaining": sum(t["remaining"] for t in tiers)}
    with _public_lock:
        _public_cache["v"] = (now, out)
    return out


def _checkout_refusal(tier: str, upgrade: bool) -> Optional[str]:
    if not (program_open() and _flag("FOUNDING_CHECKOUT_ENABLED") and pages_open()):
        return "Founder checkout isn't open yet."
    if upgrade:
        if tier != rules.FOUNDER_ELITE or not _flag("FOUNDING_UPGRADES_ENABLED"):
            return "Upgrades aren't open yet."
    if not rules.tier_available(tier):
        return f"{rules.TIER_LABEL.get(tier, 'This tier')} isn't on sale right now."
    return None


def _check_policies(accepted: List[Dict[str, Any]]) -> List[int]:
    """Version ids for the accepted policies, or 409 policy_changed (the page reloads them)."""
    try:
        current = recorded()
    except Exception as e:
        log.warning("policy versions unavailable: %s", type(e).__name__)
        raise CodedError(503, "Checkout is temporarily unavailable. Please try again.", "policies_unavailable") from e
    sent = {str(p.get("policy_type")): p for p in accepted}
    ids, stale = [], []
    for ptype in policies.CHECKOUT_POLICIES:
        row, got = current.get(ptype), sent.get(ptype)
        if not row or not row["available"]:
            raise CodedError(503, "Our policies are being updated. Please try again later.", "policies_unavailable")
        if not got or int(got.get("version") or 0) != int(row["version"]) or got.get("content_hash") != row["content_hash"]:
            stale.append(ptype)
        ids.append(int(row["id"]))
    if stale:
        raise CodedError(409, "Our policies changed since this page loaded. Review the current versions and try again.",
                         "policy_changed", changed=stale)
    return ids


def start_checkout(account: Dict[str, Any], *, tier: str, upgrade: bool, acknowledged: bool,
                   accepted: List[Dict[str, Any]]) -> Dict[str, Any]:
    from api import billing
    from db import founding as db
    from db import legal as legal_db

    refusal = _checkout_refusal(tier, upgrade)
    if refusal:
        raise CodedError(403, refusal, "checkout_closed")
    if acknowledged is not True:
        raise CodedError(400, "Tick the box to confirm you understand the membership terms.", "acknowledgment_required")
    version_ids = _check_policies(accepted)
    if not billing.embedded_enabled():
        raise CodedError(503, "Checkout is temporarily unavailable. Please try again.", "billing_unavailable")
    from billing_service import founding_checkout
    from config import APP_BASE_URL

    b = _billing()

    email = str(account["username"]).strip().lower()
    try:
        user = b._get_user_by_email(email) or {}
        live_sub = founding_checkout.has_live_subscription(b, user)
    except Exception as e:
        log.warning("founding subscription check failed: %s", type(e).__name__)
        raise CodedError(503, "Checkout is temporarily unavailable. Please try again.", "billing_unavailable") from e
    if live_sub:
        raise CodedError(409, "Cancel your monthly or yearly plan in Account before buying a founding membership.",
                         "cancel_subscription_first")
    kind = "upgrade" if upgrade else "purchase"
    try:
        res = db.reserve(email, tier, kind=kind)
    except db.FoundingError as e:
        raise CodedError(409, str(e), e.code) from e
    base = APP_BASE_URL.rstrip("/")
    try:
        url = founding_checkout.open_session_url(b, res["checkout_session_id"]) if res.get("checkout_session_id") else None
        if res.get("checkout_session_id") and not url:
            # The session ended (paid, or expired just before the hold): never open a second one for it.
            raise CodedError(409, "This checkout already finished or expired. Refresh the page to see your membership.",
                             "checkout_finished")
        if not url:
            out = founding_checkout.create_session(
                b, email=email, user=user, reservation=res,
                success_url=f"{base}/account/founder?checkout=success",
                cancel_url=f"{base}/account/founder?checkout=cancelled")
            if not out["session_id"] or not out["url"]:
                raise RuntimeError("no checkout session")
            db.attach_checkout(res["id"], out["session_id"])
            res = {**res, "checkout_session_id": out["session_id"]}
            url = out["url"]
        legal_db.record_acceptance(email, version_ids, checkout_session_id=res["checkout_session_id"],
                                   tier=tier, kind=kind, reservation_id=int(res["id"]))
    except CodedError:
        raise
    except Exception as e:
        log.warning("founding checkout failed: %s", type(e).__name__)
        raise CodedError(503, "Checkout is temporarily unavailable. Please try again.", "billing_unavailable") from e
    return {"checkout_url": url, "expires_at": res["expires_at"]}


def cancel_checkout(account: Dict[str, Any]) -> Dict[str, Any]:
    """Give back this account's open hold (the Back button on Stripe's page)."""
    from db import founding as db

    email = str(account["username"]).strip().lower()
    res = db.open_reservation(email)
    if not res:
        return {"released": False}
    if res.get("checkout_session_id"):
        from billing_service import founding_checkout

        b = _billing()
        try:
            status = founding_checkout.expire_session(b, res["checkout_session_id"])
        except Exception as e:
            raise CodedError(503, "Couldn't close the checkout. Please try again.", "billing_unavailable") from e
        if status == "complete":
            return {"released": False, "paid": True}
    return {"released": db.release(int(res["id"]), reason="cancelled")}



# ---- refunds (db/founding_refunds.py, legal/founding_refunds.v1.txt) ------------------------------
#
# Members can always ask for a refund of their own payments (a request is only a record;
# late ones go to a person). Admins approve or deny one request at a time. Money moves
# only when FOUNDING_REFUNDS_ENABLED=1: then an approval sends the refund to Stripe, and
# access changes only when Stripe confirms it (the charge.refunded webhook). Permanent-
# discontinuation (prorated) approvals also need FOUNDING_PRORATED_REFUNDS_ENABLED=1.

def refunds_live() -> bool:
    return _flag("FOUNDING_REFUNDS_ENABLED")


def _notify(name: str, *args, **kwargs) -> None:
    try:
        from services import founding_emails

        getattr(founding_emails, name)(*args, **kwargs)
    except Exception as e:  # emails never block a refund step
        log.warning("founding email %s failed: %s", name, type(e).__name__)


def _refund_error(e: Exception) -> CodedError:
    from db import founding as db

    if isinstance(e, db.NotFound):
        return CodedError(404, str(e) or "No such refund request.", "refund_not_found")
    status = 400 if getattr(e, "code", "") == "refund_error" else 409
    return CodedError(status, str(e), getattr(e, "code", "refund_error"))


_MEMBER_PAYMENT = ("kind", "tier", "amount_cents", "currency", "paid_at", "refunded_cents", "deadline",
                   "within_guarantee", "refundable_cents")
_MEMBER_REQUEST = ("id", "kind", "status", "within_guarantee", "member_reason", "amount_cents", "requested_at",
                   "due_at", "decided_at", "initiated_at", "completed_at")


def _member_request(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: r.get(k) for k in _MEMBER_REQUEST}
    out["within_guarantee"] = bool(int(r.get("within_guarantee") or 0))
    # The member sees the reason only for a decision about them (never admin-only failure notes).
    out["decision_reason"] = r.get("decision_reason") if r.get("status") == "denied" else None
    return out


def my_refunds(account: Dict[str, Any]) -> Dict[str, Any]:
    """This account's founding payments (with the 14-day deadline) and refund requests."""
    from db import founding_refunds as refunds
    from domain import founding_refunds as rrules

    user = str(account["username"]).strip().lower()
    pays = [{k: p.get(k) for k in _MEMBER_PAYMENT} for p in refunds.payments(user)]
    return {"payments": pays, "requests": [_member_request(r) for r in refunds.requests_for(user)],
            "guarantee_days": rrules.GUARANTEE_DAYS, "start_within_business_days": rrules.START_WITHIN_BUSINESS_DAYS,
            "support_email": policies.details()["support_email"]}


def request_refund(account: Dict[str, Any], *, kind: str, reason: str) -> Dict[str, Any]:
    """Only the signed-in account's own payment of that kind: the browser never names a
    payment, so it can't point at someone else's."""
    from db import founding_refunds as refunds

    user = str(account["username"]).strip().lower()
    try:
        r = refunds.request_refund(user, kind, reason=reason)
    except refunds.db.FoundingError as e:
        raise _refund_error(e) from e
    _notify("refund_received", user, r)
    return _member_request(r)


_ADMIN_REQUEST = _MEMBER_REQUEST + ("user_id", "basis", "decision_reason", "decided_by", "provider_refund_id",
                                    "failure_reason", "tier", "payment_reference", "paid_cents", "refunded_cents",
                                    "paid_at", "membership_number", "former_number", "membership_status",
                                    "activated_at", "expires_at", "upgraded_at", "updated_at")


def _admin_request(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: r.get(k) for k in _ADMIN_REQUEST}
    out["within_guarantee"] = bool(int(r.get("within_guarantee") or 0))
    out["number"] = rules.format_number(r.get("membership_number") or r.get("former_number"))
    return out


def admin_refunds(status: Optional[str]) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    return {"requests": [_admin_request(r) for r in refunds.all_requests(status)],
            "refunds_live": refunds_live(), "prorated_live": _flag("FOUNDING_PRORATED_REFUNDS_ENABLED")}


def admin_refund(request_id: int) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    r = refunds.get_request(request_id)
    if not r:
        raise CodedError(404, "No such refund request.", "refund_not_found")
    accepted = [{"policy_type": a["policy_type"], "version": a["version"], "content_hash": a["content_hash"],
                 "accepted_at": a["accepted_at"]} for a in refunds.accepted_policies(r["payment_reference"])]
    audit = [{"at": a["at"], "actor": a["actor"], "action": a["action"], "detail": a.get("detail")}
             for a in refunds.audit_for(request_id)]
    return {"request": _admin_request(r), "accepted_policies": accepted, "audit": audit,
            "refunds_live": refunds_live(), "prorated_live": _flag("FOUNDING_PRORATED_REFUNDS_ENABLED")}


def admin_quote(request_id: int, stopped_at) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    try:
        return refunds.prorated_quote(request_id, stopped_at)
    except refunds.db.FoundingError as e:
        raise _refund_error(e) from e


def _send_to_provider(r: Dict[str, Any], admin: str) -> Dict[str, Any]:
    """Create the Stripe refund for an approved request. A failure keeps it approved (and
    retryable through the same approve call); nothing is marked refunded until Stripe
    confirms it."""
    from db import founding_refunds as refunds

    if not refunds_live():
        return {"sent": False, "note": "Refunds are switched off (FOUNDING_REFUNDS_ENABLED). Approved, not sent."}
    from api import billing
    from billing_service import founding_checkout

    if not billing.embedded_enabled():
        refunds.note_provider_error(r["id"], "billing unavailable", admin)
        return {"sent": False, "note": "Billing is unavailable. Approved; try again later."}
    try:
        full = refunds.get_request(r["id"]) or r
        out = founding_checkout.create_refund(_billing(), request=full)
        if not out.get("id"):
            raise RuntimeError("no refund id")
    except Exception as e:
        refunds.note_provider_error(r["id"], f"provider error: {type(e).__name__}", admin)
        log.warning("founding refund %s not sent: %s", r["id"], type(e).__name__)
        return {"sent": False, "note": "Stripe didn't accept the refund. It stays approved; try again."}
    refunds.mark_initiated(r["id"], out["id"], admin)
    return {"sent": True, "note": "Sent to Stripe. Access changes when Stripe confirms the refund."}


def admin_approve(account: Dict[str, Any], request_id: int, *, basis: str, amount_cents: Optional[int],
                  reason: str, stopped_at) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    admin = str(account["username"]).strip().lower()
    if basis == "discontinuation" and not _flag("FOUNDING_PRORATED_REFUNDS_ENABLED"):
        raise CodedError(403, "Prorated discontinuation refunds are switched off.", "prorated_refunds_off")
    before = refunds.get_request(request_id)
    try:
        r = refunds.approve(request_id, admin, basis=basis, amount_cents=amount_cents, reason=reason,
                            stopped_at=stopped_at)
    except refunds.db.FoundingError as e:
        raise _refund_error(e) from e
    first_time = bool(before and before["status"] == "requested")
    if first_time:
        _notify("refund_approved", r["user_id"], {**r, "kind": before.get("kind")})
    sent = _send_to_provider(r, admin)
    if sent["sent"]:
        _notify("refund_initiated", r["user_id"], {**r, "kind": before.get("kind") if before else None})
    return {"request": _admin_request(refunds.get_request(request_id)), **sent}


def admin_deny(account: Dict[str, Any], request_id: int, *, reason: str) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    admin = str(account["username"]).strip().lower()
    try:
        r = refunds.deny(request_id, admin, reason=reason)
    except refunds.db.FoundingError as e:
        raise _refund_error(e) from e
    full = refunds.get_request(request_id)
    _notify("refund_denied", r["user_id"], {**r, "kind": full.get("kind") if full else None})
    return {"request": _admin_request(full)}


# ---- founder benefits (domain/founding.py; terms sections 3, 5 and 10) --------------------------

def _feature_label(name: str) -> str:
    return name.removeprefix("can_").replace("_", " ").capitalize()


def benefits(account: Dict[str, Any]) -> Dict[str, Any]:
    """What a founding member gets beyond their plan. 404 for accounts that never had a
    membership (or whose membership was refunded or revoked)."""
    from api import referrals
    from db import founding as db
    from db import founding_benefits as extras

    user = str(account["username"]).strip().lower()
    m = db.membership(user)
    if not m or m["status"] not in ("active", "expired"):
        raise CodedError(404, "This account doesn't have a founding membership.", "founding_not_member")
    active = m["status"] == "active"
    prefs = extras.prefs(user)
    elite = m["tier"] == rules.FOUNDER_ELITE
    options = rules.poll_options()
    invites_left = referrals.founder_invites_left(user) if active and referrals.enabled() else None
    return {
        "status": m["status"], "tier": m["tier"],
        "badge": rules.badge_title(m["tier"], m.get("membership_number") or m.get("former_number")),
        "show_badge": prefs["show_badge"],
        "alumni_discount_percent": rules.ALUMNI_DISCOUNT_PERCENT,
        "alumni_rate_ready": (not active) and bool(os.environ.get("HSF_FOUNDER_ALUMNI_COUPON", "").strip()),
        "early_access": [{"key": k, "label": _feature_label(k)} for k in rules.early_access_features()] if active else [],
        "invites": None if invites_left is None else {"days": rules.FOUNDER_INVITE_DAYS, "total": rules.FOUNDER_INVITES,
                                                       "left": invites_left},
        "elite_limits": {"alert_limit": rules.ELITE_ALERT_LIMIT, "ai_multiplier": rules.ELITE_AI_MULTIPLIER}
        if elite and active else None,
        "poll": {"open": active and bool(options), "choice": prefs["poll_choice"],
                 "options": [{"key": k, "label": v} for k, v in options.items()]},
        "value": value_for_membership(m, user) if active else None,
        "wall_name": extras.wall_name(user),
        "card_path": _card_path(user),
        "reply_target": reply_target(),
    }


def value_for_membership(m: Dict[str, Any], user: Optional[str] = None,
                         now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    from services import founder_value

    return founder_value.for_membership(m, user, now)


def reply_target() -> Optional[str]:
    """How soon founders hear back (HSF_FOUNDER_REPLY_TARGET, e.g. "one business day").
    Unset = not promised anywhere."""
    t = " ".join(os.environ.get("HSF_FOUNDER_REPLY_TARGET", "").split())
    return t[:60] or None


def _card_path(user: str) -> Optional[str]:
    """The member's shareable founder card: /founders/card/<invite code>."""
    try:
        from db import referrals as store

        return f"/founders/card/{store.get_or_create_code(user)}"
    except Exception:
        return None


def set_wall_name(account: Dict[str, Any], show: bool, name: Optional[str]) -> Dict[str, Any]:
    from db import founding_benefits as extras

    benefits(account)  # members only
    clean = None
    if show:
        clean = rules.clean_wall_name(name)
        if not clean:
            raise CodedError(400, "Use 2 to 30 letters, numbers, spaces, dots, dashes or apostrophes. "
                                  "No email addresses or links.", "wall_name_invalid")
    extras.set_wall_name(str(account["username"]), clean)
    with _public_lock:
        _public_cache.pop("wall", None)
    return benefits(account)


def public_wall() -> Dict[str, Any]:
    """The Founders Wall: every seat by number, names only for members who opted in.
    Cached like the public status."""
    if not program_open():
        return {"open": False, "seats": [], "taken": 0}
    now = time.monotonic()
    with _public_lock:
        hit = _public_cache.get("wall")
        if hit and now - hit[0] < PUBLIC_TTL_S:
            return hit[1]
    from db import founding_benefits as extras

    seats = extras.wall()
    out = {"open": True, "seats": seats, "taken": sum(1 for x in seats if x["taken"])}
    with _public_lock:
        _public_cache["wall"] = (now, out)
    return out


def public_card(code: str) -> Optional[Dict[str, Any]]:
    """What a shared founder card shows: number, tier, member since and valid to. Only for a
    member who shows the badge; never a name or email."""
    from db import founding as db
    from db import founding_benefits as extras
    from db import referrals as store

    owner = store.owner_of(code)
    if not owner:
        return None
    m = db.membership(owner)
    if not m or m["status"] not in ("active", "expired") or not extras.prefs(owner)["show_badge"]:
        return None
    number = m.get("membership_number") or m.get("former_number")
    return {"badge": rules.badge_title(m["tier"], number), "number": rules.format_number(number),
            "tier": m["tier"], "label": rules.TIER_LABEL[m["tier"]], "status": m["status"],
            "since": m.get("activated_at"), "until": m.get("expires_at"), "code": code}


def set_show_badge(account: Dict[str, Any], show: bool) -> Dict[str, Any]:
    from db import founding_benefits as extras

    benefits(account)  # members only
    extras.set_show_badge(str(account["username"]), show)
    return benefits(account)


def poll_vote(account: Dict[str, Any], option: str) -> Dict[str, Any]:
    from db import founding as db
    from db import founding_benefits as extras

    out = benefits(account)
    if not out["poll"]["open"]:
        raise CodedError(403, "The founder poll is for active founding members.", "poll_closed")
    try:
        extras.vote(str(account["username"]), option)
    except db.FoundingError as e:
        raise CodedError(400, str(e), "poll_unknown_option") from e
    return benefits(account)


def admin_poll() -> Dict[str, Any]:
    from db import founding_benefits as extras

    counts = extras.tally()
    options = rules.poll_options()
    return {"options": [{"key": k, "label": v, "votes": counts.get(k, 0)} for k, v in options.items()],
            "total": sum(counts.values())}


def public_badge(code: str) -> Optional[str]:
    """The founder badge behind an invite code on a shared link, if its owner shows it."""
    from db import founding_benefits as extras
    from db import referrals as store

    owner = store.owner_of(code)
    return extras.public_badge(owner) if owner else None


# ---- monthly Founder Briefing (services/founding_emails.py; admins preview, then send) ---------

def _sample_member() -> Dict[str, Any]:
    from db import founding_benefits as extras

    members = extras.active_members()
    if members:
        return members[0]
    return {"user_id": "", "tier": rules.FOUNDER_PRO, "membership_number": 1,
            "activated_at": None, "expires_at": None}


def briefing_preview(note: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """The briefing as the first active founder would get it (a sample when there are none),
    how many founders it goes to, and whether this month's has already been sent."""
    from db import founding_benefits as extras
    from services import founding_emails as mail

    month = mail.briefing_month(now)
    m = _sample_member()
    user = str(m.get("user_id") or "")
    value = value_for_membership(m, user or None, now) if m.get("activated_at") else None
    msg = mail.briefing_message(m, mail.briefing_parts(note), month, value,
                                extras.prefs(user)["poll_choice"] if user else None)
    sent = extras.briefing(month)
    return {"month": month, "subject": msg["subject"], "html": msg["html"], "text": msg["text"],
            "recipients": len(extras.active_members()), "already_sent": bool(sent),
            "sent_at": (sent or {}).get("started_at"),
            "history": [{k: b.get(k) for k in ("month", "subject", "sent_by", "started_at", "sent", "failed")}
                        for b in extras.briefings()]}


def briefing_send(account: Dict[str, Any], note: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Send this month's briefing to every active founder, once per month."""
    from db import founding_benefits as extras
    from services import founding_emails as mail

    if len(note.strip()) < 20:
        raise CodedError(400, "Write at least a sentence about what shipped.", "briefing_note_short")
    month = mail.briefing_month(now)
    admin = str(account["username"]).strip().lower()
    if not extras.claim_briefing(month, mail.briefing_subject(month), admin, now=now):
        raise CodedError(409, "This month's Founder Briefing has already been sent.", "briefing_already_sent")
    out = mail.send_briefing(note, month=month, now=now)
    extras.finish_briefing(month, out["sent"], out["failed"], now=now)
    log.info("founder briefing %s sent by %s: %s", month, admin, out)
    return {"month": month, **out}
