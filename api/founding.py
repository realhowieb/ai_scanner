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

import importlib
import logging
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
                      "disclosure": rules.disclosure(tier)})
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

