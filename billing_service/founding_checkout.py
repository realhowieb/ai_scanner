"""Stripe side of Founding 100 memberships: one-time Checkout sessions and the webhook
events that complete them. Seats, numbers and access live in db/founding.py; this
module only turns verified Stripe facts into calls there.

A founding session is mode=payment, card only (no delayed bank payments), priced
from domain.founding (never from the browser), and carries the hold's id in its
metadata and in the PaymentIntent's metadata (so later refund events can find it).
The webhook routes these sessions here before the subscription sync runs, so a
founder payment never touches users.tier.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any, Dict, Optional

from domain import founding as rules

FOUNDING_KEY = "founding_reservation_id"


def is_founding(kind: str, data: Dict[str, Any]) -> bool:
    return (str(kind or "").startswith("checkout.session.") and data.get("mode") == "payment"
            and bool((data.get("metadata") or {}).get(FOUNDING_KEY)))


def has_live_subscription(b, user: Dict[str, Any]) -> bool:
    """True when the account's Stripe customer has any subscription that isn't over.
    Errors propagate: the caller refuses checkout rather than risk paying twice."""
    customer_id = (user or {}).get("stripe_customer_id")
    if not customer_id:
        return False
    subs = b.stripe.Subscription.list(customer=customer_id, status="all", limit=100)
    return any(s.get("status") not in {"canceled", "incomplete_expired"} for s in subs.get("data", []))


def _epoch(value: Any) -> int:
    t = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp())


def create_session(b, *, email: str, user: Dict[str, Any], reservation: Dict[str, Any],
                   success_url: str, cancel_url: str) -> Dict[str, Optional[str]]:
    """A Checkout session for one hold. Every argument comes from the hold, so a retry
    with the same idempotency key sends Stripe identical parameters."""
    tier, kind = reservation["tier"], reservation["kind"]
    amount = rules.expected_amount(kind, tier)
    name = (f"HSF AI {rules.TIER_LABEL[tier]} upgrade" if kind == "upgrade"
            else f"HSF AI {rules.TIER_LABEL[tier]}")
    description = ("Upgrade from Founder Pro to Founder Elite. Keeps your founder number and end date. "
                   "One-time payment; no recurring charges." if kind == "upgrade" else
                   "Five-year HSF AI Founding Membership. One-time payment; does not renew.")
    metadata = {FOUNDING_KEY: str(reservation["id"]), "founding_tier": tier, "founding_kind": kind,
                "user_email": email}
    customer_id = (user or {}).get("stripe_customer_id")
    session = b.stripe.checkout.Session.create(
        idempotency_key=f"hsf-founding-{reservation['id']}",
        mode="payment",
        payment_method_types=["card"],
        client_reference_id=hashlib.sha256(email.encode()).hexdigest(),
        customer=customer_id or None,
        customer_email=None if customer_id else email,
        customer_creation=None if customer_id else "always",
        line_items=[{"quantity": 1, "price_data": {
            "currency": rules.CURRENCY, "unit_amount": amount,
            "product_data": {"name": name, "description": description}}}],
        payment_intent_data={"metadata": metadata, "description": name},
        metadata=metadata,
        expires_at=_epoch(reservation["created_at"]) + rules.CHECKOUT_MINUTES * 60,
        success_url=success_url,
        cancel_url=cancel_url,
    )
    sid = getattr(session, "id", None) or (session.get("id") if isinstance(session, dict) else None)
    url = getattr(session, "url", None) or (session.get("url") if isinstance(session, dict) else None)
    return {"session_id": sid if isinstance(sid, str) else None, "url": url}


def open_session_url(b, session_id: str) -> Optional[str]:
    """The URL of a hold's session while it can still be paid, else None."""
    session = b.stripe.checkout.Session.retrieve(session_id)
    return session.get("url") if session.get("status") == "open" else None


def expire_session(b, session_id: str) -> str:
    """Close an unpaid session. Returns its final status (complete means it was paid)."""
    session = b.stripe.checkout.Session.retrieve(session_id)
    status = session.get("status")
    if status == "open":
        b.stripe.checkout.Session.expire(session_id, idempotency_key=f"expire-{session_id}")
        return "expired"
    return str(status or "unknown")


def apply_event(kind: str, data: Dict[str, Any], *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """One verified founding Checkout event -> db/founding. Returns {"outcome": ...}."""
    from db import founding as db

    meta = data.get("metadata") or {}
    try:
        reservation_id = int(meta.get(FOUNDING_KEY))
    except (TypeError, ValueError):
        return {"outcome": "ignored", "reason": "bad_reservation_id"}
    res = db.reservation(reservation_id)
    if not res:
        return {"outcome": "ignored", "reason": "unknown_reservation"}
    if str(meta.get("user_email") or "").strip().lower() != res["user_id"] or \
            (res.get("checkout_session_id") and res["checkout_session_id"] != data.get("id")):
        return {"outcome": "ignored", "reason": "reservation_mismatch"}
    if kind in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        if data.get("payment_status") != "paid":
            return {"outcome": "waiting", "reason": str(data.get("payment_status") or "unpaid")}
        intent = data.get("payment_intent")
        ref = intent.get("id") if isinstance(intent, dict) else intent
        if not isinstance(ref, str) or not ref:
            raise ValueError("paid founding session without a payment reference")
        # Activation time is when Stripe tells us the payment succeeded (this event).
        out = db.fulfill(reservation_id, payment_reference=ref, amount_cents=int(data.get("amount_total") or 0),
                         currency=str(data.get("currency") or ""), now=now)
        return {"outcome": out["outcome"], "reason": out.get("reason")}
    if kind == "checkout.session.async_payment_failed":
        return {"outcome": "released" if db.release(reservation_id, reason="payment_failed", now=now) else "already_done"}
    return {"outcome": "ignored", "reason": "event_not_used"}


# ---- refunds ------------------------------------------------------------------------------------

REFUND_KEY = "founding_refund_request_id"
REFUND_EVENTS = ("charge.refunded", "refund.failed")


def _intent_id(value: Any) -> Optional[str]:
    ref = value.get("id") if isinstance(value, dict) else value
    return ref if isinstance(ref, str) and ref else None


def refund_event_user(kind: str, data: Dict[str, Any]) -> Optional[str]:
    """The account behind a founding refund event, or None when it isn't one of ours."""
    from db import founding_refunds as refunds

    if kind == "charge.refunded":
        ref = _intent_id(data.get("payment_intent"))
        if not ref:
            return None
        with_payment = refunds.payment_by_reference(ref)
        return with_payment["user_id"] if with_payment else None
    if kind == "refund.failed":
        try:
            rid = int((data.get("metadata") or {}).get(REFUND_KEY))
        except (TypeError, ValueError):
            return None
        r = refunds.get_request(rid)
        return r["user_id"] if r else None
    return None


def create_refund(b, *, request: Dict[str, Any]) -> Dict[str, Any]:
    """Send an approved refund to Stripe. The idempotency key is the request, so retrying
    an approval can never refund twice."""
    refund = b.stripe.Refund.create(
        idempotency_key=f"hsf-founding-refund-{request['id']}",
        payment_intent=request["payment_reference"],
        amount=int(request["amount_cents"]),
        metadata={REFUND_KEY: str(request["id"]), "basis": str(request.get("basis") or "")},
    )
    rid = getattr(refund, "id", None) or (refund.get("id") if isinstance(refund, dict) else None)
    status = getattr(refund, "status", None) or (refund.get("status") if isinstance(refund, dict) else None)
    return {"id": rid, "status": status}


def apply_refund_event(kind: str, data: Dict[str, Any], *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    from db import founding_refunds as refunds

    if kind == "charge.refunded":
        latest = ((data.get("refunds") or {}).get("data") or [{}])[0]
        return refunds.apply_provider_refund(_intent_id(data.get("payment_intent")) or "",
                                             refunded_total_cents=int(data.get("amount_refunded") or 0),
                                             provider_refund_id=latest.get("id"), now=now)
    if kind == "refund.failed":
        rid = int((data.get("metadata") or {}).get(REFUND_KEY))
        r = refunds.mark_failed(rid, str(data.get("failure_reason") or "failed"), "payment_provider", now=now)
        return {"outcome": "failed", "request_id": rid, "status": r["status"], "user_id": r["user_id"]}
    return {"outcome": "ignored"}
