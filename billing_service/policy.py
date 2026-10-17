"""One subscription/access policy for webhook delivery and manual reconciliation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

PAID = {"pro", "premium"}
ELIGIBLE = {"active", "trialing"}
ENDED = {"canceled", "unpaid", "incomplete_expired", "paused"}


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None


def snapshot(billing, user, subscription, previous, *, grace_days=3, now=None):
    now = now or datetime.now(timezone.utc)
    status = subscription.get("status", "none") if subscription else "none"
    state = {"status": status, "plan": "basic", "interval": None, "amount": None, "currency": None,
             "renewal_at": None, "cancellation_at": None, "cancel_at_period_end": False,
             "checked_at": now.isoformat(), "test_mode": billing._stripe_key_mode(billing.STRIPE_SECRET_KEY) == "test",
             "sync_state": "confirmed", "message": "No active subscription. Your account is on Starter."}
    if user.get("is_admin") or user.get("tier") == "admin":
        return {**state, "status": "admin", "plan": "admin", "message": "Admin access is managed by HSF."}
    if subscription is None:
        return state
    if subscription.get("customer") != user.get("stripe_customer_id"):
        raise HTTPException(502, "Subscription account could not be verified.")
    items = (subscription.get("items") or {}).get("data") or []
    if len(items) != 1 and status in (ELIGIBLE | {"past_due"}):
        raise HTTPException(409, "This subscription needs review. Contact support to confirm your plan.")
    item = items[0] if items else {}
    price = item.get("price") or {}
    mapped = billing._price_to_plan(price.get("id") or "")
    period_end = subscription.get("current_period_end") or item.get("current_period_end")
    recurring = price.get("recurring") or {}
    if status in (ELIGIBLE | {"past_due"}) and (
        recurring.get("interval") not in {None, "month", "year"} or recurring.get("interval_count", 1) != 1
        or item.get("quantity", 1) != 1
    ):
        raise HTTPException(409, "This subscription's billing interval or quantity needs review. Contact support.")
    state.update(interval=recurring.get("interval") if recurring.get("interval") in {"month", "year"} else None,
                 amount=price.get("unit_amount"),
                 currency=price.get("currency"), cancel_at_period_end=bool(subscription.get("cancel_at_period_end")),
                 cancellation_at=iso(subscription.get("cancel_at") or (period_end if subscription.get("cancel_at_period_end")
                                     else subscription.get("canceled_at"))))
    if status in ELIGIBLE or status == "past_due":
        if mapped not in PAID:
            raise HTTPException(409, "Stripe's subscription price is not configured in HSF. Contact support to sync your plan.")
        if status in ELIGIBLE:
            state.update(plan=mapped, renewal_at=iso(period_end) if not state["cancel_at_period_end"] else None,
                         message="Subscription confirmed with Stripe.")
            if state["cancel_at_period_end"]:
                state["message"] = "Cancellation scheduled. Your paid access continues until the subscription ends."
        else:
            invoice = subscription.get("latest_invoice")
            invoice = invoice if isinstance(invoice, dict) else {}
            since = iso(invoice.get("due_date") or invoice.get("created")) or previous.get("past_due_since") or now.isoformat()
            deadline = datetime.fromisoformat(since) + timedelta(days=grace_days)
            state.update(past_due_since=since, sync_state="needs_attention",
                         message="Payment is overdue. Update your payment method in Manage subscription.")
            if now < deadline and user.get("tier") in PAID:
                state["plan"] = user["tier"]  # grace never upgrades an unpaid plan change
            state["grace_ends_at"] = deadline.isoformat()
    elif status in ENDED:
        state["message"] = "Your subscription has ended or is unpaid. Your account is on Starter."
    elif status == "incomplete":
        state.update(sync_state="pending", message="Payment is pending. Finish checkout before paid access can start.")
    else:
        raise HTTPException(409, "Stripe's subscription status needs review. Contact support to confirm your plan.")
    return state


def choose_subscription(subscriptions):
    if subscriptions.get("has_more"):
        raise HTTPException(409, "Subscription history needs review. Contact support to confirm your plan.")
    rows = subscriptions.get("data") or []
    active = [s for s in rows if s.get("status") in ELIGIBLE]
    due = [s for s in rows if s.get("status") == "past_due"]
    candidates = active or due
    if len(candidates) > 1:
        raise HTTPException(409, "Multiple subscriptions need review. Contact support to confirm your plan.")
    return candidates[0] if candidates else max(rows, key=lambda s: s.get("created", 0), default=None)
