"""Stripe operations with durable account locking and recoverable checkout intents."""
from __future__ import annotations

import hashlib
import os
import time
import uuid

from fastapi import HTTPException
from fastapi.responses import JSONResponse

try:
    from .policy import choose_subscription, snapshot
except ImportError:  # standalone Render root
    from policy import choose_subscription, snapshot

EVENTS = {"checkout.session.completed", "checkout.session.async_payment_succeeded", "checkout.session.async_payment_failed",
          "customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted",
          "invoice.paid", "invoice.payment_failed"}


class Engine:
    def __init__(self, billing, store):
        self.b, self.store = billing, store

    def reconcile(self, tx, customer_id=None):
        user = dict(tx.user)
        linked = user.get("stripe_customer_id")
        if customer_id and linked and linked != customer_id:
            raise HTTPException(409, "This subscription is linked to another customer. Contact support.")
        customer_id = customer_id or linked
        if not customer_id and tx.intent.get("session_id"):
            session = self.b.stripe.checkout.Session.retrieve(tx.intent["session_id"])
            if session.get("status") == "complete":
                reference = hashlib.sha256(user["username"].encode()).hexdigest()
                metadata_email = (session.get("metadata") or {}).get("user_email")
                if (session.get("client_reference_id") and session["client_reference_id"] != reference
                        or metadata_email != user["username"]):
                    raise HTTPException(409, "Checkout account could not be verified. Contact support.")
                customer_id = session.get("customer")
                if not isinstance(customer_id, str) or not customer_id:
                    raise HTTPException(502, "Checkout customer is not available yet. Please try again.")
            elif session.get("status") == "expired":
                tx.save_intent({})
        user["stripe_customer_id"] = customer_id
        sub = None
        if customer_id and not (user.get("is_admin") or user.get("tier") == "admin"):
            rows = self.b.stripe.Subscription.list(customer=customer_id, status="all", limit=100,
                                                   expand=["data.latest_invoice", "data.items.data.price"])
            sub = choose_subscription(rows)
        if not customer_id and tx.user.get("tier") in {"pro", "premium"}:
            # Manual/admin grants have no Stripe link; do not revoke them.
            return {"status": "unlinked", "plan": tx.user["tier"], "sync_state": "needs_attention",
                    "message": "This paid plan has no Stripe customer link. Contact support to review billing."}
        grace = int(os.getenv("BILLING_PAST_DUE_GRACE_DAYS", "3"))
        if not 0 <= grace <= 30:
            raise RuntimeError("BILLING_PAST_DUE_GRACE_DAYS must be between 0 and 30")
        state = snapshot(self.b, user, sub, tx.snapshot, grace_days=grace)
        price = (((sub or {}).get("items") or {}).get("data") or [{}])[0].get("price") or {}
        tx.save_snapshot(state, customer_id, (sub or {}).get("id"), price.get("id"))
        if state["plan"] in {"pro", "premium"} and tx.intent:
            tx.save_intent({})
        return state

    def sync(self, email):
        try:
            with self.store.transaction(email) as tx:
                state = self.reconcile(tx)
                return self.present(state, tx.intent)
        except Exception as exc:
            self._sync_failure(email, exc)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(502, "Subscription status is temporarily unavailable. Please try again.") from exc

    def _sync_failure(self, email, error):
        try:
            code = "configuration" if isinstance(error, HTTPException) and error.status_code == 409 else "provider_unavailable"
            self.store.sync_failure(email, code)
        except Exception:
            self.b._log.warning("could not persist billing sync failure")

    def status(self, email):
        state, intent = self.store.read(email)
        return self.present(state, intent)

    def reconcile_due(self):
        checked, failed = 0, 0
        for email in self.store.claim_reconciliation():
            try:
                self.sync(email)
                checked += 1
            except Exception:
                failed += 1
        return {"checked": checked, "failed": failed}

    @staticmethod
    def present(state, intent):
        state = dict(state) or {"status": "unknown", "sync_state": "pending", "message": "Refresh plan to check your billing details."}
        if intent and intent.get("expires_at", 0) > time.time():
            state.update(checkout_pending=True, checkout_url=intent.get("url"), sync_state="pending",
                         message="A checkout is open. Finish payment, or start over to choose a different plan.")
        return state

    def checkout(self, email, payload):
        plan, interval = payload.get("plan"), payload.get("interval") or "month"
        if plan not in {"pro", "premium"} or interval not in {"month", "year"}:
            raise HTTPException(400, "Choose a valid plan and billing interval.")
        # Persist the request key BEFORE contacting Stripe. A timeout/rollback in
        # the second transaction must not lose Stripe's idempotency key.
        fingerprint = hashlib.sha256(f"{plan}:{interval}:{payload.get('success_url', '')}:{payload.get('return_url', '')}".encode()).hexdigest()
        with self.store.transaction(email) as tx:
            intent = tx.intent
            if intent and intent.get("expires_at", 0) > time.time():
                if intent["fingerprint"] != fingerprint:
                    raise HTTPException(409, "A checkout is already open. Finish it or choose Start over in Account before changing plans.")
            else:
                intent = {"key": f"hsf-{uuid.uuid4().hex}", "fingerprint": fingerprint,
                          "expires_at": int(time.time()) + 3600, "plan": plan, "interval": interval}
                tx.save_intent(intent)
        with self.store.transaction(email) as tx:
            if tx.intent.get("key") != intent["key"]:
                raise HTTPException(409, "Checkout changed in another window. Refresh your account and try again.")
            if tx.intent.get("url"):
                return {"checkout_url": tx.intent["url"], "mode": "checkout"}
            result = dict(self.b._create_checkout_for_account(email, payload, user=tx.user, idempotency_key=intent["key"],
                                                              expires_at=intent["expires_at"]))
            result.setdefault("mode", "portal" if result.get("portal_url") else "checkout")
            if result.get("checkout_url"):
                tx.save_intent({**intent, "url": result["checkout_url"], "session_id": result.get("session_id")})
            else:
                tx.save_intent({})  # existing subscriptions go to the portal
            result.pop("session_id", None)
            return result

    def cancel_checkout(self, email):
        with self.store.transaction(email) as tx:
            intent = tx.intent
            if intent.get("session_id"):
                session = self.b.stripe.checkout.Session.retrieve(intent["session_id"])
                if session.get("status") == "open":
                    self.b.stripe.checkout.Session.expire(intent["session_id"], idempotency_key=f"expire-{intent['key']}")
                elif session.get("status") == "complete":
                    raise HTTPException(409, "Payment is already processing. Refresh your plan instead of starting a second checkout.")
                elif session.get("status") != "expired":
                    raise HTTPException(502, "Checkout status could not be confirmed. Please try again.")
            elif intent:
                # Recover a create timeout with the same key before expiring it;
                # never forget an uncertain request and allow a second charge.
                raise HTTPException(409, "Checkout is still being prepared. Retry the same plan before starting over.")
            tx.save_intent({})
        return {"message": "Checkout closed. You can choose a new plan or billing interval."}

    def webhook(self, payload, sig):
        try:
            event = self.b.stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=self.b.STRIPE_WEBHOOK_SECRET)
        except Exception as exc:
            raise HTTPException(400, "Webhook signature verification failed") from exc
        event_id, kind = event.get("id"), event.get("type")
        if not event_id or not kind:
            raise HTTPException(400, "Invalid Stripe event")
        if kind not in EVENTS:
            return JSONResponse({"received": True, "type": kind, "note": "ignored"})
        mode = self.b._stripe_key_mode(self.b.STRIPE_SECRET_KEY)
        if isinstance(event.get("livemode"), bool) and event["livemode"] != (mode == "live"):
            raise HTTPException(400, "Stripe event mode does not match this service")
        try:
            if self.store.processed(event_id):
                return JSONResponse({"received": True, "type": kind, "note": "already_processed"})
            data = event.get("data", {}).get("object", {})
            customer_id = data.get("customer")
            if not isinstance(customer_id, str) or not customer_id:
                raise HTTPException(400, "Missing Stripe customer")
            email = self.b._account_for_customer(customer_id, kind)
            with self.store.transaction(email) as tx:
                if tx.processed(event_id):
                    return JSONResponse({"received": True, "type": kind, "note": "already_processed"})
                # Fetch current Stripe state UNDER the account lock. Event
                # timestamps/snapshots never roll back a newer plan or cancel it.
                state = self.reconcile(tx, customer_id)
                tx.mark(event_id, kind)  # same commit as the account update
            if state["plan"] in {"pro", "premium"} and kind.startswith("checkout.session."):
                self.b._record_paid_conversion(email, state["plan"], stripe_event_id=event_id)
            return JSONResponse({"received": True, "type": kind})
        except Exception as exc:
            code = "configuration" if isinstance(exc, HTTPException) and exc.status_code == 409 else "processing_failed"
            try:
                self.store.failure(event_id, kind, code)
            except Exception:
                self.b._log.warning("could not persist webhook failure")
            self.b._log.warning("billing webhook failed: event=%s type=%s code=%s", event_id, kind, code)
            raise HTTPException(503, "Billing event could not be processed. Please retry.") from exc
