"""Lifecycle, ordering and retry contracts against mocked Stripe boundaries."""
import importlib.util
import json
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

DEPS = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(DEPS, "needs fastapi")
class ReliabilityTest(unittest.TestCase):
    def setUp(self):
        from test_billing_service import _configured_module

        self.b = _configured_module()
        self.user = {"tier": "pro", "stripe_customer_id": "cus_member", "is_admin": False}
        self.b._get_user_by_email = MagicMock(side_effect=lambda email: self.user)
        self.b._get_username_by_customer = MagicMock(return_value="member@example.com")
        self.b._set_user_plan_by_email = MagicMock()
        self.b._record_paid_conversion = MagicMock()
        self.engine = self.b._billing_engine
        self.store = self.engine.store
        self.email = "member@example.com"
        self.sub = {"id": "sub_current", "customer": "cus_member", "status": "active", "created": 100,
                    "current_period_end": 1800000000,
                    "items": {"data": [{"price": {"id": "price_premium_fake", "unit_amount": 4000,
                                                    "currency": "usd", "recurring": {"interval": "month"}}}]}}
        self.b.stripe.Subscription.list.return_value = {"data": [self.sub], "has_more": False}

    def event(self, kind="customer.subscription.updated", event_id="evt_1", data=None):
        event = {"id": event_id, "type": kind, "livemode": False,
                 "data": {"object": data or {"customer": "cus_member", "status": "active"}}}
        self.b.stripe.Webhook.construct_event.return_value = event
        return self.engine.webhook(b"raw body", "sig")

    def test_missed_cancellation_and_empty_customer_history_recover_to_starter(self):
        for subscriptions in ([{**self.sub, "status": "canceled"}], []):
            with self.subTest(subscriptions=subscriptions):
                self.b.stripe.Subscription.list.return_value = {"data": subscriptions, "has_more": False}
                state = self.engine.sync(self.email)
                self.assertEqual(state["plan"], "basic")
                self.assertEqual(self.b._set_user_plan_by_email.call_args.kwargs["tier"], "basic")

    def test_scheduled_cancellation_keeps_access_and_reports_end_date(self):
        self.sub["cancel_at_period_end"] = True
        state = self.engine.sync(self.email)
        self.assertEqual(state["plan"], "premium")
        self.assertIsNone(state["renewal_at"])
        self.assertTrue(state["cancellation_at"])

    def test_terminal_and_pending_statuses_never_grant_paid_access(self):
        for status in ("unpaid", "canceled", "incomplete_expired", "paused", "incomplete"):
            with self.subTest(status=status):
                self.sub["status"] = status
                self.assertEqual(self.engine.sync(self.email)["plan"], "basic")

    def test_past_due_grace_keeps_only_existing_plan_then_expires(self):
        self.sub["status"] = "past_due"
        self.sub["latest_invoice"] = {"created": int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())}
        self.assertEqual(self.engine.sync(self.email)["plan"], "pro")  # unpaid upgrade cannot grant Premium
        self.sub["latest_invoice"]["created"] = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp())
        self.assertEqual(self.engine.sync(self.email)["plan"], "basic")

    def test_first_past_due_observation_does_not_restart_grace_on_refresh(self):
        self.sub["status"] = "past_due"
        first = self.engine.sync(self.email)
        second = self.engine.sync(self.email)
        self.assertEqual(first["grace_ends_at"], second["grace_ends_at"])

    def test_delayed_deleted_event_cannot_cancel_newer_active_subscription(self):
        response = self.event("customer.subscription.deleted", data={"customer": "cus_member", "id": "sub_old"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.b._set_user_plan_by_email.call_args.kwargs["tier"], "premium")

    def test_delayed_active_event_cannot_restore_canceled_access(self):
        self.sub["status"] = "canceled"
        self.event(data={"customer": "cus_member", "status": "active"})
        self.assertEqual(self.b._set_user_plan_by_email.call_args.kwargs["tier"], "basic")

    def test_duplicate_delivery_does_not_reapply_or_contact_stripe(self):
        self.event()
        self.b._set_user_plan_by_email.reset_mock()
        self.b.stripe.Subscription.list.reset_mock()
        response = self.event()
        self.assertEqual(json.loads(response.body)["note"], "already_processed")
        self.b._set_user_plan_by_email.assert_not_called()
        self.b.stripe.Subscription.list.assert_not_called()

    def test_unknown_price_is_retryable_and_recovers_after_mapping_fixed(self):
        from fastapi import HTTPException

        self.sub["items"]["data"][0]["price"]["id"] = "price_legacy"
        with self.assertRaises(HTTPException) as error:
            self.event()
        self.assertEqual(error.exception.status_code, 503)
        self.assertNotIn("evt_1", self.store.events)
        self.b._set_user_plan_by_email.assert_not_called()
        self.assertEqual(self.store.failures["evt_1"]["code"], "configuration")
        self.b.STRIPE_PRICE_PREMIUM_LEGACY = frozenset({"price_legacy"})
        self.assertEqual(self.event().status_code, 200)
        self.assertNotIn("evt_1", self.store.failures)

    def test_invoice_payment_failure_reconciles_same_policy(self):
        self.sub["status"] = "unpaid"
        self.event("invoice.payment_failed")
        self.assertEqual(self.b._set_user_plan_by_email.call_args.kwargs["tier"], "basic")

    def test_status_read_does_not_contact_stripe_or_return_customer_ids(self):
        self.engine.sync(self.email)
        self.b.stripe.Subscription.list.reset_mock()
        status = self.engine.status(self.email)
        self.assertEqual(status["interval"], "month")
        self.assertEqual(status["amount"], 4000)
        self.assertTrue(status["renewal_at"])
        self.assertNotIn("cus_member", json.dumps(status))
        self.b.stripe.Subscription.list.assert_not_called()

    def checkout(self, plan="pro"):
        return self.engine.checkout(self.email, {"plan": plan, "interval": "month"})

    def test_repeated_checkout_reuses_session_across_engine_restarts(self):
        self.b._create_checkout_for_account = MagicMock(return_value={"checkout_url": "https://checkout.stripe.com/session", "session_id": "cs_1"})
        first = self.checkout()
        self.engine = type(self.engine)(self.b, self.store)
        second = self.checkout()
        self.assertEqual(first, second)
        self.b._create_checkout_for_account.assert_called_once()

    def test_checkout_timeout_keeps_request_key_for_retry(self):
        self.b._create_checkout_for_account = MagicMock(side_effect=[RuntimeError("response lost"),
            {"checkout_url": "https://checkout.stripe.com/session", "session_id": "cs_1"}])
        with self.assertRaises(RuntimeError):
            self.checkout()
        key = self.store.intents[self.email]["key"]
        self.checkout()
        keys = [c.kwargs["idempotency_key"] for c in self.b._create_checkout_for_account.call_args_list]
        self.assertEqual(keys, [key, key])

    def test_different_plan_cannot_silently_reuse_old_checkout(self):
        from fastapi import HTTPException

        self.b._create_checkout_for_account = MagicMock(return_value={"checkout_url": "https://checkout.stripe.com/session", "session_id": "cs_1"})
        self.checkout()
        with self.assertRaises(HTTPException) as error:
            self.checkout("premium")
        self.assertEqual(error.exception.status_code, 409)
        self.b._create_checkout_for_account.assert_called_once()

    def test_start_over_expires_only_open_session_and_preserves_completed_payment(self):
        from fastapi import HTTPException

        self.store.intents[self.email] = {"key": "one", "session_id": "cs_1", "expires_at": time.time() + 100}
        self.b.stripe.checkout.Session.retrieve.return_value = {"status": "complete"}
        with self.assertRaises(HTTPException):
            self.engine.cancel_checkout(self.email)
        self.b.stripe.checkout.Session.expire.assert_not_called()
        self.assertTrue(self.store.intents[self.email])
        self.b.stripe.checkout.Session.retrieve.return_value = {"status": "open"}
        self.engine.cancel_checkout(self.email)
        self.b.stripe.checkout.Session.expire.assert_called_once_with("cs_1", idempotency_key="expire-one")
        self.assertEqual(self.store.intents[self.email], {})

    def test_completed_first_checkout_recovers_customer_link_without_webhook(self):
        self.user.update(tier="basic", stripe_customer_id=None)
        self.store.intents[self.email] = {"session_id": "cs_owned", "expires_at": time.time() + 100}
        self.b.stripe.checkout.Session.retrieve.return_value = {
            "status": "complete", "customer": "cus_member", "metadata": {"user_email": self.email}}
        self.assertEqual(self.engine.sync(self.email)["plan"], "premium")
        self.assertEqual(self.b._set_user_plan_by_email.call_args.kwargs["stripe_customer_id"], "cus_member")
        self.assertEqual(self.store.intents[self.email], {})

    def test_first_checkout_cannot_link_another_account(self):
        from fastapi import HTTPException

        self.user.update(tier="basic", stripe_customer_id=None)
        self.store.intents[self.email] = {"session_id": "cs_owned", "expires_at": time.time() + 100}
        self.b.stripe.checkout.Session.retrieve.return_value = {
            "status": "complete", "customer": "cus_other", "metadata": {"user_email": "someone@example.com"}}
        with self.assertRaises(HTTPException):
            self.engine.sync(self.email)
        self.b._set_user_plan_by_email.assert_not_called()
