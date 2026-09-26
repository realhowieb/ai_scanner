"""Smoke tests for billing_service.main.

These tests mock out stripe, psycopg2, and env vars so the billing service
module can be imported and its pure-logic functions exercised without real
credentials or a live DB.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


def _make_stripe_mock() -> types.ModuleType:
    m = types.ModuleType("stripe")
    m.api_key = None
    m.Customer = MagicMock()
    m.Subscription = MagicMock()
    m.Webhook = MagicMock()
    m.checkout = MagicMock()
    m.billing_portal = MagicMock()
    return m


def _make_psycopg2_mock(*, db_reachable: bool = False) -> types.ModuleType:
    m = types.ModuleType("psycopg2")
    if db_reachable:
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchone.return_value = (1,)
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value = cur
        m.connect = MagicMock(return_value=conn)
    else:
        m.connect = MagicMock()
    m.extensions = MagicMock()
    return m


def _load_billing_module(env_overrides: dict[str, str] | None = None, *, db_reachable: bool = False):
    """Import billing_service.main with all external deps mocked."""
    for mod in list(sys.modules):
        if mod.startswith("billing_service"):
            del sys.modules[mod]

    env = {
        "STRIPE_SECRET_KEY": "sk_test_fake",
        "STRIPE_WEBHOOK_SECRET": "whsec_fake",
        "STRIPE_PRICE_PRO": "price_pro_fake",
        "STRIPE_PRICE_PREMIUM": "price_premium_fake",
        "DATABASE_URL": "",
        "APP_SUCCESS_URL": "",
        "APP_CANCEL_URL": "",
        "APP_PORTAL_RETURN_URL": "",
    }
    env.update(env_overrides or {})
    stripe_mock = _make_stripe_mock()
    psycopg2_mock = _make_psycopg2_mock(db_reachable=db_reachable)

    with patch.dict(sys.modules, {"stripe": stripe_mock, "psycopg2": psycopg2_mock}):
        with patch.dict("os.environ", env):
            sys.modules.pop("billing_service.main", None)
            import billing_service.main as bm
            return bm


def _configured_module():
    return _load_billing_module(
        {
            "DATABASE_URL": "postgresql://example.test/db",
            "APP_SUCCESS_URL": "https://hsf-beta.streamlit.app",
            "APP_CANCEL_URL": "https://hsf-beta.streamlit.app",
            "APP_PORTAL_RETURN_URL": "https://hsf-beta.streamlit.app/billing",
        },
        db_reachable=True,
    )


def _client(module):
    from fastapi.testclient import TestClient

    return TestClient(module.app, raise_server_exceptions=False)


def _stripe_event(event_type: str, data: dict, *, event_id: str = "evt_test") -> dict:
    return {"id": event_id, "type": event_type, "data": {"object": data}}


def _webhook(module, event: dict):
    module.stripe.Webhook.construct_event.return_value = event
    return _client(module).post(
        "/webhook",
        content=b"{}",
        headers={"content-type": "application/json", "stripe-signature": "test_signature"},
    )


@contextmanager
def _idempotency_db(*, processed: bool = False):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = (1,) if processed else None
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value = cursor
    yield connection


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServiceImportTest(unittest.TestCase):
    def test_module_imports_without_real_credentials(self):
        bm = _load_billing_module()
        self.assertTrue(hasattr(bm, "app"))
        self.assertTrue(hasattr(bm, "health"))

    def test_health_endpoint_reports_missing_env_and_db(self):
        bm = _load_billing_module()
        result = bm.health()
        self.assertEqual(result.status_code, 503)
        body = json.loads(result.body.decode("utf-8"))
        self.assertFalse(body["ok"])
        self.assertIn("missing_env", body)
        self.assertIn("db", body)
        self.assertIn("DATABASE_URL", body["missing_env"])
        self.assertEqual(body["db"]["error"], "DATABASE_URL is missing")

    def test_health_endpoint_returns_ok_with_env_and_reachable_db(self):
        bm = _load_billing_module(
            {
                "DATABASE_URL": "postgresql://example.test/db",
                "APP_SUCCESS_URL": "https://hsf-beta.streamlit.app",
                "APP_CANCEL_URL": "https://hsf-beta.streamlit.app",
            },
            db_reachable=True,
        )

        result = bm.health()

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_env"], [])
        self.assertEqual(result["db"], {"reachable": True, "error": None})
        self.assertIn("billing_readiness", result["features"])


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServicePriceToPlanTest(unittest.TestCase):
    def setUp(self):
        self.bm = _load_billing_module()

    def test_price_pro_maps_to_pro(self):
        self.assertEqual(self.bm._price_to_plan("price_pro_fake"), "pro")

    def test_price_premium_maps_to_premium(self):
        self.assertEqual(self.bm._price_to_plan("price_premium_fake"), "premium")

    def test_unknown_price_maps_to_basic(self):
        self.assertEqual(self.bm._price_to_plan("price_unknown"), "basic")

    def test_empty_price_maps_to_basic(self):
        self.assertEqual(self.bm._price_to_plan(""), "basic")


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServiceWebhookSignatureTest(unittest.TestCase):
    def setUp(self):
        self.bm = _load_billing_module(
            {
                "DATABASE_URL": "postgresql://example.test/db",
                "APP_SUCCESS_URL": "https://hsf-beta.streamlit.app",
                "APP_CANCEL_URL": "https://hsf-beta.streamlit.app",
            },
            db_reachable=True,
        )

    def test_webhook_rejects_missing_signature(self):
        """Endpoint must raise 400 when stripe-signature header is absent."""
        from fastapi.testclient import TestClient

        self.bm.stripe.Webhook.construct_event.side_effect = Exception("missing signature")
        client = TestClient(self.bm.app, raise_server_exceptions=False)
        resp = client.post("/webhook", content=b"{}", headers={"content-type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_webhook_rejects_bad_signature(self):
        """Endpoint must raise 400 on tampered payload."""
        self.bm.stripe.Webhook.construct_event.side_effect = Exception("invalid signature")
        resp = _client(self.bm).post(
            "/webhook",
            content=b'{"type":"test"}',
            headers={"content-type": "application/json", "stripe-signature": "bad"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_webhook_rejects_missing_webhook_configuration(self):
        bm = _load_billing_module(
            {
                "STRIPE_WEBHOOK_SECRET": "",
                "DATABASE_URL": "postgresql://example.test/db",
            }
        )
        response = _client(bm).post("/webhook", content=b"{}")
        self.assertEqual(response.status_code, 500)
        bm.stripe.Webhook.construct_event.assert_not_called()


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServiceCheckoutTest(unittest.TestCase):
    def setUp(self):
        self.bm = _configured_module()
        self.bm._get_user_by_email = MagicMock(
            return_value={"username": "member@example.com", "tier": "basic", "stripe_customer_id": None}
        )
        self.bm.stripe.checkout.Session.create.return_value = types.SimpleNamespace(
            url="https://checkout.test/session"
        )

    def _checkout(self, plan: str):
        return _client(self.bm).post(
            "/create-checkout-session",
            json={"email": " Member@Example.com ", "plan": plan},
        )

    def test_pro_checkout_uses_only_configured_pro_price_and_metadata(self):
        response = self._checkout("pro")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"checkout_url": "https://checkout.test/session", "mode": "checkout"})
        kwargs = self.bm.stripe.checkout.Session.create.call_args.kwargs
        self.assertEqual(kwargs["line_items"], [{"price": "price_pro_fake", "quantity": 1}])
        self.assertEqual(kwargs["metadata"], {"user_email": "member@example.com", "requested_plan": "pro"})
        self.assertEqual(kwargs["customer_email"], "member@example.com")
        self.assertIn("checkout=success", kwargs["success_url"])
        self.assertIn("checkout=cancel", kwargs["cancel_url"])
        self.assertNotIn("price_premium_fake", repr(kwargs))

    def test_premium_checkout_uses_only_configured_premium_price(self):
        response = self._checkout("premium")
        self.assertEqual(response.status_code, 200)
        kwargs = self.bm.stripe.checkout.Session.create.call_args.kwargs
        self.assertEqual(kwargs["line_items"], [{"price": "price_premium_fake", "quantity": 1}])
        self.assertEqual(kwargs["metadata"]["requested_plan"], "premium")
        self.assertNotIn("price_pro_fake", repr(kwargs["line_items"]))

    def test_forged_and_unknown_plans_are_rejected_before_stripe(self):
        for plan in ("admin", "price_pro_fake", "enterprise", ""):
            with self.subTest(plan=plan):
                response = self._checkout(plan)
                self.assertEqual(response.status_code, 400)
        self.bm.stripe.checkout.Session.create.assert_not_called()

    def test_missing_price_configuration_fails_before_user_or_stripe_lookup(self):
        bm = _configured_module()
        bm.STRIPE_PRICE_PRO = ""
        bm._get_user_by_email = MagicMock()
        response = _client(bm).post(
            "/create-checkout-session",
            json={"email": "member@example.com", "plan": "pro"},
        )
        self.assertEqual(response.status_code, 500)
        bm._get_user_by_email.assert_not_called()
        bm.stripe.checkout.Session.create.assert_not_called()

    def test_missing_user_and_database_failure_do_not_call_stripe(self):
        self.bm._get_user_by_email.return_value = {}
        self.assertEqual(self._checkout("pro").status_code, 404)
        self.bm._get_user_by_email.side_effect = RuntimeError("database unavailable")
        self.assertEqual(self._checkout("pro").status_code, 503)
        self.bm.stripe.checkout.Session.create.assert_not_called()

    def test_existing_active_customer_is_sent_to_portal(self):
        self.bm._get_user_by_email.return_value = {
            "username": "member@example.com", "tier": "pro", "stripe_customer_id": "cus_existing"
        }
        self.bm.stripe.Subscription.list.return_value = {"data": [{"id": "sub_existing"}]}
        self.bm.stripe.billing_portal.Session.create.return_value = types.SimpleNamespace(
            url="https://billing.test/portal"
        )
        response = self._checkout("premium")
        self.assertEqual(response.json(), {"portal_url": "https://billing.test/portal", "mode": "portal"})
        self.bm.stripe.checkout.Session.create.assert_not_called()


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServiceWebhookLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.bm = _configured_module()
        self.bm._set_user_plan_by_email = MagicMock()
        self.bm._mark_event_processed = MagicMock()
        self.bm._is_event_processed = MagicMock(return_value=False)

        @contextmanager
        def db_conn():
            with _idempotency_db() as connection:
                yield connection

        self.bm._db_conn = db_conn

    def test_checkout_completion_grants_pro_from_configured_price(self):
        self.bm.stripe.Subscription.retrieve.return_value = {
            "items": {"data": [{"price": {"id": "price_pro_fake"}}]}
        }
        response = _webhook(
            self.bm,
            _stripe_event(
                "checkout.session.completed",
                {
                    "id": "cs_test",
                    "customer": "cus_test",
                    "subscription": "sub_test",
                    "metadata": {"user_email": "Member@Example.com", "requested_plan": "premium"},
                },
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.bm._set_user_plan_by_email.assert_called_once_with(
            email="member@example.com",
            tier="pro",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            stripe_price_id="price_pro_fake",
        )

    def test_checkout_completion_grants_premium_and_entitlements_match(self):
        from ui.app_session import compute_entitlements

        self.bm.stripe.Subscription.retrieve.return_value = {
            "items": {"data": [{"price": {"id": "price_premium_fake"}}]}
        }
        response = _webhook(
            self.bm,
            _stripe_event(
                "checkout.session.completed",
                {
                    "id": "cs_test", "customer": "cus_test", "subscription": "sub_test",
                    "metadata": {"user_email": "member@example.com", "requested_plan": "premium"},
                },
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bm._set_user_plan_by_email.call_args.kwargs["tier"], "premium")
        flags = compute_entitlements(tier_obj="premium", is_admin=False)
        self.assertTrue(flags["can_ai_notes"])
        self.assertTrue(flags["can_scan_nasdaq"])

    def test_subscription_update_handles_upgrade_and_downgrade(self):
        from ui.app_session import compute_entitlements

        self.bm.stripe.Customer.retrieve.return_value = {"email": "member@example.com"}
        for price_id, expected in (("price_premium_fake", "premium"), ("price_pro_fake", "pro")):
            with self.subTest(expected=expected):
                self.bm._set_user_plan_by_email.reset_mock()
                response = _webhook(
                    self.bm,
                    _stripe_event(
                        "customer.subscription.updated",
                        {
                            "id": f"sub_{expected}", "customer": "cus_test", "status": "active",
                            "cancel_at_period_end": False,
                            "items": {"data": [{"price": {"id": price_id}}]},
                        },
                        event_id=f"evt_{expected}",
                    ),
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.bm._set_user_plan_by_email.call_args.kwargs["tier"], expected)
        pro = compute_entitlements(tier_obj="pro", is_admin=False)
        self.assertTrue(pro["can_scan_nasdaq"])
        self.assertFalse(pro["can_ai_notes"])

    def test_scheduled_cancel_keeps_tier_until_effective(self):
        self.bm.stripe.Customer.retrieve.return_value = {"email": "member@example.com"}
        response = _webhook(
            self.bm,
            _stripe_event(
                "customer.subscription.updated",
                {
                    "id": "sub_test", "customer": "cus_test", "status": "active",
                    "cancel_at_period_end": True,
                },
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "cancel_scheduled_keep_tier")
        self.bm._set_user_plan_by_email.assert_not_called()

    def test_effective_cancel_downgrades_to_free_entitlements(self):
        from ui.app_session import compute_entitlements

        self.bm.stripe.Customer.retrieve.return_value = {"email": "member@example.com"}
        response = _webhook(
            self.bm,
            _stripe_event(
                "customer.subscription.deleted",
                {"id": "sub_test", "customer": "cus_test"},
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bm._set_user_plan_by_email.call_args.kwargs["tier"], "basic")
        free = compute_entitlements(tier_obj="basic", is_admin=False)
        self.assertTrue(free["can_scan_sp500"])
        self.assertFalse(free["can_scan_nasdaq"])
        self.assertFalse(free["can_ai_notes"])

    def test_unknown_price_fails_closed_to_basic(self):
        self.bm.stripe.Customer.retrieve.return_value = {"email": "member@example.com"}
        response = _webhook(
            self.bm,
            _stripe_event(
                "customer.subscription.updated",
                {
                    "id": "sub_test", "customer": "cus_test", "status": "active",
                    "cancel_at_period_end": False,
                    "items": {"data": [{"price": {"id": "price_forged"}}]},
                },
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bm._set_user_plan_by_email.call_args.kwargs["tier"], "basic")

    def test_missing_customer_mapping_and_database_write_failure_do_not_grant(self):
        self.bm.stripe.Customer.retrieve.return_value = {"email": ""}
        event = _stripe_event(
            "customer.subscription.deleted", {"id": "sub_test", "customer": "cus_unknown"}
        )
        self.assertEqual(_webhook(self.bm, event).status_code, 500)
        self.bm._set_user_plan_by_email.assert_not_called()

        self.bm.stripe.Customer.retrieve.return_value = {"email": "member@example.com"}
        self.bm._set_user_plan_by_email.side_effect = RuntimeError("write failed")
        self.assertEqual(_webhook(self.bm, event).status_code, 500)

    def test_duplicate_event_is_skipped_without_reapplying_tier(self):
        self.bm._is_event_processed.return_value = True
        response = _webhook(
            self.bm,
            _stripe_event("customer.subscription.deleted", {"id": "sub_test", "customer": "cus_test"}),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["note"], "already_processed")
        self.bm._set_user_plan_by_email.assert_not_called()

    def test_malformed_signed_checkout_event_fails_without_grant(self):
        response = _webhook(
            self.bm,
            _stripe_event("checkout.session.completed", {"id": "cs_bad", "metadata": {}}),
        )
        self.assertEqual(response.status_code, 500)
        self.bm._set_user_plan_by_email.assert_not_called()


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingContractTest(unittest.TestCase):
    def test_customer_prices_tiers_and_checkout_mapping_stay_aligned(self):
        from ui.pricing import PRICES, TIER_NAMES, TIERS

        bm = _configured_module()
        self.assertEqual(TIERS, ("basic", "pro", "premium"))
        self.assertEqual(TIER_NAMES, {"basic": "Free", "pro": "Pro", "premium": "Premium"})
        self.assertEqual(PRICES, {"basic": "Free", "pro": "$19/mo", "premium": "$39/mo"})
        self.assertEqual(bm._price_to_plan(bm.STRIPE_PRICE_PRO), "pro")
        self.assertEqual(bm._price_to_plan(bm.STRIPE_PRICE_PREMIUM), "premium")
        self.assertEqual(bm._price_to_plan("unconfigured"), "basic")


if __name__ == "__main__":
    unittest.main()
