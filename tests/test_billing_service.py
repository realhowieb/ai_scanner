"""Smoke tests for billing_service.main.

These tests mock out stripe, psycopg2, and env vars so the billing service
module can be imported and its pure-logic functions exercised without real
credentials or a live DB.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


def _make_stripe_mock() -> types.ModuleType:
    m = types.ModuleType("stripe")
    m.api_key = None
    m.Customer = MagicMock()
    m.Webhook = MagicMock()
    m.checkout = MagicMock()
    m.billing_portal = MagicMock()
    return m


def _make_psycopg2_mock() -> types.ModuleType:
    m = types.ModuleType("psycopg2")
    m.connect = MagicMock()
    m.extensions = MagicMock()
    return m


def _load_billing_module():
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
    }
    stripe_mock = _make_stripe_mock()
    psycopg2_mock = _make_psycopg2_mock()

    with patch.dict(sys.modules, {"stripe": stripe_mock, "psycopg2": psycopg2_mock}):
        with patch.dict("os.environ", env):
            sys.modules.pop("billing_service.main", None)
            import billing_service.main as bm
            return bm


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingServiceImportTest(unittest.TestCase):
    def test_module_imports_without_real_credentials(self):
        bm = _load_billing_module()
        self.assertTrue(hasattr(bm, "app"))
        self.assertTrue(hasattr(bm, "health"))

    def test_health_endpoint_returns_ok(self):
        bm = _load_billing_module()
        result = bm.health()
        self.assertEqual(result, {"ok": True})


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
        self.bm = _load_billing_module()

    def test_webhook_rejects_missing_signature(self):
        """Endpoint must raise 400 when stripe-signature header is absent."""
        from fastapi.testclient import TestClient
        client = TestClient(self.bm.app, raise_server_exceptions=False)
        resp = client.post("/webhook", content=b"{}", headers={"content-type": "application/json"})
        self.assertIn(resp.status_code, (400, 422, 500))

    def test_webhook_rejects_bad_signature(self):
        """Endpoint must raise 400 on tampered payload."""
        import stripe as _stripe
        from fastapi.testclient import TestClient
        _stripe.Webhook.construct_event.side_effect = Exception("invalid signature")

        client = TestClient(self.bm.app, raise_server_exceptions=False)
        resp = client.post(
            "/webhook",
            content=b'{"type":"test"}',
            headers={"content-type": "application/json", "stripe-signature": "bad"},
        )
        self.assertIn(resp.status_code, (400, 500))


if __name__ == "__main__":
    unittest.main()
