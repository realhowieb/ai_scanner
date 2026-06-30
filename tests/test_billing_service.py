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
        from fastapi.testclient import TestClient

        self.bm.stripe.Webhook.construct_event.side_effect = Exception("invalid signature")

        client = TestClient(self.bm.app, raise_server_exceptions=False)
        resp = client.post(
            "/webhook",
            content=b'{"type":"test"}',
            headers={"content-type": "application/json", "stripe-signature": "bad"},
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
