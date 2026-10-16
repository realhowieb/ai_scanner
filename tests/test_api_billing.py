"""Embedded billing contracts without live Stripe or database access."""
import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "jwt"))


@unittest.skipUnless(DEPS, "needs API test dependencies")
class EmbeddedBillingTest(unittest.TestCase):
    def setUp(self):
        # Import the API before enabling billing so module startup stays inert.
        from test_billing_service import _configured_module

        from api import main, ratelimit
        from api.settings import Settings

        self.main = main
        self.settings = Settings(jwt_secret="billing-test-" + "x" * 40, access_ttl_s=900,
                                 refresh_ttl_s=3600, cors_origins=())
        self.billing = _configured_module()
        package = types.ModuleType("billing_service")
        package.main = self.billing
        modules = patch.dict(sys.modules, {"billing_service": package, "billing_service.main": self.billing})
        modules.start()
        self.addCleanup(modules.stop)
        self.env = patch.dict("os.environ", {"BILLING_MODE": "embedded"})
        self.env.start()
        self.addCleanup(self.env.stop)
        ratelimit.reset()
        self.addCleanup(ratelimit.reset)
        self.app = main.create_app(self.settings)
        from fastapi.testclient import TestClient

        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        returns = patch("services.billing._build_return_urls", return_value=(None, None))
        returns.start()
        self.addCleanup(returns.stop)

    def sign_in(self):
        from api.deps import current_account

        self.app.dependency_overrides[current_account] = lambda: {"username": "member@example.com", "tier": "basic"}

    def test_external_mode_does_not_load_billing_or_mount_routes(self):
        with patch.dict("os.environ", {"BILLING_MODE": "external", "STRIPE_SECRET_KEY": ""}):
            app = self.main.create_app(self.settings)
        self.assertFalse(any(getattr(r, "path", "") == "/billing" for r in app.routes))

    def test_invalid_mode_fails_configuration(self):
        with patch.dict("os.environ", {"BILLING_MODE": "embeded"}):
            with self.assertRaisesRegex(RuntimeError, "BILLING_MODE"):
                self.main.create_app(self.settings)

    def test_missing_configuration_blocks_embedded_startup(self):
        with patch.object(self.billing, "STRIPE_WEBHOOK_SECRET", ""):
            with self.assertRaisesRegex(RuntimeError, "STRIPE_WEBHOOK_SECRET"):
                self.main.create_app(self.settings)

    def test_session_routes_require_api_authentication(self):
        for path in ("checkout", "portal"):
            response = self.client.post(f"/v1/billing/{path}", json={"plan": "pro"})
            self.assertEqual(response.status_code, 401)

    def test_unverified_checkout_is_blocked(self):
        self.sign_in()
        with patch("api.account.verification_status", return_value=False), \
                patch.object(self.billing, "checkout_for_account") as checkout:
            self.assertEqual(self.client.post("/v1/billing/checkout", json={"plan": "pro"}).status_code, 403)
        checkout.assert_not_called()

    def test_checkout_uses_authenticated_account_without_remote_call_or_token(self):
        self.sign_in()
        with patch("api.account.verification_status", return_value=True), \
                patch.object(self.billing, "checkout_for_account", return_value={"checkout_url": "https://checkout.stripe.com/test"}) as checkout, \
                patch("services.billing.create_checkout_url") as remote, \
                patch("services.billing.billing_auth_headers") as token:
            response = self.client.post("/v1/billing/checkout", json={"plan": "pro", "email": "other@example.com"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"url": "https://checkout.stripe.com/test", "mode": "checkout"})
        checkout.assert_called_once_with("member@example.com", {"plan": "pro", "interval": "month"})
        remote.assert_not_called()
        token.assert_not_called()

    def test_portal_cancel_uses_authenticated_account_directly(self):
        self.sign_in()
        with patch.object(self.billing, "portal_for_account", return_value={"portal_url": "https://billing.stripe.com/test"}) as portal, \
                patch("services.billing.billing_auth_headers") as token:
            response = self.client.post("/v1/billing/portal", json={"flow": "cancel"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "portal")
        portal.assert_called_once_with("member@example.com", {"flow": "cancel"})
        token.assert_not_called()

    def test_checkout_failure_is_safe(self):
        from fastapi import HTTPException

        self.sign_in()
        with patch("api.account.verification_status", return_value=True), \
                patch.object(self.billing, "checkout_for_account", side_effect=HTTPException(500, "private Stripe error")):
            response = self.client.post("/v1/billing/checkout", json={"plan": "pro"})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertNotIn("private Stripe", response.text)

    def test_portal_without_subscription_preserves_api_404(self):
        from fastapi import HTTPException

        self.sign_in()
        with patch.object(self.billing, "portal_for_account", side_effect=HTTPException(400, "No subscription")):
            response = self.client.post("/v1/billing/portal", json={})
        self.assertEqual(response.status_code, 404, response.text)

    def test_webhook_preserves_raw_body_and_does_not_require_jwt(self):
        from fastapi.responses import JSONResponse

        raw = b'{ "id": "evt_raw", "data": {} }'
        with patch.object(self.billing, "process_webhook", return_value=JSONResponse({"received": True})) as process:
            response = self.client.post("/billing/webhook", content=raw, headers={"stripe-signature": "signature"})
        self.assertEqual(response.status_code, 200, response.text)
        process.assert_called_once_with(raw, "signature")

    def test_invalid_webhook_signature_does_not_write_account(self):
        self.billing.stripe.Webhook.construct_event.side_effect = ValueError("bad signature")
        with patch.object(self.billing, "_set_user_plan_by_email") as update:
            response = self.client.post("/billing/webhook", content=b"{}")
        self.assertEqual(response.status_code, 400, response.text)
        update.assert_not_called()

    def test_existing_customer_failure_cannot_create_second_subscription(self):
        self.billing._get_user_by_email = MagicMock(return_value={"stripe_customer_id": "cus_existing"})
        self.billing.stripe.Subscription.list.side_effect = RuntimeError("Stripe unavailable")
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as error:
            self.billing.checkout_for_account("member@example.com", {"plan": "pro"})
        self.assertEqual(error.exception.status_code, 502)
        self.billing.stripe.checkout.Session.create.assert_not_called()

    def test_embedded_preflight_checks_exact_endpoint(self):
        self.billing.stripe.WebhookEndpoint = MagicMock()
        self.billing.stripe.WebhookEndpoint.list.return_value.auto_paging_iter.return_value = iter([
            {"url": "https://old.example.com/webhook", "status": "enabled", "enabled_events": ["*"]},
            {"url": "https://wrong.example.com/billing/webhook", "status": "enabled", "enabled_events": ["*"]},
        ])
        with patch.dict("os.environ", {"BILLING_WEBHOOK_URL": "https://api.example.com/billing/webhook"}):
            self.assertTrue(self.billing._check_webhooks()["problems"])

    def test_mount_does_not_start_second_alert_worker(self):
        with patch.object(self.billing, "_start_realtime_alerts") as worker, self.client:
            self.assertEqual(self.client.get("/billing/healthz").status_code, 200)
        worker.assert_not_called()
