"""Run 83 (B1) — billing checkout/portal endpoints require the signed-in account.

Exercises billing_service.main through FastAPI's TestClient with Stripe mocked.
The account token check runs the service's real SQL against an in-memory SQLite
stand-in populated by the app-side issuer (ui/auth_tokens), so issue → consume →
replay is tested end to end. No network, no real Stripe. Runs in the
`billing-contract` CI job.
"""
import importlib.util
import types
import unittest
from unittest.mock import MagicMock

from _token_db import TokenDB
from test_billing_service import _client, _configured_module

from ui import auth_tokens as at

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

USERS = {
    "alice@example.com": {"username": "alice@example.com", "tier": "pro", "stripe_customer_id": "cus_ALICE"},
    "bob@example.com": {"username": "bob@example.com", "tier": "premium", "stripe_customer_id": "cus_BOB"},
    "newbie@example.com": {"username": "newbie@example.com", "tier": "basic", "stripe_customer_id": None},
}


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingAuthTests(unittest.TestCase):
    def setUp(self):
        self.db = TokenDB()
        self.addCleanup(self.db.shutdown)
        self.bm = _configured_module()
        self.bm._db_conn = lambda: self.db                          # real token SQL, fake DB
        self.bm._get_user_by_email = MagicMock(side_effect=lambda e: dict(USERS.get(e, {})))
        self.bm._set_user_plan_by_email = MagicMock()
        self.bm.stripe.billing_portal.Session.create.side_effect = (
            lambda customer, return_url: types.SimpleNamespace(url=f"https://portal.test/{customer}"))
        self.bm.stripe.Subscription.list.return_value = {"data": [{"id": "sub_1"}]}
        self.bm.stripe.checkout.Session.create.return_value = types.SimpleNamespace(url="https://checkout.test/s")
        self.client = _client(self.bm)

    def token(self, user, purpose="billing"):
        return at.issue_token(user, purpose, conn=self.db)

    def portal(self, token=None, **payload):
        headers = {"X-HSF-Auth": token} if token is not None else {}
        return self.client.post("/create-portal-session", json=payload, headers=headers)

    def assert_no_stripe(self):
        self.bm.stripe.billing_portal.Session.create.assert_not_called()
        self.bm.stripe.checkout.Session.create.assert_not_called()
        self.bm.stripe.Subscription.list.assert_not_called()

    # --- signed out / bad identity -------------------------------------------
    def test_signed_out_request_is_refused_without_calling_stripe(self):
        r = self.portal(email="alice@example.com")
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("portal_url", r.json())
        self.assert_no_stripe()
        self.bm._get_user_by_email.assert_not_called()

    def test_malformed_unknown_and_expired_tokens_are_refused(self):
        for bad in ("", "garbage", "x" * 400):
            self.assertEqual(self.portal(token=bad).status_code, 401, bad)
        restore = self.token("alice@example.com", purpose="restore")  # wrong purpose
        self.assertEqual(self.portal(token=restore).status_code, 401)
        self.assert_no_stripe()

    def test_token_is_single_use(self):
        tok = self.token("alice@example.com")
        self.assertEqual(self.portal(token=tok).status_code, 200)
        r = self.portal(token=tok)
        self.assertEqual(r.status_code, 401)
        self.assertEqual(self.bm.stripe.billing_portal.Session.create.call_count, 1)

    # --- valid user -----------------------------------------------------------
    def test_valid_user_gets_their_own_portal(self):
        r = self.portal(token=self.token("alice@example.com"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"portal_url": "https://portal.test/cus_ALICE"})
        self.bm._get_user_by_email.assert_called_once_with("alice@example.com")

    def test_cancel_flow_uses_the_authenticated_accounts_subscription(self):
        self.bm.stripe.billing_portal.Session.create.side_effect = None
        self.bm.stripe.billing_portal.Session.create.return_value = types.SimpleNamespace(
            url="https://portal.test/cancel"
        )
        self.bm.stripe.Subscription.list.return_value = {"data": [{"id": "sub_ALICE"}]}

        r = self.portal(token=self.token("alice@example.com"), flow="cancel")

        self.assertEqual(r.status_code, 200)
        self.bm.stripe.Subscription.list.assert_called_once_with(
            customer="cus_ALICE", status="active", limit=1
        )
        kwargs = self.bm.stripe.billing_portal.Session.create.call_args.kwargs
        self.assertEqual(kwargs["customer"], "cus_ALICE")
        self.assertEqual(kwargs["flow_data"]["subscription_cancel"]["subscription"], "sub_ALICE")

    # --- cross-account --------------------------------------------------------
    def test_user_a_cannot_request_user_b_portal(self):
        r = self.portal(token=self.token("alice@example.com"), email="bob@example.com")
        self.assertEqual(r.status_code, 403)
        self.assertNotIn("cus_BOB", r.text)
        self.assert_no_stripe()

    def test_forged_customer_id_is_ignored(self):
        r = self.portal(token=self.token("alice@example.com"),
                        customer="cus_BOB", customer_id="cus_BOB", stripe_customer_id="cus_BOB")
        self.assertEqual(r.json(), {"portal_url": "https://portal.test/cus_ALICE"})
        kwargs = self.bm.stripe.billing_portal.Session.create.call_args.kwargs
        self.assertEqual(kwargs["customer"], "cus_ALICE")

    # --- failure modes --------------------------------------------------------
    def test_missing_customer_mapping_fails_safely(self):
        r = self.portal(token=self.token("newbie@example.com"))
        self.assertEqual(r.status_code, 400)
        self.assert_no_stripe()

    def test_account_database_down_fails_closed(self):
        def down():
            raise RuntimeError("connection refused to db.internal:5432")
        self.bm._db_conn = down
        r = self.portal(token="anything")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("db.internal", r.text)                     # no internals in user-facing error
        self.assert_no_stripe()

    def test_stripe_error_fails_safely_without_entitlement_change(self):
        self.bm.stripe.billing_portal.Session.create.side_effect = RuntimeError("No such customer: cus_ALICE")
        r = self.portal(token=self.token("alice@example.com"))
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("cus_ALICE", r.text)
        self.bm._set_user_plan_by_email.assert_not_called()

    # --- checkout endpoint (returns a portal for active subscribers) ----------
    def test_checkout_requires_the_account_and_uses_its_identity(self):
        r = self.client.post("/create-checkout-session", json={"email": "bob@example.com", "plan": "pro"})
        self.assertEqual(r.status_code, 401)
        self.assert_no_stripe()
        r = self.client.post("/create-checkout-session", json={"email": "bob@example.com", "plan": "pro"},
                             headers={"X-HSF-Auth": self.token("alice@example.com")})
        self.assertEqual(r.status_code, 403)
        self.assert_no_stripe()
        r = self.client.post("/create-checkout-session", json={"plan": "premium"},
                             headers={"X-HSF-Auth": self.token("alice@example.com")})
        self.assertEqual(r.json(), {"portal_url": "https://portal.test/cus_ALICE", "mode": "portal"})

    def test_debug_status_hidden_unless_enabled(self):
        self.assertEqual(self.client.get("/debug/status").status_code, 404)


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class TokenSqlParityTests(unittest.TestCase):
    def test_billing_service_uses_the_same_token_sql_as_the_app(self):
        bm = _configured_module()
        self.assertEqual(bm._TOKENS_SCHEMA_SQL, at.SCHEMA_SQL)
        self.assertEqual(bm._TOKENS_CONSUME_SQL, at.CONSUME_SQL)
        self.assertEqual(bm.AUTH_HEADER, "X-HSF-Auth")


if __name__ == "__main__":
    unittest.main()
