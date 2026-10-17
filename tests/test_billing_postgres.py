"""Real Postgres rollback/concurrency tests; only an explicit throwaway URL is used."""
import concurrent.futures
import importlib.util
import os
import time
import unittest
import uuid
from unittest.mock import MagicMock

ENABLED = bool(os.getenv("HSF_TEST_BILLING_PG_URL")) and all(
    importlib.util.find_spec(m) for m in ("psycopg2", "fastapi"))


@unittest.skipUnless(ENABLED, "set HSF_TEST_BILLING_PG_URL to a throwaway Postgres")
class BillingPostgresTest(unittest.TestCase):
    def setUp(self):
        import psycopg2
        from psycopg2 import sql
        from test_billing_service import _configured_module

        self.driver, self.sql = psycopg2, sql
        self.url = os.environ["HSF_TEST_BILLING_PG_URL"]
        self.schema = "hsf_billing_test_" + uuid.uuid4().hex
        with psycopg2.connect(self.url) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        self.connect = lambda: psycopg2.connect(self.url, options=f"-csearch_path={self.schema}")
        conn = self.connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("CREATE TABLE users(username TEXT PRIMARY KEY,tier TEXT,is_admin BOOLEAN DEFAULT FALSE,"
                            "stripe_customer_id TEXT,stripe_subscription_id TEXT,stripe_price_id TEXT,plan_updated_at TIMESTAMPTZ)")
                cur.execute("INSERT INTO users(username,tier,stripe_customer_id) VALUES ('member@example.com','pro','cus_member')")
        finally:
            conn.close()
        self.b = _configured_module()
        # Import again after fixture module isolation; use the real driver factory.
        from billing_service.state import PostgresStore

        self.store = PostgresStore(self.connect)
        self.engine = type(self.b._billing_engine)(self.b, self.store)
        self.email = "member@example.com"
        self.b._account_for_customer = MagicMock(return_value=self.email)
        self.b._record_paid_conversion = MagicMock()
        self.b.stripe.Subscription.list.side_effect = self.current_subscription
        self.b.stripe.Webhook.construct_event.return_value = {
            "id": "evt_pg", "type": "customer.subscription.updated", "livemode": False,
            "data": {"object": {"customer": "cus_member", "status": "active"}}}

    def drop_schema(self):
        conn = self.driver.connect(self.url)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self.sql.SQL("DROP SCHEMA {} CASCADE").format(self.sql.Identifier(self.schema)))
        finally:
            conn.close()

    def current_subscription(self, **kwargs):
        time.sleep(0.03)  # expose races between independent DB connections
        return {"data": [{"id": "sub_current", "customer": "cus_member", "status": "active",
                          "items": {"data": [{"price": {"id": "price_premium_fake"}}]}}], "has_more": False}

    def tier(self):
        with self.store.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT tier FROM users WHERE username=%s", (self.email,))
            return cur.fetchone()[0]

    def test_schema_is_repeatable_and_duplicate_delivery_applies_once(self):
        self.store.ensure_schema()
        self.store._ready = False
        self.store.ensure_schema()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: self.engine.webhook(b"raw", "sig"), range(4)))
        self.assertTrue(all(reply.status_code == 200 for reply in replies))
        self.assertEqual(self.tier(), "premium")
        self.b.stripe.Subscription.list.assert_called_once()
        with self.store.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM stripe_processed_events")
            self.assertEqual(cur.fetchone()[0], 1)

    def test_plan_and_event_marker_rollback_together(self):
        from fastapi import HTTPException

        self.store.ensure_schema()
        # Preexisting marker causes a unique violation AFTER the account update.
        # Bypass read guards here to exercise the database transaction boundary.
        with self.store.connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO stripe_processed_events(event_id,event_type) VALUES ('evt_pg','test')")
        self.store.processed = MagicMock(return_value=False)
        from unittest.mock import patch

        from billing_service.state import Transaction

        with patch.object(Transaction, "processed", return_value=False):
            with self.assertRaises(HTTPException) as error:
                self.engine.webhook(b"raw", "sig")
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.tier(), "pro")
        self.assertEqual(self.store.read(self.email)[0], {})

    def test_concurrent_checkout_and_restart_reuse_one_intent(self):
        self.b._create_checkout_for_account = MagicMock(return_value={
            "checkout_url": "https://checkout.stripe.com/one", "session_id": "cs_1"})
        payload = {"plan": "pro", "interval": "month"}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: self.engine.checkout(self.email, payload), range(4)))
        self.assertTrue(all(reply["checkout_url"] == "https://checkout.stripe.com/one" for reply in replies))
        self.b._create_checkout_for_account.assert_called_once()
        from billing_service.state import PostgresStore

        restarted = type(self.engine)(self.b, PostgresStore(self.connect))
        self.assertEqual(restarted.checkout(self.email, payload)["checkout_url"], "https://checkout.stripe.com/one")
        self.b._create_checkout_for_account.assert_called_once()

    def test_account_deletion_removes_private_billing_state(self):
        self.engine.sync(self.email)
        with self.store.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (self.email,))
        self.assertEqual(self.store.read(self.email), ({}, {}))

    def test_reconciliation_leases_are_shared_between_instances(self):
        from billing_service.state import PostgresStore

        self.assertEqual(self.store.claim_reconciliation(), [self.email])
        self.assertEqual(PostgresStore(self.connect).claim_reconciliation(), [])


    def test_research_preferences_merge_sections_and_isolate_users(self):
        from unittest.mock import patch

        from api import workspace
        with patch.dict(os.environ, {"AI_SCANNER_SCHEMA_ONCE": "0"}), patch("api.workspace._conn", side_effect=self.connect):
            initial = workspace.preferences(self.email)
            self.assertEqual(initial["scanner_views"], [])
            workspace.preferences(self.email, {"scanner_views": [{"name": "Morning", "min": 60}]})
            workspace.preferences(self.email, {"table": {"compact": True}})
            result = workspace.preferences(self.email)
            self.assertTrue(result["table"]["compact"])
            self.assertEqual(result["scanner_views"][0]["min"], 60)
            self.assertEqual(workspace.preferences("other@example.com")["scanner_views"], [])
            # Independent section updates must not clobber one another.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda change: workspace.preferences(self.email, change), [
                    {"table": {"compact": False}}, {"quiet_hours": {"enabled": True}}]))
            result = workspace.preferences(self.email)
            self.assertFalse(result["table"]["compact"])
            self.assertTrue(result["quiet_hours"]["enabled"])

    def test_notification_retry_claims_only_failed_owned_receipts_once(self):
        from unittest.mock import patch

        from api import alert_delivery as delivery
        with patch.dict(os.environ, {"AI_SCANNER_SCHEMA_ONCE": "0"}), patch("api.alert_delivery._conn", side_effect=self.connect):
            key = delivery.begin(self.email, 1, {"body": "Private alert"})
            delivery.finish(key, "failed")
            self.assertEqual(delivery.recent("other@example.com"), [])
            self.assertTrue(delivery.recent(self.email)[0]["retryable"])
            with patch("api.webpush.enabled", return_value=True), patch("api.webpush.plan_allows", return_value=True), \
                 patch("api.store.get_account", return_value={}), patch("api.workspace.is_quiet", return_value=False), \
                 patch("api.devices.devices_for_user", return_value=[{"id": 1, "provider": "webpush", "token": "private-token"}]), \
                 patch("api.webpush.send", return_value=(True, 201)) as send:
                self.assertIsNone(delivery.retry("other@example.com", key))
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    results = list(pool.map(lambda _: delivery.retry(self.email, key), range(3)))
                self.assertEqual(sum(r is not None for r in results), 1)
                send.assert_called_once()
            self.assertEqual(delivery.recent(self.email)[0]["status"], "accepted")
            self.assertFalse(delivery.recent(self.email)[0]["retryable"])
            self.assertNotIn("private-token", str(delivery.recent(self.email)))
