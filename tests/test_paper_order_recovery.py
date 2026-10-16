import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from services import paper_orders as orders
from tests.paper_fakes import MemoryOrders, PaperBroker


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.store, self.broker, self.fills = MemoryOrders(), PaperBroker(), {}
        for target, fn in (
            ("services.paper_orders.store.reserve", self.store.reserve),
            ("services.paper_orders.store.get", self.store.get),
            ("services.paper_orders.store.save", self.store.save),
            ("services.paper_orders.store.pending", self.store.pending),
            ("services.paper_orders.broker.get_order_by_client_id", self.broker.lookup),
            ("services.paper_orders.broker.submit_market_order", self.broker.submit),
            ("services.paper_orders.broker._base_url", lambda: "https://paper-api.alpaca.markets"),
            ("db.paper_trading.get_paper_account", lambda user: {"api_key": user, "api_secret": "fake"}),
            ("services.paper_orders.reconcile_paper_fill", self.import_fill),
        ):
            handle = patch(target, side_effect=fn)
            handle.start()
            self.addCleanup(handle.stop)

    def import_fill(self, user, oid, ticker, price, qty, context):
        self.fills[user, oid] = (ticker, price, qty)

    def send(self, user="a", key="confirm-123", ticker="AAPL", qty=2):
        return orders.submit(user, ticker, qty, key)

    def fill(self, user="a", key="confirm-123", qty="2", status="filled"):
        cid = orders.client_id(user, key)
        self.broker.orders[user, cid].update(filled_avg_price="101.25", filled_qty=qty, status=status)

    def test_retry_and_concurrent_confirmations_submit_once(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.send(), range(8)))
        self.assertEqual(len(self.broker.posts), 1)
        self.assertEqual({r["order_id"] for r in results}, {"o1"})
        self.assertEqual(self.fills, {})  # acceptance is not a fill
        self.fill()
        orders.reconcile_pending("a")
        orders.reconcile_pending("a")
        self.assertEqual(self.fills, {("a", "o1"): ("AAPL", 101.25, 2)})

    def test_lost_broker_response_is_recovered_without_resubmission(self):
        def lost(*args, **kw):
            self.broker.submit(*args, **kw)
            return {"ok": False, "uncertain": True}

        with patch.object(orders.broker, "submit_market_order", side_effect=lost):
            self.assertEqual(self.send()["order_id"], "o1")
        self.assertEqual(len(self.broker.posts), 1)

    def test_durable_intent_survives_response_storage_failure(self):
        with patch.object(orders.store, "save", side_effect=RuntimeError("db unavailable")):
            with self.assertRaises(orders.PaperUnavailable):
                self.send()
        self.assertEqual(self.send()["order_id"], "o1")
        self.assertEqual(len(self.broker.posts), 1)

    def test_journal_failure_and_partial_fills_reconcile_later(self):
        self.send()
        self.fill(qty="1", status="partially_filled")
        with patch.object(orders, "reconcile_paper_fill", side_effect=RuntimeError("db unavailable")):
            result = self.send()
        self.assertFalse(result["journal_synced"])
        self.assertEqual(self.fills, {})
        orders.reconcile_pending("a")
        self.assertEqual(self.fills["a", "o1"][2], 1)
        self.fill()
        orders.reconcile_pending("a")
        self.assertEqual(self.fills["a", "o1"][2], 2)
        self.assertEqual(len(self.fills), 1)

    def test_same_key_different_payload_or_account_is_refused(self):
        self.send()
        with self.assertRaises(orders.OrderConflict):
            self.send(qty=3)
        with patch("db.paper_trading.get_paper_account", return_value={"api_key": "changed", "api_secret": "fake"}):
            with self.assertRaises(orders.OrderConflict):
                self.send()
        self.send(user="b")
        self.assertEqual(len(self.broker.posts), 2)

    def test_unknown_lookup_or_unavailable_storage_never_submits(self):
        with patch.object(orders.store, "reserve", side_effect=RuntimeError("db down")):
            with self.assertRaises(orders.PaperUnavailable):
                self.send()
        with patch.object(orders.broker, "get_order_by_client_id", return_value={"ok": False}):
            with self.assertRaises(orders.PaperUnavailable):
                self.send()
        self.assertEqual(self.broker.posts, [])

    def test_live_endpoint_is_refused_before_reserving_or_submitting(self):
        with patch.object(orders.broker, "_base_url", return_value="https://api.alpaca.markets"):
            with self.assertRaises(orders.PaperUnavailable):
                self.send()
        self.assertEqual(self.store.rows, {})

    def test_confirmed_rejection_does_not_resubmit_on_retry(self):
        with patch.object(
            orders.broker, "submit_market_order", return_value={"ok": False, "status_code": 403}
        ) as submit:
            for _ in range(2):
                with self.assertRaises(orders.PaperRejected):
                    self.send()
            self.assertEqual(submit.call_count, 1)
        self.assertEqual(self.fills, {})

    def test_key_rotation_can_recover_an_existing_order_but_never_resubmit_it(self):
        self.send()
        self.fill()
        cid = orders.client_id("a", "confirm-123")
        self.broker.orders["rotated", cid] = self.broker.orders["a", cid]
        with patch("db.paper_trading.get_paper_account", return_value={"api_key": "rotated", "api_secret": "fake"}):
            orders.reconcile_pending("a")
            self.assertTrue(self.send()["journal_synced"])
        self.assertEqual(len(self.broker.posts), 1)
        self.assertEqual(self.fills, {("a", "o1"): ("AAPL", 101.25, 2)})

    def test_outage_after_key_rotation_remains_retryable_not_an_account_conflict(self):
        self.send()
        with (
            patch("db.paper_trading.get_paper_account", return_value={"api_key": "rotated", "api_secret": "fake"}),
            patch.object(orders.broker, "get_order_by_client_id", return_value={"ok": False}),
        ):
            with self.assertRaises(orders.PaperUnavailable):
                self.send()
        self.assertEqual(len(self.broker.posts), 1)


    def test_status_is_owned_and_never_submits_an_unknown_confirmation(self):
        self.assertIsNone(orders.status("a", "confirm-123"))
        self.send()
        self.assertIsNone(orders.status("b", "confirm-123"))
        self.assertEqual(len(self.broker.posts), 1)

    def test_status_recovers_partial_and_full_fills_without_resubmitting(self):
        self.send()
        self.assertEqual(orders.status("a", "confirm-123")["status"], "accepted")
        self.fill(qty="1", status="partially_filled")
        partial = orders.status("a", "confirm-123")
        self.assertEqual(partial["filled_qty"], 1)
        self.assertTrue(partial["journal_synced"])
        self.fill()
        complete = orders.status("a", "confirm-123")
        self.assertEqual(complete["state"], "synced")
        self.assertTrue(complete["journal_synced"])
        self.assertEqual(len(self.broker.posts), 1)

    def test_status_keeps_saved_receipt_when_broker_cannot_confirm(self):
        self.send()
        with patch.object(orders.broker, "get_order_by_client_id", return_value={"ok": False}):
            receipt = orders.status("a", "confirm-123")
        self.assertEqual(receipt["status"], "accepted")
        self.assertFalse(receipt["broker_available"])
        self.assertEqual(len(self.broker.posts), 1)

    def test_status_rejects_malformed_broker_fill_quantities(self):
        self.send()
        for quantity in ("nan", "inf", "-1", "3", "0.5"):
            with self.subTest(quantity=quantity):
                self.fill(qty=quantity)
                with self.assertRaises(orders.PaperUnavailable):
                    orders.status("a", "confirm-123")
        self.assertEqual(len(self.broker.posts), 1)

    def test_completed_checks_and_retries_do_not_recreate_deleted_journal_rows(self):
        self.send()
        self.fill()
        orders.status("a", "confirm-123")
        self.fills.clear()  # user removed the imported trade
        self.assertTrue(orders.status("a", "confirm-123")["journal_synced"])
        self.assertTrue(self.send()["journal_synced"])
        self.assertFalse(self.fills)

    def test_status_with_changed_credentials_never_imports_an_unconfirmed_order(self):
        self.send()
        with patch("db.paper_trading.get_paper_account", return_value={"api_key": "other", "api_secret": "fake"}):
            self.assertFalse(orders.status("a", "confirm-123")["broker_available"])
        self.assertEqual(len(self.broker.posts), 1)
        self.assertFalse(self.fills)
