import json
import unittest
from unittest import mock

from db import opportunity_snapshots as osnap


class OpportunitySnapshotStoreTests(unittest.TestCase):
    def test_save_upserts_by_snapshot_time(self):
        cur = mock.MagicMock()
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            ok = osnap.save_opportunity_snapshot(
                "2026-09-12T12:00:00Z", [{"ticker": "NTAP", "score": 92, "status": "STRONG"}]
            )
        self.assertTrue(ok)
        insert = [c for c in cur.execute.call_args_list if "INSERT INTO opportunity_snapshots" in c[0][0]][0]
        self.assertIn("ON CONFLICT", insert[0][0])
        payload = json.loads(insert[0][1][1])
        self.assertEqual(payload[0]["ticker"], "NTAP")

    def test_save_no_op_without_time_or_db(self):
        self.assertFalse(osnap.save_opportunity_snapshot(None, [{"ticker": "X"}]))
        with mock.patch.object(osnap, "get_neon_conn", return_value=None):
            self.assertFalse(osnap.save_opportunity_snapshot("t", [{"ticker": "X"}]))

    def test_load_previous_returns_payload(self):
        cur = mock.MagicMock()
        cur.fetchone.return_value = ("2026-09-11T12:00:00Z",
                                     [{"ticker": "NTAP", "score": 81, "status": "WATCH"}])
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            prev = osnap.load_previous_opportunity_snapshot("2026-09-12T12:00:00Z")
        self.assertIsNotNone(prev)
        self.assertEqual(prev["opportunities"][0]["ticker"], "NTAP")

    def test_load_previous_handles_string_payload_and_none(self):
        cur = mock.MagicMock()
        cur.fetchone.return_value = ("t", '[{"ticker": "A", "score": 50, "status": "WATCH"}]')
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            prev = osnap.load_previous_opportunity_snapshot("t2")
        self.assertEqual(prev["opportunities"][0]["ticker"], "A")
        # No prior row.
        cur.fetchone.return_value = None
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            self.assertIsNone(osnap.load_previous_opportunity_snapshot("t2"))
        # DB down.
        with mock.patch.object(osnap, "get_neon_conn", return_value=None):
            self.assertIsNone(osnap.load_previous_opportunity_snapshot("t2"))
        self.assertIsNone(osnap.load_previous_opportunity_snapshot(None))


    def test_save_dedupes_tickers_and_tags_context(self):
        cur = mock.MagicMock()
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            osnap.save_opportunity_snapshot("t", [
                {"ticker": "A", "score": 90, "status": "STRONG"},
                {"ticker": "A", "score": 10, "status": "CAUTION"},  # dup -> dropped
                {"ticker": "B", "score": 70, "status": "WATCH"},
            ], context="scanner")
        insert = [c for c in cur.execute.call_args_list if "INSERT INTO opportunity_snapshots" in c[0][0]][0]
        payload = json.loads(insert[0][1][1])
        self.assertEqual([r["ticker"] for r in payload], ["A", "B"])  # deduped, first wins
        self.assertEqual(insert[0][1][2], "scanner")                  # context tagged

    def test_load_filters_by_context_when_given(self):
        cur = mock.MagicMock()
        cur.fetchone.return_value = ("t2", [{"ticker": "A", "score": 80, "status": "WATCH"}], "market_brief")
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            osnap.load_previous_opportunity_snapshot("t3", context="market_brief")
        sel = [c for c in cur.execute.call_args_list if "SELECT snapshot_time" in c[0][0]][0]
        self.assertIn("context = %s", sel[0][0])            # same-context filter applied
        self.assertIn("snapshot_time < %s", sel[0][0])      # strictly earlier

    def test_load_returns_context_and_survives_old_rows_without_it(self):
        cur = mock.MagicMock()
        cur.fetchone.return_value = ("t", [{"ticker": "A", "score": 80, "status": "WATCH"}])  # legacy 2-col row
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(osnap, "get_neon_conn", return_value=conn):
            prev = osnap.load_previous_opportunity_snapshot("t2")
        self.assertEqual(prev["opportunities"][0]["ticker"], "A")
        self.assertIsNone(prev["context"])  # graceful on old-shaped rows


if __name__ == "__main__":
    unittest.main()
