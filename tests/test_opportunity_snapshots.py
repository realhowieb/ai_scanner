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


if __name__ == "__main__":
    unittest.main()
