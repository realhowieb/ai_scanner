"""Tests for the alert edge scorecard aggregation (scorecards_by_type)."""
from __future__ import annotations

import unittest
from unittest import mock

from db import alert_outcomes as ao


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        return None

    def fetchall(self):
        return self._rows

    def close(self):
        return None


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self, *a, **k):
        return _FakeCursor(self._rows)

    def close(self):
        return None


class ScorecardsByTypeTests(unittest.TestCase):
    def _run(self, rows):
        conn = _FakeConn(rows)
        with mock.patch.object(ao, "get_neon_conn", return_value=conn), \
             mock.patch.object(ao, "_ensure_schema", lambda c: None):
            return ao.scorecards_by_type("u@example.com")

    def test_aggregates_and_computes_hit_rate(self):
        # dict_row shape from the SQL: type, fires, hits, avg_return, avg_max_gain
        rows = [
            {"alert_type": "breakout", "fires": 10, "hits": 7,
             "avg_return_pct": 2.5, "avg_max_gain_pct": 6.0},
            {"alert_type": "move", "fires": 4, "hits": 1,
             "avg_return_pct": -1.0, "avg_max_gain_pct": 3.0},
        ]
        out = self._run(rows)
        self.assertAlmostEqual(out["breakout"]["hit_rate"], 0.7)
        self.assertEqual(out["breakout"]["hits"], 7)
        self.assertAlmostEqual(out["breakout"]["avg_return_pct"], 2.5)
        self.assertAlmostEqual(out["move"]["hit_rate"], 0.25)

    def test_tuple_rows_supported(self):
        rows = [("price", 6, 3, None, 4.2)]  # avg_return None → passthrough
        out = self._run(rows)
        self.assertEqual(out["price"]["fires"], 6)
        self.assertAlmostEqual(out["price"]["hit_rate"], 0.5)
        self.assertIsNone(out["price"]["avg_return_pct"])
        self.assertAlmostEqual(out["price"]["avg_max_gain_pct"], 4.2)

    def test_zero_fire_rows_skipped(self):
        out = self._run([{"alert_type": "rvol", "fires": 0, "hits": 0,
                          "avg_return_pct": None, "avg_max_gain_pct": None}])
        self.assertEqual(out, {})

    def test_no_conn_returns_empty(self):
        with mock.patch.object(ao, "get_neon_conn", return_value=None):
            self.assertEqual(ao.scorecards_by_type("u"), {})


class RecentFireCountsTests(unittest.TestCase):
    def test_aggregates_fire_counts_by_alert(self):
        from db import alerts as A

        conn = _FakeConn([{"alert_id": 5, "n": 4}, {"alert_id": 7, "n": 1}])
        with mock.patch.object(A, "_get_conn", return_value=conn):
            out = A.recent_fire_counts_for_user("u@example.com")
        self.assertEqual(out, {5: 4, 7: 1})

    def test_tuple_rows_and_null_alert_id(self):
        from db import alerts as A

        conn = _FakeConn([(5, 3), (None, 9)])  # null alert_id ignored
        with mock.patch.object(A, "_get_conn", return_value=conn):
            out = A.recent_fire_counts_for_user("u")
        self.assertEqual(out, {5: 3})


if __name__ == "__main__":
    unittest.main()
