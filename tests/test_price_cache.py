"""Neon OHLCV price cache: load/upsert flow, staleness, key preservation."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None


class _Cursor:
    def __init__(self, fetch_rows):
        self._rows = fetch_rows
        self.executed = []
        self.executemany_calls = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def executemany(self, sql, seq):
        self.executemany_calls.append((sql, list(seq)))

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _Conn:
    def __init__(self, fetch_rows=None):
        self.cur = _Cursor(fetch_rows or [])

    def cursor(self, *a, **k):
        return self.cur

    def commit(self):
        pass


@unittest.skipUnless(_PANDAS, "price cache round-trips DataFrames")
class PriceCacheTests(unittest.TestCase):
    def _df(self):
        import pandas as pd

        return pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
            index=pd.to_datetime(["2026-07-01"]),
        )

    def test_load_returns_frames_by_original_symbol_and_marks_rest_stale(self):
        import db.prices as p

        payload = p._serialize(self._df())
        conn = _Conn(fetch_rows=[{"symbol": "AAPL", "payload": payload}])
        with mock.patch.object(p, "get_neon_conn", return_value=conn):
            cached, stale = p.get_price_data_snapshot(["aapl", "msft"], max_age_minutes=30)
        # Original-case key preserved; MSFT had no row → stale.
        self.assertIn("aapl", cached)
        self.assertEqual(list(cached["aapl"].columns),
                         ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(stale, {"msft"})

    def test_no_connection_marks_everything_stale(self):
        import db.prices as p

        with mock.patch.object(p, "get_neon_conn", return_value=None):
            cached, stale = p.get_price_data_snapshot(["AAA", "BBB"])
        self.assertEqual(cached, {})
        self.assertEqual(stale, {"AAA", "BBB"})

    def test_upsert_serializes_and_batches(self):
        import db.prices as p

        conn = _Conn()
        with mock.patch.object(p, "get_neon_conn", return_value=conn):
            p.upsert_price_data_snapshot({"AAPL": self._df(), "MSFT": self._df()})
        self.assertEqual(len(conn.cur.executemany_calls), 1)
        _sql, rows = conn.cur.executemany_calls[0]
        self.assertEqual({r[0] for r in rows}, {"AAPL", "MSFT"})
        self.assertTrue(all(isinstance(r[1], str) for r in rows))  # serialized

    def test_upsert_skips_empty_frames(self):
        import pandas as pd

        import db.prices as p

        conn = _Conn()
        with mock.patch.object(p, "get_neon_conn", return_value=conn):
            p.upsert_price_data_snapshot({"AAPL": pd.DataFrame(), "MSFT": None})
        # Nothing serializable → no executemany call.
        self.assertEqual(conn.cur.executemany_calls, [])

    def test_class_shares_stay_distinct(self):
        import db.prices as p

        self.assertEqual(p.normalize_symbol("brk-b"), "BRK-B")
        self.assertNotEqual(p.normalize_symbol("BRK-A"), p.normalize_symbol("BRK-B"))


if __name__ == "__main__":
    unittest.main()
