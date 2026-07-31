"""15-min BTC outcome logger: storage idempotency + settle-from-Kalshi logic."""
from __future__ import annotations

import unittest
from unittest import mock


class _Cur:
    def __init__(self, fetchone=None, fetchall=None):
        self._fetchone = fetchone
        self._fetchall = fetchall or []
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self._fetchone

    def fetchall(self):
        return self._fetchall

    def close(self):
        pass


class _Conn:
    def __init__(self, **kw):
        self.cur = _Cur(**kw)

    def cursor(self, *a, **k):
        return self.cur

    def commit(self):
        pass


class StorageTests(unittest.TestCase):
    def test_log_window_upserts_on_conflict_do_nothing(self):
        import db.btc_outcomes as bo

        conn = _Conn()
        with mock.patch.object(bo, "get_neon_conn", return_value=conn):
            ok = bo.log_window(
                "KXBTC15M-w1", close_time="2026-07-31T05:30:00Z", strike=64000.0,
                spot=63950.0, pred_direction="up", pred_confidence=72,
                pred_win_prob=75, kalshi_yes_pct=48.0, features={"rsi": 61},
            )
        self.assertTrue(ok)
        ins = [c for c in conn.cur.calls if "INSERT INTO btc_outcomes" in c[0]]
        self.assertEqual(len(ins), 1)
        self.assertIn("ON CONFLICT (window_ticker) DO NOTHING", ins[0][0])
        self.assertEqual(ins[0][1][0], "KXBTC15M-w1")   # window ticker bound first

    def test_stats_computes_accuracy(self):
        import db.btc_outcomes as bo

        # logged=10, settled=6, decided=4, correct=3 → 75%
        conn = _Conn(fetchone=(10, 6, 4, 3))
        with mock.patch.object(bo, "get_neon_conn", return_value=conn):
            s = bo.outcome_stats()
        self.assertEqual(s["settled"], 6)
        self.assertEqual(s["decided"], 4)
        self.assertAlmostEqual(s["accuracy"], 0.75)


class SettleTests(unittest.TestCase):
    def test_settles_only_finalized_and_records_result(self):
        import analytics.btc_outcome_logger as lg

        due = [{"window_ticker": "KXBTC15M-done", "close_time": None, "strike": 64000.0},
               {"window_ticker": "KXBTC15M-pending", "close_time": None, "strike": 64000.0}]

        def fake_market(tk):
            if tk == "KXBTC15M-done":
                return {"settled": True, "result_up": True, "expiration_value": 64120.0}
            return {"settled": False, "result_up": None}   # not finalized yet

        recorded = []
        with mock.patch("db.btc_outcomes.pending_due", return_value=due), \
             mock.patch("data.kalshi_markets.fetch_market", side_effect=fake_market), \
             mock.patch("db.btc_outcomes.record_result",
                        side_effect=lambda tk, up, val: recorded.append((tk, up, val)) or True):
            n = lg.settle_due()

        self.assertEqual(n, 1)                              # only the finalized one
        self.assertEqual(recorded, [("KXBTC15M-done", True, 64120.0)])


if __name__ == "__main__":
    unittest.main()
