"""Signal leaderboard: ranking math (compute) + persistence upsert/load."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None


class _Cur:
    def __init__(self, fetch=None, fetchall=None):
        self._fetch = fetch
        self._fetchall = fetchall or []
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self._fetch

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

    def close(self):
        pass


class LeaderboardStoreTests(unittest.TestCase):
    def test_save_upserts_each_row_then_load_maps_tuples(self):
        import db.signal_leaderboard as lb

        rows = [
            {"signal": "breakout", "display": "Breakout score", "avg_excess": 0.03,
             "median_excess": 0.02, "win_rate": 0.6, "sample_size": 40,
             "runs_used": 8, "benchmark": "SPY", "top_n": 5},
            {"signal": "gap", "display": "Gap %", "avg_excess": -0.01,
             "median_excess": 0.0, "win_rate": 0.45, "sample_size": 40,
             "runs_used": 8, "benchmark": "SPY", "top_n": 5},
        ]
        save_conn = _Conn()
        with mock.patch.object(lb, "get_neon_conn", return_value=save_conn):
            n = lb.save_leaderboard(5, rows)
        self.assertEqual(n, 2)
        # one INSERT ... ON CONFLICT per signal row (plus the CREATE TABLE)
        inserts = [c for c in save_conn.cur.calls if "INSERT INTO signal_leaderboard" in c[0]]
        self.assertEqual(len(inserts), 2)

        # load maps tuple rows to dicts in the declared column order
        tup = ("breakout", 5, "Breakout score", 0.03, 0.02, 0.6, 40, 8, "SPY", 5, None)
        load_conn = _Conn(fetchall=[tup])
        with mock.patch.object(lb, "get_neon_conn", return_value=load_conn):
            out = lb.load_leaderboard(5)
        self.assertEqual(out[0]["signal"], "breakout")
        self.assertEqual(out[0]["avg_excess"], 0.03)
        self.assertEqual(out[0]["sample_size"], 40)

    def test_save_empty_is_noop(self):
        import db.signal_leaderboard as lb
        with mock.patch.object(lb, "get_neon_conn",
                               side_effect=AssertionError("must not connect")):
            self.assertEqual(lb.save_leaderboard(5, []), 0)


@unittest.skipUnless(_PANDAS, "pandas required")
class LeaderboardComputeTests(unittest.TestCase):
    def test_ranks_signals_by_forward_excess(self):
        import pandas as pd

        import analytics.signal_leaderboard as sl

        # One snapshot: BreakoutScore favors AAA (the winner), GapPct favors CCC
        # (the loser). SPY is flat, so excess == raw forward return.
        df = pd.DataFrame({
            "Ticker": ["AAA", "BBB", "CCC"],
            "BreakoutScore": [90.0, 50.0, 10.0],
            "GapPct": [1.0, 2.0, 9.0],
        })
        run_date = __import__("datetime").date(2026, 1, 5)

        # forward returns: AAA +10%, BBB 0, CCC -10%, SPY 0.
        fwd = {"AAA": 0.10, "BBB": 0.0, "CCC": -0.10, "SPY": 0.0}

        def fake_forward(bars, rd, horizon, entry_mode="close"):
            return fwd.get(bars)  # we stash the symbol string as "bars"

        bars_by_symbol = {s: s for s in ("AAA", "BBB", "CCC", "SPY")}

        with mock.patch.object(sl, "TOP_N", 1), \
             mock.patch.object(sl, "_eligible_snapshots", return_value=[(run_date, df)]), \
             mock.patch.object(sl, "_forward_return", side_effect=fake_forward), \
             mock.patch.object(sl, "_bars_for", side_effect=lambda m, s: m.get(s)), \
             mock.patch("data.price_alpaca.download_multi_alpaca",
                        return_value=bars_by_symbol):
            rows = sl.compute_signal_leaderboard(horizon_days=5, max_snapshots=1)

        self.assertIsNotNone(rows)
        by_sig = {r["signal"]: r for r in rows}
        # Breakout's top pick AAA (+10%) should beat Gap's top pick CCC (-10%).
        self.assertGreater(by_sig["breakout"]["avg_excess"], by_sig["gap"]["avg_excess"])
        self.assertEqual(rows[0]["signal"], "breakout")   # sorted best-first
        self.assertAlmostEqual(by_sig["breakout"]["avg_excess"], 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
