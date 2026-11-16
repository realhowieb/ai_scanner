"""scripts/diag_rs_backtest.py: rebuilt RS has no look-ahead and the verdict follows the declared rule."""
import datetime as dt
import unittest
from types import SimpleNamespace

from analytics.research_dataset import MATURED
from scripts import diag_rs_backtest as bt


def _spy():
    days = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(40)]
    return {"dates": days, "open": [400.0 + i for i in range(40)], "close": [401.0 + i for i in range(40)]}


def _rec(ticker, when, trend20, x5, x3=None, matured=True):
    m = MATURED if matured else "PENDING"
    return {"observation": {"ticker": ticker, "observed_at": when},
            "features": SimpleNamespace(values={"trend_20d_pct": trend20, "trend_10d_pct": trend20}),
            "outcome": SimpleNamespace(values={"excess_return_5d": x5, "excess_return_3d": x3 if x3 is not None else x5,
                                               "excess_return_1d": x5},
                                       maturity={"1d": m, "3d": m, "5d": m})}


class RebuildTest(unittest.TestCase):
    def test_uses_day_open_and_close_n_sessions_before(self):
        spy = _spy()
        day = spy["dates"][25]
        # SPY: open that day 425, close 20 sessions earlier 406 -> +4.6798%
        self.assertAlmostEqual(bt.rebuild_rs(10.0, day, 20, spy), round(10.0 - (425 / 406 - 1) * 100, 4))
        self.assertIsNone(bt.rebuild_rs(10.0, spy["dates"][5], 20, spy))      # not enough history
        self.assertIsNone(bt.rebuild_rs(10.0, dt.date(2027, 1, 1), 20, spy))  # SPY has no bar
        self.assertIsNone(bt.rebuild_rs(None, day, 20, spy))

    def test_picks_regular_session_one_per_day_and_window(self):
        spy = _spy()
        recs = [_rec("AAA", "2026-10-30T14:00:00+00:00", 9.0, 0.01),   # 10:00 ET
                _rec("AAA", "2026-10-30T15:00:00+00:00", 9.5, 0.02),   # same day: dropped
                _rec("BBB", "2026-10-30T12:00:00+00:00", 5.0, 0.01),   # 08:00 ET pre-market: dropped
                _rec("CCC", "2026-11-17T15:00:00+00:00", 5.0, 0.01),   # training window: dropped
                _rec("DDD", "2026-10-30T15:00:00+00:00", 1.0, 0.03, matured=False)]
        rows = bt.picks(recs, spy)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["x5"], 0.01)
        self.assertIsNone(rows[1]["x5"])


class VerdictTest(unittest.TestCase):
    def _rows(self, beat_x, lag_x, days=30):
        rows = []
        for d in range(days):
            for i in range(6):
                beat = i < 3
                x = (beat_x if beat else lag_x) + (0.001 * ((d + i) % 3 - 1))
                rows.append({"day": f"2026-10-{d:02d}", "rs_10d": 1.0 if beat else -1.0, "rs_20d": 1.0 if beat else -1.0,
                             "trend_20d": 2.0 if beat else -2.0, "x1": x, "x3": x, "x5": x})
        return rows

    def test_clear_gap_helps(self):
        self.assertEqual(bt.analyze(self._rows(0.02, 0.0))["verdict"], "HELPS")

    def test_no_gap_is_no_edge(self):
        self.assertEqual(bt.analyze(self._rows(0.001, 0.0))["verdict"], "NO_EDGE")

    def test_reverse_gap_hurts(self):
        self.assertEqual(bt.analyze(self._rows(-0.02, 0.0))["verdict"], "HURTS")


if __name__ == "__main__":
    unittest.main()
