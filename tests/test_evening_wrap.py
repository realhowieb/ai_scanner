"""Tests for the evening wrap sections (summary, movers, tomorrow's setups)."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None


class WatchlistSummaryTests(unittest.TestCase):
    def test_best_worst_and_breadth(self):
        from scheduler.evening_wrap import _watchlist_summary

        rows = [
            {"ticker": "AEHR", "chg_pct": -5.57},
            {"ticker": "GGLS", "chg_pct": 1.41},
            {"ticker": "PEP", "chg_pct": 0.51},
        ]
        s = _watchlist_summary(rows)
        self.assertIn("Best GGLS +1.4%", s)
        self.assertIn("Worst AEHR -5.6%", s)
        self.assertIn("2 up / 1 down", s)

    def test_empty_returns_none(self):
        from scheduler.evening_wrap import _watchlist_summary

        self.assertIsNone(_watchlist_summary([{"ticker": "X", "chg_pct": None}]))


@unittest.skipUnless(_PANDAS, "movers/setups read a DataFrame")
class MoversAndSetupsTests(unittest.TestCase):
    def _df(self):
        import pandas as pd

        return pd.DataFrame({
            "Ticker": ["GGLS", "ATAI", "AEHR", "DAL", "AMDX"],
            "PctChange": [12.4, 8.1, -9.2, -3.0, 5.0],
            "BreakoutScore": [54.8, 47.4, 30.0, 10.0, 45.1],
            "EMACross": ["Golden", None, "Death", "Golden", None],
        })

    def test_day_movers_splits_gainers_and_losers(self):
        from scheduler.evening_wrap import _day_movers

        gainers, losers = _day_movers(self._df(), limit=3)
        self.assertEqual([t for t, _ in gainers], ["GGLS", "ATAI", "AMDX"])
        self.assertEqual([t for t, _ in losers], ["AEHR", "DAL"])

    def test_tomorrow_setups_golden_and_top_scores(self):
        from scheduler.evening_wrap import _tomorrow_setups

        golden, top = _tomorrow_setups(self._df(), limit=4)
        self.assertEqual(golden, ["GGLS", "DAL"])           # only "Golden" rows
        self.assertEqual([t for t, _ in top][:2], ["GGLS", "ATAI"])  # ranked by score

    def test_none_df_is_safe(self):
        from scheduler.evening_wrap import _day_movers, _tomorrow_setups

        self.assertEqual(_day_movers(None), ([], []))
        self.assertEqual(_tomorrow_setups(None), ([], []))


if __name__ == "__main__":
    unittest.main()
