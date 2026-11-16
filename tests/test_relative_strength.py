"""Real relative strength vs SPY: return minus SPY's, and the RS-line high (Nov 2026)."""
import unittest

import pandas as pd

from scan.indicators import relative_strength_vs_spy


def _series(values, start="2026-09-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


class RelativeStrengthTests(unittest.TestCase):
    def test_return_minus_spy_over_same_dates(self):
        stock = _series([100.0] * 40 + [110.0])   # +10% over 10 and 20 sessions
        spy = _series([400.0] * 40 + [404.0])     # +1%
        rs = relative_strength_vs_spy(stock, spy)
        self.assertAlmostEqual(rs["rs_10d"], 9.0, places=3)
        self.assertAlmostEqual(rs["rs_20d"], 9.0, places=3)
        self.assertTrue(rs["rs_line_high"])

    def test_price_level_does_not_matter(self):
        cheap = _series([5.0 + 0.01 * i for i in range(41)])
        dear = _series([500.0 + 1.0 * i for i in range(41)])
        spy = _series([400.0] * 41)
        self.assertAlmostEqual(relative_strength_vs_spy(cheap, spy)["rs_20d"],
                               relative_strength_vs_spy(dear, spy)["rs_20d"], places=3)

    def test_lagging_stock_is_negative_and_line_not_at_high(self):
        stock = _series([100.0] * 30 + [99.0] * 11)
        spy = _series([400.0 + i for i in range(41)])
        rs = relative_strength_vs_spy(stock, spy)
        self.assertLess(rs["rs_20d"], 0)
        self.assertFalse(rs["rs_line_high"])

    def test_spy_aligned_onto_stock_dates(self):
        # SPY has an extra day the stock lacks; SPY's value carries forward to the stock's dates.
        stock = _series([100.0] * 25 + [105.0])
        spy = pd.concat([_series([400.0] * 26), pd.Series([999.0], index=[pd.Timestamp("2025-01-01")])])
        rs = relative_strength_vs_spy(stock, spy)
        self.assertAlmostEqual(rs["rs_10d"], 5.0, places=3)

    def test_missing_or_short_inputs_give_none(self):
        empty = {"rs_10d": None, "rs_20d": None, "rs_line_high": None}
        self.assertEqual(relative_strength_vs_spy(_series([1.0, 2.0]), None), empty)
        short = relative_strength_vs_spy(_series([10.0] * 5), _series([400.0] * 5))
        self.assertEqual(short, empty)
        # SPY with no bar at or before the stock's last date
        stale = relative_strength_vs_spy(_series([10.0] * 30), _series([400.0] * 30, start="2027-01-01"))
        self.assertEqual(stale, empty)


if __name__ == "__main__":
    unittest.main()
