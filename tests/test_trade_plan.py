"""Tests for the deterministic trade-plan math."""
import unittest

from ui.trade_plan import build_trade_plan


class TradePlanTests(unittest.TestCase):
    def test_basic_plan_math(self):
        plan = build_trade_plan(
            {"Last": 100.0, "Volatility20D%": 8.0},
            account_size=10_000.0,
            risk_pct=1.0,
        )
        # stop = 4% below entry (half of 8% vol), risk $4/share, $100 budget.
        self.assertEqual(plan["entry"], 100.0)
        self.assertEqual(plan["stop"], 96.0)
        self.assertEqual(plan["stop_pct"], 4.0)
        self.assertEqual(plan["risk_per_share"], 4.0)
        self.assertEqual(plan["shares"], 25)
        self.assertEqual(plan["targets"], [106.0, 112.0])  # 1.5R / 3R

    def test_stop_clamped_to_bounds(self):
        tight = build_trade_plan({"Last": 50.0, "Volatility20D%": 1.0})
        self.assertEqual(tight["stop_pct"], 2.0)  # floor
        wild = build_trade_plan({"Last": 50.0, "Volatility20D%": 40.0})
        self.assertEqual(wild["stop_pct"], 8.0)  # ceiling

    def test_missing_volatility_uses_default(self):
        plan = build_trade_plan({"Last": 100.0})
        self.assertEqual(plan["stop_pct"], 3.0)  # 6.0 default vol * 0.5

    def test_no_price_returns_none(self):
        self.assertIsNone(build_trade_plan({}))
        self.assertIsNone(build_trade_plan({"Last": 0.0}))

    def test_zero_risk_budget_zero_shares(self):
        plan = build_trade_plan({"Last": 100.0}, account_size=0.0)
        self.assertEqual(plan["shares"], 0)


if __name__ == "__main__":
    unittest.main()
