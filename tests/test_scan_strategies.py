import importlib.util
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from scan.strategies import apply_strategy_filter


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for strategy tests")
class ScanStrategiesTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {
                    "Ticker": "UP",
                    "GapPct": 4.0,
                    "DollarVol20": 2000,
                    "Volume": 100,
                    "VolRel20": 1.5,
                    "Trend20D%": 3.0,
                    "Trend10D%": 2.0,
                    "IsBreakout": True,
                    "BreakoutScore": 8.0,
                },
                {
                    "Ticker": "DOWN",
                    "GapPct": -3.0,
                    "DollarVol20": 5000,
                    "Volume": 200,
                    "VolRel20": 2.5,
                    "Trend20D%": -1.0,
                    "Trend10D%": 1.0,
                    "IsBreakout": False,
                    "BreakoutScore": 2.0,
                },
                {
                    "Ticker": "MOMO",
                    "GapPct": 1.0,
                    "DollarVol20": 3000,
                    "Volume": 150,
                    "VolRel20": 3.0,
                    "Trend20D%": 6.0,
                    "Trend10D%": 5.0,
                    "IsBreakout": True,
                    "BreakoutScore": 12.0,
                },
            ]
        )

    def test_gap_up_filters_positive_gaps_descending(self):
        result = apply_strategy_filter("gap_up", self.frame)

        self.assertEqual(list(result["Ticker"]), ["UP", "MOMO"])

    def test_gap_down_filters_negative_gaps_ascending(self):
        result = apply_strategy_filter("gap_down", self.frame)

        self.assertEqual(list(result["Ticker"]), ["DOWN"])

    def test_most_active_prefers_dollar_volume(self):
        result = apply_strategy_filter("most_active", self.frame)

        self.assertEqual(list(result["Ticker"]), ["DOWN", "MOMO", "UP"])

    def test_unusual_volume_threshold(self):
        result = apply_strategy_filter("unusual_vol", self.frame)

        self.assertEqual(list(result["Ticker"]), ["MOMO", "DOWN"])

    def test_momentum_requires_positive_ten_and_twenty_day_trends(self):
        result = apply_strategy_filter("momentum", self.frame)

        self.assertEqual(list(result["Ticker"]), ["MOMO", "UP"])

    def test_breakout_only_sorts_by_score(self):
        result = apply_strategy_filter("breakout_only", self.frame)

        self.assertEqual(list(result["Ticker"]), ["MOMO", "UP"])

    def test_unknown_strategy_returns_original_frame(self):
        result = apply_strategy_filter("unknown", self.frame)

        self.assertIs(result, self.frame)

    def test_missing_required_column_returns_empty_frame(self):
        result = apply_strategy_filter("gap_up", self.frame.drop(columns=["GapPct"]))

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
