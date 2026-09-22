"""Run 33 — DT Score validation building blocks (direction-aware, no lookahead)."""
import unittest
from unittest.mock import patch

import pandas as pd

from analytics import day_trade_validation as v


class DirectionalReturnTests(unittest.TestCase):
    def test_bullish_uses_raw_return(self):
        self.assertEqual(v.directional_return("bullish", 0.02), 0.02)

    def test_bearish_inverts(self):
        self.assertEqual(v.directional_return("bearish", -0.02), 0.02)  # fall = favorable
        self.assertEqual(v.directional_return("bearish", 0.02), -0.02)

    def test_neutral_excluded(self):
        self.assertIsNone(v.directional_return("neutral", 0.02))

    def test_missing_return(self):
        self.assertIsNone(v.directional_return("bullish", None))


class ForwardReturnTests(unittest.TestCase):
    def test_forward_return(self):
        prices = [100, 101, 102, 99]
        self.assertAlmostEqual(v.forward_return(prices, 0, 2), 0.02)   # 100 -> 102
        self.assertAlmostEqual(v.forward_return(prices, 1, 2), (99 - 101) / 101)

    def test_missing_future_bar_is_none(self):
        prices = [100, 101]
        self.assertIsNone(v.forward_return(prices, 1, 5))  # no bar 5 bars ahead

    def test_multi_horizon(self):
        prices = [100, 101, 103, 104, 106]
        out = v.forward_returns(prices, 0, {"a": 2, "z": 99})
        self.assertAlmostEqual(out["a"], 0.03)
        self.assertIsNone(out["z"])  # beyond data


class MfeMaeTests(unittest.TestCase):
    def test_bullish_mfe_mae(self):
        prices = [100, 103, 98, 101]  # up +3%, down -2%, up +1%
        r = v.mfe_mae(prices, 0, 3, "bullish")
        self.assertAlmostEqual(r["mfe"], 0.03)
        self.assertAlmostEqual(r["mae"], -0.02)

    def test_bearish_inverts_excursions(self):
        prices = [100, 103, 98, 101]
        r = v.mfe_mae(prices, 0, 3, "bearish")
        self.assertAlmostEqual(r["mfe"], 0.02)   # the -2% move is favorable for a short
        self.assertAlmostEqual(r["mae"], -0.03)  # the +3% move is adverse

    def test_neutral_none(self):
        self.assertEqual(v.mfe_mae([100, 101], 0, 1, "neutral"), {"mfe": None, "mae": None})

    def test_no_future_window(self):
        self.assertEqual(v.mfe_mae([100], 0, 5, "bullish"), {"mfe": None, "mae": None})


class BucketTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        self.assertEqual(v.assign_score_bucket(0), "0-39")
        self.assertEqual(v.assign_score_bucket(39.9), "0-39")
        self.assertEqual(v.assign_score_bucket(40), "40-59")
        self.assertEqual(v.assign_score_bucket(69.9), "60-69")
        self.assertEqual(v.assign_score_bucket(90), "90-100")
        self.assertEqual(v.assign_score_bucket(100), "90-100")
        self.assertIsNone(v.assign_score_bucket(None))

    def test_bucket_report_excludes_neutral_from_hit_rate(self):
        obs = [
            {"direction": "bullish", "score": 85, "directional_return_15m": 0.01, "mfe": 0.02, "mae": -0.01},
            {"direction": "bearish", "score": 82, "directional_return_15m": 0.015, "mfe": 0.02, "mae": 0.0},
            {"direction": "neutral", "score": 88, "directional_return_15m": None},  # excluded
        ]
        rep = v.bucket_report(obs, horizons=("15m",), min_n=1)
        b = next(r for r in rep if r["bucket"] == "80-89")
        self.assertEqual(b["n"], 3)
        self.assertEqual(b["neutral"], 1)
        self.assertEqual(b["n_15m"], 2)              # only the 2 directional rows
        self.assertEqual(b["hit_rate_15m"], 1.0)     # both favorable
        self.assertTrue(b["sufficient"])

    def test_bucket_report_sample_flag(self):
        obs = [{"direction": "bullish", "score": 75, "directional_return_5m": 0.01}]
        rep = v.bucket_report(obs, horizons=("5m",), min_n=10)
        self.assertFalse(rep[0]["sufficient"])  # 1 < 10


class DistributionTests(unittest.TestCase):
    def test_distribution_and_bunching(self):
        d = v.score_distribution([72, 72, 72, 72, None])
        self.assertEqual(d["n"], 4)
        self.assertEqual(d["n_insufficient"], 1)
        self.assertEqual(d["std"], 0.0)   # everything at 72 -> bunching
        self.assertEqual(d["median"], 72)

    def test_percentiles_spread(self):
        d = v.score_distribution(list(range(0, 101)))
        self.assertAlmostEqual(d["p50"], 50)
        self.assertAlmostEqual(d["p90"], 90)
        self.assertEqual(d["min"], 0)
        self.assertEqual(d["max"], 100)

    def test_empty(self):
        self.assertEqual(v.score_distribution([None, None])["n"], 0)


class SpearmanTests(unittest.TestCase):
    def test_perfect_rank(self):
        self.assertAlmostEqual(v.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_inverse_rank(self):
        self.assertAlmostEqual(v.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_insufficient(self):
        self.assertIsNone(v.spearman([1], [2]))


class DeterminismTests(unittest.TestCase):
    def test_repeatable(self):
        obs = [{"direction": "bullish", "score": 71, "directional_return_5m": 0.01}]
        self.assertEqual(v.bucket_report(obs, horizons=("5m",), min_n=1),
                         v.bucket_report(obs, horizons=("5m",), min_n=1))


if __name__ == "__main__":
    unittest.main()


class ScriptPipelineTests(unittest.TestCase):
    def test_daily_frame_lookup_does_not_evaluate_dataframe_truthiness(self):
        from scripts.validate_day_trade_score import _fetch_daily_frame

        daily = pd.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0],
                              "close": [10.5], "volume": [100]})
        with patch("data.price_alpaca.download_multi_alpaca", return_value={"NVDA": daily}):
            result = _fetch_daily_frame("NVDA")
        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])

    def test_build_observation_no_lookahead(self):
        from scripts.validate_day_trade_score import build_observation
        # bullish features; prices_after[0] is signal price, rest are outcomes only
        feat = {"vs_vwap_pct": 0.8, "supertrend_direction": "green", "ewo": 6,
                "chg_pct": 1.2, "adx": 28, "rvol": 2.2}
        prices = [100.0] + [100.0 + i for i in range(1, 61)]  # rising
        obs = build_observation(timestamp="T", ticker="NVDA", features=feat, prices_after=prices)
        self.assertEqual(obs["direction"], "bullish")
        self.assertGreater(obs["directional_return_15m"], 0)  # rising + bullish -> favorable
        self.assertGreaterEqual(obs["mfe"], 0)

    def test_build_report_shape(self):
        from scripts.validate_day_trade_score import build_report
        obs = [
            {"direction": "bullish", "score": 85, "setup_quality": "strong",
             "directional_return_15m": 0.01, "mfe": 0.02, "mae": -0.01},
            {"direction": "bearish", "score": 45, "setup_quality": "weak",
             "directional_return_15m": -0.005, "mfe": 0.0, "mae": -0.02},
        ]
        rep = build_report(obs, min_n=1)
        self.assertEqual(rep["sample_size"], 2)
        self.assertIn("score_distribution", rep)
        self.assertIn("bullish", rep)
        self.assertIn("bearish", rep)
        self.assertIn("strong", rep["by_setup_quality"])
