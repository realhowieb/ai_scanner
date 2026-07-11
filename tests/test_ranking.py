"""Tests for the default result ranking (PreBreakout first, BreakoutScore fallback)."""
import unittest

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from scan.ranking import apply_default_ranking

requires_pandas = unittest.skipIf(pd is None, "pandas not installed")


@requires_pandas
class DefaultRankingTests(unittest.TestCase):
    def test_prebreakout_wins_when_model_scored(self):
        df = pd.DataFrame(
            {
                "Ticker": ["A", "B", "C"],
                "BreakoutScore": [9.0, 5.0, 7.0],
                "PreBreakoutProb%": [10.0, 90.0, 50.0],
            }
        )
        out = apply_default_ranking(df)
        self.assertEqual(list(out["Ticker"]), ["B", "C", "A"])

    def test_all_zero_probs_fall_back_to_breakout_score(self):
        df = pd.DataFrame(
            {
                "Ticker": ["A", "B"],
                "BreakoutScore": [5.0, 9.0],
                "PreBreakoutProb%": [0.0, 0.0],
            }
        )
        out = apply_default_ranking(df)
        self.assertEqual(list(out["Ticker"]), ["B", "A"])

    def test_breakout_score_only(self):
        df = pd.DataFrame({"Ticker": ["A", "B"], "BreakoutScore": [1.0, 2.0]})
        self.assertEqual(list(apply_default_ranking(df)["Ticker"]), ["B", "A"])

    def test_none_and_empty_passthrough(self):
        self.assertIsNone(apply_default_ranking(None))
        empty = pd.DataFrame()
        self.assertTrue(apply_default_ranking(empty).empty)


if __name__ == "__main__":
    unittest.main()
