"""PreBreakout pick follow-through by calibrated-probability band."""
import unittest

from analytics.hsf_calibration import (
    prebreakout_followthrough,
    summarize_prebreakout_buckets,
)


def rec(prob, matured, positive):
    # normalize_row shape: prob + matured flag + positive (mfe_5d>=0.04) + mfe/mae/return
    return {"prob": prob, "matured": matured, "positive": positive,
            "mfe_5d": 0.05 if positive else 0.0, "mae_5d": -0.01, "return_5d": 0.03}


class BucketTests(unittest.TestCase):
    def test_buckets_by_prob(self):
        recs = [rec(12, True, False), rec(13, True, True), rec(22, True, True), rec(27, True, True)]
        out = {b["band"]: b for b in summarize_prebreakout_buckets(recs)}
        self.assertEqual(out["<15%"]["n_matured"], 2)
        self.assertEqual(out["<15%"]["positive_count"], 1)
        self.assertEqual(out["20-25%"]["n_matured"], 1)
        self.assertEqual(out["25%+"]["n_matured"], 1)

    def test_pending_not_counted(self):
        recs = [rec(27, False, None), rec(27, True, True)]  # one pending, one matured
        ft = prebreakout_followthrough(recs, 27, min_sample=1)
        self.assertEqual(ft["n_matured"], 1)
        self.assertEqual(ft["positive_rate"], 1.0)


class FollowthroughTests(unittest.TestCase):
    def test_sufficient_gate(self):
        recs = [rec(22, True, True) for _ in range(12)]
        ft = prebreakout_followthrough(recs, 21.8, min_sample=10)
        self.assertTrue(ft["sufficient"])
        self.assertEqual(ft["band"], "20-25%")
        self.assertEqual(ft["positive_rate"], 1.0)

    def test_insufficient_below_min(self):
        recs = [rec(22, True, True) for _ in range(5)]
        ft = prebreakout_followthrough(recs, 21.8, min_sample=10)
        self.assertFalse(ft["sufficient"])

    def test_no_prob(self):
        self.assertFalse(prebreakout_followthrough([], None)["sufficient"])


if __name__ == "__main__":
    unittest.main()
