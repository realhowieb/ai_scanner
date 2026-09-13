"""V1 bug fix — PreBreakout calibration display/ranking behavior.

Documents WHY many Scanner rows share one calibrated % (the ~13.1% plateau): it
is a legitimate isotonic calibration plateau plus np.interp's flat left tail
(raw <= first knot -> first knot's y), NOT a bug/fallback/broadcast/rounding.
The calibrated value is PRIMARY for ranking; the raw model probability is a
deterministic secondary tie-break. These tests guard against someone "fixing"
valid calibration and against a regression of the tie-break.
"""
import unittest

import numpy as np
import pandas as pd

import ml_prebreakout as m
from scan.ranking import apply_default_ranking

# Isotonic map mimicking the live shape reported in the 200-row export:
# first knot ~0.0345 -> 0.131, then increasing.
_MAP = {"method": "isotonic", "n": 500,
        "x": [0.0345, 0.08, 0.1809, 0.35, 0.60],
        "y": [0.131, 0.18, 0.250, 0.45, 0.72]}


class CalibrationPlateauTests(unittest.TestCase):
    def test_flat_left_tail_is_the_plateau(self):
        # Every raw <= first knot maps to the first knot's y (np.interp clamps).
        raws = [0.003, 0.010, 0.020, 0.030, 0.0340, 0.0345]
        out = m.apply_calibration_map(raws, _MAP)
        self.assertTrue(np.allclose(out, 0.131))  # all identical -> 13.1%

    def test_above_plateau_varies(self):
        self.assertAlmostEqual(float(m.apply_calibration_map([0.1809], _MAP)[0]), 0.250, places=3)
        self.assertGreater(float(m.apply_calibration_map([0.05], _MAP)[0]), 0.131)

    def test_values_are_truly_identical_not_just_rounded(self):
        # Rules OUT rounding: raw 0.003 and 0.030 produce the exact same float.
        a = float(m.apply_calibration_map([0.003], _MAP)[0])
        b = float(m.apply_calibration_map([0.030], _MAP)[0])
        self.assertEqual(a, b)

    def test_monotonic_and_deterministic(self):
        raws = [0.01, 0.05, 0.1809, 0.5]
        out1 = m.apply_calibration_map(raws, _MAP)
        out2 = m.apply_calibration_map(raws, _MAP)
        self.assertTrue(np.array_equal(out1, out2))  # deterministic
        self.assertTrue(np.all(np.diff(out1) >= 0))   # monotonic (ranking-preserving)

    def test_vector_alignment_no_broadcast(self):
        # N distinct raws -> N calibrated (position-preserving), no scalar broadcast.
        raws = [0.0122, 0.1809, 0.0067, 0.50]
        out = m.apply_calibration_map(raws, _MAP)
        self.assertEqual(len(out), len(raws))
        self.assertAlmostEqual(float(out[1]), 0.250, places=3)  # 0.1809 stays at index 1
        self.assertEqual(float(out[0]), float(out[2]))          # both in plateau


class FallbackAuditTests(unittest.TestCase):
    def test_missing_map_returns_raw_unchanged(self):
        raws = [0.0122, 0.1809]
        out = m.apply_calibration_map(raws, None)
        self.assertTrue(np.allclose(out, raws))  # never a fake per-ticker constant

    def test_degenerate_map_returns_raw(self):
        out = m.apply_calibration_map([0.0122], {"x": [0.1], "y": [0.5]})  # <2 knots
        self.assertAlmostEqual(float(out[0]), 0.0122)


class RankingTieBreakTests(unittest.TestCase):
    def _df(self):
        # Three rows share the calibrated plateau (13.1%) but differ in raw.
        return pd.DataFrame({
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "PreBreakoutProb%": [13.1, 13.1, 13.1, 25.0],
            "PreBreakoutProbRaw": [0.0122, 0.0345, 0.0067, 0.1809],
            "BreakoutScore": [40, 50, 30, 60],
        })

    def test_primary_calibrated_order_preserved(self):
        out = apply_default_ranking(self._df())
        self.assertEqual(out.iloc[0]["Ticker"], "DDD")  # 25% ranks first (unchanged)

    def test_plateau_ties_broken_by_raw(self):
        out = apply_default_ranking(self._df())
        plateau = list(out[out["PreBreakoutProb%"] == 13.1]["Ticker"])
        # within the 13.1% plateau, higher raw ranks higher (BBB>AAA>CCC), deterministic
        self.assertEqual(plateau, ["BBB", "AAA", "CCC"])

    def test_calibrated_values_not_mutated_by_ranking(self):
        out = apply_default_ranking(self._df())
        self.assertEqual(sorted(out["PreBreakoutProb%"].tolist()), [13.1, 13.1, 13.1, 25.0])

    def test_fallback_to_breakout_score_when_all_zero(self):
        df = self._df()
        df["PreBreakoutProb%"] = 0.0
        out = apply_default_ranking(df)
        self.assertEqual(out.iloc[0]["Ticker"], "DDD")  # sorted by BreakoutScore (60)

    def test_missing_raw_column_still_ranks(self):
        df = self._df().drop(columns=["PreBreakoutProbRaw"])
        out = apply_default_ranking(df)  # must not raise
        self.assertEqual(out.iloc[0]["Ticker"], "DDD")


if __name__ == "__main__":
    unittest.main()
