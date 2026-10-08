"""V1 bug fix — PreBreakout calibration display/ranking behavior.

Documents WHY many Scanner rows share one calibrated % (the ~13.1% plateau): it
is a legitimate isotonic calibration plateau plus np.interp's flat left tail
(raw <= first knot -> first knot's y), NOT a bug/fallback/broadcast/rounding.
The calibrated value is PRIMARY for ranking; the raw model probability is a
deterministic secondary tie-break. These tests guard against someone "fixing"
valid calibration and against a regression of the tie-break.
"""
import importlib.util
import unittest
from unittest import mock

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


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
class SigmoidCalibrationTests(unittest.TestCase):
    def setUp(self):
        m._load_ml_libs()
        rng = np.random.default_rng(7)
        self.raw = rng.beta(1.2, 6, 4000)
        self.y = (rng.random(4000) < self.raw).astype(int)

    def test_sigmoid_map_keeps_low_scores_distinct(self):
        cmap = m.fit_sigmoid_calibration_map(self.y, self.raw)
        self.assertEqual(cmap["method"], "sigmoid")
        out = m.apply_calibration_map([0.003, 0.010, 0.020, 0.030], cmap)
        self.assertTrue(np.all(np.diff(out) > 0))  # strictly increasing, no plateau

    def test_smoothing_report_scores_out_of_time(self):
        report = m.calibration_smoothing_report(self.y, self.raw, live_map=_MAP)
        self.assertEqual(report["fit_rows"] + report["test_rows"], 4000)
        for key in ("raw", "isotonic", "sigmoid", "live_map"):
            self.assertIn("brier", report[key])
            self.assertEqual(len(report[key]["deciles"]), 10)
        self.assertGreater(report["sigmoid"]["spread"]["distinct_whole_pcts"], report["live_map"]["spread"]["distinct_whole_pcts"])
        self.assertEqual(report["sigmoid_map_all_rows"]["n"], 4000)

    def test_smoothing_report_skips_tiny_inputs(self):
        self.assertIn("skipped", m.calibration_smoothing_report([0, 1] * 10, [0.1, 0.2] * 10))

    def test_live_report_uses_only_matured_rows_since_training(self):
        now = pd.Timestamp("2026-10-07", tz="UTC")
        stamps = pd.date_range("2026-08-01", periods=len(self.raw), freq="15min", tz="UTC")
        labeled = pd.DataFrame({"Timestamp": stamps, m.PREBREAKOUT_TARGET_COLUMN: self.y})
        raw_by_ts = dict(zip(stamps, self.raw))

        def fake_score(df):
            df["PreBreakoutProbRaw"] = [raw_by_ts[t] for t in df["Timestamp"]]
            return df

        bundle = {"model": object(), "trained_at": "2026-08-10T00:00:00Z", "model_version": "v", "calibration_map": _MAP}
        with mock.patch.object(m, "load_prebreakout_model", return_value=bundle), \
                mock.patch.object(m, "load_run_history", return_value=pd.DataFrame({"x": [1]})), \
                mock.patch.object(m, "add_prebreakout_target_label", return_value=labeled), \
                mock.patch.object(m, "score_prebreakout", side_effect=fake_score), \
                mock.patch.object(m, "_utc_now", return_value=now.to_pydatetime()):
            report = m.live_calibration_report(days_back=90)
        expected = int(((stamps > pd.Timestamp("2026-08-10", tz="UTC")) & (stamps <= now - pd.Timedelta(days=10))).sum())
        self.assertEqual(report["rows_since_trained"], expected)
        self.assertIn("sigmoid", report)
        self.assertIn("live_auc", report)

    def test_live_report_retries_a_dropped_model_load(self):
        loads = mock.Mock(side_effect=[None, None, None])
        with mock.patch.object(m, "load_prebreakout_model", loads), mock.patch("time.sleep"):
            report = m.live_calibration_report()
        self.assertEqual(report, {"skipped": "no live model"})
        self.assertEqual(loads.call_count, 3)

    def test_serving_skew_audit_compares_pipelines(self):
        n = 600
        stamps = pd.date_range("2026-08-20", periods=n, freq="h", tz="UTC")
        rows = pd.DataFrame({
            "Symbol": [f"S{i % 40}" for i in range(n)], "Timestamp": stamps,
            "run_time": stamps.floor("D"), "Last": 10 + self.raw[:n],
            "PreBreakoutProbRaw": self.raw[:n], m.PREBREAKOUT_TARGET_COLUMN: self.y[:n],
        })

        class Model:
            def predict_proba(self, X):
                v = 1 / (1 + np.exp(-X["F1"].to_numpy()))
                return np.column_stack([1 - v, v])

        def feats(df, **_):
            df = df.copy()
            df["F1"] = df["PreBreakoutProbRaw"] * (2 if "OHLC" in df.columns else 1)
            return df

        bundle = {"model": Model(), "features": ["F1"], "trained_at": "2026-08-01T00:00:00Z", "auc": 0.68}
        with mock.patch.object(m, "load_prebreakout_model", return_value=bundle), \
                mock.patch.object(m, "load_run_history", return_value=rows), \
                mock.patch.object(m, "add_prebreakout_target_label", return_value=rows), \
                mock.patch.object(m, "add_prebreakout_features", side_effect=feats), \
                mock.patch.object(m, "load_benchmark_regime_context", return_value={}), \
                mock.patch.object(m, "add_historical_ohlcv_context", side_effect=lambda df, **_: df.assign(OHLC=1)), \
                mock.patch.object(m, "_utc_now", return_value=pd.Timestamp("2026-10-08", tz="UTC").to_pydatetime()):
            report = m.serving_skew_audit()
        self.assertEqual(report["rows"], n)
        self.assertEqual(set(report["auc"]), {"stored", "per_scan", "history", "training"})
        self.assertEqual(report["auc"]["stored"], report["auc"]["per_scan"])
        self.assertEqual(report["most_different_features"][0]["feature"], "F1")


if __name__ == "__main__":
    unittest.main()
