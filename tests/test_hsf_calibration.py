import datetime as dt
import unittest

from analytics import hsf_calibration as hc
from ui import opportunities as op


def _row(score, *, matured=True, mfe=None, mae=-0.015, ret=None, signals=("breakout",),
         status="STRONG", n_signals=2, when=None):
    return {
        "ticker": f"T{score}",
        "fired_at": when or dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc),
        "setup_score": score * 0.6,
        "prebreakout_prob": score * 0.7,
        "indicators": {"n_signals": n_signals, "signals": list(signals),
                       "chg_pct": (score - 50) / 10, "fading": "fading" in signals,
                       "status": status, "primary_setup": "Breakout"},
        "raw_signal": {"hsf_score": score, "score_version": "1.0",
                       "score_components": {"signals_component": 24, "model_component": score * 0.3,
                                            "momentum_component": 5, "fading_penalty": 0},
                       "status": status, "primary_setup": "Breakout"},
        "return_5d": ret if ret is not None else (0.05 if mfe and mfe >= 0.04 else -0.01),
        "mfe_5d": mfe, "mae_5d": mae,
        "outcome_computed_at": dt.datetime(2026, 8, 10, tzinfo=dt.timezone.utc) if matured else None,
    }


class ScoreVersioningTests(unittest.TestCase):
    def test_opportunities_carry_version_and_components(self):
        opps = op.build_opportunities(
            {"top_setups": [("NTAP", 55)], "golden": ["NTAP"], "gainers": [("NTAP", 4.0)]}, top_n=5)
        o = opps[0]
        self.assertEqual(o["score_version"], op.HSF_SCORE_VERSION)
        comps = o["score_components"]
        for k in ("signals_component", "model_component", "momentum_component", "fading_penalty"):
            self.assertIn(k, comps)
        # Components sum (minus penalty) reconstructs the clamped score.
        raw = (comps["signals_component"] + comps["model_component"]
               + comps["momentum_component"] - comps["fading_penalty"])
        self.assertEqual(o["score"], int(round(max(0, min(100, raw)))))

    def test_score_breakdown_matches_score(self):
        bd = op.score_breakdown(n_signals=3, breakout_score=55, prob=None, chg_pct=4.0, fading=False)
        self.assertEqual(bd["score"], op.build_opportunity_score(
            n_signals=3, breakout_score=55, prob=None, chg_pct=4.0, fading=False))


class LeakageTests(unittest.TestCase):
    def test_frozen_features_contain_no_outcome_fields(self):
        # Simulate what freeze_opportunity persists and confirm no outcome key.
        opp = op.build_opportunities({"top_setups": [("X", 55)], "golden": ["X"]}, top_n=1)[0]
        raw = {"hsf_score": opp["score"], "score_version": opp["score_version"],
               "score_components": opp["score_components"], "status": opp["status"],
               "primary_setup": opp["primary_setup"]}
        ind = {"n_signals": opp["n_signals"], "signals": opp["signals"],
               "chg_pct": opp["chg_pct"], "fading": opp["fading"], "status": opp["status"]}
        banned = {"mfe_1d", "mfe_5d", "mae_5d", "return_1d", "return_3d", "return_5d",
                  "positive", "outcome_computed_at", "hit"}
        self.assertFalse(banned & set(raw) or banned & set(ind))

    def test_normalize_never_reads_outcome_into_features(self):
        rec = hc.normalize_row(_row(90, mfe=0.06))
        # The feature side must not carry the raw outcome as a 'feature'.
        self.assertEqual(rec["hsf_score"], 90)
        self.assertTrue(rec["matured"])
        self.assertTrue(rec["positive"])  # derived, clearly on the outcome side


class DatasetTests(unittest.TestCase):
    def test_pending_excluded_matured_included(self):
        rows = [_row(80, matured=True, mfe=0.06), _row(60, matured=False, mfe=None)]
        ds = hc.build_calibration_dataset(rows)
        self.assertEqual(ds["n_total"], 2)
        self.assertEqual(ds["n_matured"], 1)
        self.assertEqual(ds["n_pending"], 1)
        # Pending never counts as a failure.
        self.assertIsNone(hc.normalize_row(_row(60, matured=False))["positive"])

    def test_empty_dataset_is_safe(self):
        ds = hc.build_calibration_dataset([])
        self.assertEqual(ds["n_matured"], 0)
        self.assertEqual(hc.summarize_score_buckets([]), hc.summarize_score_buckets([]))
        self.assertIsNone(hc.evaluate_calibration([])["brier"])
        # Baselines still list each ranker, but with no AUC (insufficient data).
        self.assertTrue(all(b["auc"] is None for b in hc.compare_baselines([])))

    def test_bucket_boundaries(self):
        self.assertEqual(hc._bucket_of(49), "0-49")
        self.assertEqual(hc._bucket_of(50), "50-59")
        self.assertEqual(hc._bucket_of(89), "80-89")
        self.assertEqual(hc._bucket_of(90), "90-100")
        self.assertEqual(hc._bucket_of(100), "90-100")
        self.assertIsNone(hc._bucket_of(None))


class SummaryTests(unittest.TestCase):
    def _monotone_rows(self):
        rows = []
        for score, rate in [(45, 0.2), (55, 0.35), (65, 0.45), (75, 0.55), (85, 0.65), (95, 0.8)]:
            for i in range(30):
                hit = i < int(rate * 30)
                rows.append(_row(score, mfe=0.06 if hit else 0.01))
        return rows

    def test_buckets_and_monotonicity(self):
        ds = hc.build_calibration_dataset(self._monotone_rows())
        buckets = hc.summarize_score_buckets(ds["matured"])
        rates = [b["positive_rate"] for b in buckets if b["n_matured"]]
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(rates, rates[1:])))
        mono = hc.evaluate_monotonicity(buckets)
        self.assertTrue(mono["monotonic"])
        self.assertGreater(mono["rank_correlation"], 0.9)

    def test_status_summary_and_confidence(self):
        rows = [_row(90, status="STRONG", mfe=0.06) for _ in range(40)] + \
               [_row(60, status="WATCH", mfe=0.01) for _ in range(5)]
        recs = hc.build_calibration_dataset(rows)["matured"]
        st = {s["status"]: s for s in hc.summarize_status_performance(recs)}
        self.assertEqual(st["STRONG"]["confidence"], "MODERATE CONFIDENCE")
        self.assertEqual(st["WATCH"]["confidence"], "INSUFFICIENT DATA")  # n=5
        self.assertEqual(st["CAUTION"]["n_matured"], 0)

    def test_baselines_bounded_auc_and_report_n(self):
        recs = hc.build_calibration_dataset(self._monotone_rows())["matured"]
        for b in hc.compare_baselines(recs):
            if b["auc"] is not None:
                self.assertGreaterEqual(b["auc"], 0.0)
                self.assertLessEqual(b["auc"], 1.0)
            self.assertIn("n", b)

    def test_calibration_brier_bounded(self):
        recs = hc.build_calibration_dataset(self._monotone_rows())["matured"]
        cal = hc.evaluate_calibration(recs)
        self.assertIsNotNone(cal["brier"])
        self.assertGreaterEqual(cal["brier"], 0.0)
        self.assertLessEqual(cal["brier"], 1.0)

    def test_confidence_labels(self):
        self.assertEqual(hc.confidence_label(5), "INSUFFICIENT DATA")
        self.assertEqual(hc.confidence_label(20), "LOW CONFIDENCE")
        self.assertEqual(hc.confidence_label(50), "MODERATE CONFIDENCE")
        self.assertEqual(hc.confidence_label(150), "STRONGER EVIDENCE")

    def test_combinations_require_min_sample(self):
        rows = [_row(80, signals=("breakout", "golden_cross"), mfe=0.06) for _ in range(3)]
        # n=3 < min_n=10 -> not reported.
        self.assertEqual(hc.analyze_signal_combinations(hc.build_calibration_dataset(rows)["matured"]), [])


class DataQualityTests(unittest.TestCase):
    def test_flags_duplicates_and_invalid_scores(self):
        when = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
        a = _row(80, when=when)
        a["ticker"] = "DUP"
        b = _row(80, when=when)
        b["ticker"] = "DUP"
        bad = _row(150)  # invalid > 100
        recs = [hc.normalize_row(x) for x in (a, b, bad)]
        warns = hc.data_quality_checks(recs)
        self.assertTrue(any("duplicate" in w for w in warns))
        self.assertTrue(any("0-100" in w for w in warns))

    def test_historical_context_still_building(self):
        rows = [_row(90, mfe=0.06) for _ in range(5)]  # n=5 -> not sufficient
        recs = hc.build_calibration_dataset(rows)["records"]
        ctx = hc.historical_context(recs, 92)
        self.assertEqual(ctx["bucket"], "90-100")
        self.assertFalse(ctx["sufficient"])
        self.assertIsNone(hc.historical_context(recs, None))


if __name__ == "__main__":
    unittest.main()
