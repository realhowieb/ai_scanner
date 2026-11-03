"""ML v4 shadow mode (Milestone 5): forward scoring, summaries, promotion gates and
the shadow script. No database, no network, no training libraries."""
import datetime as dt
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from research_fixtures import opp, scan  # noqa: E402

from analytics import market_calendar as mc  # noqa: E402
from analytics import ml_v3_audit as A  # noqa: E402
from analytics import ml_v4_shadow as S  # noqa: E402
from analytics import research_dataset as rd  # noqa: E402

UTC = dt.timezone.utc
MODEL = {"id": 7, "model_version": "ml-v4-xgboost-production-f5d-aaaaaaaaaaaa", "target_label": "F_clean_path_5d",
         "horizon_days": 5, "feature_schema_version": "fs1", "status": "shadow", "calibration": None,
         "artifact_uri": "github-release://ml-v4-models-1/x.joblib", "artifact_sha256": "a" * 64}


def _rows(n_days=12, per_day=4):
    rng = random.Random(3)
    rows, scans, oid, d = [], [], 1, dt.date(2026, 6, 1)
    days = []
    while len(days) < n_days:
        if mc.is_trading_day(d):
            days.append(d)
        d += dt.timedelta(days=1)
    for day in days:
        for k in range(per_day):
            t = f"T{oid:03d}"
            r5 = rng.gauss(0.01, 0.04)
            fired = f"{day.isoformat()}T15:{10 + k:02d}:00"
            rows.append(opp(oid, t, fired, score=50 + k, scored=True, r1=r5 / 3, r3=r5 / 2, r5=r5,
                            mfe=max(r5, 0) + 0.02, mae=-0.01, b1=0.001, b3=0.002, b5=0.003))
            scans.append(scan(t, (dt.datetime.fromisoformat(fired) - dt.timedelta(minutes=10)).isoformat()))
            oid += 1
    built = rd.build_dataset(rows, scans)
    return A.build_audit_rows(built["records"], rows, unit="signal_day")


class FakeEstimator:
    def predict(self, rows):
        return np.array([0.2 + 0.01 * (r.observation_id % 10) for r in rows])


class AucTest(unittest.TestCase):
    def test_matches_rank_definition_with_ties(self):
        y = [0, 0, 1, 1, 0, 1]
        p = [0.1, 0.4, 0.35, 0.8, 0.4, 0.4]
        # pairs (pos, neg): 9; wins + half ties counted by hand = 6
        self.assertAlmostEqual(S.auc(y, p), 6 / 9)
        self.assertIsNone(S.auc([1, 1], [0.2, 0.3]))


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()

    def test_forward_rows_only_after_shadow_start(self):
        start = self.rows[len(self.rows) // 2].observed_at
        fwd = S.forward_rows(self.rows, start)
        self.assertTrue(fwd and all(r.observed_at >= start for r in fwd))
        self.assertEqual(S.forward_rows(self.rows, None), [])

    def test_score_uses_registered_calibrator(self):
        preds = S.score(MODEL, FakeEstimator(), self.rows[:3], scored_at="2026-11-03T21:40:00+00:00")
        self.assertEqual(len(preds), 3)
        self.assertIsNone(preds[0]["calibrated_probability"])
        self.assertEqual(preds[0]["model_id"], 7)
        cal = {"method": "sigmoid", "a": 1.0, "b": 1.0}
        preds = S.score({**MODEL, "calibration": cal}, FakeEstimator(), self.rows[:3], scored_at="x")
        self.assertGreater(preds[0]["calibrated_probability"], preds[0]["raw_probability"])
        self.assertEqual(S.score(MODEL, FakeEstimator(), [], scored_at="x"), [])


def _preds(n, *, model_id=7, edge=0.3, served_edge=0.1, days=25, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = int(rng.random() < 0.4)
        p = float(np.clip(0.4 + (edge if y else -edge) * 0.5 + rng.normal(0, 0.15), 0.01, 0.99))
        s = float(np.clip(0.4 + (served_edge if y else -served_edge) * 0.5 + rng.normal(0, 0.15), 0.01, 0.99))
        out.append({"model_id": model_id, "observation_id": i, "entry_day": f"2026-11-{i % days + 1:02d}",
                    "scored_at": f"2026-11-{i % days + 1:02d}T21:40:00", "raw_probability": p,
                    "calibrated_probability": None, "label": y, "ai_confidence": s * 100})
    return out


class SummaryAndGatesTest(unittest.TestCase):
    def test_summary_compares_with_ai_confidence_on_same_rows(self):
        s = S.summarize_model(_preds(300))
        self.assertEqual((s["scored"], s["labeled"], s["entry_days"]), (300, 300, 25))
        vs = s["vs_ai_confidence_as_served"]
        self.assertEqual(vs["rows"], 300)
        self.assertGreater(vs["auc_delta"], 0)
        self.assertIn("reliability", s)
        empty = S.summarize_model([{**p, "label": None} for p in _preds(5)])
        self.assertEqual(empty["labeled"], 0)
        self.assertIn("note", empty)

    def test_disagreement_on_shared_observations(self):
        d = S.disagreement({7: _preds(50), 8: _preds(50, model_id=8, seed=2)})
        self.assertEqual((d["models"], d["rows"]), (2, 50))
        self.assertIn("not independent", d["note"])
        self.assertEqual(S.disagreement({7: _preds(5)})["rows"], 0)

    def _model(self, **kw):
        return {**MODEL, "metrics": {"evaluation_version": "hsf-ml-v4-eval-1", "holdout": {"raw": {"n": 120}}}, **kw}

    def test_gates_pass_only_when_every_check_passes(self):
        g = S.evaluate_gates(self._model(), S.summarize_model(_preds(300)), gates={"max_ece": 1.0})
        self.assertTrue(g["passed"], [c for c in g["checks"] if not c["passed"]])
        few = S.evaluate_gates(self._model(), S.summarize_model(_preds(50)), gates={"max_ece": 1.0})
        self.assertFalse(few["passed"])
        self.assertIn("labeled_shadow_rows", [c["gate"] for c in few["checks"] if not c["passed"]])
        worse = S.evaluate_gates(self._model(), S.summarize_model(_preds(300, edge=0.05, served_edge=0.3)),
                                 gates={"max_ece": 1.0})
        self.assertIn("auc_vs_ai_confidence", [c["gate"] for c in worse["checks"] if not c["passed"]])
        no_eval = S.evaluate_gates({**MODEL, "metrics": {}}, S.summarize_model(_preds(300)), gates={"max_ece": 1.0})
        self.assertFalse(no_eval["passed"])
        self.assertFalse(S.evaluate_gates(self._model(status="validated"), S.summarize_model(_preds(300)),
                                          gates={"max_ece": 1.0})["passed"])

    def test_evaluation_evidence(self):
        self.assertIsNone(S.evaluation_evidence(MODEL))
        ev = S.evaluation_evidence({"metrics": {"evaluation_version": "e1", "walk_forward_raw": {
            "n": 200, "roc_auc": 0.6, "reliability": [1, 2]}, "holdout": {"train_n": 300, "raw": {"n": 100}}}})
        self.assertEqual(ev["evaluation"]["walk_forward_raw"]["roc_auc"], 0.6)
        self.assertNotIn("reliability", ev["evaluation"]["walk_forward_raw"])
        self.assertEqual(ev["evaluation"]["holdout"]["train_n"], 300)


class ShadowScriptTest(unittest.TestCase):
    def setUp(self):
        import ml_v4_shadow as script

        self.script = script
        self.rows = _rows()
        self.inserted, self.filled = [], []
        start = self.rows[len(self.rows) // 2].observed_at
        p = mock.patch
        p.object(script, "scored_models", return_value=[MODEL]).start()
        p.object(script, "shadow_start", return_value=start).start()
        p.object(script, "recent_rows", return_value=self.rows).start()
        p.object(script, "load_estimator", return_value=FakeEstimator()).start()
        p("db.ml_shadow_predictions.insert_predictions", side_effect=lambda preds: self.inserted.extend(preds)
          or len(preds)).start()
        p("db.ml_shadow_predictions.unlabeled",
          side_effect=lambda label, h: [r.observation_id for r in self.rows]).start()
        p("db.ml_shadow_predictions.fill_labels",
          side_effect=lambda label, h, labels: self.filled.extend(labels) or len(labels)).start()
        p("db.ml_shadow_predictions.prune", return_value=0).start()
        self.addCleanup(mock.patch.stopall)
        self.start = start

    def test_count(self):
        with mock.patch("builtins.print") as pr:
            self.assertEqual(self.script.main(["--count"]), 0)
        pr.assert_called_with(1)

    def test_scores_forward_rows_and_fills_matured_labels(self):
        self.assertEqual(self.script.main([]), 0)
        self.assertTrue(self.inserted)
        self.assertTrue(all(p["observed_at"] >= self.start for p in self.inserted))
        expect = {r.observation_id for r, _ in A.eligible(self.rows, 5, "F_clean_path_5d")}
        self.assertEqual({oid for oid, _ in self.filled}, expect)

    def test_broken_model_is_reported_not_fatal(self):
        with mock.patch.object(self.script, "load_estimator", side_effect=ValueError("sha256 mismatch")):
            self.assertEqual(self.script.main([]), 1)
        self.assertEqual(self.inserted, [])

    def test_no_models_is_a_no_op(self):
        with mock.patch.object(self.script, "scored_models", return_value=[]):
            self.assertEqual(self.script.main([]), 0)
        self.assertEqual(self.inserted, [])

    def test_release_uri_is_validated(self):
        with self.assertRaises(ValueError):
            self.script.fetch_release_asset("https://example.com/x.joblib", Path("."))


class RegisterEvaluationTest(unittest.TestCase):
    def test_evaluation_matched_by_provider_feature_set_and_dataset(self):
        import ml_v4_register as R

        report = {"evaluation_version": "hsf-ml-v4-eval-1", "dataset": {"dataset_version": "ds1"},
                  "primary": {"models": {"xgboost_production": {
                      "fold_auc": {"mean": 0.6}, "holdout": {"raw": {"n": 120}},
                      "serving_calibrator": {"method": "sigmoid", "a": 1, "b": 0}}}},
                  "head_to_head_ai_confidence": {"comparisons": {"xgboost_production": {"walk_forward": {}}}}}
        man = {"provider": "xgboost", "feature_set": "production", "dataset_version": "ds1"}
        metrics, cal = R.evaluation_for(report, man)
        self.assertEqual((metrics["evaluation_version"], cal["method"]), ("hsf-ml-v4-eval-1", "sigmoid"))
        self.assertEqual(R.evaluation_for(report, {**man, "dataset_version": "other"}), (None, None))
        self.assertEqual(R.evaluation_for(report, {**man, "provider": "catboost"}), (None, None))
        self.assertEqual(R.evaluation_for(None, man), (None, None))
