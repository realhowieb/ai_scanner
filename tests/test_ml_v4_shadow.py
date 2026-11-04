"""ML v4 shadow mode (Milestone 5): forward scoring, summaries, promotion gates and
the shadow script. No database, no network, no training libraries."""
import datetime as dt
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from research_fixtures import opp, scan  # noqa: E402

from analytics import market_calendar as mc  # noqa: E402
from analytics import ml_v3_audit as A  # noqa: E402
from analytics import ml_v4_monitoring as M  # noqa: E402
from analytics import ml_v4_providers as P  # noqa: E402
from analytics import ml_v4_shadow as S  # noqa: E402
from analytics import research_dataset as rd  # noqa: E402

UTC = dt.timezone.utc
MODEL = {"id": 7, "model_version": "ml-v4-xgboost-production-f5d-aaaaaaaaaaaa", "target_label": "F_clean_path_5d",
         "horizon_days": 5, "feature_schema_version": P.feature_schema_version(P.FEATURE_SETS["production"]),
         "feature_set": "production", "status": "shadow", "calibration": None,
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
        self.assertGreaterEqual(preds[0]["scoring_ms"], 0)
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
                    "calibrated_probability": None, "label": y, "ai_confidence": s * 100, "scoring_ms": 0.2})
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
        return {**MODEL, "metrics": {"evaluation_version": "hsf-ml-v4-eval-1", "holdout": {"raw": {"n": 120}},
                                     "fold_auc": {"folds": 4, "worst": 0.55}}, **kw}

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
        unstable = self._model(metrics={**self._model()["metrics"], "fold_auc": {"folds": 4, "worst": 0.45}})
        bad = S.evaluate_gates(unstable, S.summarize_model(_preds(300)), gates={"max_ece": 1.0})
        self.assertEqual([c["gate"] for c in bad["checks"] if not c["passed"]], ["worst_fold_auc"])
        slow = [{**p, "scoring_ms": 500.0} for p in _preds(300)]
        bad = S.evaluate_gates(self._model(), S.summarize_model(slow), gates={"max_ece": 1.0})
        self.assertEqual([c["gate"] for c in bad["checks"] if not c["passed"]], ["scoring_latency_p95_ms"])

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
        p("db.ml_shadow_predictions.predictions", side_effect=lambda ids, since: list(self.inserted)).start()
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

    def test_feature_schema_change_refuses_to_score(self):
        with mock.patch.object(self.script, "scored_models",
                               return_value=[{**MODEL, "feature_schema_version": "research-v0/old"}]):
            self.assertEqual(self.script.main([]), 1)
        self.assertEqual(self.inserted, [])

    def test_step_summary_lists_monitoring(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "summary.md"
            with mock.patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": str(out)}):
                self.assertEqual(self.script.main([]), 0)
            text = out.read_text()
        self.assertIn("ML v4 shadow scoring", text)
        self.assertIn("scoring_fresh", text)

    def test_broken_model_is_reported_not_fatal(self):
        with mock.patch.object(self.script, "load_estimator", side_effect=ValueError("sha256 mismatch")):
            self.assertEqual(self.script.main([]), 1)
        self.assertEqual(self.inserted, [])

    def test_no_models_is_a_no_op(self):
        with mock.patch.object(self.script, "scored_models", return_value=[]):
            self.assertEqual(self.script.main([]), 0)
        self.assertEqual(self.inserted, [])

    def test_switches(self):
        script = self.script
        mock.patch.stopall()
        models = [MODEL, {**MODEL, "id": 8, "provider": "catboost", "model_version": "ml-v4-catboost-x"}]
        with mock.patch("db.ml_model_registry.list_models", side_effect=lambda status: models if status == "shadow" else []):
            with mock.patch.dict("os.environ", {"HSF_ML_V4_SHADOW_ENABLED": "", "HSF_ML_V4_SHADOW_DISABLED": ""}):
                self.assertEqual(len(script.scored_models()), 2)
            with mock.patch.dict("os.environ", {"HSF_ML_V4_SHADOW_DISABLED": "CatBoost"}):
                self.assertEqual([m["id"] for m in script.scored_models()], [7])
            with mock.patch.dict("os.environ", {"HSF_ML_V4_SHADOW_DISABLED": "7, ml-v4-catboost-x"}):
                self.assertEqual(script.scored_models(), [])
            with mock.patch.dict("os.environ", {"HSF_ML_V4_SHADOW_ENABLED": "0"}):
                self.assertEqual(script.scored_models(), [])

    def test_release_uri_is_validated(self):
        with self.assertRaises(ValueError):
            self.script.fetch_release_asset("https://example.com/x.joblib", Path("."))


class RegisterEvaluationTest(unittest.TestCase):
    def test_evaluation_matched_by_provider_feature_set_and_dataset(self):
        import ml_v4_register as R

        target = {"label": "F_clean_path_5d", "horizon_days": 5}
        binding = {"target": target, "git_sha": "abc", "providers": {"xgboost": {"seed": 1}},
                   "feature_sets": ["production"], "training_slice_fingerprint": "slice",
                   "train_ids_sha256": "ids", "library_versions": {"xgboost": "test"}, "engine_version": "v4"}
        report = {"evaluation_version": "hsf-ml-v4-eval-1", "binding": binding, "holdout_claim": "claim",
                  "dataset": {"dataset_version": "ds1", "dataset_fingerprint": "fp"},
                  "primary": {"target": target, "models": {"xgboost_production": {
                      "fold_auc": {"mean": 0.6}, "holdout": {"raw": {"n": 120}},
                      "serving_calibrator": {"method": "sigmoid", "a": 1, "b": 0}}}},
                  "head_to_head_ai_confidence": {"comparisons": {"xgboost_production": {"walk_forward": {}}}}}
        man = {"provider": "xgboost", "feature_set": "production", "dataset_version": "ds1",
               "dataset_fingerprint": "fp", "evaluation_binding": binding, "target": target,
               "git_sha": "abc", "params": {"seed": 1}, "training_slice_fingerprint": "slice",
               "train_ids_sha256": "ids", "library_versions": {"xgboost": "test"}, "engine_version": "v4"}
        metrics, cal = R.evaluation_for(report, man)
        self.assertEqual((metrics["evaluation_version"], cal["method"]), ("hsf-ml-v4-eval-1", "sigmoid"))
        for field, value in (("dataset_version", "other"), ("dataset_fingerprint", "wrong"),
                             ("provider", "catboost"), ("git_sha", "other"), ("params", {"seed": 9}),
                             ("target", {"label": "A_return_gt_0", "horizon_days": 5})):
            with self.subTest(field=field), self.assertRaises(SystemExit):
                R.evaluation_for(report, {**man, field: value})
        with self.assertRaises(SystemExit):
            R.evaluation_for({**report, "holdout_claim": None}, man)
        self.assertEqual(R.evaluation_for(None, man), (None, None))


NOW = dt.datetime(2026, 12, 10, 23, 0, tzinfo=UTC)  # a Thursday evening


def _series(n, *, start=dt.date(2026, 9, 1), shift=0.0, label_from=None, seed=4):
    rng = np.random.default_rng(seed)
    out, d = [], start
    while len(out) < n:
        if mc.is_trading_day(d):
            for _ in range(5):
                y = int(rng.random() < 0.4)
                p = float(np.clip(0.35 + 0.15 * y + shift + rng.normal(0, 0.1), 0.01, 0.99))
                out.append({"observation_id": len(out), "entry_day": d.isoformat(),
                            "scored_at": f"{d.isoformat()}T21:40:00+00:00", "raw_probability": p,
                            "calibrated_probability": None, "label": y, "ai_confidence": p * 100})
        d += dt.timedelta(days=1)
    return out[:n]


class MonitoringTest(unittest.TestCase):
    def _checks(self, res):
        return {c["check"]: c for c in res["checks"]}

    def test_waiting_when_new_and_warning_when_stale(self):
        m = {**MODEL, "status_changed_at": NOW - dt.timedelta(hours=2)}
        res = M.monitor_model(m, [], now=NOW)
        self.assertEqual(res["status"], "WAITING")
        old = {**MODEL, "status_changed_at": dt.datetime(2026, 11, 2, tzinfo=UTC)}
        self.assertEqual(self._checks(M.monitor_model(old, [], now=NOW))["scoring_fresh"]["status"], "WARNING")
        preds = _series(40, start=dt.date(2026, 12, 1))  # last scored Dec 8 or so
        c = self._checks(M.monitor_model(old, preds, now=dt.datetime(2026, 12, 21, 23, 0, tzinfo=UTC)))
        self.assertEqual(c["scoring_fresh"]["status"], "WARNING")

    def test_overdue_labels_and_schema_mismatch(self):
        preds = _series(400, start=dt.date(2026, 9, 1))
        preds[0] = {**preds[0], "label": None}
        res = M.monitor_model(MODEL, preds, now=NOW, current_feature_schema="research-v9/other")
        c = self._checks(res)
        self.assertEqual((c["labels_filling"]["status"], c["labels_filling"]["value"]), ("WARNING", 1))
        self.assertEqual(c["feature_schema"]["status"], "WARNING")
        self.assertEqual(res["status"], "WARNING")

    def test_drift_detected_and_calm_series_ok(self):
        base = _series(300, start=dt.date(2026, 9, 1))
        recent = [{**p, "observation_id": 10000 + i, "scored_at": (NOW - dt.timedelta(days=2)).isoformat()}
                  for i, p in enumerate(_series(150, start=dt.date(2026, 11, 20), seed=9))]
        calm = self._checks(M.monitor_model(MODEL, base + recent, now=NOW))
        self.assertIn(calm["probability_drift"]["status"], ("OK", "WATCH"))  # sampling noise, never a warning
        shifted = [{**p, "raw_probability": min(0.99, p["raw_probability"] + 0.3)} for p in recent]
        drift = self._checks(M.monitor_model(MODEL, base + shifted, now=NOW))
        self.assertEqual(drift["probability_drift"]["status"], "WARNING")
        self.assertIn(drift["calibration_recent"]["status"], ("WATCH", "WARNING"))

    def test_psi(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.4, 0.1, 2000)
        self.assertLess(M.psi(a, rng.normal(0.4, 0.1, 2000)), 0.05)
        self.assertGreater(M.psi(a, rng.normal(0.6, 0.1, 2000)), 0.25)
        self.assertIsNone(M.psi([], [0.1]))
        self.assertEqual(M.psi([0.2] * 10, [0.3] * 10), 0.0)

    def test_monitor_rolls_up_worst_status(self):
        res = M.monitor([{**MODEL, "status_changed_at": NOW}], {}, now=NOW)
        self.assertEqual(res["status"], "WAITING")
        self.assertEqual(res["monitoring_version"], M.MONITORING_VERSION)
