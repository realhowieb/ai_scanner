"""ML v4 registry + shadow admin endpoints (Milestone 5): admin gate, no artifact
locations, server-side evidence for validated/champion, rollback approval."""
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

EVAL = {"evaluation_version": "hsf-ml-v4-eval-1", "holdout": {"raw": {"n": 120}},
        "walk_forward_raw": {"n": 300, "roc_auc": 0.61}}


def _model(mid=1, status="candidate", metrics=None):
    return {"id": mid, "model_version": f"ml-v4-xgboost-production-f5d-{mid:012d}", "provider": "xgboost",
            "kind": "model", "feature_set": "production", "target_label": "F_clean_path_5d", "horizon_days": 5,
            "status": status, "metrics": metrics or {}, "calibration": None, "training_config": {},
            "artifact_uri": "github-release://ml-v4-models-1/secret-path.joblib", "artifact_sha256": "a" * 64,
            "status_changed_at": "2026-11-03T00:00:00+00:00"}


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class MlModelsApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.models = {1: _model(1), 2: _model(2, "shadow", EVAL)}
        self.transitions = []
        p = mock.patch
        p("db.ml_model_registry.list_models", side_effect=self._list).start()
        p("db.ml_model_registry.get_model", side_effect=lambda mid: self.models.get(mid)).start()
        p("db.ml_model_registry.model_events", return_value=[{"to_status": "candidate", "actor": "ci"}]).start()
        p("db.ml_model_registry.transition", side_effect=self._transition).start()
        p("db.ml_model_registry.rollback", return_value={"restored": 1, "retired": 2}).start()
        p("db.ml_shadow_predictions.predictions", return_value=[]).start()
        self.boss = self.auth(self.login("boss@example.com").json()["access_token"])

    def _list(self, status=None, target_label=None, horizon_days=None):
        return [m for m in self.models.values() if status in (None, m["status"])]

    def _transition(self, mid, to, *, actor, reason, evidence=None, approved=False):
        self.transitions.append((mid, to, actor, evidence, approved))
        return {"id": mid, "from": self.models[mid]["status"], "to": to, "superseded": None}

    def test_admin_only(self):
        pro = self.auth(self.login().json()["access_token"])
        for path in ("/v1/admin/ml/models", "/v1/admin/ml/models/1", "/v1/admin/ml/shadow", "/v1/admin/ml/production", "/v1/admin/ml/production/prebreakout", "/v1/admin/ml/ai-confidence", "/v1/admin/ml/prebreakout", "/v1/admin/ml/prebreakout/inputs/TEST"):
            self.assertEqual(self.client.get(path, headers=pro).status_code, 403, path)
            self.assertEqual(self.client.get(path).status_code, 401, path)
        r = self.client.post("/v1/admin/ml/models/1/transition", headers=pro,
                             json={"to_status": "retired", "reason": "nope"})
        self.assertEqual(r.status_code, 403)

    def test_production_evidence_is_read_only_and_failure_is_explicit(self):
        value = {"configured_model": {"model_version": "xgb", "trained_at": None}, "latest_capture": None}
        with mock.patch("db.ml_operations.production_confidence", return_value=value):
            r = self.client.get("/v1/admin/ml/production", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), value)
        from db.ml_model_registry import RegistryUnavailable
        with mock.patch("db.ml_operations.production_confidence", side_effect=RegistryUnavailable("down")):
            self.assertEqual(self.client.get("/v1/admin/ml/production", headers=self.boss).status_code, 503)

    def test_prebreakout_summary_inputs_and_failures(self):
        value = {"model": None, "latest": {"scanned_at": None, "rows": []}, "warnings": []}
        with mock.patch("db.prebreakout_admin.summary", return_value=value):
            r = self.client.get("/v1/admin/ml/prebreakout", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), value)
        from db.ml_model_registry import RegistryUnavailable
        with mock.patch("db.prebreakout_admin.summary", side_effect=RegistryUnavailable("down")):
            self.assertEqual(self.client.get("/v1/admin/ml/prebreakout", headers=self.boss).status_code, 503)
        with mock.patch("db.prebreakout_admin.inputs", return_value=None):
            self.assertEqual(self.client.get("/v1/admin/ml/prebreakout/inputs/TEST", headers=self.boss).status_code, 404)
        with mock.patch("db.prebreakout_admin.inputs", return_value={"inputs": []}) as load:
            r = self.client.get("/v1/admin/ml/prebreakout/inputs/TEST?scanned_at=2026-11-12T20:00:00Z", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(load.call_args.args[0], "TEST")
        self.assertEqual(load.call_args.args[1].hour, 20)
        self.assertEqual(self.client.get("/v1/admin/ml/prebreakout/inputs/NOT_VALID!", headers=self.boss).status_code, 422)

    def test_ai_confidence_summary_and_explicit_failure(self):
        value = {"model": None, "warnings": ["No active AI Confidence model in the database."],
                 "latest": {"scanned_at": None, "rows": [], "tiers": {}}, "live": {"days": 60, "buckets": []}}
        with mock.patch("db.ai_confidence_admin.summary", return_value=value):
            r = self.client.get("/v1/admin/ml/ai-confidence", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), value)
        from db.ml_model_registry import RegistryUnavailable
        with mock.patch("db.ai_confidence_admin.summary", side_effect=RegistryUnavailable("down")):
            self.assertEqual(self.client.get("/v1/admin/ml/ai-confidence", headers=self.boss).status_code, 503)

    def test_ai_confidence_explain_is_admin_only_and_grounded_on_saved_inputs(self):
        pro = self.auth(self.login().json()["access_token"])
        self.assertEqual(self.client.post("/v1/admin/ml/ai-confidence/explain", headers=pro, json={"ticker": "NVDA"}).status_code, 403)
        with mock.patch("db.ai_confidence_admin.ticker_features", return_value=None):
            r = self.client.post("/v1/admin/ml/ai-confidence/explain", headers=self.boss, json={"ticker": "ZZZZ"})
        self.assertEqual(r.status_code, 404)
        found = {"ticker": "NVDA", "scanned_at": "2026-11-12T19:35:00+00:00", "ai_confidence": 46.2,
                 "features": {"Trend10D%": 8.1, "VolRel20": 2.4}}
        seen = {}

        def fake_explain(row, *, earnings_days=None, username=None):
            seen.update(row=row, days=earnings_days, user=username)
            return "Trend and volume lift it.", None

        with mock.patch("db.ai_confidence_admin.ticker_features", return_value=found), \
                mock.patch("ui.ai_confidence_explain._earnings_days_for", return_value=2), \
                mock.patch("ui.ai_confidence_explain.explain_confidence", side_effect=fake_explain):
            r = self.client.post("/v1/admin/ml/ai-confidence/explain", headers=self.boss, json={"ticker": "nvda"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual((body["explanation"], body["tier"], body["earnings_in_days"]), ("Trend and volume lift it.", "High", 2))
        self.assertEqual(seen["row"], {"ticker": "NVDA", "AI Confidence": 46.2, "Trend10D%": 8.1, "VolRel20": 2.4})
        self.assertEqual(seen["user"], "boss@example.com")
        bad = self.client.post("/v1/admin/ml/ai-confidence/explain", headers=self.boss, json={"ticker": "NV DA;"})
        self.assertEqual(bad.status_code, 422)

    def test_prebreakout_evidence_and_explicit_failure(self):
        value = {"configured_model": {"model_version": "configured-prebreakout"},
                 "latest_capture": {"model_version": "actual-local-prebreakout"}}
        with mock.patch("db.ml_operations.production_prebreakout", return_value=value):
            r = self.client.get("/v1/admin/ml/production/prebreakout", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), value)
        from db.ml_model_registry import RegistryUnavailable
        with mock.patch("db.ml_operations.production_prebreakout", side_effect=RegistryUnavailable("down")):
            self.assertEqual(self.client.get("/v1/admin/ml/production/prebreakout", headers=self.boss).status_code, 503)

    def test_list_and_detail_hide_artifact_location(self):
        r = self.client.get("/v1/admin/ml/models", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 2)
        self.assertNotIn("secret-path", r.text)
        self.assertNotIn("github-release://", r.text)
        self.assertTrue(r.json()["models"][0]["has_artifact"])
        d = self.client.get("/v1/admin/ml/models/2", headers=self.boss).json()
        self.assertNotIn("artifact_uri", d["model"])
        self.assertFalse(d["promotion_gates"]["passed"])  # no shadow predictions yet
        self.assertEqual(d["shadow"]["labeled"], 0)
        self.assertEqual(self.client.get("/v1/admin/ml/models/99", headers=self.boss).status_code, 404)

    def test_validated_needs_stored_evaluation(self):
        r = self.client.post("/v1/admin/ml/models/1/transition", headers=self.boss,
                             json={"to_status": "validated", "reason": "looks fine"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.transitions, [])
        self.models[1]["metrics"] = EVAL
        r = self.client.post("/v1/admin/ml/models/1/transition", headers=self.boss,
                             json={"to_status": "validated", "reason": "walk-forward reviewed"})
        self.assertEqual(r.status_code, 200)
        mid, to, actor, evidence, _ = self.transitions[0]
        self.assertEqual((mid, to, actor), (1, "validated", "boss@example.com"))
        self.assertEqual(evidence["evaluation"]["evaluation_version"], "hsf-ml-v4-eval-1")

    def test_champion_refused_until_gates_pass(self):
        r = self.client.post("/v1/admin/ml/models/2/transition", headers=self.boss,
                             json={"to_status": "champion", "reason": "ship it", "approve": True})
        self.assertEqual(r.status_code, 422)
        self.assertIn("promotion_gates", r.json()["detail"])
        self.assertEqual(self.transitions, [])
        passing = {"passed": True, "checks": []}
        with mock.patch("analytics.ml_v4_shadow.evaluate_gates", return_value=passing):
            r = self.client.post("/v1/admin/ml/models/2/transition", headers=self.boss,
                                 json={"to_status": "champion", "reason": "gates passed", "approve": True})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.transitions[0][3]["promotion_gates_passed"], True)
        self.assertTrue(self.transitions[0][4])

    def test_lifecycle_refusal_is_422_and_outage_503(self):
        from analytics.ml_v4_lifecycle import LifecycleError
        from db.ml_model_registry import RegistryUnavailable

        with mock.patch("db.ml_model_registry.transition", side_effect=LifecycleError("not allowed")):
            r = self.client.post("/v1/admin/ml/models/1/transition", headers=self.boss,
                                 json={"to_status": "shadow", "reason": "skip"})
        self.assertEqual((r.status_code, r.json()["detail"]), (422, "not allowed"))
        with mock.patch("db.ml_model_registry.list_models", side_effect=RegistryUnavailable("db down")):
            r = self.client.get("/v1/admin/ml/models", headers=self.boss)
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("db down", r.text)

    def test_rollback_needs_approval(self):
        body = {"target_label": "F_clean_path_5d", "horizon_days": 5, "reason": "live drift"}
        self.assertEqual(self.client.post("/v1/admin/ml/rollback", headers=self.boss, json=body).status_code, 422)
        r = self.client.post("/v1/admin/ml/rollback", headers=self.boss, json={**body, "approve": True})
        self.assertEqual(r.json()["restored"], 1)

    def test_shadow_overview(self):
        r = self.client.get("/v1/admin/ml/shadow", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual([m["model"]["id"] for m in out["models"]], [2])
        self.assertNotIn("github-release://", r.text)
        self.models = {1: _model(1)}
        self.assertEqual(self.client.get("/v1/admin/ml/shadow", headers=self.boss).json()["models"], [])

    def test_monitoring(self):
        r = self.client.get("/v1/admin/ml/monitoring", headers=self.boss)
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual([m["model_id"] for m in out["models"]], [2])
        self.assertIn("feature_schema", [c["check"] for c in out["models"][0]["checks"]])
        pro = self.auth(self.login().json()["access_token"])
        self.assertEqual(self.client.get("/v1/admin/ml/monitoring", headers=pro).status_code, 403)

    def test_api_never_imports_training_libraries(self):
        import subprocess
        import sys

        code = ("import sys; import api.routes.ml_models, analytics.ml_v4_shadow, db.ml_model_registry, "
                "db.ml_shadow_predictions, analytics.ml_v4_monitoring, analytics.ml_v4_providers; "
                "print(','.join(m for m in ('lightgbm','catboost','xgboost','sklearn') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "")

    def test_production_capture_requires_admin_and_masks_outage(self):
        from db.ml_model_registry import RegistryUnavailable
        url = '/v1/admin/ml/production/health'
        with mock.patch('db.ml_operations.production_capture_health', return_value={'models': []}) as read:
            self.assertEqual(self.client.get(url).status_code, 401)
            pro = self.auth(self.login().json()['access_token'])
            self.assertEqual(self.client.get(url, headers=pro).status_code, 403)
            read.assert_not_called()
            self.assertEqual(self.client.get(url, headers=self.boss).json(), {'models': []})
        with mock.patch('db.ml_operations.production_capture_health', side_effect=RegistryUnavailable('private detail')):
            response = self.client.get(url, headers=self.boss)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private detail', response.text)
