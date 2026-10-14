"""Admin analytics API (/v1/admin/analytics, signals, evidence, stair-stepper, ml-readiness)."""
import datetime as dt
import unittest
from unittest import mock

try:
    from tests.test_api_v1 import DEPS, ApiTestCase
except ImportError:  # pragma: no cover
    DEPS, ApiTestCase = False, unittest.TestCase

NOW = dt.datetime.now(dt.timezone.utc)


def _obs(oid, signal, cohort, score, ret, horizon="5d", status="MATURED"):
    outcome = {"data_status": status, "directional_return": ret, "raw_return": ret, "mfe": 0.05, "mae": -0.02,
               "hit": None if ret is None else ret > 0, "evaluation_time": NOW.isoformat()}
    return {"observation_id": oid, "symbol": "NVDA", "timestamp": (NOW - dt.timedelta(days=2)).isoformat(),
            "context": "scheduled:intraday", "session": "regular", "research_cohort": cohort,
            "research_metadata": {"row_features": {"hsf_score": score}}, "scanners": [{"name": signal, "triggered": True}],
            "outcomes": {horizon: outcome} if ret is not None else {}}


OBS = [_obs("a", "Breakout", "candidate", 82, 0.04), _obs("b", "Breakout", "control", 55, -0.01),
       _obs("c", "Momentum", "candidate", 91, 0.02), _obs("d", "Momentum", "near_miss", None, None)]

BUNDLE = {
    "available": True, "loaded_at": NOW.isoformat(), "days": 30, "errors": {"acquisition_events": "UndefinedTable"},
    "users": [{"tier": "pro", "is_active": True, "created_at": NOW - dt.timedelta(days=1), "has_subscription": True},
              {"tier": "basic", "is_active": True, "created_at": NOW - dt.timedelta(days=60), "has_subscription": False},
              {"tier": "premium", "is_active": False, "created_at": NOW - dt.timedelta(days=3), "has_subscription": True}],
    "events": [{"event_name": "signup_completed", "user_hash": "h1", "plan": None, "occurred_at": NOW},
               {"event_name": "signup_completed", "user_hash": "h2", "plan": None, "occurred_at": NOW},
               {"event_name": "successful_paid_conversion", "user_hash": "h1", "plan": "pro", "occurred_at": NOW},
               {"event_name": "return_session", "user_hash": "h2", "plan": None, "occurred_at": NOW}],
    "runs": [{"username": None, "created_at": NOW}, {"username": "pro@example.com", "created_at": NOW},
             {"username": None, "created_at": NOW - dt.timedelta(days=1)}],
    "health": {"subsystems": {"scanner": {"status": "HEALTHY", "reason": "all slots ran"}}},
    "recovery": [{"started_at": "2026-10-13T12:00:00Z", "incident_id": "SCAN_MISSED", "action": "rerun", "result": "ok"}],
}


@unittest.skipUnless(DEPS, "needs API dependencies")
class AdminAnalyticsApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        from api import admin_analytics

        admin_analytics.clear_cache()

    def auth(self, who="boss@example.com"):
        return {"Authorization": "Bearer " + self.login(who).json()["access_token"]}

    def test_every_analytics_endpoint_needs_an_admin(self):
        pro = self.auth("pro@example.com")
        with mock.patch("db.admin_analytics.load_admin_data", side_effect=AssertionError("must not load")), \
             mock.patch("api.admin_analytics._rows", side_effect=AssertionError("must not load")):
            for path in ("/v1/admin/analytics", "/v1/admin/signals", "/v1/admin/evidence",
                         "/v1/admin/stair-stepper", "/v1/admin/ml-readiness"):
                self.assertEqual(self.client.get(path).status_code, 401, path)
                self.assertEqual(self.client.get(path, headers=pro).status_code, 403, path)

    def test_product_analytics(self):
        with mock.patch("db.admin_analytics.load_admin_data", return_value=BUNDLE) as load:
            r = self.client.get("/v1/admin/analytics?days=7", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        load.assert_called_once_with(days=7, include_research=False)
        b = r.json()
        self.assertEqual(b["overview"], {"total_users": 3, "active_users_30d": 1, "new_users_7d": 2,
                                         "scans_today": 2, "paying_users": 1})
        self.assertEqual(len(b["users_daily"]), 7)
        self.assertEqual(b["users_daily"][-1]["total"], 3)
        self.assertEqual(sum(d["scheduled"] for d in b["scans_daily"]), 2)
        self.assertEqual({p["plan"]: p["users"] for p in b["plans"]}, {"Basic": 1, "Pro": 1, "Premium": 0, "Admin": 0})
        self.assertEqual(b["conversions"]["signup_to_paid"], 0.5)
        self.assertEqual(b["conversions"]["destinations"], [{"plan": "Pro", "users": 1}, {"plan": "Premium", "users": 0}])
        self.assertEqual(b["unavailable"], ["acquisition_events"])
        self.assertEqual(b["systems"][1], {"system": "Scanner", "status": "HEALTHY", "reason": "all slots ran"})
        self.assertEqual(b["recovery"][0]["incident"], "SCAN_MISSED")

    def test_period_must_be_listed_and_database_must_be_up(self):
        h = self.auth()
        self.assertEqual(self.client.get("/v1/admin/analytics?days=12", headers=h).status_code, 422)
        with mock.patch("db.admin_analytics.load_admin_data", return_value={"available": False}):
            self.assertEqual(self.client.get("/v1/admin/analytics", headers=h).status_code, 503)

    def test_signal_performance_filters_and_calibration(self):
        h = self.auth()
        with mock.patch("api.admin_analytics.scanner_observations", return_value=OBS):
            all_ = self.client.get("/v1/admin/signals?min_n=1", headers=h).json()
            only = self.client.get("/v1/admin/signals?min_n=1&signal=Breakout&score_min=60", headers=h).json()
        self.assertEqual(all_["options"]["signals"], ["Breakout", "Momentum"])
        self.assertEqual(all_["options"]["cohorts"], ["CANDIDATE", "CONTROL", "NEAR_MISS"])
        self.assertEqual((all_["options"]["score_low"], all_["options"]["score_high"]), (55, 91))
        self.assertEqual(all_["summary"]["observations"], 4)
        self.assertEqual(all_["summary"]["matured"], 3)
        self.assertAlmostEqual(all_["summary"]["positive_rate"], 2 / 3)
        self.assertEqual(only["summary"]["observations"], 1)
        self.assertAlmostEqual(only["summary"]["median_return"], 0.04)
        cells = {(c["key"], c["column"]): c["n"] for c in all_["by_signal_horizon"]}
        self.assertEqual(cells, {("Breakout", "5d"): 2, ("Momentum", "5d"): 1})
        self.assertTrue(all_["score_available"])
        self.assertEqual(all_["score_horizon"], "5d")
        self.assertEqual([b["key"] for b in all_["score_buckets"]], ["50-59", "80-89", "90-100"])

    def test_evidence_funnel_and_readiness(self):
        report = {"subsystems": {"forward_evidence": {"detail_state": "COLLECTING", "reason": "need more days",
                                                      "metrics": {"trading_days_collected": 9, "trading_days_preferred": 20}},
                                 "maturation": {"metrics": {"last_success": "2026-10-14T03:00:00Z"}}}}
        with mock.patch("api.admin_analytics.scanner_observations", return_value=OBS), \
             mock.patch("api.operations.load_snapshot", return_value={"report": report}):
            r = self.client.get("/v1/admin/evidence?days=90", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertEqual([s["count"] for s in b["funnel"]], [4, 4, 3, 3, 3])
        self.assertEqual(b["pending"], 1)
        self.assertEqual(b["readiness"]["state"], "COLLECTING")
        self.assertEqual(b["readiness"]["trading_days"], 9)
        self.assertEqual(b["last_maturation_run"], "2026-10-14T03:00:00+00:00")
        self.assertEqual({c["cohort"]: c["observations"] for c in b["cohorts"]}, {"CANDIDATE": 2, "CONTROL": 1, "NEAR_MISS": 1})

    def test_stair_stepper_without_evidence(self):
        with mock.patch("api.admin_analytics.stair_observations", return_value=[]):
            r = self.client.get("/v1/admin/stair-stepper", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["observations"], 0)
        self.assertIsNone(r.json()["best_window"])
        self.assertIn("+15m", r.json()["horizons"])

    def test_ml_readiness_compact_view(self):
        view = {"status": "NOT_READY", "recommendation": "Keep collecting.", "generated_at": "2026-10-14T12:00:00Z",
                "observations": {"matured_signal_days": 134, "matured": 600},
                "coverage": {"matured_trading_days": 12, "matured_symbols": 80, "outcome_coverage_recent": 0.9},
                "validation": {"usable_walk_forward_folds": 0},
                "gates": [{"gate": "Matured signal-days", "kind": "accumulating", "value": 134, "operator": ">=",
                           "threshold": 1000, "status": "FAIL", "progress": 0.134, "why": "volume"}],
                "blocking_reasons": ["Matured signal-days 134 < 1000"],
                "projection": {"estimate_trading_days_to_accumulating_targets": 40}}
        with mock.patch("api.ml_readiness.readiness", return_value=view):
            r = self.client.get("/v1/admin/ml-readiness", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["gates"][0]["needs"], ">= 1000")
        self.assertEqual(r.json()["gates"][0]["kind"], "Accumulating")
        self.assertEqual(r.json()["estimate_trading_days"], 40)


if __name__ == "__main__":
    unittest.main()
