"""Admin console API (/v1/admin/*): admin gate, account actions, health and tools."""
import unittest
from unittest import mock

try:
    from tests.test_api_v1 import DEPS, ApiTestCase
except ImportError:  # pragma: no cover
    DEPS, ApiTestCase = False, unittest.TestCase

USERS = [
    {"username": "boss@example.com", "full_name": "Boss", "tier": "basic", "is_active": True,
     "is_admin": True, "email_verified": True, "created_at": None},
    {"username": "pro@example.com", "full_name": "Pro User", "tier": "pro", "is_active": True,
     "is_admin": False, "email_verified": False, "created_at": None},
]


@unittest.skipUnless(DEPS, "needs API dependencies")
class AdminApiTests(ApiTestCase):
    def auth(self, who="boss@example.com"):
        return {"Authorization": "Bearer " + self.login(who).json()["access_token"]}

    def test_every_admin_endpoint_needs_an_admin(self):
        gets = ["/v1/admin/users", "/v1/admin/audit", "/v1/admin/health", "/v1/admin/providers",
                "/v1/admin/activity", "/v1/admin/earnings-refresh"]
        posts = ["/v1/admin/users", "/v1/admin/users/update", "/v1/admin/users/verify",
                 "/v1/admin/users/resend-verification", "/v1/admin/users/role", "/v1/admin/db-check",
                 "/v1/admin/earnings-refresh"]
        pro = self.auth("pro@example.com")
        with mock.patch("db.user_admin.fetch_users_for_admin", side_effect=AssertionError("must not load")):
            for path in gets:
                self.assertEqual(self.client.get(path).status_code, 401, path)
                self.assertEqual(self.client.get(path, headers=pro).status_code, 403, path)
            for path in posts:
                self.assertEqual(self.client.post(path, json={}).status_code, 401, path)
                self.assertEqual(self.client.post(path, json={}, headers=pro).status_code, 403, path)

    def test_lists_users(self):
        with mock.patch("db.user_admin.fetch_users_for_admin", return_value=USERS):
            r = self.client.get("/v1/admin/users", headers=self.auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual([u["username"] for u in r.json()["users"]], ["boss@example.com", "pro@example.com"])
        self.assertFalse(r.json()["users"][1]["email_verified"])

    def test_update_user_changes_plan_and_audits(self):
        h = self.auth()
        with mock.patch("db.user_admin.fetch_users_for_admin", return_value=USERS), \
             mock.patch("db.user_admin._update", return_value=True) as upd, \
             mock.patch("db.admin_audit.record_admin_event") as audit:
            r = self.client.post("/v1/admin/users/update", headers=h,
                                 json={"username": "PRO@example.com", "tier": "premium"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(upd.call_args.args[1], ("premium", True, "pro@example.com"))
        audit.assert_called_once_with("boss@example.com", "update_user", "pro@example.com", {"tier": ["pro", "premium"]})

    def test_update_refuses_bad_changes(self):
        h = self.auth()
        with mock.patch("db.user_admin.fetch_users_for_admin", return_value=USERS), \
             mock.patch("db.user_admin._update", side_effect=AssertionError("must not write")):
            self.assertEqual(self.client.post("/v1/admin/users/update", headers=h,
                                              json={"username": "nobody@example.com", "tier": "pro"}).status_code, 422)
            r = self.client.post("/v1/admin/users/update", headers=h,
                                 json={"username": "boss@example.com", "is_active": False})
            self.assertEqual(r.status_code, 422)
            self.assertIn("your own account", r.json()["detail"])
            self.assertEqual(self.client.post("/v1/admin/users/update", headers=h,
                                              json={"username": "pro@example.com", "tier": "admin"}).status_code, 422)
            r = self.client.post("/v1/admin/users/update", headers=h, json={"username": "pro@example.com", "tier": "pro"})
            self.assertEqual(r.json()["message"], "Nothing to change.")

    def test_role_verify_and_create_use_db_user_admin(self):
        h = self.auth()
        with mock.patch("db.user_admin.set_admin", return_value=(False, "You can't remove your own admin role.")) as role:
            r = self.client.post("/v1/admin/users/role", headers=h, json={"username": "boss@example.com", "is_admin": False})
        self.assertEqual(r.status_code, 422)
        role.assert_called_once_with("boss@example.com", False, acting_user="boss@example.com")
        with mock.patch("db.user_admin.mark_email_verified", return_value=(True, "ok")) as ver:
            r = self.client.post("/v1/admin/users/verify", headers=h, json={"username": "pro@example.com"})
        self.assertEqual(r.json(), {"ok": True, "message": "ok"})
        ver.assert_called_once_with("pro@example.com", actor="boss@example.com")
        with mock.patch("db.user_admin.create_user", return_value=(True, "Created new@example.com.")) as create:
            r = self.client.post("/v1/admin/users", headers=h, json={"email": "new@example.com", "full_name": "New",
                                                                      "password": "a long pass phrase", "tier": "pro"})
        self.assertEqual(r.status_code, 200)
        create.assert_called_once_with("new@example.com", "New", "a long pass phrase", tier="pro", active=True,
                                       actor="boss@example.com")

    def test_resend_verification(self):
        h = self.auth()
        with mock.patch("api.account.send_verification", return_value=True) as send:
            r = self.client.post("/v1/admin/users/resend-verification", headers=h, json={"username": "Pro@example.com"})
        self.assertEqual(r.status_code, 200)
        send.assert_called_once_with("pro@example.com")
        r = self.client.post("/v1/admin/users/resend-verification", headers=h, json={"username": "legacy-name"})
        self.assertEqual(r.status_code, 422)

    def test_health_combines_snapshot_and_email(self):
        report = {"health_score": 90, "human_action": "NONE", "market": {"next_expected_scan": "2026-10-15T12:35:00Z"},
                  "subsystems": {"scanner": {"status": "HEALTHY", "reason": "ok", "metrics": {}}},
                  "incidents": [{"severity": "WARNING", "incident_id": "X1", "age_hours": 2, "summary": "s",
                                 "recommended_action": "r", "human_action": "NONE", "automation_candidate": False}]}
        email = {"status": "HEALTHY", "day": "2026-10-14",
                 "jobs": {"digest": {"label": "Morning digest", "status": "HEALTHY", "detail": "sent 3"}}}
        with mock.patch("api.operations.load_snapshot", return_value={"report": report, "refresh_failed": False}), \
             mock.patch("api.operations.get_summary", return_value={"available": True}), \
             mock.patch("analytics.email_health.evaluate", return_value=email), \
             mock.patch("db.email_job_runs.recent_email_runs", return_value=[]):
            r = self.client.get("/v1/admin/health", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["health_score"], 90)
        self.assertEqual([s["name"] for s in body["subsystems"]], ["Scanner"])
        self.assertEqual(body["incidents"][0]["id"], "X1")
        self.assertEqual(body["email"]["jobs"][0]["label"], "Morning digest")

    def test_earnings_refresh_runs_once_in_background(self):
        from api import admin

        admin._earnings_job.update(state="idle")
        started = []
        with mock.patch("api.admin._universe", return_value=["AAPL", "MSFT"]), \
             mock.patch("api.admin.threading.Thread") as thread, \
             mock.patch("db.admin_audit.record_admin_event"):
            thread.return_value.start.side_effect = lambda: started.append(1)
            h = self.auth()
            first = self.client.post("/v1/admin/earnings-refresh", headers=h).json()
            second = self.client.post("/v1/admin/earnings-refresh", headers=h).json()
        self.assertEqual(first["state"], "running")
        self.assertEqual(first["symbols"], 2)
        self.assertEqual(second["state"], "running")
        self.assertEqual(len(started), 1)
        admin._earnings_job.update(state="idle")

    def test_earnings_refresh_job_records_result(self):
        from api import admin

        info = mock.Mock(earnings_date="2026-10-20")
        with mock.patch("db.earnings.populate_earnings_calendar", return_value={"AAPL": info}) as pop, \
             mock.patch("db.earnings.mark_earnings_refreshed_today") as mark:
            admin._run_earnings_refresh(["AAPL"])
        pop.assert_called_once_with(["AAPL"], use_yf_fallback=False, sleep_s=0)
        mark.assert_called_once_with("cron_earnings")
        self.assertEqual(admin.earnings_refresh_status()["state"], "done")
        self.assertEqual(admin.earnings_refresh_status()["dated"], 1)
        admin._earnings_job.update(state="idle")

    def test_providers_reports_each_check(self):
        with mock.patch("api.admin._probe_database", return_value={"name": "Database", "status": "ok", "detail": "neon", "ms": 4.0}), \
             mock.patch("api.admin._probe_alpaca", return_value={"name": "Alpaca", "status": "not_configured", "detail": None, "ms": None}), \
             mock.patch("api.admin._probe_earnings", side_effect=lambda w: {"name": w, "status": "ok", "detail": "1", "ms": 1.0}), \
             mock.patch("api.admin._probe_billing", return_value={"name": "Billing service", "status": "error", "detail": "down", "ms": None}), \
             mock.patch("api.admin._earnings_cache", return_value={"last_refresh": None, "dated_rows": 5}):
            from api import admin

            admin._probe_cache.clear()
            r = self.client.get("/v1/admin/providers", headers=self.auth())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([c["status"] for c in r.json()["checks"]], ["ok", "not_configured", "ok", "ok", "error"])
        self.assertEqual(r.json()["earnings_cache"]["dated_rows"], 5)


if __name__ == "__main__":
    unittest.main()
