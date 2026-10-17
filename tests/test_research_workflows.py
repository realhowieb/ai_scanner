"""Research workflow boundaries: ownership, limits, timezone handling and retry safety."""
import datetime as dt
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase


class WorkspaceRoutes(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.users = {}
        from api.workspace import WorkspacePreferences
        def prefs(user, changes=None):
            value = self.users.setdefault(user, WorkspacePreferences().model_dump())
            value.update(changes or {})
            return value
        self.prefs = mock.patch("api.workspace.preferences", side_effect=prefs).start()
        self.headers = self.auth(self.login().json()["access_token"])

    def test_preferences_are_authenticated_and_separate_for_each_account(self):
        path = "/v1/me/research-preferences"
        self.assertEqual(self.client.get(path).status_code, 401)
        saved = {"scanner_views": [{"name": "Morning", "min": 60}]}
        result = self.client.patch(path, headers=self.headers, json=saved)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["scanner_views"][0]["min"], 60)
        other = self.auth(self.login("boss@example.com").json()["access_token"])
        self.assertEqual(self.client.get(path, headers=other).json()["scanner_views"], [])
        self.assertEqual(self.client.get(path, headers=self.headers).json()["scanner_views"][0]["name"], "Morning")

    def test_preferences_reject_unbounded_and_invalid_settings(self):
        for changes in ({"scanner_views": [{"name": "x"}] * 21}, {"scanner_views": [{"name": "  "}]},
                        {"table": {"columns": ["password"]}}, {"quiet_hours": {"timezone": "not/a/timezone"}},
                        {"custom_views": [{"name": "x", "min_price": -1}]}, {"username": "victim"}):
            r = self.client.patch("/v1/me/research-preferences", headers=self.headers, json=changes)
            self.assertEqual(r.status_code, 422, r.text)

    def test_operations_are_admin_only_and_sanitize_routes(self):
        self.assertEqual(self.client.get("/v1/admin/requests", headers=self.headers).status_code, 403)
        admin = self.auth(self.login("boss@example.com").json()["access_token"])
        self.client.get("/healthz?secret=never-store-this", headers={"X-Request-ID": "web-support-code-123"})
        with mock.patch("api.scan_jobs._execute", return_value=[]):
            r = self.client.get("/v1/admin/requests?q=web-support-code-123", headers=admin)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["requests"]), 1)
        self.assertEqual(r.json()["requests"][0]["route"], "/healthz")
        self.assertNotIn("never-store", r.text)

    def test_retry_uses_session_owner_never_an_input_account(self):
        with mock.patch("api.alert_delivery.retry", return_value=None) as retry:
            r = self.client.post("/v1/alerts/deliveries/not-mine/retry", headers=self.headers,
                                 json={"username": "victim"})
        self.assertEqual(r.status_code, 409)
        retry.assert_called_once_with("pro@example.com", "not-mine")


class QuietHoursPolicyTests(unittest.TestCase):
    def test_overnight_local_time_and_daylight_saving(self):
        from billing_service.notification_preferences import quiet_at
        q = {"enabled": True, "start": "22:00", "end": "07:00", "timezone": "America/Los_Angeles"}
        for when, expected in [("2026-07-01T06:00:00+00:00", True), ("2026-07-01T14:00:00+00:00", False),
                               ("2026-01-01T07:00:00+00:00", True), ("2026-01-01T15:00:00+00:00", False)]:
            self.assertEqual(quiet_at(q, dt.datetime.fromisoformat(when)), expected)
        self.assertFalse(quiet_at({**q, "start": "07:00"}))
        self.assertFalse(quiet_at({**q, "enabled": False}))


@unittest.skipUnless(DEPS, "needs API test dependencies")
class QuietHoursDeliveryTests(unittest.TestCase):
    def test_quiet_hours_suppress_push_and_alert_email(self):
        from api import webpush
        with mock.patch("api.webpush.enabled", return_value=True), \
             mock.patch("api.devices.devices_for_user", return_value=[{"id": 1, "provider": "webpush", "token": "private"}]), \
             mock.patch("api.store.get_account", return_value={"tier": "pro"}), \
             mock.patch("api.workspace.is_quiet", return_value=True), \
             mock.patch("api.alert_delivery.begin", return_value="receipt"), \
             mock.patch("api.alert_delivery.finish") as finish, mock.patch("api.webpush.send") as send:
            self.assertEqual(webpush.notify_user("u", "Alert", "Message"), 0)
            send.assert_not_called()
            finish.assert_called_once_with("receipt", "quiet_hours")
            from db.email_prefs import wants_email
            self.assertFalse(wants_email("u", "alerts"))

    def test_realtime_worker_keeps_in_app_event_during_quiet_hours(self):
        from billing_service import realtime_alerts as rt
        conn = mock.MagicMock()
        alert = {"id": 1, "user_id": "member@example.com", "ticker": "AAPL", "tier": "pro", "email_verified": True}
        with mock.patch.object(rt, "_conn", return_value=conn), \
             mock.patch.object(rt, "_due_price_alerts", return_value=[alert]), \
             mock.patch.object(rt, "_latest_snapshots", return_value={"AAPL": {"last": 100}}), \
             mock.patch.object(rt, "_quiet_users", return_value={alert["user_id"]}), \
             mock.patch.object(rt, "evaluate_alert", return_value="Triggered"), \
             mock.patch.object(rt, "_record_fire") as record, mock.patch.object(rt, "run_fire_hooks"), \
             mock.patch.object(rt, "_send_email") as email:
            self.assertEqual(rt.check_once(), 1)
            record.assert_called_once()
            email.assert_not_called()

    def test_ambiguous_push_outcome_is_not_marked_retryable(self):
        from api import webpush
        with mock.patch("api.webpush.enabled", return_value=True), \
             mock.patch("api.devices.devices_for_user", return_value=[{"id": 1, "provider": "webpush", "token": "private"}]), \
             mock.patch("api.store.get_account", return_value={"tier": "pro"}), \
             mock.patch("api.workspace.is_quiet", return_value=False), \
             mock.patch("api.alert_delivery.begin", return_value="receipt"), \
             mock.patch("api.alert_delivery.finish") as finish, mock.patch("api.webpush.send", return_value=(False, None)):
            self.assertEqual(webpush.notify_user("u", "Alert", "Message"), 0)
            finish.assert_called_once_with("receipt", "unconfirmed")

    def test_retry_does_not_resend_accepted_or_unknown_receipt(self):
        from api.alert_delivery import retry
        with mock.patch("api.webpush.enabled", return_value=True), \
             mock.patch("api.webpush.plan_allows", return_value=True), \
             mock.patch("api.store.get_account", return_value={}), mock.patch("api.workspace.is_quiet", return_value=False), \
             mock.patch("api.alert_delivery._run", return_value=[]) as run, mock.patch("api.webpush.send") as send:
            self.assertIsNone(retry("u", "unknown"))
            send.assert_not_called()
            self.assertIn("user_id = %s", run.call_args.args[0])
            self.assertIn("status = 'failed'", run.call_args.args[0])
            self.assertIn("attempts < 3", run.call_args.args[0])


class SymbolSearchTests(unittest.TestCase):
    def test_company_names_exact_tickers_and_outage_keep_last_directory(self):
        from api import symbol_search as s
        assets = [{"symbol": "AAPL", "name": "Apple Inc.", "tradable": True, "status": "active"},
                  {"symbol": "MSFT", "name": "Microsoft Corporation", "tradable": True, "status": "active"}]
        with mock.patch.object(s, "_assets", []), mock.patch.object(s, "_expires", 0), \
             mock.patch("data.us_market_universe._fetch_us_equity_assets", return_value=assets) as fetch:
            self.assertEqual(s.search("apple")["results"][0]["ticker"], "AAPL")
            self.assertEqual(s.search("MSFT")["results"][0]["name"], "Microsoft Corporation")
            fetch.assert_called_once()
            s._expires = 0
            fetch.return_value = None
            self.assertEqual(s.search("apple")["results"][0]["ticker"], "AAPL")
