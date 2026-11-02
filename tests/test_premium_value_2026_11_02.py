"""Premium value pack (2026-11-02): the pricing card lists what Premium already does,
scheduled custom scans (db.scheduled_scans, services.scheduled_scans, /v1/scheduled-scans)
and the Premium CSV entitlement."""
import datetime as dt
import sqlite3
import unittest
from unittest import mock

from tests.test_api_paid_features import PaidApiTestCase

UTC = dt.timezone.utc


class PlanTests(unittest.TestCase):
    def test_new_flags_are_premium(self):
        from domain.plans import FEATURE_MIN_TIER

        self.assertEqual(FEATURE_MIN_TIER["can_scheduled_scans"], "premium")
        self.assertEqual(FEATURE_MIN_TIER["can_export_research"], "premium")

    def test_premium_card_lists_the_built_features(self):
        from ui.pricing import plan_highlights

        prem = " | ".join(plan_highlights()["premium"])
        for want in ("Ask HSF", "Scheduled custom scans", "AI scan summaries", "Describe a scan in plain English",
                     "AI watchlist rundowns", "\"why\" line", "AI journal coach", "Early Breakout",
                     "CSV export of watchlists", "Every earnings reporter scored"):
            self.assertIn(want, prem)
        self.assertNotIn("Scheduled custom scans", " | ".join(plan_highlights()["pro"]))

    def test_upgrade_messages(self):
        from domain.upgrades import upgrade_message

        self.assertIn("emails you the results", upgrade_message("can_scheduled_scans"))
        self.assertIn("Premium", upgrade_message("can_export_research"))


class StoreTests(unittest.TestCase):
    def setUp(self):
        from db import scheduled_scans

        self.conn = sqlite3.connect(":memory:")
        scheduled_scans.set_connection_factory(lambda: self.conn)
        self.addCleanup(scheduled_scans.set_connection_factory, None)
        self.s = scheduled_scans

    def test_create_list_limit_delete(self):
        a = self.s.create("p@x.com", "Gappers", {"universe": "nasdaq"}, "open")
        self.assertEqual((a["name"], a["run_at"], a["request"]), ("Gappers", "open", {"universe": "nasdaq"}))
        self.s.create("p@x.com", "Two", {"universe": "sp500"}, "midday")
        self.s.create("p@x.com", "Three", {"universe": "sp500"}, "close")
        with self.assertRaises(self.s.ScheduleLimitReached):
            self.s.create("p@x.com", "Four", {"universe": "sp500"}, "close")
        with self.assertRaises(ValueError):
            self.s.create("q@x.com", "Bad", {}, "night")
        self.assertEqual(len(self.s.list_for_user("p@x.com")), 3)
        self.assertFalse(self.s.delete("q@x.com", a["id"]))          # not yours
        self.assertTrue(self.s.delete("p@x.com", a["id"]))
        self.assertEqual(len(self.s.list_for_user("p@x.com")), 2)

    def test_claim_once_per_run_and_release(self):
        a = self.s.create("p@x.com", "Gappers", {"universe": "nasdaq"}, "open")
        key = "2026-11-02:open"
        self.assertEqual([r["id"] for r in self.s.list_due("open", key)], [a["id"]])
        self.assertEqual(self.s.list_due("midday", "2026-11-02:midday"), [])
        prev = self.s.claim(a["id"], key)
        self.assertEqual(prev, "")
        self.assertIsNone(self.s.claim(a["id"], key))               # a second pass gets nothing
        self.assertEqual(self.s.list_due("open", key), [])
        self.s.release(a["id"], prev)
        self.assertEqual(self.s.claim(a["id"], key), "")             # released: claimable again
        self.assertEqual(self.s.claim(a["id"], "2026-11-03:open"), key)
        self.s.record(a["id"], "sent", "4 setups", scan_id="j1")
        row = self.s.get("p@x.com", a["id"])
        self.assertEqual((row["last_status"], row["last_detail"], row["last_scan_id"]), ("sent", "4 setups", "j1"))

    def test_account_deletion_covers_the_table(self):
        from db import account_deletion

        self.assertIn(("hsf_scheduled_scans", "username"), account_deletion.ACCOUNT_TABLES)


class SlotTests(unittest.TestCase):
    def slot(self, y, mo, d, h, mi):
        from services.scheduled_scans import slot_for

        return slot_for(dt.datetime(y, mo, d, h, mi, tzinfo=UTC))

    def test_windows_in_eastern_time_on_trading_days(self):
        # 2026-11-02 is a Monday, EST (UTC-5)
        self.assertIsNone(self.slot(2026, 11, 2, 13, 0))                       # 8:00 ET: too early
        self.assertEqual(self.slot(2026, 11, 2, 13, 5), ("open", "2026-11-02:open"))
        self.assertIsNone(self.slot(2026, 11, 2, 14, 25))                      # 9:25 ET: closed
        self.assertEqual(self.slot(2026, 11, 2, 17, 10)[0], "midday")
        self.assertEqual(self.slot(2026, 11, 2, 21, 30)[0], "close")
        self.assertIsNone(self.slot(2026, 11, 1, 13, 30))                      # Sunday
        self.assertIsNone(self.slot(2026, 11, 26, 13, 30))                     # Thanksgiving


class EmailTests(unittest.TestCase):
    RESULT = {"label": "NASDAQ", "symbols_scanned": 3200, "total": 140,
              "setups": [{"ticker": f"T{i}", "score": 90 - i, "primary_setup": "breakout", "chg_pct": 1.5}
                         for i in range(12)]}

    def test_compose(self):
        from services.scheduled_scans import compose

        subject, text, html, glance = compose("Gappers", "open", self.RESULT, base_url="https://app.example")
        self.assertEqual(subject, 'Your scan "Gappers": 12 setups before the open')
        self.assertIn("1. T0  HSF 90  Breakout", text)
        self.assertNotIn("T10", html)                                          # top 10 only
        self.assertIn("Showing the top 10 of 12", html)
        self.assertIn("https://app.example/scanner/custom", html)
        self.assertIn("aren't predictions or advice", html)
        self.assertIn("3,200 stocks", glance)
        for bad in ("guarantee", "profit", "win rate", "buy now"):
            self.assertNotIn(bad, (subject + text + html).lower())

    def test_compose_empty(self):
        from services.scheduled_scans import compose

        subject, text, html, _ = compose("Quiet", "close", {"setups": [], "symbols_scanned": 500}, base_url="https://a")
        self.assertIn("no setups matched after the close", subject)
        self.assertIn("Nothing passed your filters", html)


class RunPassTests(unittest.TestCase):
    NOW = dt.datetime(2026, 11, 2, 13, 30, tzinfo=UTC)   # 8:30 ET Monday: the open window

    def setUp(self):
        from db import scheduled_scans

        self.conn = sqlite3.connect(":memory:")
        scheduled_scans.set_connection_factory(lambda: self.conn)
        self.addCleanup(scheduled_scans.set_connection_factory, None)
        self.store = scheduled_scans
        self.tiers = {"prem@x.com": "premium", "pro@x.com": "pro"}
        p = mock.patch
        p("api.store.get_account", side_effect=lambda u: {"username": u, "tier": self.tiers.get(u, "basic"),
                                                           "is_active": True, "is_admin": False}).start()
        p("api.plan_access.apply_to_account", side_effect=lambda a: a).start()
        self.create = p("api.scan_jobs.create_job", side_effect=lambda u, uni, params: {"id": f"job-{u}"}).start()
        self.submitted = []
        p("api.scan_jobs.submit", side_effect=lambda job_id, work: self.submitted.append((job_id, work))).start()
        self.addCleanup(mock.patch.stopall)

    def run_pass(self):
        from services.scheduled_scans import run_pass

        return run_pass(self.NOW)

    def test_premium_schedule_is_queued_once_and_emails_results(self):
        s = self.store.create("prem@x.com", "Gappers", {"universe": "sp500", "filters": {"top_n": 25}}, "open")
        out = self.run_pass()
        self.assertEqual(out["queued"], 1)
        self.assertEqual(self.run_pass()["queued"], 0)                         # claimed for today
        job_id, work = self.submitted[0]
        self.assertEqual(job_id, "job-prem@x.com")
        self.assertEqual(self.store.get("prem@x.com", s["id"])["last_status"], "running")
        result = {"label": "SP500", "symbols_scanned": 500, "total": 2, "setups": [{"ticker": "AAA", "score": 80}]}
        with mock.patch("api.custom_scans.run_scan", return_value=result), \
                mock.patch("services.scheduled_scans._email", return_value="sent") as email:
            self.assertEqual(work(lambda p: None), result)
        email.assert_called_once_with("prem@x.com", "Gappers", "open", result)
        row = self.store.get("prem@x.com", s["id"])
        self.assertEqual((row["last_status"], row["last_detail"]), ("sent", "1 setups"))

    def test_lapsed_plan_pauses(self):
        s = self.store.create("pro@x.com", "Old", {"universe": "sp500"}, "open")
        self.assertEqual(self.run_pass()["skipped"], {"not_premium": 1})
        self.create.assert_not_called()
        self.assertEqual(self.store.get("pro@x.com", s["id"])["last_status"], "paused")

    def test_busy_is_retried_next_pass(self):
        from api import scan_jobs

        self.store.create("prem@x.com", "Gappers", {"universe": "sp500"}, "open")
        self.create.side_effect = scan_jobs.ScanInProgress("other")
        self.assertEqual(self.run_pass()["skipped"], {"busy": 1})
        self.create.side_effect = lambda u, uni, params: {"id": "job-2"}
        self.assertEqual(self.run_pass()["queued"], 1)

    def test_plan_refusal_is_recorded(self):
        s = self.store.create("prem@x.com", "Bad", {"universe": "watchlist"}, "open")   # no watchlist_id
        self.assertEqual(self.run_pass()["skipped"], {"plan": 1})
        self.assertIn("watchlist_id", self.store.get("prem@x.com", s["id"])["last_detail"])

    def test_outside_a_window_does_nothing(self):
        from services.scheduled_scans import run_pass

        self.store.create("prem@x.com", "Gappers", {"universe": "sp500"}, "open")
        self.assertEqual(run_pass(dt.datetime(2026, 11, 2, 15, 0, tzinfo=UTC)), {"slot": None})
        self.create.assert_not_called()

    def test_per_pass_cap(self):
        for i in range(3):
            self.tiers[f"u{i}@x.com"] = "premium"
            self.store.create(f"u{i}@x.com", "S", {"universe": "sp500"}, "open")
        with mock.patch.dict("os.environ", {"HSF_SCHEDULED_SCANS_PER_PASS": "2"}):
            self.assertEqual(self.run_pass()["queued"], 2)
            self.assertEqual(self.run_pass()["queued"], 1)


class ScheduledScanApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from db import scheduled_scans

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        scheduled_scans.set_connection_factory(lambda: self.conn)
        self.addCleanup(scheduled_scans.set_connection_factory, None)

    def post(self, email, body):
        return self.client.post("/v1/scheduled-scans", headers=self.h(email), json=body)

    def test_premium_only(self):
        body = {"name": "Gappers", "run_at": "open", "scan": {"universe": "sp500"}}
        for email in ("free@example.com", "pro@example.com"):
            self.assertEqual(self.post(email, body).status_code, 403)
            self.assertEqual(self.get(email, "/v1/scheduled-scans").status_code, 403)

    def test_create_list_delete(self):
        r = self.post("prem@example.com", {"name": "Gappers", "run_at": "open",
                                           "scan": {"universe": "nasdaq", "filters": {"session": "premarket", "top_n": 50}}})
        self.assertEqual(r.status_code, 201, r.text)
        sid = r.json()["id"]
        self.assertEqual(r.json()["request"]["filters"]["session"], "premarket")
        listed = self.get("prem@example.com", "/v1/scheduled-scans").json()
        self.assertEqual([x["id"] for x in listed], [sid])
        self.assertEqual(self.client.delete(f"/v1/scheduled-scans/{sid}", headers=self.h("pro@example.com")).status_code, 404)
        self.assertEqual(self.client.delete(f"/v1/scheduled-scans/{sid}", headers=self.h("prem@example.com")).status_code, 204)
        self.assertEqual(self.get("prem@example.com", "/v1/scheduled-scans").json(), [])

    def test_plan_rules_checked_on_save_and_limit(self):
        bad = self.post("prem@example.com", {"name": "Big", "run_at": "open", "scan": {"universe": "sp500", "filters": {"top_n": 500}}})
        self.assertEqual(bad.status_code, 403)
        for i in range(3):
            self.assertEqual(self.post("prem@example.com", {"name": f"S{i}", "run_at": "close", "scan": {"universe": "sp500"}}).status_code, 201)
        self.assertEqual(self.post("prem@example.com", {"name": "S4", "run_at": "close", "scan": {"universe": "sp500"}}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
