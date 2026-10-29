"""Founding 100 members get their founder plan in background jobs that read the stored
plan (users.tier): digest/wrap/recap emails, real-time and scheduled alerts, trial
emails and the weekly business report."""
import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from domain import founding as rules

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)


class FounderBackgroundTests(unittest.TestCase):
    def setUp(self):
        from api import plan_access
        from db import founding

        self.fdb, self.pa = founding, plan_access
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))
        founding.set_connection_factory(lambda: self.conn)
        self.addCleanup(lambda: founding.set_connection_factory(None))
        plan_access.clear_cache()
        self.addCleanup(plan_access.clear_cache)

    def buy(self, user, tier=rules.FOUNDER_PRO, ref="pi_1"):
        res = self.fdb.reserve(user, tier, now=NOW)
        out = self.fdb.fulfill(res["id"], payment_reference=ref, amount_cents=rules.PRICE_CENTS[tier], paid_at=NOW, now=NOW)
        self.assertEqual(out["outcome"], "activated")
        self.pa.clear_cache()

    def test_with_founder(self):
        self.assertEqual(self.pa.with_founder("a@example.com", "basic"), "basic")
        self.buy("a@example.com")
        self.buy("e@example.com", rules.FOUNDER_ELITE, "pi_2")
        self.assertEqual(self.pa.with_founder("a@example.com", "basic"), "pro")
        self.assertEqual(self.pa.with_founder("a@example.com", "premium"), "premium", "never lowers a plan")
        self.assertEqual(self.pa.with_founder("e@example.com", "pro"), "premium")
        self.assertEqual(self.pa.with_founder("e@example.com", "admin"), "admin")
        self.assertEqual(self.pa.with_founder("x@example.com", None), "basic")

    def test_refunded_founder_is_back_to_their_own_plan(self):
        self.buy("a@example.com")
        self.fdb.refund("pi_1", now=NOW)
        self.pa.clear_cache()
        self.assertEqual(self.pa.with_founder("a@example.com", "basic"), "basic")

    def test_lookup_error_keeps_the_stored_plan(self):
        self.buy("a@example.com")
        with mock.patch.object(self.pa, "founder_access", side_effect=RuntimeError("db down")):
            self.assertEqual(self.pa.with_founder("a@example.com", "basic"), "basic")

    def test_digest_wrap_and_recap_email_plan(self):
        from scheduler.morning_digest import _email_tier_key

        basic = lambda email, users: type("T", (), {"key": "basic"})()  # noqa: E731
        self.buy("a@example.com")
        with mock.patch("db.users.is_admin_from_db", return_value=False):
            self.assertEqual(_email_tier_key("a@example.com", {"tier": "basic"}, {}, basic), "pro")
            self.assertEqual(_email_tier_key("b@example.com", {"tier": "basic"}, {}, basic), "basic")

    def test_realtime_alerts_use_the_founder_plan(self):
        from billing_service import realtime_alerts as rt

        self.buy("a@example.com")
        alerts = [{"id": i, "user_id": u, "tier": "basic"} for u in ("a@example.com", "b@example.com") for i in range(6)]
        rt._apply_founder_plans(alerts)
        kept = rt._apply_plan_limits(alerts)
        self.assertEqual(sum(a["user_id"] == "a@example.com" for a in kept), rt.ALERT_LIMITS["pro"])
        self.assertEqual(sum(a["user_id"] == "b@example.com" for a in kept), rt.ALERT_LIMITS["basic"])
        self.assertIn(next(a for a in kept if a["user_id"] == "a@example.com")["tier"], rt.EMAIL_TIERS)

    def test_scheduled_alert_runner_uses_the_founder_plan(self):
        from scheduler import alert_runner

        self.buy("a@example.com")
        with mock.patch("auth.tier_sync.resolve_user_tier", return_value={"tier_key": "basic"}):
            self.assertEqual(alert_runner._tier_key_for_user("a@example.com"), "pro")
            self.assertEqual(alert_runner._tier_key_for_user("b@example.com"), "basic")

    def test_trial_emails_skip_founders(self):
        from services.lifecycle_emails import _still_on_trial

        self.buy("a@example.com")
        lookup = lambda email: {"tier": "basic", "is_active": True}  # noqa: E731
        self.assertFalse(_still_on_trial("a@example.com", lookup))
        self.assertTrue(_still_on_trial("b@example.com", lookup))

    def test_business_report_lists_founders(self):
        from db import founding_refunds
        from services.business_report import _founder_lines

        self.assertEqual(_founder_lines(founding_refunds.weekly_summary(NOW - dt.timedelta(days=7))), [])
        self.buy("a@example.com")
        self.buy("e@example.com", rules.FOUNDER_ELITE, "pi_2")
        (line,) = _founder_lines(founding_refunds.weekly_summary(NOW - dt.timedelta(days=7)))
        self.assertIn("1 Founder Pro, 1 Founder Elite", line)
        self.assertIn("2 payments this week $1,498.00", line)


if __name__ == "__main__":
    unittest.main()
