"""Lifecycle emails (services/lifecycle_emails.py): the welcome email after the first
email verification, and the "free Pro ends soon" reminder for invite trials.

No network, no Postgres: db.referrals runs on SQLite and the SMTP send is patched.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

UTC = dt.timezone.utc


class _Store(unittest.TestCase):
    def setUp(self):
        from api import referrals as svc
        from db import referrals

        self.r = referrals
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(lambda: (referrals.set_connection_factory(None), svc.clear_cache(), self.conn.close()))
        env = mock.patch.dict(os.environ, {"HSF_LIFECYCLE_EMAILS_ENABLED": "1", "HSF_TRIAL_REMINDER_DAYS": "3"})
        env.start()
        self.addCleanup(env.stop)
        self.send = mock.patch("ui.email_utils.send_digest_email", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def grant(self, user, until):
        self.conn.execute("INSERT INTO hsf_plan_grants (user_id, tier, until_at, reason, updated_at) "
                          "VALUES (?, 'pro', ?, 'referral:invited', ?)",
                          (user, until.isoformat(timespec="seconds"), dt.datetime.now(UTC).isoformat()))
        self.conn.commit()


class WelcomeTests(_Store):
    def test_sent_once_per_account(self):
        from services import lifecycle_emails as le

        self.assertTrue(le.send_welcome("New@Example.com"))
        self.assertFalse(le.send_welcome("new@example.com"))
        self.assertEqual(self.send.call_count, 1)
        to, subject, inner, text = self.send.call_args.args
        self.assertEqual(to, "new@example.com")
        self.assertIn("/today", inner)
        self.assertNotIn("free Pro", text)

    def test_mentions_a_running_invite_trial(self):
        from services import lifecycle_emails as le

        self.grant("new@example.com", dt.datetime.now(UTC) + dt.timedelta(days=30))
        le.send_welcome("new@example.com")
        self.assertIn("free Pro until", self.send.call_args.args[3])

    def test_failed_send_is_retried_later(self):
        from services import lifecycle_emails as le

        self.send.return_value = False
        self.assertFalse(le.send_welcome("new@example.com"))
        self.send.return_value = True
        self.assertTrue(le.send_welcome("new@example.com"))

    def test_kill_switch_and_non_email_accounts(self):
        from services import lifecycle_emails as le

        self.assertFalse(le.send_welcome("admin"))
        with mock.patch.dict(os.environ, {"HSF_LIFECYCLE_EMAILS_ENABLED": "0"}):
            self.assertFalse(le.send_welcome("new@example.com"))
        self.send.assert_not_called()

    def test_storage_errors_never_raise(self):
        from services import lifecycle_emails as le

        self.r.set_connection_factory(lambda: None)
        self.assertFalse(le.send_welcome("new@example.com"))


class TrialReminderTests(_Store):
    users = {"soon@example.com": {"tier": "basic", "is_active": True},
             "paid@example.com": {"tier": "pro", "is_active": True},
             "later@example.com": {"tier": "basic", "is_active": True},
             "gone@example.com": None}

    def lookup(self, user):
        return self.users.get(user)

    def test_only_starter_accounts_ending_soon_get_one_email(self):
        from services import lifecycle_emails as le

        now = dt.datetime.now(UTC)
        self.grant("soon@example.com", now + dt.timedelta(days=2, hours=5))
        self.grant("paid@example.com", now + dt.timedelta(days=1))
        self.grant("gone@example.com", now + dt.timedelta(days=1))
        self.grant("later@example.com", now + dt.timedelta(days=10))
        counts = le.run_trial_reminders(users_lookup=self.lookup)
        self.assertEqual((counts["sent"], counts["skipped"]), (1, 2))
        to, subject, inner, text = self.send.call_args.args
        self.assertEqual(to, "soon@example.com")
        self.assertEqual(subject, "Your free Pro ends in 3 days")
        self.assertIn("/pricing", inner)
        # The next scan slot sends nothing new.
        self.assertEqual(le.run_trial_reminders(users_lookup=self.lookup)["sent"], 0)
        self.assertEqual(self.send.call_count, 1)

    def test_an_extended_trial_gets_a_new_reminder(self):
        from services import lifecycle_emails as le

        now = dt.datetime.now(UTC)
        self.grant("soon@example.com", now + dt.timedelta(days=1))
        le.run_trial_reminders(users_lookup=self.lookup)
        self.conn.execute("UPDATE hsf_plan_grants SET until_at = ? WHERE user_id = 'soon@example.com'",
                          ((now + dt.timedelta(days=2)).isoformat(timespec="seconds"),))
        self.conn.commit()
        self.assertEqual(le.run_trial_reminders(users_lookup=self.lookup)["sent"], 1)

    def test_ended_trials_are_left_alone(self):
        from services import lifecycle_emails as le

        self.grant("soon@example.com", dt.datetime.now(UTC) - dt.timedelta(hours=1))
        self.assertEqual(le.run_trial_reminders(users_lookup=self.lookup)["sent"], 0)

    def test_failed_send_retries_next_run(self):
        from services import lifecycle_emails as le

        self.grant("soon@example.com", dt.datetime.now(UTC) + dt.timedelta(days=1))
        self.send.return_value = False
        self.assertEqual(le.run_trial_reminders(users_lookup=self.lookup)["failed"], 1)
        self.send.return_value = True
        self.assertEqual(le.run_trial_reminders(users_lookup=self.lookup)["sent"], 1)

    def test_wording_for_the_last_day(self):
        from services import lifecycle_emails as le

        now = dt.datetime(2026, 10, 20, 15, tzinfo=UTC)
        subject, text, _ = le.trial_ending_content(now + dt.timedelta(hours=20), now)
        self.assertEqual(subject, "Your free Pro ends in 1 day")
        self.assertIn("Wednesday, October 21", text)

    def test_storage_errors_never_raise(self):
        from services import lifecycle_emails as le

        self.r.set_connection_factory(lambda: None)
        self.assertIn("error", le.run_trial_reminders(users_lookup=self.lookup))


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class VerifyRouteWelcomeTest(ApiTestCase):
    def setUp(self):
        super().setUp()
        from api import referrals as svc
        from db import referrals

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(lambda: (referrals.set_connection_factory(None), svc.clear_cache(), self.conn.close()))

    def test_verify_email_sends_the_welcome(self):
        with mock.patch("api.account.verify_email", return_value="free@example.com"), \
                mock.patch("services.lifecycle_emails.send_welcome") as welcome:
            r = self.client.post("/v1/auth/verify-email", json={"token": "t" * 20})
        self.assertEqual(r.status_code, 200)
        welcome.assert_called_once_with("free@example.com")


if __name__ == "__main__":
    unittest.main()
