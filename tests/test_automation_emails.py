"""Automation and email batch (2026-10-23): post-promote live check, ops jobs and
notices, billing emails, trial-ended, first-week tips and come-back, weekly recap,
business email, last-active tracking, new email preferences, Dependabot auto-merge.

No network, no Postgres: SQLite stands in for the database and every send is patched.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

UTC = dt.timezone.utc


class _Store(unittest.TestCase):
    """db.referrals (and its lifecycle email log) on SQLite."""

    def setUp(self):
        from db import referrals

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        referrals._ensure_lifecycle_schema.__wrapped__(self.conn, True)
        self.addCleanup(lambda: (referrals.set_connection_factory(None), self.conn.close()))
        self.send = mock.patch("ui.email_utils.send_digest_email", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def grant(self, user, until):
        self.conn.execute("INSERT INTO hsf_plan_grants (user_id, tier, until_at, reason, updated_at) "
                          "VALUES (?, 'pro', ?, 'referral:invited', ?)",
                          (user, until.isoformat(timespec="seconds"), dt.datetime.now(UTC).isoformat()))
        self.conn.commit()


# ---- 1. post-promote live check -------------------------------------------------------------------
class WaitForDeployTests(unittest.TestCase):
    def test_both_on_the_commit(self):
        from scripts.wait_for_deploy import ready

        self.assertEqual(ready("abcdef1234567890", "abcdef123456", "abcdef123456", lambda a, b: False), (True, True))

    def test_web_counts_when_web_files_did_not_change(self):
        from scripts.wait_for_deploy import ready

        self.assertEqual(ready("abcdef1234567890", "abcdef123456", "000000000000", lambda a, b: True), (True, True))
        self.assertEqual(ready("abcdef1234567890", "abcdef123456", "000000000000", lambda a, b: False), (True, False))

    def test_old_api_or_no_build_waits(self):
        from scripts.wait_for_deploy import ready

        self.assertEqual(ready("abcdef1234567890", "999999999999", None, lambda a, b: True), (False, False))


class ReadyzBuildTests(unittest.TestCase):
    def test_readyz_reports_the_commit(self):
        from fastapi.testclient import TestClient

        from api.main import create_app

        with mock.patch.dict(os.environ, {"API_JWT_SECRET": "x" * 40, "RENDER_GIT_COMMIT": "0123456789abcdef"}), \
                mock.patch("api.store.ping"), mock.patch("api.today.market_runs", return_value=[]):
            r = TestClient(create_app()).get("/readyz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["build"], "0123456789ab")


# ---- live check notices + nightly note -------------------------------------------------------------
def _run(**kw):
    base = {"id": 1, "status": "completed", "conclusion": "success", "event": "push", "head_sha": "abc1234def",
            "head_commit": {"message": "Merge pull request #300\n\nbody"}, "html_url": "https://x/run/1",
            "updated_at": "2026-10-23T12:00:00Z"}
    base.update(kw)
    return base


class LiveCheckNoticeTests(unittest.TestCase):
    def test_messages(self):
        from api.ops_jobs import live_check_message

        self.assertIn("passed (after promote abc1234)", live_check_message(_run())[0])
        self.assertIn("failed", live_check_message(_run(conclusion="failure"))[0])
        self.assertIsNone(live_check_message(_run(event="schedule")))  # passing daily run: quiet
        self.assertIn("daily run", live_check_message(_run(event="schedule", conclusion="failure"))[0])
        self.assertIsNone(live_check_message(_run(status="in_progress")))
        self.assertIsNone(live_check_message(_run(conclusion="cancelled")))

    def test_each_run_once_and_only_recent(self):
        from api.ops_jobs import live_check_notices

        now = dt.datetime(2026, 10, 23, 12, 30, tzinfo=UTC)
        claimed = set()
        claim = lambda job, key: (job, key) not in claimed and not claimed.add((job, key))  # noqa: E731
        notify = mock.Mock()
        runs = [_run(id=1), _run(id=2, conclusion="failure"), _run(id=3, updated_at="2026-10-22T01:00:00Z")]
        self.assertEqual(len(live_check_notices(now, runs, notify, claim)), 2)
        self.assertEqual(live_check_notices(now, runs, notify, claim), [])
        self.assertTrue(notify.call_args_list[1].kwargs["email"])  # failures also email
        self.assertFalse(notify.call_args_list[0].kwargs["email"])


class NightlyNoteTests(unittest.TestCase):
    def test_due_window(self):
        from api.ops_jobs import nightly_due

        self.assertEqual(nightly_due(dt.datetime(2026, 10, 27, 0, 30, tzinfo=UTC)), "2026-10-26")  # Tue -> Mon
        self.assertIsNone(nightly_due(dt.datetime(2026, 10, 26, 0, 30, tzinfo=UTC)))  # Mon (Sunday evening)
        self.assertIsNone(nightly_due(dt.datetime(2026, 10, 27, 4, 0, tzinfo=UTC)))

    def test_note_lists_incidents_but_not_the_known_parity_one(self):
        from api.ops_jobs import nightly_note

        now = dt.datetime(2026, 10, 27, 0, 30, tzinfo=UTC)
        health = {"generated_at": "2026-10-26T23:46:00+00:00", "system_status": "DEGRADED", "health_score": 90,
                  "human_action": "WATCH",
                  "incidents": [{"incident_id": "cohort_parity:RETIREMENT_POLICY_EFFECT", "severity": "CRITICAL",
                                 "subsystem": "cohort_parity", "summary": "old"},
                                {"incident_id": "scanner:SCAN_MISSED", "severity": "WARNING",
                                 "subsystem": "scanner", "summary": "one slot missed"}]}
        subject, text = nightly_note(health, flags=[("2026-10-26T14:30+00:00", "missed", "", "")],
                                     memory={"container_peak_mb": 380.0, "memory_limit_mb": 512.0},
                                     live="passed (Mon 11:15 AM ET)", now=now)
        self.assertIn("DEGRADED 90/100", subject)
        self.assertIn("one slot missed", text)
        self.assertNotIn("old", text)
        self.assertIn("flagged by the backup check today: 1", text)
        self.assertIn("380 of 512 MB", text)

    def test_missing_report(self):
        from api.ops_jobs import nightly_note

        subject, _ = nightly_note(None, flags=[], memory={}, live="no recent run",
                                  now=dt.datetime(2026, 10, 27, 1, 30, tzinfo=UTC))
        self.assertIn("missing", subject)


class OpsScheduleTests(_Store):
    def test_due_keys(self):
        from api import ops_jobs as oj

        sat = dt.datetime(2026, 10, 24, 15, 0, tzinfo=UTC)
        self.assertEqual(oj.weekly_recap_due(sat), "2026-W43")
        self.assertIsNone(oj.weekly_recap_due(sat.replace(hour=13)))
        self.assertEqual(oj.business_due(dt.datetime(2026, 10, 26, 12, 5, tzinfo=UTC)), "2026-W44")
        self.assertIsNone(oj.business_due(sat))
        self.assertEqual(oj.daily_mail_due(sat), "2026-10-24")

    def test_run_once_claims_and_releases_on_crash(self):
        from api.ops_jobs import _run_once

        fn = mock.Mock(return_value={})
        _run_once("job", "k", fn)
        _run_once("job", "k", fn)
        self.assertEqual(fn.call_count, 1)
        boom = mock.Mock(side_effect=RuntimeError("x"))
        _run_once("job2", "k", boom)
        _run_once("job2", "k", boom)
        self.assertEqual(boom.call_count, 2)  # released after the crash, so it retried

    def test_ops_notify_chat_then_email_fallback(self):
        from services import ops_notify

        with mock.patch.object(ops_notify, "admin_accounts", return_value=["boss@x.com"]), \
                mock.patch("api.chat_delivery.send_all", return_value=0) as chat:
            out = ops_notify.notify_ops("Hi", "body")
        chat.assert_called_once()
        self.assertEqual(out, {"chats": 0, "emails": 1})
        with mock.patch.object(ops_notify, "admin_accounts", return_value=["boss@x.com"]), \
                mock.patch("api.chat_delivery.send_all", return_value=1):
            self.assertEqual(ops_notify.notify_ops("Hi", "body"), {"chats": 1, "emails": 0})
        with mock.patch.dict(os.environ, {"HSF_OPS_NOTIFY_ENABLED": "0"}):
            self.assertTrue(ops_notify.notify_ops("Hi", "body")["disabled"])


# ---- 3. billing emails --------------------------------------------------------------------------------
class BillingEmailTests(_Store):
    def test_changes(self):
        from services.billing_emails import changes

        due = {"status": "past_due", "plan": "pro", "past_due_since": "2026-10-20T00:00:00+00:00",
               "grace_ends_at": "2026-10-23T00:00:00+00:00"}
        self.assertEqual(changes({"status": "active", "plan": "pro"}, due)[0][:2],
                         ("payment_failed", "2026-10-20T00:00:00+00:00"))
        self.assertEqual(changes(due, due), [])  # same spell: nothing new
        ended = {"status": "canceled", "plan": "basic", "cancellation_at": "2026-10-22T00:00:00+00:00"}
        self.assertEqual(changes({"status": "active", "plan": "premium"}, ended),
                         [("plan_ended", "2026-10-22T00:00:00+00:00", "premium")])
        self.assertEqual(changes({"status": "none", "plan": "basic"}, {"status": "none", "plan": "basic"}), [])

    def test_sends_once(self):
        from services.billing_emails import notify_change

        before = {"status": "active", "plan": "pro"}
        after = {"status": "past_due", "plan": "pro", "past_due_since": "2026-10-20T00:00:00+00:00",
                 "grace_ends_at": "2026-10-23T00:00:00+00:00"}
        notify_change("A@x.com", before, after, background=False)
        notify_change("a@x.com", before, after, background=False)
        self.assertEqual(self.send.call_count, 1)
        to, subject, inner, text = self.send.call_args.args
        self.assertEqual((to, subject), ("a@x.com", "Your payment didn't go through"))
        self.assertIn("/account", inner)
        self.assertIn("until", text)

    def test_engine_calls_the_hook(self):
        from billing_service import engine

        with mock.patch("services.billing_emails.notify_change") as hook:
            engine._billing_emails("a@x.com", {"plan": "pro"}, {"plan": "basic"}, "pro")
        hook.assert_called_once_with("a@x.com", {"plan": "pro"}, {"plan": "basic"}, "pro")

    def test_banner_field(self):
        from api import account

        class Cur:
            def __init__(self, row): self.row = row
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a): pass
            def fetchone(self): return self.row

        class Conn:
            def __init__(self, row): self.row = row
            def cursor(self): return Cur(self.row)
            def close(self): pass

        snap = {"status": "past_due", "past_due_since": "x", "grace_ends_at": "2026-10-25T00:00:00+00:00"}
        with mock.patch("api.store._conn", return_value=Conn((snap,))):
            self.assertEqual(account.billing_attention("a@x.com", "pro"),
                             {"state": "past_due", "grace_ends_at": "2026-10-25T00:00:00+00:00"})
            self.assertIsNone(account.billing_attention("a@x.com", "basic"))
        with mock.patch("api.store._conn", return_value=Conn(({"status": "active"},))):
            self.assertIsNone(account.billing_attention("a@x.com", "pro"))


class TrialEndedTests(_Store):
    def test_ended_trial_emailed_once_unless_upgraded(self):
        from services import lifecycle_emails as le

        now = dt.datetime.now(UTC)
        self.grant("gone@x.com", now - dt.timedelta(days=1))
        self.grant("paid@x.com", now - dt.timedelta(days=1))
        self.grant("old@x.com", now - dt.timedelta(days=10))
        lookup = {"gone@x.com": {"tier": "basic"}, "paid@x.com": {"tier": "pro"}, "old@x.com": {"tier": "basic"}}.get
        self.assertEqual(le.run_trial_ended(users_lookup=lookup)["sent"], 1)
        self.assertEqual(le.run_trial_ended(users_lookup=lookup)["sent"], 0)
        self.assertEqual(self.send.call_args.args[0], "gone@x.com")
        self.assertIn("has ended", self.send.call_args.args[1])


# ---- 4. tips and come-back ---------------------------------------------------------------------------
class OnboardingTests(_Store):
    NOW = dt.datetime(2026, 10, 23, 16, 0, tzinfo=UTC)

    def acct(self, user, days_old, **kw):
        return {"user": user, "created_at": (self.NOW - dt.timedelta(days=days_old)).isoformat(),
                "verified": True, "is_admin": False, "is_active": True, **kw}

    def test_pick(self):
        from services.onboarding_emails import pick

        accounts = [self.acct("new@x.com", 3), self.acct("week@x.com", 6), self.acct("set@x.com", 6),
                    self.acct("unv@x.com", 3, verified=False), self.acct("away@x.com", 60)]
        due = pick(accounts, engaged_watchlist={"set@x.com"}, engaged_alerts={"set@x.com"},
                   last_active={"away@x.com": (self.NOW - dt.timedelta(days=15)).isoformat(),
                                "new@x.com": self.NOW.isoformat()}, now=self.NOW)
        self.assertIn(("new@x.com", "tip_watchlist", ""), due)
        self.assertIn(("week@x.com", "tip_watchlist", ""), due)
        self.assertIn(("week@x.com", "tip_alert", ""), due)
        self.assertNotIn("set@x.com", [d[0] for d in due])
        self.assertNotIn("unv@x.com", [d[0] for d in due])
        self.assertIn(("away@x.com", "comeback", "2026-10-08"), due)

    def test_run_respects_tips_preference_and_sends_once(self):
        from services import onboarding_emails as ob

        inputs = {"accounts": [self.acct("new@x.com", 3), self.acct("quiet@x.com", 3)], "watch": set(),
                  "alerts": set(), "last_active": {}}
        with mock.patch("db.email_prefs.wants_email", side_effect=lambda u, k: u != "quiet@x.com"), \
                mock.patch("db.email_prefs.unsubscribe_url", return_value="https://u"):
            first = ob.run(now=self.NOW, inputs=inputs)
            second = ob.run(now=self.NOW, inputs=inputs)
        self.assertEqual((first["sent"], first["skipped"]), (1, 1))
        self.assertEqual(second["sent"], 0)
        self.assertEqual(self.send.call_args.kwargs["unsubscribe_url"], "https://u")

    def test_comeback_content(self):
        from services.onboarding_emails import comeback_content

        _, text, inner = comeback_content([{"ticker": "NVDA", "score": 88.0}])
        self.assertIn("NVDA: HSF Score 88", text)
        self.assertIn("/today", inner)


class UserActivityTests(unittest.TestCase):
    def test_touch_is_throttled_and_readable(self):
        from db import user_activity as ua

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        ua.set_connection_factory(lambda: conn)
        ua._ensure_schema.__wrapped__(conn, True)
        self.addCleanup(lambda: (ua.set_connection_factory(None), conn.close()))
        t = dt.datetime(2026, 10, 1, tzinfo=UTC)
        self.assertTrue(ua.touch("A@x.com", now=t))
        self.assertFalse(ua.touch("a@x.com", now=t + dt.timedelta(minutes=5)))  # within the hour
        self.assertEqual(ua.last_active_map(), {"a@x.com": "2026-10-01T00:00:00+00:00"})


# ---- 2. weekly recap ------------------------------------------------------------------------------------
class WeeklyRecapTests(unittest.TestCase):
    def test_week_bounds(self):
        from services.weekly_recap import iso_week, week_bounds

        mon, fri = week_bounds(dt.datetime(2026, 10, 24, 15, 0, tzinfo=UTC))
        self.assertEqual((mon.isoformat(), fri.isoformat()), ("2026-10-19", "2026-10-23"))
        self.assertEqual(iso_week(mon), "2026-W43")

    def test_summarize_picks_uses_each_days_first_run_top_five(self):
        from services.weekly_recap import summarize_picks

        first = dt.datetime(2026, 10, 19, 13, 40, tzinfo=UTC)
        later = dt.datetime(2026, 10, 19, 19, 40, tzinfo=UTC)
        rows = [(f"T{i}", first, 90 - i, 0.01 * i, 0.002) for i in range(7)]
        rows.append(("LATE", later, 99, 0.5, 0.002))  # later run the same day: ignored
        rows.append(("NORES", dt.datetime(2026, 10, 20, 13, 40, tzinfo=UTC), 95, None, None))
        s = summarize_picks(rows)
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["days"], 2)
        self.assertAlmostEqual(s["avg"], 0.02)
        self.assertEqual(s["best"][0]["ticker"], "T4")
        self.assertEqual(s["beat"], 4)  # T0 at 0% did not beat SPY's +0.2%

    def test_change_since(self):
        from services.weekly_recap import change_since

        series = [("2026-10-16", 100.0), ("2026-10-19", 101.0), ("2026-10-23", 110.0)]
        self.assertAlmostEqual(change_since(series, dt.date(2026, 10, 19), before=True), 0.10)
        self.assertAlmostEqual(change_since(series, dt.date(2026, 10, 19), before=False), 110 / 101 - 1)
        self.assertIsNone(change_since([], dt.date(2026, 10, 19), before=True))

    def test_compose_pro_and_starter(self):
        from services.weekly_recap import compose

        picks = {"n": 5, "days": 1, "avg": 0.012, "spy": 0.003, "beat": 3, "with_spy": 5,
                 "best": [{"ticker": "AMD", "ret": 0.05}]}
        args = dict(picks=picks, movers=[("AAPL", 0.031)], alerts=[("TSLA", -0.02)],
                    earnings=[("MSFT", "Tue")], monday=dt.date(2026, 10, 19), base_url="https://app")
        _, text, inner = compose(pro=True, **args)
        for part in ("+1.2%", "3 of 5 beat SPY", "AMD +5.0%", "AAPL: +3.1%", "TSLA (-2.0% since)", "MSFT Tue"):
            self.assertIn(part, text)
        _, text, inner = compose(pro=False, **args)
        self.assertNotIn("TSLA", text)
        self.assertIn("/pricing", inner)

    def test_account_parts(self):
        from services.weekly_recap import _account_parts

        closes = {"AAPL": [("2026-10-16", 100.0), ("2026-10-23", 103.0)],
                  "TSLA": [("2026-10-20", 200.0), ("2026-10-23", 190.0)]}
        movers, alerts, earnings = _account_parts(
            "a@x.com", pro=True, closes=closes, monday=dt.date(2026, 10, 19), friday=dt.date(2026, 10, 23),
            watch=["AAPL"], alert_rows=[("tsla", dt.datetime(2026, 10, 20, 15, tzinfo=UTC))],
            earnings_rows=[{"symbol": "AAPL", "earnings_date": dt.date(2026, 10, 27)},
                           {"symbol": "ZZZ", "earnings_date": dt.date(2026, 10, 27)}])
        self.assertEqual(movers[0][0], "AAPL")
        self.assertEqual(alerts[0][0], "TSLA")
        self.assertAlmostEqual(alerts[0][1], -0.05)
        self.assertEqual(earnings, [("AAPL", "Tue")])


# ---- 5. business email --------------------------------------------------------------------------------
class BusinessReportTests(unittest.TestCase):
    def test_revenue(self):
        from services.business_report import revenue

        r = revenue([{"status": "active", "plan": "pro", "amount": 1900, "interval": "month", "currency": "usd"},
                     {"status": "active", "plan": "premium", "amount": 24000, "interval": "year", "currency": "usd",
                      "test_mode": True},
                     {"status": "canceled", "plan": "basic", "amount": 1900}])
        self.assertEqual(r["paying"], 2)
        self.assertAlmostEqual(r["mrr"]["USD"], 39.0)
        self.assertTrue(r["test_mode"])

    def test_compose(self):
        from services.business_report import compose

        subject, text, _ = compose({"signups": 4, "verified": 3, "engaged": 2, "trials": 1, "sources": [("x", 2)],
                                    "revenue": {"paying": 2, "mrr": {"USD": 39.0}}, "new_paid": 1,
                                    "cancel_scheduled": 0, "past_due": 0, "emails": {"tip_alert": 2}, "errors": []},
                                   dt.datetime(2026, 10, 26, 12, tzinfo=UTC))
        self.assertIn("4 sign-ups, 2 paying, USD 39.00/mo", subject)
        self.assertIn("tip_alert 2", text)


# ---- email preferences -----------------------------------------------------------------------------
class EmailPrefKindsTests(unittest.TestCase):
    def test_new_kinds(self):
        from db import email_prefs

        self.assertIn("weekly", email_prefs.KINDS)
        self.assertIn("tips", email_prefs.KINDS)
        with mock.patch.object(email_prefs, "_conn", return_value=None):
            self.assertEqual(email_prefs.get_prefs("a@x.com"),
                             {"digest": True, "evening": True, "alerts": True, "weekly": True, "tips": True})


# ---- 7. Dependabot -----------------------------------------------------------------------------------
class DependabotMergeTests(unittest.TestCase):
    def test_decide(self):
        from scripts.dependabot_merge_check import decide

        self.assertEqual(decide("Bump jsdom from 30.1.2 to 30.1.3 in /web", "dev"), "merge")
        self.assertEqual(decide("Bump httpx from 0.28.1 to 0.28.2", "dev"), "merge")
        self.assertTrue(decide("Bump httpx from 0.28.1 to 0.29.0", "dev").startswith("hold"))
        self.assertTrue(decide("Bump eslint from 9.39.5 to 10.13.0 in /web", "dev").startswith("hold: major"))
        self.assertTrue(decide("Bump jsdom from 30.1.2 to 30.1.3 in /web", "main").startswith("hold: targets"))
        self.assertTrue(decide("Bump typescript from 5.9.3 to 5.9.4", "dev", "typescript").startswith("hold"))
        self.assertTrue(decide("Bump the npm group across 1 directory with 3 updates", "dev").startswith("hold"))
        self.assertEqual(decide("Bump actions/checkout from 7.0.0 to 7.1.0", "dev"), "merge")


if __name__ == "__main__":
    unittest.main()
