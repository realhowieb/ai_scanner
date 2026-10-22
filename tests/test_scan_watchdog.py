"""Backup scan check (api/scan_watchdog.py, P2-80). No network, no Postgres."""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from api import scan_watchdog as wd

UTC = dt.timezone.utc
SLOT = dt.datetime(2026, 10, 22, 16, 35, tzinfo=UTC)  # a Thursday, regular hours


def run(minutes_after_slot, status="completed", conclusion="success"):
    t = SLOT + dt.timedelta(minutes=minutes_after_slot)
    return {"created_at": t.isoformat().replace("+00:00", "Z"), "status": status, "conclusion": conclusion}


class DueSlotTests(unittest.TestCase):
    def test_window(self):
        self.assertIsNone(wd.due_slot(SLOT + dt.timedelta(minutes=19)))
        self.assertEqual(wd.due_slot(SLOT + dt.timedelta(minutes=20)), SLOT)
        self.assertEqual(wd.due_slot(SLOT + dt.timedelta(minutes=59)), SLOT)
        self.assertIsNone(wd.due_slot(SLOT + dt.timedelta(minutes=60)))

    def test_premarket_and_last_slot(self):
        self.assertEqual(wd.due_slot(dt.datetime(2026, 10, 22, 12, 56, tzinfo=UTC)),
                         dt.datetime(2026, 10, 22, 12, 35, tzinfo=UTC))
        self.assertEqual(wd.due_slot(dt.datetime(2026, 10, 22, 22, 0, tzinfo=UTC)),
                         dt.datetime(2026, 10, 22, 21, 35, tzinfo=UTC))

    def test_no_slots_on_weekends_or_holidays(self):
        self.assertIsNone(wd.due_slot(dt.datetime(2026, 10, 24, 16, 56, tzinfo=UTC)))  # Saturday
        self.assertIsNone(wd.due_slot(dt.datetime(2026, 11, 26, 16, 56, tzinfo=UTC)))  # Thanksgiving


class ClassifyTests(unittest.TestCase):
    def test_statuses(self):
        self.assertEqual(wd.classify(SLOT, []), wd.MISSED)
        self.assertEqual(wd.classify(SLOT, [run(1)]), wd.OK)
        self.assertEqual(wd.classify(SLOT, [run(-5)]), wd.OK)  # cron-job.org a little early
        self.assertEqual(wd.classify(SLOT, [run(1, "in_progress", None)]), wd.RUNNING)
        self.assertEqual(wd.classify(SLOT, [run(1, "queued", None)]), wd.RUNNING)
        self.assertEqual(wd.classify(SLOT, [run(1, conclusion="failure")]), wd.FAILED)
        self.assertEqual(wd.classify(SLOT, [run(1, conclusion="cancelled")]), wd.FAILED)

    def test_runs_from_other_slots_dont_count(self):
        self.assertEqual(wd.classify(SLOT, [run(-180), run(-11), run(46)]), wd.MISSED)

    def test_a_failed_run_then_a_good_rerun_is_ok(self):
        self.assertEqual(wd.classify(SLOT, [run(1, conclusion="failure"), run(21)]), wd.OK)


class _Resp:
    def __init__(self, status, data=None):
        self.status_code, self._data = status, data or {}

    def json(self):
        return self._data


class FakeGitHub:
    def __init__(self, runs=None, read_status=200, dispatch_status=204):
        self.runs, self.read_status, self.dispatch_status = runs or [], read_status, dispatch_status
        self.gets, self.posts = [], []

    def get(self, url, **kw):
        self.gets.append((url, kw))
        return _Resp(self.read_status, {"workflow_runs": self.runs})

    def post(self, url, **kw):
        self.posts.append((url, kw))
        return _Resp(self.dispatch_status)


class CheckTests(unittest.TestCase):
    def setUp(self):
        from db import scan_watchdog as store

        self.store = store
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE users (username TEXT, tier TEXT, is_admin INTEGER, is_active INTEGER)")
        self.conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", [
            ("Boss@Example.com", "admin", 1, 1), ("old@example.com", "admin", 1, 0),
            ("user@example.com", "basic", 0, 1), ("admin", "admin", 1, 1)])
        self.conn.commit()
        store.set_connection_factory(lambda: self.conn)
        store._ensure_schema.__wrapped__(self.conn, True)
        self.addCleanup(lambda: (store.set_connection_factory(None), self.conn.close()))
        env = mock.patch.dict(os.environ, {"HSF_GITHUB_DISPATCH_TOKEN": "", "HSF_OPS_ALERT_EMAIL": ""})
        env.start()
        self.addCleanup(env.stop)
        self.send = mock.patch("ui.email_utils.send_digest_email", return_value=True).start()
        self.addCleanup(mock.patch.stopall)
        self.now = SLOT + dt.timedelta(minutes=20)

    def test_idle_outside_the_window(self):
        gh = FakeGitHub()
        self.assertEqual(wd.check(SLOT + dt.timedelta(minutes=5), session=gh)["status"], "idle")
        self.assertEqual(gh.gets, [])

    def test_ok_slot_is_settled_with_one_read(self):
        gh, done = FakeGitHub([run(1)]), set()
        self.assertEqual(wd.check(self.now, done=done, session=gh)["status"], wd.OK)
        self.assertEqual(wd.check(self.now + dt.timedelta(minutes=2), done=done, session=gh)["status"], "settled")
        self.assertEqual(len(gh.gets), 1)
        self.send.assert_not_called()

    def test_missed_slot_is_flagged_once_and_admins_emailed(self):
        gh = FakeGitHub([run(-180)])
        out = wd.check(self.now, done=set(), session=gh)
        self.assertEqual((out["status"], out["flagged"], out["redispatched"], out["emailed"]),
                         (wd.MISSED, True, None, 1))
        self.assertEqual(self.send.call_args.args[0], "boss@example.com")
        self.assertIn("never started", self.send.call_args.args[3])
        self.assertIn("12:35 PM ET", self.send.call_args.args[1])
        self.assertEqual(gh.posts, [])  # no token, no re-run
        # A restarted API (empty memory) doesn't flag the same slot again.
        self.assertFalse(wd.check(self.now + dt.timedelta(minutes=2), done=set(), session=gh)["flagged"])
        self.assertEqual(self.send.call_count, 1)
        self.assertEqual(self.store.recent_flags()[0][:2], ("2026-10-22T16:35+00:00", wd.MISSED))

    def test_token_re_runs_the_scan_like_cron_job_org(self):
        gh = FakeGitHub([run(1, conclusion="failure")])
        with mock.patch.dict(os.environ, {"HSF_GITHUB_DISPATCH_TOKEN": "t", "HSF_OPS_ALERT_EMAIL": "ops@x.com, b@y.com"}):
            out = wd.check(self.now, done=set(), session=gh)
        self.assertEqual((out["status"], out["redispatched"], out["emailed"]), (wd.FAILED, True, 2))
        url, kw = gh.posts[0]
        self.assertTrue(url.endswith("/actions/workflows/scheduled-scans.yml/dispatches"))
        self.assertEqual(kw["json"], {"ref": "main", "inputs": {"force": "false", "session": "auto"}})
        self.assertEqual(gh.gets[0][1]["headers"]["Authorization"], "Bearer t")
        self.assertIn("started the scan again", self.send.call_args.args[3])

    def test_redispatch_kill_switch(self):
        gh = FakeGitHub([])
        with mock.patch.dict(os.environ, {"HSF_GITHUB_DISPATCH_TOKEN": "t", "HSF_SCAN_WATCHDOG_REDISPATCH": "0"}):
            self.assertIsNone(wd.check(self.now, done=set(), session=gh)["redispatched"])
        self.assertEqual(gh.posts, [])

    def test_running_scan_is_checked_again(self):
        gh, done = FakeGitHub([run(1, "in_progress", None)]), set()
        self.assertEqual(wd.check(self.now, done=done, session=gh)["status"], wd.RUNNING)
        gh.runs = [run(1, conclusion="failure")]
        self.assertTrue(wd.check(self.now + dt.timedelta(minutes=2), done=done, session=gh)["flagged"])

    def test_github_unreadable_flags_nothing(self):
        gh = FakeGitHub(read_status=403)
        self.assertEqual(wd.check(self.now, done=set(), session=gh)["status"], wd.UNKNOWN)
        self.send.assert_not_called()

    def test_storage_down_flags_nothing_and_retries(self):
        self.store.set_connection_factory(lambda: None)
        done = set()
        self.assertEqual(wd.check(self.now, done=done, session=FakeGitHub())["status"], wd.UNKNOWN)
        self.assertEqual(done, set())
        self.send.assert_not_called()


class StartTests(unittest.TestCase):
    def test_only_on_render_and_when_enabled(self):
        with mock.patch("threading.Thread") as thread:
            with mock.patch.dict(os.environ, {"RENDER": ""}):
                wd.start()
            with mock.patch.dict(os.environ, {"RENDER": "true", "HSF_SCAN_WATCHDOG_ENABLED": "0"}):
                wd.start()
            thread.assert_not_called()
            with mock.patch.dict(os.environ, {"RENDER": "true", "HSF_SCAN_WATCHDOG_ENABLED": "1"}):
                wd.start()
            thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
