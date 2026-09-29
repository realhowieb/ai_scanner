"""Daily emails are throttled per New York day, inside their own ET windows.

Incident 2026-09-29: GitHub fired the 21:10 UTC scheduled scan at 00:27 UTC
(8:27 PM ET Monday). The UTC-dated once-a-day throttle saw a new day, sent the
MORNING digest in the evening, and marked Tuesday as done, which would have
skipped Tuesday's real morning digest and evening wrap.
"""
import datetime as dt
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock
from zoneinfo import ZoneInfo

import scheduler.evening_wrap as ew
import scheduler.morning_digest as md

UTC = dt.timezone.utc
ET = ZoneInfo("America/New_York")


def at(utc_iso):
    return dt.datetime.fromisoformat(utc_iso).replace(tzinfo=UTC).astimezone(ET)


class Log:
    """In-memory earnings_refresh_log keyed like the real table (existence matters)."""

    def __init__(self, keys=()):
        self.keys = set(keys)

    def already_sent(self, key):
        return key in self.keys

    def mark_sent(self, key):
        self.keys.add(key)


def run_job(job, when_utc, log, force=False):
    users = {"pro@example.com": {"tier": "pro"}}
    sends = []

    def fake_send(to_address, **kw):
        sends.append(to_address)
        return True

    patches = [
        mock.patch.object(md, "et_now", side_effect=lambda now=None: at(when_utc)),
        mock.patch.object(md, "already_sent", side_effect=log.already_sent),
        mock.patch.object(md, "mark_sent", side_effect=log.mark_sent),
        mock.patch("config.MORNING_DIGEST_ENABLED", True), mock.patch("config.MORNING_DIGEST_MAX_USERS", 500),
        mock.patch("db.users.load_users", return_value=users),
        mock.patch("db.users.is_admin_from_db", return_value=False),
        mock.patch("db.email_verification.is_email_verified", return_value=True),
        mock.patch("db.watchlists.list_watchlists", return_value=[{"id": 1}]),
        mock.patch("db.watchlists.get_watchlist_tickers", return_value=["AAPL"]),
        mock.patch("market_data.build_day_trader_metrics", return_value=[]),
        mock.patch("ui.email_utils.send_digest_email", side_effect=fake_send),
        mock.patch("db.email_prefs.wants_email", return_value=True),
        mock.patch("db.email_prefs.unsubscribe_url", return_value=None),
        mock.patch("db.email_job_runs.record_email_run"),
    ]
    if job == "digest":
        patches += [mock.patch.object(md, n, return_value=v) for n, v in (
            ("_latest_snapshot_df", None), ("_market_gappers", []), ("_earnings_today", set()),
            ("_earnings_days_map", {}), ("_watchlist_notes", []), ("_compose", ("h", "t")))]
        run = lambda: md.run_morning_digest(force=force)  # noqa: E731
    else:
        patches += [mock.patch.object(ew, n, return_value=v) for n, v in (
            ("_market_close_context", {}), ("_day_movers", ([], [])), ("_tomorrow_setups", ([], [])),
            ("_todays_events", []), ("_tomorrows_earnings", []), ("_compose_wrap", ("h", "t")))]
        patches.append(mock.patch.object(md, "_latest_snapshot_df", return_value=None))
        run = lambda: ew.run_evening_wrap(force=force)  # noqa: E731
    out = io.StringIO()
    with redirect_stdout(out):
        for p in patches:
            p.start()
        try:
            run()
        finally:
            for p in reversed(patches):
                p.stop()
    return sends, out.getvalue()


class KeyTests(unittest.TestCase):
    def test_keys_are_dated_in_new_york(self):
        mon_evening = dt.datetime(2026, 9, 29, 0, 27, tzinfo=UTC)          # Mon 20:27 ET
        self.assertEqual(md.daily_send_key("morning_digest", mon_evening), "morning_digest:2026-09-28")
        tue_morning = dt.datetime(2026, 9, 29, 12, 35, tzinfo=UTC)         # Tue 08:35 ET
        self.assertEqual(md.daily_send_key("evening_wrap", tue_morning), "evening_wrap:2026-09-29")

    def test_already_sent_checks_existence_and_fails_open(self):
        cur = mock.MagicMock()
        cur.__enter__.return_value = cur
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch("db.earnings._get_conn", return_value=conn), \
             mock.patch("db.earnings.ensure_earnings_refresh_log_table"):
            cur.fetchone.return_value = (1,)
            self.assertTrue(md.already_sent("morning_digest:2026-09-29"))
            cur.fetchone.return_value = None
            self.assertFalse(md.already_sent("morning_digest:2026-09-29"))
        with mock.patch("db.earnings._get_conn", side_effect=RuntimeError("db down")):
            self.assertFalse(md.already_sent("morning_digest:2026-09-29"))


class DigestTimelineTests(unittest.TestCase):
    def test_last_nights_timeline(self):
        # Last night's run left the OLD undated UTC record; it must not matter now.
        log = Log({"morning_digest"})
        sends, out = run_job("digest", "2026-09-29T00:27:00", log)            # Mon 20:27 ET
        self.assertEqual(sends, [])
        self.assertIn("outside the morning window (20:27 ET)", out)
        self.assertNotIn("morning_digest:2026-09-29", log.keys)                  # Tuesday not claimed
        sends, _ = run_job("digest", "2026-09-29T12:35:00", log)               # Tue 08:35 ET
        self.assertEqual(sends, ["pro@example.com"])
        self.assertIn("morning_digest:2026-09-29", log.keys)
        sends, out = run_job("digest", "2026-09-29T13:35:00", log)             # Tue 09:35 ET
        self.assertEqual(sends, [])
        self.assertIn("already sent today", out)

    def test_forced_runs_bypass_window_and_throttle(self):
        log = Log({"morning_digest:2026-09-28"})
        sends, _ = run_job("digest", "2026-09-29T00:27:00", log, force=True)
        self.assertEqual(sends, ["pro@example.com"])


class WrapTimelineTests(unittest.TestCase):
    def test_once_per_new_york_evening_across_utc_midnight(self):
        log = Log({"evening_wrap"})                                              # stale UTC record
        sends, out = run_job("wrap", "2026-09-29T18:00:00", log)               # Tue 14:00 ET
        self.assertEqual(sends, [])
        self.assertIn("before the evening window (14:00 ET)", out)
        sends, _ = run_job("wrap", "2026-09-29T20:35:00", log)                 # Tue 16:35 ET
        self.assertEqual(sends, ["pro@example.com"])
        sends, out = run_job("wrap", "2026-09-30T00:27:00", log)               # Tue 20:27 ET (UTC Wed)
        self.assertEqual(sends, [])
        self.assertIn("already sent today", out)
        self.assertNotIn("evening_wrap:2026-09-30", log.keys)                    # Wednesday not claimed


if __name__ == "__main__":
    unittest.main()
