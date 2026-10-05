"""2026-09-29: one alert email per user per run, and no duplicate alerts.

An account had three identical watchlist alerts and two breakout alerts
(≥30 and ≥50); one scan sent five emails. Now identical alerts are refused at
creation, only the newest of any existing identical set is evaluated, and every
alert that fires for a user in a run goes out in a single email.
"""
import contextlib
import io
import unittest
from unittest import mock

try:
    import pandas as pd

    _PANDAS = True
except Exception:
    _PANDAS = False

from scheduler.alert_email import compose, make_section
from scheduler.alert_runner import alert_signature

ME = "me@example.com"


def alert(aid, kind, threshold=None, user=ME, **kw):
    return {"id": aid, "user_id": user, "alert_type": kind, "ticker": kw.get("ticker"),
            "threshold": threshold, "direction": kw.get("direction"),
            "watchlist_only": kw.get("watchlist_only", False), "enabled": True, "last_fired_at": kw.get("last")}


class ComposeTests(unittest.TestCase):
    def test_single_alert(self):
        subject, text, html = compose([make_section(alert(1, "breakout", 30.0), ["IOVA: BreakoutScore 105.8 (≥ 30)"])])
        self.assertEqual(subject, "📈 Breakout alert: IOVA")
        self.assertIn("Breakout alert · BreakoutScore ≥ 30", text)
        self.assertIn("IOVA", html)

    def test_several_alerts_in_one_email_without_repeats(self):
        watch = make_section(alert(1, "watchlist"), ["SMTC: in scan results (BreakoutScore 33.9)"])
        subject, text, _ = compose([watch, dict(watch),
                                    make_section(alert(2, "breakout", 30.0), ["IOVA: BreakoutScore 105.8 (≥ 30)"])])
        self.assertEqual(subject, "📈 2 alerts: SMTC, IOVA")
        self.assertEqual(text.count("SMTC"), 1)

    def test_signature(self):
        self.assertEqual(alert_signature(alert(1, "watchlist")), alert_signature(alert(2, "watchlist")))
        self.assertNotEqual(alert_signature(alert(1, "breakout", 30)), alert_signature(alert(2, "breakout", 50)))
        self.assertNotEqual(alert_signature(alert(1, "watchlist")),
                            alert_signature(alert(2, "watchlist", user="other@example.com")))
        self.assertEqual(alert_signature(alert(1, "price", 5, ticker="aapl")),
                         alert_signature(alert(2, "price", 5.0, ticker="AAPL")))


@unittest.skipUnless(_PANDAS, "needs pandas")
class RunnerTests(unittest.TestCase):
    DF = None

    def run_alerts(self, alerts):
        import scheduler.alert_runner as ar

        df = pd.DataFrame([{"Ticker": "SMTC", "BreakoutScore": 33.9, "Last": 50.0},
                           {"Ticker": "IOVA", "BreakoutScore": 105.8, "Last": 3.0}])
        fired, sent = [], []
        patches = [
            mock.patch("config.ALERTS_ENABLED", True), mock.patch("config.ALERT_THROTTLE_HOURS", 12),
            mock.patch("db.alerts.list_all_enabled_alerts", return_value=alerts),
            mock.patch("db.alerts.mark_alert_fired", side_effect=fired.append),
            mock.patch("db.alerts.record_alert_event", return_value=None),
            mock.patch("db.watchlists.list_watchlists", return_value=[{"id": 1}]),
            mock.patch("db.watchlists.get_watchlist_tickers", return_value=["SMTC"]),
            mock.patch("ui.email_utils.send_digest_email", side_effect=lambda **k: sent.append(k) or True),
            mock.patch.object(ar, "_latest_snapshot_df", return_value=df),
            mock.patch.object(ar, "_alert_limit_for_user", return_value=25),
            mock.patch.object(ar, "_annotate_earnings", side_effect=lambda lines: lines),
            mock.patch.object(ar, "_freeze_signal_outcomes"),
            mock.patch.object(ar, "_is_verified", return_value=True),
            mock.patch.object(ar, "_email_allowed_for_tier", return_value=True),
            mock.patch.object(ar, "_alert_emails_on", return_value=True),
            mock.patch.object(ar, "_alert_unsubscribe_link", return_value=None),
            mock.patch("scheduler.morning_digest.record_email_job"),
        ]
        out = io.StringIO()
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(out):
            for p in patches:
                stack.enter_context(p)
            ar.run_alerts()
        return fired, sent, out.getvalue()

    def test_owner_case_sends_one_email(self):
        # newest first, as list_all_enabled_alerts returns them
        alerts = [alert(23, "watchlist"), alert(13, "watchlist"), alert(12, "breakout", 30.0),
                  alert(11, "watchlist"), alert(8, "breakout", 50.0)]
        fired, sent, log = self.run_alerts(alerts)
        self.assertEqual(fired, [23, 12, 8])                      # duplicates 13 and 11 skipped
        self.assertEqual(len(sent), 1)
        # ≥50 lists only IOVA, already under ≥30, so its section is dropped (P2-50)
        self.assertEqual(sent[0]["subject"], "📈 2 alerts: SMTC, IOVA")
        self.assertEqual(sent[0]["text_inner"].count("Watchlist alert"), 1)
        self.assertNotIn("≥ 50", sent[0]["html_inner"])
        self.assertIn("skipped 2 duplicate alert(s)", log)

    def test_throttled_owner_keeps_duplicates_quiet(self):
        import datetime as dt

        recent = dt.datetime.now(dt.timezone.utc)
        fired, sent, _ = self.run_alerts([alert(23, "watchlist", last=recent), alert(13, "watchlist")])
        self.assertEqual((fired, sent), ([], []))

    def test_each_user_gets_their_own_email(self):
        other = "you@example.com"
        fired, sent, _ = self.run_alerts([alert(2, "watchlist", user=other), alert(1, "watchlist")])
        self.assertEqual(sorted(s["to_address"] for s in sent), [ME, other])


class CreateTests(unittest.TestCase):
    def _conn(self, existing):
        cur = mock.MagicMock()
        cur.fetchone.return_value = (1,) if existing else None
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        return conn, cur

    def test_identical_alert_is_refused(self):
        from db import alerts

        conn, cur = self._conn(existing=True)
        with mock.patch.object(alerts, "_get_conn", return_value=conn):
            with self.assertRaisesRegex(ValueError, "already have this alert"):
                alerts.create_alert(ME, "watchlist")
        self.assertFalse(any("INSERT" in str(c.args[0]) for c in cur.execute.call_args_list))

    def test_new_alert_is_inserted(self):
        from db import alerts

        conn, cur = self._conn(existing=False)
        with mock.patch.object(alerts, "_get_conn", return_value=conn):
            alerts.create_alert(ME, "price", ticker="aapl", threshold=5.0, direction="above")
        calls = [c.args for c in cur.execute.call_args_list]
        self.assertIn("pg_advisory_xact_lock", calls[0][0])  # per-user lock before the checks
        sql, params = next(a for a in calls if "IS NOT DISTINCT FROM" in a[0])
        self.assertEqual(params, (ME, "price", "AAPL", 5.0, "above", False))
        self.assertIn("INSERT", str(cur.execute.call_args_list[-1].args[0]))
        conn.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
