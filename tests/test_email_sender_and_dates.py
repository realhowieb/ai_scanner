"""HSF email shows "HSF Alerts" as the sender and New York dates.

2026-09-29: emails sent at 8:34 PM ET Monday showed the sender as "alerts"
(bare address) and were dated "Tuesday, Sep 29" (UTC date).
"""
import datetime as dt
import email
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MON_EVENING_ET = dt.datetime(2026, 9, 29, 0, 34, tzinfo=dt.timezone.utc).astimezone(ET)   # Mon 20:34 ET


class SenderTests(unittest.TestCase):
    def test_display_name_defaults_to_hsf_alerts(self):
        from ui.email_utils import _sender

        self.assertEqual(_sender("alerts@ai.hsfinest.com"),
                         ("HSF Alerts <alerts@ai.hsfinest.com>", "alerts@ai.hsfinest.com"))

    def test_existing_name_is_kept_and_override_is_honoured(self):
        from ui.email_utils import _sender

        self.assertEqual(_sender("HSFinest Team <team@ai.hsfinest.com>"),
                         ("HSFinest Team <team@ai.hsfinest.com>", "team@ai.hsfinest.com"))
        with mock.patch("config.SMTP_FROM_NAME", "HSF Desk", create=True):
            self.assertEqual(_sender("alerts@ai.hsfinest.com")[0], "HSF Desk <alerts@ai.hsfinest.com>")

    def test_sent_message_header_and_envelope(self):
        import ui.email_utils as eu

        server = mock.MagicMock()
        server.__enter__.return_value = server
        with mock.patch("config.SMTP_HOST", "smtp.test"), mock.patch("config.SMTP_USER", "u"), \
             mock.patch("config.SMTP_PASS", "p"), mock.patch("config.SMTP_FROM", "alerts@ai.hsfinest.com"), \
             mock.patch("smtplib.SMTP", return_value=server), redirect_stdout(io.StringIO()):
            self.assertTrue(eu.send_alert_email("a@example.com", "Breakout alert triggered", "body"))
        envelope_from, _to, raw = server.sendmail.call_args.args
        self.assertEqual(envelope_from, "alerts@ai.hsfinest.com")
        self.assertEqual(email.message_from_string(raw)["From"], "HSF Alerts <alerts@ai.hsfinest.com>")

    def test_billing_live_alerts_use_the_same_name(self):
        from billing_service import realtime_alerts as ra

        server = mock.MagicMock()
        server.__enter__.return_value = server
        env = {"SMTP_HOST": "smtp.test", "SMTP_USER": "u", "SMTP_PASS": "p", "SMTP_FROM": "alerts@ai.hsfinest.com"}
        with mock.patch.dict(os.environ, env), mock.patch("smtplib.SMTP", return_value=server):
            self.assertTrue(ra._send_email("a@example.com", "Live alert", "body"))
        envelope_from, _to, raw = server.sendmail.call_args.args
        self.assertEqual(envelope_from, "alerts@ai.hsfinest.com")
        self.assertEqual(email.message_from_string(raw)["From"], "HSF Alerts <alerts@ai.hsfinest.com>")


class NewYorkDateTests(unittest.TestCase):
    def test_digest_and_wrap_are_dated_in_new_york(self):
        import scheduler.evening_wrap as ew
        import scheduler.morning_digest as md

        with mock.patch.object(md, "et_now", return_value=MON_EVENING_ET), \
             mock.patch.object(md, "_track_record_line", return_value=("", "")):
            _html, text = md._compose("a@example.com", [], [], [], [])
            wrap_html, wrap_text = ew._compose_wrap([], [], [])
        self.assertIn("Monday, Sep 28", text)
        self.assertIn("Monday, Sep 28", wrap_text + wrap_html)
        self.assertNotIn("Tuesday", text)

    def test_evening_alerts_count_for_the_new_york_day(self):
        import scheduler.evening_wrap as ew
        import scheduler.morning_digest as md

        fired_mon_9pm_et = dt.datetime(2026, 9, 29, 1, 0, tzinfo=dt.timezone.utc)       # UTC Tuesday
        fired_sun = dt.datetime(2026, 9, 27, 15, 0, tzinfo=dt.timezone.utc)
        events = [{"fired_at": fired_mon_9pm_et, "message": "GRAL breakout"},
                  {"fired_at": fired_sun, "message": "old"}]
        with mock.patch.object(md, "et_now", return_value=MON_EVENING_ET), \
             mock.patch("db.alerts.list_recent_events", return_value=events):
            self.assertEqual(ew._todays_events("a@example.com"), ["GRAL breakout"])


if __name__ == "__main__":
    unittest.main()
