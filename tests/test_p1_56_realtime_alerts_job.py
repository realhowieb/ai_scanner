"""P1-56: the 5-minute Realtime Alerts job (scripts/realtime_alerts_once.py)
runs one check_once pass during extended hours on trading days only."""
import datetime as dt
import unittest
from pathlib import Path
from unittest import mock

from scripts import realtime_alerts_once as job

UTC = dt.timezone.utc
TUE_10AM_ET = dt.datetime(2026, 9, 29, 14, 0, tzinfo=UTC)
TUE_5AM_ET = dt.datetime(2026, 9, 29, 9, 0, tzinfo=UTC)       # pre-market, extended hours
TUE_9PM_ET = dt.datetime(2026, 9, 30, 1, 0, tzinfo=UTC)       # after 20:00 ET
SAT_NOON_ET = dt.datetime(2026, 10, 3, 16, 0, tzinfo=UTC)
THANKSGIVING_NOON_ET = dt.datetime(2026, 11, 26, 17, 0, tzinfo=UTC)


class ScheduleGateTests(unittest.TestCase):
    def test_runs_in_regular_and_extended_hours(self):
        self.assertEqual(job.should_run(TUE_10AM_ET), (True, ""))
        self.assertTrue(job.should_run(TUE_5AM_ET)[0])

    def test_skips_after_hours_weekends_and_holidays(self):
        self.assertFalse(job.should_run(TUE_9PM_ET)[0])
        self.assertFalse(job.should_run(SAT_NOON_ET)[0])
        self.assertEqual(job.should_run(THANKSGIVING_NOON_ET), (False, "market holiday"))


class MainTests(unittest.TestCase):
    def test_one_pass_when_open(self):
        with mock.patch("billing_service.realtime_alerts.check_once", return_value=2) as check:
            self.assertEqual(job.main(now=TUE_10AM_ET), 0)
        check.assert_called_once_with()

    def test_no_pass_when_closed(self):
        with mock.patch("billing_service.realtime_alerts.check_once") as check:
            self.assertEqual(job.main(now=SAT_NOON_ET), 0)
        check.assert_not_called()

    def test_failed_pass_exits_1_without_details(self):
        err = RuntimeError("password=secret user@example.com")
        with mock.patch("billing_service.realtime_alerts.check_once", side_effect=err), \
                mock.patch("builtins.print") as out:
            self.assertEqual(job.main(now=TUE_10AM_ET), 1)
        printed = " ".join(str(c.args[0]) for c in out.call_args_list)
        self.assertIn("RuntimeError", printed)
        self.assertNotIn("secret", printed)
        self.assertNotIn("example.com", printed)


class WorkflowTests(unittest.TestCase):
    SRC = Path(".github/workflows/realtime-alerts.yml").read_text()

    def test_dispatch_only_with_lock_and_short_timeout(self):
        import re

        self.assertRegex(self.SRC, r"(?m)^  workflow_dispatch:$")
        self.assertNotRegex(self.SRC, r"(?m)^\s+schedule:")  # comments may mention it
        self.assertIn("group: realtime-alerts", self.SRC)
        self.assertIn("timeout-minutes: 4", self.SRC)
        self.assertIn("python scripts/realtime_alerts_once.py", self.SRC)

    def test_secrets_match_scheduled_scans(self):
        scans = Path(".github/workflows/scheduled-scans.yml").read_text()
        for name in ("DATABASE_URL", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "SMTP_HOST", "SMTP_PASS"):
            self.assertIn(f"secrets.{name} }}}}", self.SRC)
            self.assertIn(f"secrets.{name} }}}}", scans)


if __name__ == "__main__":
    unittest.main()
