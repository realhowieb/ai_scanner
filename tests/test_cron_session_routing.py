"""Cron session routing + skip logic, and the DB-cache max-age helper."""
from __future__ import annotations

import datetime as dt
import importlib.util
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

_PANDAS = importlib.util.find_spec("pandas") is not None

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def _utc_for_et(y, m, d, hour, minute=0):
    """A UTC datetime corresponding to the given wall-clock ET time (DST-safe)."""
    return dt.datetime(y, m, d, hour, minute, tzinfo=ET).astimezone(UTC)


class ResolveSessionTests(unittest.TestCase):
    def _resolve(self, now_utc):
        from scheduler import cron_runner

        # Force auto-inference (ignore any ambient CRON_SESSION).
        with mock.patch.dict("os.environ", {"CRON_SESSION": "auto"}):
            return cron_runner._resolve_session(now_utc)

    def test_premarket_before_930_et(self):
        self.assertEqual(self._resolve(_utc_for_et(2026, 7, 15, 8, 35)), "premarket")

    def test_regular_midday_et(self):
        self.assertEqual(self._resolve(_utc_for_et(2026, 7, 15, 12, 0)), "regular")

    def test_boundary_930_is_regular(self):
        self.assertEqual(self._resolve(_utc_for_et(2026, 7, 15, 9, 30)), "regular")

    def test_postmarket_at_and_after_1600_et(self):
        self.assertEqual(self._resolve(_utc_for_et(2026, 7, 15, 16, 0)), "postmarket")
        self.assertEqual(self._resolve(_utc_for_et(2026, 7, 15, 17, 30)), "postmarket")

    def test_explicit_session_overrides_time(self):
        from scheduler import cron_runner

        with mock.patch.dict("os.environ", {"CRON_SESSION": "postmarket"}):
            # Midday, but the explicit env wins.
            self.assertEqual(
                cron_runner._resolve_session(_utc_for_et(2026, 7, 15, 12, 0)),
                "postmarket",
            )


class SkipReasonTests(unittest.TestCase):
    def _skip(self, now_utc, force=None):
        from scheduler import cron_runner

        env = {"CRON_FORCE": force} if force is not None else {"CRON_FORCE": ""}
        with mock.patch.dict("os.environ", env):
            return cron_runner._skip_reason(now_utc)

    def test_weekend_skipped(self):
        # 2026-07-18 is a Saturday.
        self.assertIn("Weekend", self._skip(_utc_for_et(2026, 7, 18, 12, 0)))

    def test_too_early_skipped(self):
        self.assertIn("early", self._skip(_utc_for_et(2026, 7, 15, 5, 0)))

    def test_weekday_daytime_runs(self):
        self.assertIsNone(self._skip(_utc_for_et(2026, 7, 15, 12, 0)))

    def test_cron_force_bypasses_skip(self):
        # Saturday but forced → runs anyway.
        self.assertIsNone(self._skip(_utc_for_et(2026, 7, 18, 3, 0), force="1"))


class DedupeTests(unittest.TestCase):
    def test_dedupe_preserves_first_occurrence_order(self):
        from scheduler import cron_runner

        self.assertEqual(
            cron_runner._dedupe(["AAPL", "MSFT", "AAPL", "NVDA", "MSFT"]),
            ["AAPL", "MSFT", "NVDA"],
        )


@unittest.skipUnless(_PANDAS, "scan.engine imports pandas")
class DbCacheMaxAgeTests(unittest.TestCase):
    def test_env_overrides_default(self):
        from scan import engine

        with mock.patch.dict("os.environ", {"DB_PRICE_CACHE_MAX_AGE_MINUTES": "180"}):
            self.assertEqual(engine._db_cache_max_age_minutes(), 180)

    def test_junk_env_falls_back_to_default(self):
        from scan import engine

        with mock.patch.dict("os.environ", {"DB_PRICE_CACHE_MAX_AGE_MINUTES": "oops"}):
            self.assertEqual(engine._db_cache_max_age_minutes(default=30), 30)


if __name__ == "__main__":
    unittest.main()
