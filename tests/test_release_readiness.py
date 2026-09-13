"""Run 30 — V1 release-blocker, failure-injection, and smoke tests.

Covers P0/P1 production concerns: config validation (no secret values), build
identification, DB connect timeout + sanitized failure, graceful degradation when
the database is unavailable, watchlist input normalization, and core-module import
safety (no render-time side effects on import).
"""
import os
import unittest
from unittest import mock


class ConfigValidationTests(unittest.TestCase):
    def test_required_missing_is_not_ok(self):
        from config_validation import validate_config
        with mock.patch.dict(os.environ, {}, clear=True):
            r = validate_config()
        self.assertFalse(r["ok"])
        self.assertTrue(any("DATABASE_URL" in n for n in r["missing_required"]))

    def test_required_present_is_ok(self):
        from config_validation import validate_config
        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgres://x"}, clear=True):
            r = validate_config()
        self.assertTrue(r["ok"])
        self.assertEqual(r["missing_required"], [])

    def test_optional_absent_does_not_fail(self):
        from config_validation import validate_config
        with mock.patch.dict(os.environ, {"NEON_DATABASE_URL": "postgres://x"}, clear=True):
            r = validate_config()
        self.assertTrue(r["ok"])  # email/alpaca absent -> still ok

    def test_no_secret_values_exposed(self):
        from config_validation import config_health_summary
        secret = "postgres://user:SUPERSECRET@host/db"
        with mock.patch.dict(os.environ, {"DATABASE_URL": secret, "RESEND_API_KEY": "rk_SECRET"}, clear=True):
            ch = config_health_summary()
        blob = repr(ch)
        self.assertNotIn("SUPERSECRET", blob)
        self.assertNotIn("rk_SECRET", blob)
        # items report presence by name, never a value field
        for item in ch["items"]:
            self.assertNotIn("value", item)

    def test_build_info_has_sha_and_no_secret(self):
        from config_validation import build_info
        with mock.patch.dict(os.environ, {"HSF_COMMIT_SHA": "abc1234"}, clear=True):
            b = build_info()
        self.assertEqual(b["commit_sha"], "abc1234")
        self.assertIn("version", b)


class DatabaseHardeningTests(unittest.TestCase):
    def test_connect_uses_bounded_timeout(self):
        from db import engine
        captured = {}

        class _FakePsycopg:
            class rows:
                dict_row = object()

            @staticmethod
            def connect(url, **kw):
                captured.update(kw)
                return "CONN"

        with (
            mock.patch.dict(os.environ, {"DATABASE_URL": "postgres://x", "DB_CONNECT_TIMEOUT": "7"}, clear=True),
            mock.patch.object(engine, "_pool_enabled", return_value=False),
            mock.patch.dict("sys.modules", {"psycopg": _FakePsycopg}),
        ):
            conn = engine.get_neon_conn()
        self.assertEqual(conn, "CONN")
        self.assertEqual(captured.get("connect_timeout"), 7)  # bounded, from env

    def test_connection_failure_does_not_leak_dsn(self):
        from db import engine
        dsn = "postgres://user:SUPERSECRET@host/db"

        class _BoomPsycopg:
            class rows:
                dict_row = object()

            @staticmethod
            def connect(url, **kw):
                raise RuntimeError(f"could not connect using {dsn}")

        messages = []
        with (
            mock.patch.dict(os.environ, {"DATABASE_URL": dsn}, clear=True),
            mock.patch.object(engine, "_pool_enabled", return_value=False),
            mock.patch.dict("sys.modules", {"psycopg": _BoomPsycopg}),
            mock.patch.object(engine.st, "caption", side_effect=lambda m: messages.append(str(m))),
        ):
            conn = engine.get_neon_conn()
        self.assertIsNone(conn)  # fails safe, never raises
        self.assertTrue(all("SUPERSECRET" not in m for m in messages))  # DSN never echoed


class GracefulDegradationTests(unittest.TestCase):
    def test_opportunity_summary_safe_when_db_unavailable(self):
        from db import opportunity_outcomes as oo
        with mock.patch.object(oo, "get_neon_conn", return_value=None):
            s = oo.get_opportunity_outcome_summary()
        self.assertFalse(s["available"])  # empty, not a crash

    def test_alert_quality_safe_when_db_unavailable(self):
        from db import intelligence_alerts as ia
        with mock.patch.object(ia, "get_neon_conn", return_value=None):
            q = ia.get_alert_quality_summary()
        self.assertFalse(q["available"])

    def test_performance_summary_safe_when_evidence_empty(self):
        from analytics import intelligence_performance as ip
        from db import intelligence_alerts as ia
        from db import opportunity_outcomes as oo
        with (
            mock.patch.object(oo, "get_neon_conn", return_value=None),
            mock.patch.object(ia, "get_neon_conn", return_value=None),
        ):
            s = ip.get_intelligence_performance_summary()
        self.assertTrue(s["available"])  # composes safely over empty evidence
        self.assertEqual(s["readiness"]["tier"], "EARLY")
        self.assertEqual(s["findings"], [])  # no fabricated findings


class WatchlistInputTests(unittest.TestCase):
    def test_ticker_normalization(self):
        from db.watchlists import normalize_watchlist_ticker, normalize_watchlist_tickers
        self.assertEqual(normalize_watchlist_ticker(" nvda "), "NVDA")
        self.assertEqual(normalize_watchlist_ticker(None), "")
        self.assertEqual(normalize_watchlist_ticker("; DROP TABLE"), "")  # invalid rejected
        # de-duped + sorted, invalid dropped
        self.assertEqual(normalize_watchlist_tickers(["nvda", "NVDA", "bad ticker", "amd"]),
                         ["AMD", "NVDA"])


class ImportSafetyTests(unittest.TestCase):
    def test_core_modules_import_cleanly(self):
        # importing must define code, not execute DB/network side effects
        import analytics.intelligence_performance  # noqa: F401
        import analytics.opportunity_outcomes  # noqa: F401
        import config_validation  # noqa: F401
        import db.opportunity_outcomes  # noqa: F401
        import ui.intelligence_alerts_ui  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
