"""Realtime Alerts: the due-alerts query runs on real Postgres.

2026-10-05, first weekday run: every pass failed with UndefinedFunction because
the query used make_interval(hours => %s) with a float (12.0), and make_interval
only takes an integer for hours. The tests that existed mocked the database, so
nothing ran the SQL. The Postgres part needs HSF_TEST_PG_URL and psycopg2 (the
driver the worker uses).
"""
import importlib.util
import os
import unittest
from unittest import mock

PG_URL = os.environ.get("HSF_TEST_PG_URL")
HAS_PSYCOPG2 = importlib.util.find_spec("psycopg2") is not None


class ThrottleParameterTests(unittest.TestCase):
    def test_throttle_is_passed_in_seconds(self):
        from billing_service import realtime_alerts as ra

        cur = mock.MagicMock()
        cur.fetchall.return_value = []
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(ra, "_email_prefs_ready", return_value=False), \
                mock.patch.object(ra, "THROTTLE_HOURS", 12.0):
            ra._due_price_alerts(conn)
        sql, params = cur.execute.call_args.args
        self.assertIn("make_interval(secs => %s)", sql)
        self.assertNotIn("hours =>", sql)
        self.assertEqual(params, (43200.0,))


@unittest.skipUnless(PG_URL and HAS_PSYCOPG2, "set HSF_TEST_PG_URL (throwaway Postgres) and install psycopg2")
class DueAlertsPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg2

        self.conn = psycopg2.connect(PG_URL)
        self.addCleanup(self.conn.close)
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS rt_due_users, rt_due_alerts")
        cur.execute("CREATE TEMP TABLE users (username TEXT PRIMARY KEY, email_verified BOOLEAN, tier TEXT)")
        cur.execute("""CREATE TEMP TABLE user_alerts (id BIGSERIAL PRIMARY KEY, user_id TEXT, alert_type TEXT,
                       ticker TEXT, threshold DOUBLE PRECISION, direction TEXT, enabled BOOLEAN DEFAULT TRUE,
                       last_fired_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW())""")
        cur.execute("INSERT INTO users VALUES ('rt@example.invalid', TRUE, 'pro')")
        cur.execute("INSERT INTO user_alerts (user_id, alert_type, ticker, threshold, direction, last_fired_at) VALUES "
                    "('rt@example.invalid', 'price', 'NVDA', 150, 'above', NULL), "
                    "('rt@example.invalid', 'move', 'AMD', 5, NULL, NOW() - INTERVAL '1 hour'), "
                    "('rt@example.invalid', 'rvol', 'TSLA', 2, NULL, NOW() - INTERVAL '13 hours')")
        self.conn.commit()

    def test_query_runs_and_applies_the_throttle(self):
        from billing_service import realtime_alerts as ra

        with mock.patch.object(ra, "_email_prefs_ready", return_value=False), \
                mock.patch.object(ra, "THROTTLE_HOURS", 12.0):
            due = ra._due_price_alerts(self.conn)
        # fired 1 h ago is still throttled; never fired and fired 13 h ago are due
        self.assertEqual(sorted(a["ticker"] for a in due), ["NVDA", "TSLA"])

    def test_fractional_throttle(self):
        from billing_service import realtime_alerts as ra

        with mock.patch.object(ra, "_email_prefs_ready", return_value=False), \
                mock.patch.object(ra, "THROTTLE_HOURS", 0.5):
            due = ra._due_price_alerts(self.conn)
        self.assertEqual(sorted(a["ticker"] for a in due), ["AMD", "NVDA", "TSLA"])


if __name__ == "__main__":
    unittest.main()
