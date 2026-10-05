"""API acceptance run: db.alerts.create_alert is atomic per user.

Concurrent requests used to pass the duplicate and plan-limit checks together
(a Pro account reached 10 of 5 alerts; identical alerts saved twice). The checks
and the insert now run in one transaction under a per-user advisory lock.
"""
import concurrent.futures as cf
import os
import unittest
from unittest import mock

PG_URL = os.environ.get("HSF_TEST_PG_URL")


class FakeCursor:
    def __init__(self, log, count=0, duplicate=False):
        self.log, self.count, self.duplicate, self.last = log, count, duplicate, ""

    def execute(self, sql, params=None):
        self.log.append(" ".join(sql.split())[:60])
        self.last = sql

    def fetchone(self):
        if "SELECT 1 FROM user_alerts" in self.last:
            return {"?": 1} if self.duplicate else None
        if "COUNT(*)" in self.last:
            return {"n": self.count}
        if "RETURNING id" in self.last:
            return {"id": 42}
        return None

    def close(self):
        pass


class FakeConn:
    def __init__(self, cur):
        self.cur, self.committed, self.rolled_back = cur, False, False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class CreateAlertUnitTests(unittest.TestCase):
    def run_create(self, **kw):
        from db import alerts

        log = []
        conn = FakeConn(FakeCursor(log, count=kw.pop("count", 0), duplicate=kw.pop("duplicate", False)))
        with mock.patch.object(alerts, "_get_conn", return_value=conn):
            try:
                result = alerts.create_alert("u@example.com", "move", ticker="nvda", threshold=5.0, **kw)
            except ValueError as e:
                result = e
        return result, log, conn

    def test_lock_is_taken_before_any_check(self):
        result, log, conn = self.run_create(max_alerts=5)
        self.assertEqual(result, 42)
        self.assertIn("pg_advisory_xact_lock", log[0])
        self.assertTrue(conn.committed)

    def test_limit_reached_rolls_back_and_releases_the_lock(self):
        from db.alerts import AlertLimitReached

        result, log, conn = self.run_create(max_alerts=1, count=1)
        self.assertIsInstance(result, AlertLimitReached)
        self.assertIn("maximum of 1 alert on", str(result))
        self.assertTrue(conn.rolled_back)
        self.assertFalse(any("INSERT" in line for line in log))

    def test_duplicate_rolls_back(self):
        result, _log, conn = self.run_create(duplicate=True)
        self.assertEqual(str(result), "You already have this alert.")
        self.assertTrue(conn.rolled_back)

    def test_web_callers_without_a_limit_still_work(self):
        result, log, _conn = self.run_create()
        self.assertEqual(result, 42)
        self.assertFalse(any("COUNT(*)" in line for line in log))


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class CreateAlertConcurrencyTests(unittest.TestCase):
    def setUp(self):
        os.environ["DATABASE_URL"] = PG_URL
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)
        os.environ["AI_SCANNER_DB_POOL"] = "0"  # one connection per call, like separate requests
        self.addCleanup(os.environ.pop, "AI_SCANNER_DB_POOL", None)
        import psycopg

        from db import alerts

        self.alerts = alerts
        with psycopg.connect(PG_URL) as conn:
            alerts._ensure_alerts_schema.__wrapped__(conn)
            conn.execute("DELETE FROM user_alerts WHERE user_id LIKE 'race-%%'")

    def count(self, user, ticker=None):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            q = "SELECT count(*) FROM user_alerts WHERE user_id = %s" + (" AND ticker = %s" if ticker else "")
            return conn.execute(q, (user, ticker) if ticker else (user,)).fetchone()[0]

    def test_concurrent_creates_never_exceed_the_limit(self):
        def make(i):
            try:
                return self.alerts.create_alert("race-limit", "move", ticker=f"T{i:02d}", threshold=9e5, max_alerts=5)
            except ValueError:
                return None

        with cf.ThreadPoolExecutor(30) as ex:
            made = [x for x in ex.map(make, range(30)) if x]
        self.assertEqual(len(made), 5)
        self.assertEqual(self.count("race-limit"), 5)

    def test_concurrent_identical_alerts_saved_once(self):
        def make(_):
            try:
                return self.alerts.create_alert("race-dupe", "move", ticker="DUPE", threshold=9e5)
            except ValueError:
                return None

        with cf.ThreadPoolExecutor(15) as ex:
            list(ex.map(make, range(15)))
        self.assertEqual(self.count("race-dupe", "DUPE"), 1)



@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class CreateWatchlistConcurrencyTests(unittest.TestCase):
    def setUp(self):
        os.environ["DATABASE_URL"] = PG_URL
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)
        os.environ["AI_SCANNER_DB_POOL"] = "0"
        self.addCleanup(os.environ.pop, "AI_SCANNER_DB_POOL", None)
        import psycopg

        from db import watchlists
        from db.schema import ensure_neon_watchlists_schema

        self.wl = watchlists
        with psycopg.connect(PG_URL) as conn:
            ensure_neon_watchlists_schema(conn)
            conn.execute("DELETE FROM watchlists WHERE user_id LIKE 'race-%%'")

    def rows(self, user):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            return conn.execute("SELECT name, is_default FROM watchlists WHERE user_id = %s", (user,)).fetchall()

    def test_concurrent_same_name_saved_once_and_one_default(self):
        def make(_):
            try:
                return self.wl.create_watchlist("race-wl", "Momentum")
            except ValueError:
                return None

        with cf.ThreadPoolExecutor(12) as ex:
            list(ex.map(make, range(12)))
        rows = self.rows("race-wl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(sum(1 for r in rows if r[1]), 1)

    def test_concurrent_creates_respect_max_watchlists(self):
        def make(i):
            try:
                return self.wl.create_watchlist("race-cap", f"L{i}", max_watchlists=3)
            except ValueError:
                return None

        with cf.ThreadPoolExecutor(12) as ex:
            list(ex.map(make, range(12)))
        self.assertEqual(len(self.rows("race-cap")), 3)

    def test_default_repair_fixes_two_defaults_and_leaves_consistent_rows_alone(self):
        import psycopg

        a = self.wl.create_watchlist("race-repair", "A")
        self.wl.create_watchlist("race-repair", "B")
        with psycopg.connect(PG_URL) as conn:
            conn.execute("UPDATE watchlists SET is_default = TRUE WHERE user_id = 'race-repair'")
        lists = self.wl.list_watchlists("race-repair")
        self.assertEqual(sum(1 for w in lists if w["is_default"]), 1)

        def xmins():
            with psycopg.connect(PG_URL) as conn:
                return conn.execute("SELECT id, xmin::text FROM watchlists WHERE user_id = 'race-repair' "
                                    "ORDER BY id").fetchall()

        before = xmins()
        self.wl.list_watchlists("race-repair")
        self.wl.get_default_watchlist_id("race-repair")
        self.assertEqual(xmins(), before)  # reads no longer rewrite the rows
        self.assertTrue(any(w["id"] == a for w in lists))


if __name__ == "__main__":
    unittest.main()
