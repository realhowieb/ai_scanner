"""P2-81: earnings and email-preference schema setup runs once per process and
database, not on every call (Neon Query Stats 2026-10-05: CREATE TABLE and two
full-table normalization UPDATEs on every earnings read)."""
import itertools
import unittest
from unittest import mock

_db_ids = itertools.count()


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append(" ".join(str(sql).split())[:40])

    def fetchall(self):
        return []

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    """Looks like a real psycopg connection to schema_once (host, port, dbname)."""

    def __init__(self, log, dbname):
        self.log, self.closed = log, False
        self.info = mock.Mock(host="db.example", port=5432, dbname=dbname)

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        pass

    def close(self):
        self.closed = True


def _fresh_db():
    return f"p2_81_{next(_db_ids)}"


class EarningsSchemaOnceTests(unittest.TestCase):
    def test_table_setup_and_cleanup_run_once_per_database(self):
        from db import earnings

        log, db = [], _fresh_db()
        for _ in range(3):
            earnings.ensure_earnings_table(FakeConn(log, db))
        self.assertEqual(sum("CREATE TABLE" in x for x in log), 1)
        self.assertEqual(sum(x.startswith("UPDATE earnings_calendar") for x in log), 1)
        earnings.ensure_earnings_table(FakeConn(log, _fresh_db()))  # another database: runs again
        self.assertEqual(sum("CREATE TABLE" in x for x in log), 2)

    def test_reads_no_longer_repeat_the_setup(self):
        from db import earnings

        log, db = [], _fresh_db()
        for _ in range(4):
            earnings.load_earnings_map(["NVDA"], conn=FakeConn(log, db))
        self.assertEqual(sum("CREATE TABLE" in x for x in log), 1)
        self.assertEqual(sum(x.startswith("SELECT symbol AS sym_key") for x in log), 4)

    def test_refresh_log_table_once(self):
        from db import earnings

        log, db = [], _fresh_db()
        for _ in range(3):
            earnings.ensure_earnings_refresh_log_table(FakeConn(log, db))
        self.assertEqual(sum("CREATE TABLE" in x for x in log), 1)

    def test_opened_connection_is_closed_injected_one_is_not(self):
        from db import earnings

        mine, theirs = FakeConn([], _fresh_db()), FakeConn([], _fresh_db())
        with mock.patch.object(earnings, "_get_conn", return_value=mine):
            earnings.ensure_earnings_table()
            earnings.ensure_earnings_refresh_log_table()
        earnings.ensure_earnings_table(theirs)
        self.assertTrue(mine.closed)
        self.assertFalse(theirs.closed)


class EmailPrefsSchemaOnceTests(unittest.TestCase):
    def test_prefs_table_created_once(self):
        from db import email_prefs

        log, db = [], _fresh_db()
        with mock.patch.object(email_prefs, "get_neon_conn", side_effect=lambda: FakeConn(log, db)):
            for _ in range(5):
                email_prefs._conn()
        self.assertEqual(sum("CREATE TABLE" in x for x in log), 1)


if __name__ == "__main__":
    unittest.main()
