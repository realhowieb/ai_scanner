"""P2-52: daily purge of expired sessions and used/expired tokens."""
import unittest
from pathlib import Path
from unittest import mock

from db import credential_cleanup as cc

ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, db):
        self.db, self.rowcount, self._row = db, 0, None

    def execute(self, sql, params=None):
        self.db.sql.append(sql)
        if sql.startswith("SELECT to_regclass"):
            name = params[0].split(".")[1]
            self._row = (name if name in self.db.tables else None,)
        elif sql.startswith("DELETE"):
            table = sql.split()[2]
            if table in self.db.fail:
                raise RuntimeError("boom")
            self.rowcount = self.db.tables[table]

    def fetchone(self):
        return self._row

    def close(self):
        pass


class FakeConn:
    def __init__(self, tables, fail=()):
        self.tables, self.fail, self.sql, self.commits, self.rollbacks = tables, set(fail), [], 0, 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class PurgeTests(unittest.TestCase):
    def test_counts_per_table_and_missing_tables_skipped(self):
        conn = FakeConn({"auth_sessions": 428, "hsf_auth_tokens": 10, "password_reset_tokens": 3})
        self.assertEqual(cc.purge_expired_credentials(conn),
                         {"auth_sessions": 428, "hsf_auth_tokens": 10, "password_reset_tokens": 3})
        self.assertEqual(conn.commits, 3)

    def test_one_failure_does_not_stop_the_rest(self):
        conn = FakeConn({"auth_sessions": 5, "hsf_auth_tokens": 2}, fail={"auth_sessions"})
        with mock.patch("builtins.print"):
            self.assertEqual(cc.purge_expired_credentials(conn), {"hsf_auth_tokens": 2})
        self.assertEqual(conn.rollbacks, 1)

    def test_only_rows_past_expiry_with_grace(self):
        sql = dict(cc.PURGES)
        self.assertIn("expires_at < NOW() - INTERVAL '1 day'", sql["auth_sessions"])
        self.assertIn("expires_at < NOW() - INTERVAL '1 day'", sql["hsf_auth_tokens"])
        self.assertIn("created_at < NOW() - INTERVAL '1 day'", sql["password_reset_tokens"])  # > the 1h rate-limit window
        self.assertIn("verified_at IS NULL", sql["email_verifications"])                     # never un-verifies anyone
        self.assertNotIn("users", " ".join(sql.values()))

    def test_cron_runs_it_daily_next_to_the_login_purge(self):
        src = (ROOT / "scheduler" / "cron_runner.py").read_text()
        self.assertIn('key = "cron_credential_purge"', src)
        self.assertLess(src.index("_purge_old_login_attempts()\n    except"), src.index("_purge_expired_credentials()\n    except"))


if __name__ == "__main__":
    unittest.main()
