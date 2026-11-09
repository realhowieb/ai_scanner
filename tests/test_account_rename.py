"""P2-36: Admin "Change email" moves an account to a new address across every table."""
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

PG_URL = os.environ.get("HSF_TEST_PG_URL")
ROOT = Path(__file__).resolve().parent.parent
_ACCOUNT_COLUMNS = ("username", "user_id", "user_email", "referrer")


class CoverageTests(unittest.TestCase):
    def test_every_account_table_is_handled(self):
        """A new table with an account column must be added to the rename (or KEEP) lists."""
        from db.account_rename import CLEAR_TABLES, KEEP_TABLES, RENAME_TABLES

        handled = {t for t, _ in RENAME_TABLES + CLEAR_TABLES} | set(KEEP_TABLES) | {"users"}
        found = set()
        create = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\)\s*[\"']{0,3}\s*[,)]?", re.S)
        for path in list((ROOT / "db").glob("*.py")) + list((ROOT / "api").glob("*.py")) + \
                list((ROOT / "services").glob("*.py")) + list((ROOT / "ui").glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in create.finditer(text):
                cols = m.group(2)
                if re.search(r"\b(%s)\s+text\b" % "|".join(_ACCOUNT_COLUMNS), cols, re.I):
                    found.add(m.group(1))
        missing = sorted(found - handled)
        self.assertEqual(missing, [], "add these to db/account_rename.py RENAME_TABLES, CLEAR_TABLES or KEEP_TABLES")


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class ChangeEmailRouteTests(ApiTestCase):
    def test_admin_only_and_self_blocked(self):
        h = self.auth(self.login().json()["access_token"])   # pro, not admin
        body = {"username": "x@example.com", "new_email": "y@example.com"}
        self.assertEqual(self.client.post("/v1/admin/users/email", headers=h, json=body).status_code, 403)
        self.assertEqual(self.client.post("/v1/admin/users/email/preview", headers=h, json=body).status_code, 403)

        from api.admin import AdminError, change_email, preview_email_change

        with self.assertRaises(AdminError):
            change_email("boss@example.com", "Boss@example.com", "new@example.com", False)
        with self.assertRaises(AdminError):
            preview_email_change("boss@example.com", "boss@example.com", "new@example.com")

    def test_change_reports_and_updates_stripe(self):
        from api import admin

        out = {"old_email": "a@example.com", "new_email": "b@example.com", "moved": {"users": 1, "watchlists": 2},
               "cleared": {}, "stripe_customer_id": "cus_1"}
        with mock.patch("api.admin._rename_conn", return_value=mock.MagicMock()), \
                mock.patch("db.account_rename.rename_account", return_value=out), \
                mock.patch("db.admin_audit.record_admin_event") as audit, \
                mock.patch("api.admin._stripe_email", return_value=" Stripe receipts now go to the new address.") as stripe:
            res = admin.change_email("boss@example.com", "a@example.com", "b@example.com", False)
        self.assertIn("3 rows", res["message"])
        self.assertIn("needs verifying", res["message"])
        stripe.assert_called_once_with("cus_1", "b@example.com")
        self.assertEqual(audit.call_args.args[1], "change_email")


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class RenamePostgresTests(unittest.TestCase):
    TABLES = ["hsf_workspace_preferences", "watchlists", "user_alerts", "auth_sessions", "api_refresh_tokens",
              "hsf_referrals", "runs", "users"]

    def setUp(self):
        import psycopg

        self.conn = psycopg.connect(PG_URL)
        self.addCleanup(self.conn.close)

        def drop():
            self.conn.rollback()
            with self.conn.cursor() as c:
                for t in self.TABLES:
                    c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")   # test-only fixed names
            self.conn.commit()
        drop()
        self.addCleanup(drop)
        from db.referrals import referred_key

        with self.conn.cursor() as c:
            c.execute("CREATE TABLE users (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, tier TEXT, "
                      "is_admin BOOLEAN DEFAULT FALSE, stripe_customer_id TEXT, email_verified BOOLEAN DEFAULT TRUE)")
            c.execute("CREATE TABLE hsf_workspace_preferences (user_id TEXT PRIMARY KEY REFERENCES users(username) "
                      "ON DELETE CASCADE, data TEXT)")
            c.execute("CREATE TABLE watchlists (id SERIAL PRIMARY KEY, user_id TEXT, name TEXT)")
            c.execute("CREATE TABLE user_alerts (id SERIAL PRIMARY KEY, user_id TEXT)")
            c.execute("CREATE TABLE runs (id SERIAL PRIMARY KEY, username TEXT)")
            c.execute("CREATE TABLE auth_sessions (username TEXT)")
            c.execute("CREATE TABLE api_refresh_tokens (username TEXT)")
            c.execute("CREATE TABLE hsf_referrals (id SERIAL PRIMARY KEY, referrer TEXT, referred TEXT UNIQUE)")
            for u in ("old@example.com", "other@example.com"):
                c.execute("INSERT INTO users (username, tier) VALUES (%s, 'pro')", (u,))
                c.execute("INSERT INTO hsf_workspace_preferences VALUES (%s, '{}')", (u,))
                c.execute("INSERT INTO watchlists (user_id, name) VALUES (%s, 'a'), (%s, 'b')", (u, u))
                c.execute("INSERT INTO user_alerts (user_id) VALUES (%s)", (u,))
                c.execute("INSERT INTO runs (username) VALUES (%s)", (u.upper(),))
                c.execute("INSERT INTO auth_sessions VALUES (%s)", (u,))
                c.execute("INSERT INTO api_refresh_tokens VALUES (%s)", (u,))
            c.execute("INSERT INTO hsf_referrals (referrer, referred) VALUES ('other@example.com', %s)",
                      (referred_key("old@example.com"),))
        self.conn.commit()

    def count(self, table, col, user):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE lower({col}) = %s", (user,)).fetchone()[0]

    def test_preview_then_rename(self):
        from db.account_rename import preview_rename, rename_account
        from db.referrals import referred_key

        p = preview_rename(self.conn, "Old@Example.com", "New@Example.com")
        self.assertEqual(p["problems"], [])
        self.assertEqual(p["move"]["watchlists"], 2)
        self.assertEqual(p["clear"], {"auth_sessions": 1, "api_refresh_tokens": 1})
        self.assertEqual(self.count("users", "username", "old@example.com"), 1)   # preview wrote nothing

        old_id = self.conn.execute("SELECT id FROM users WHERE username = 'old@example.com'").fetchone()[0]
        out = rename_account(self.conn, "old@example.com", "new@example.com")
        self.assertEqual(out["moved"]["users"], 1)
        self.assertEqual(self.conn.execute("SELECT id, email_verified FROM users WHERE username = 'new@example.com'")
                         .fetchone(), (old_id, False))
        for t, col, n in (("hsf_workspace_preferences", "user_id", 1), ("watchlists", "user_id", 2),
                          ("user_alerts", "user_id", 1), ("runs", "username", 1)):
            self.assertEqual(self.count(t, col, "new@example.com"), n, t)
            self.assertEqual(self.count(t, col, "old@example.com"), 0, t)
            self.assertEqual(self.count(t, col, "other@example.com"), n, t)
        for t in ("auth_sessions", "api_refresh_tokens"):
            self.assertEqual(self.count(t, "username", "old@example.com"), 0, t)
            self.assertEqual(self.count(t, "username", "other@example.com"), 1, t)
        self.assertEqual(self.conn.execute("SELECT referred FROM hsf_referrals").fetchone()[0], referred_key("new@example.com"))

    def test_blocked_changes_nothing(self):
        from db.account_rename import RenameBlocked, preview_rename, rename_account

        self.assertIn("already exists", preview_rename(self.conn, "old@example.com", "other@example.com")["problems"][0])
        with self.assertRaises(RenameBlocked):
            rename_account(self.conn, "old@example.com", "other@example.com")
        with self.assertRaises(RenameBlocked):
            rename_account(self.conn, "old@example.com", "not-an-email")
        with self.assertRaises(RenameBlocked):
            rename_account(self.conn, "nobody@example.com", "x@example.com")
        self.conn.execute("INSERT INTO watchlists (user_id, name) VALUES ('stale@example.com', 'z')")
        self.conn.commit()
        self.assertIn("watchlists", preview_rename(self.conn, "old@example.com", "stale@example.com")["problems"][0])
        with self.assertRaises(RenameBlocked):
            rename_account(self.conn, "old@example.com", "stale@example.com")
        self.assertEqual(self.count("watchlists", "user_id", "old@example.com"), 2)
        self.assertEqual(self.count("auth_sessions", "username", "old@example.com"), 1)

    def test_keep_verified(self):
        from db.account_rename import rename_account

        rename_account(self.conn, "old@example.com", "new@example.com", keep_verified=True)
        self.assertTrue(self.conn.execute("SELECT email_verified FROM users WHERE username = 'new@example.com'").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
