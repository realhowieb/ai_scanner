"""P2-82: account deletion (DELETE /v1/me and Settings), App Store requirement."""
import os
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

PG_URL = os.environ.get("HSF_TEST_PG_URL")


class BlockerTests(unittest.TestCase):
    def test_rules(self):
        from db.account_deletion import blocker

        self.assertIsNone(blocker({"tier": "basic"}))
        self.assertIsNone(blocker({"tier": "basic", "stripe_subscription_id": "sub_old"}))   # cancelled -> Free
        self.assertIsNone(blocker({"tier": "pro", "stripe_subscription_id": None}))          # comped plan
        self.assertIn("Cancel your subscription", blocker({"tier": "premium", "stripe_subscription_id": "sub_1"}))
        self.assertIn("Admin", blocker({"tier": "basic", "is_admin": True}))


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class DeleteMeRouteTests(ApiTestCase):
    def test_password_confirm_and_blocked(self):
        h = self.auth(self.login().json()["access_token"])
        url = "/v1/me"
        with mock.patch("api.store._conn", return_value=mock.MagicMock()), \
                mock.patch("db.account_deletion.delete_account", return_value={"users": 1}) as delete:
            r = self.client.request("DELETE", url, headers=h, json={"password": "wrong", "confirm": "DELETE"})
            self.assertEqual(r.status_code, 400)
            r = self.client.request("DELETE", url, headers=h, json={"password": "right pw", "confirm": "yes"})
            self.assertEqual(r.status_code, 422)
            delete.assert_not_called()
            r = self.client.request("DELETE", url, headers=h, json={"password": "right pw", "confirm": "DELETE"})
            self.assertEqual(r.status_code, 204, r.text)
            self.assertEqual(delete.call_args.args[1], "pro@example.com")
        from db.account_deletion import DeletionBlocked

        with mock.patch("api.store._conn", return_value=mock.MagicMock()), \
                mock.patch("db.account_deletion.delete_account", side_effect=DeletionBlocked("Cancel your subscription first")):
            r = self.client.request("DELETE", url, headers=h, json={"password": "right pw", "confirm": "DELETE"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("Cancel", r.json()["detail"])
        self.assertEqual(self.client.request("DELETE", url, json={"password": "x", "confirm": "DELETE"}).status_code, 401)


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class DeletionPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        self.conn = psycopg.connect(PG_URL)
        self.addCleanup(self.conn.close)
        tables = ["users", "watchlists", "watchlist_items", "user_alerts", "user_trades", "api_push_devices",
                  "auth_sessions", "runs", "alpaca_paper_accounts"]

        def drop():   # these are minimal stand-ins; other tests create the real schemas
            self.conn.rollback()
            with self.conn.cursor() as c:
                for t in tables:
                    c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            self.conn.commit()
        self.addCleanup(drop)
        with self.conn.cursor() as c:
            for t in tables:
                c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")   # test-only fixed names
            c.execute("CREATE TABLE users (username TEXT PRIMARY KEY, tier TEXT, is_admin BOOLEAN DEFAULT FALSE, "
                      "stripe_subscription_id TEXT, password TEXT)")
            c.execute("CREATE TABLE watchlists (id SERIAL PRIMARY KEY, user_id TEXT, name TEXT)")
            c.execute("CREATE TABLE watchlist_items (id SERIAL PRIMARY KEY, watchlist_id INT, ticker TEXT)")
            c.execute("CREATE TABLE user_alerts (id SERIAL PRIMARY KEY, user_id TEXT)")
            c.execute("CREATE TABLE user_trades (id SERIAL PRIMARY KEY, user_id TEXT)")
            c.execute("CREATE TABLE api_push_devices (id SERIAL PRIMARY KEY, username TEXT)")
            c.execute("CREATE TABLE auth_sessions (username TEXT)")
            c.execute("CREATE TABLE runs (id SERIAL PRIMARY KEY, username TEXT)")
            c.execute("CREATE TABLE alpaca_paper_accounts (user_id TEXT PRIMARY KEY)")
            for u, tier, sub in (("gone@example.com", "basic", None), ("keep@example.com", "pro", None),
                                 ("paying@example.com", "premium", "sub_1")):
                c.execute("INSERT INTO users (username, tier, stripe_subscription_id) VALUES (%s, %s, %s)", (u, tier, sub))
                wid = c.execute("INSERT INTO watchlists (user_id, name) VALUES (%s, 'l') RETURNING id", (u,)).fetchone()[0]
                c.execute("INSERT INTO watchlist_items (watchlist_id, ticker) VALUES (%s, 'AAPL')", (wid,))
                for t, col in (("user_alerts", "user_id"), ("user_trades", "user_id"), ("api_push_devices", "username"),
                               ("auth_sessions", "username"), ("runs", "username"), ("alpaca_paper_accounts", "user_id")):
                    c.execute(f"INSERT INTO {t} ({col}) VALUES (%s)", (u.upper() if t == "runs" else u,))
        self.conn.commit()

    def count(self, table, col, user):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE lower({col}) = %s", (user,)).fetchone()[0]

    def test_deletes_only_that_account(self):
        from db.account_deletion import delete_account

        counts = delete_account(self.conn, "Gone@Example.com")
        self.assertEqual(counts["users"], 1)
        self.assertEqual(counts["runs"], 1)                       # case-insensitive match
        for t, col in (("users", "username"), ("watchlists", "user_id"), ("user_alerts", "user_id"),
                       ("user_trades", "user_id"), ("api_push_devices", "username"), ("auth_sessions", "username"),
                       ("alpaca_paper_accounts", "user_id")):
            self.assertEqual(self.count(t, col, "gone@example.com"), 0, t)
            self.assertEqual(self.count(t, col, "keep@example.com"), 1, t)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM watchlist_items").fetchone()[0], 2)

    def test_paid_subscription_blocks_and_changes_nothing(self):
        from db.account_deletion import DeletionBlocked, delete_account

        with self.assertRaises(DeletionBlocked):
            delete_account(self.conn, "paying@example.com")
        self.assertEqual(self.count("users", "username", "paying@example.com"), 1)
        self.assertEqual(self.count("user_alerts", "user_id", "paying@example.com"), 1)
        with self.assertRaises(DeletionBlocked):
            delete_account(self.conn, "nobody@example.com")


if __name__ == "__main__":
    unittest.main()
