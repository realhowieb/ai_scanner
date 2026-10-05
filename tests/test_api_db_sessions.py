"""API acceptance run: worker threads must not sit idle in a transaction.

The warm per-thread connection pool (db.engine) used to keep each read
transaction open until the thread's next request, holding table locks; an
ALTER TABLE from any other process then waited, and every query queued behind
it stalled. Needs a throwaway Postgres (HSF_TEST_PG_URL).
"""
import importlib.util
import os
import threading
import unittest
from unittest import mock

PG_URL = os.environ.get("HSF_TEST_PG_URL")
DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "jwt", "bcrypt", "psycopg"))


@unittest.skipUnless(PG_URL and DEPS, "set HSF_TEST_PG_URL to a throwaway Postgres (and install the API deps)")
class NoIdleTransactionsTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        from fastapi.testclient import TestClient

        from api import main
        from api.settings import Settings
        from db.alerts import _ensure_alerts_schema
        from db.schema import ensure_neon_watchlists_schema

        for k, v in (("DATABASE_URL", PG_URL), ("AI_SCANNER_DB_POOL", "1")):
            mock.patch.dict(os.environ, {k: v}).start()
        self.addCleanup(mock.patch.stopall)
        with psycopg.connect(PG_URL) as conn:
            conn.execute("SET lock_timeout = '5s'")  # fail, don't hang, if an idle connection holds a lock
            ensure_neon_watchlists_schema.__wrapped__(conn)
            _ensure_alerts_schema.__wrapped__(conn)
            conn.execute("DELETE FROM watchlists WHERE user_id = 'idle-test'")
            conn.execute("DELETE FROM user_alerts WHERE user_id = 'idle-test'")
        app = main.create_app(Settings(jwt_secret="s" * 40, access_ttl_s=900, refresh_ttl_s=3600, cors_origins=()))
        account = {"username": "idle-test", "tier": "pro", "is_admin": False, "is_active": True}
        app.dependency_overrides[main.current_account] = lambda: account
        self.client = TestClient(app)

    def idle_in_transaction(self):
        import psycopg

        with psycopg.connect(PG_URL, autocommit=True) as conn:
            return conn.execute("SELECT count(*) FROM pg_stat_activity WHERE state LIKE 'idle in transaction%%' "
                                "AND pid <> pg_backend_pid()").fetchone()[0]

    def test_requests_leave_no_open_transaction(self):
        wid = self.client.post("/v1/watchlists", json={"name": "Idle"}).json()["id"]
        self.client.post(f"/v1/watchlists/{wid}/tickers", json={"tickers": ["NVDA"]})
        self.client.get("/v1/watchlists")
        self.client.get(f"/v1/watchlists/{wid}")
        self.client.post("/v1/alerts", json={"type": "move", "ticker": "NVDA", "threshold": 999})
        self.client.get("/v1/alerts")
        self.assertEqual(self.idle_in_transaction(), 0)

    def test_schema_change_elsewhere_is_not_blocked(self):
        import psycopg

        self.client.get("/v1/watchlists")
        self.client.get("/v1/alerts")
        done = threading.Event()

        def alter():
            with psycopg.connect(PG_URL, autocommit=True) as conn:
                conn.execute("SET lock_timeout = '3s'")
                conn.execute("ALTER TABLE watchlists ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE")
            done.set()

        t = threading.Thread(target=alter)
        t.start()
        t.join(10)
        self.assertTrue(done.is_set(), "ALTER TABLE waited on a lock held by an idle API connection")

    def test_pool_close_ends_the_transaction(self):
        from db.engine import get_neon_conn

        conn = get_neon_conn()
        conn.cursor().execute("SELECT 1 FROM watchlists LIMIT 1")
        conn.close()
        self.assertEqual(self.idle_in_transaction(), 0)



class EarningsConnectionTests(unittest.TestCase):
    """db.earnings read helpers closed nothing: one leaked connection per stock page."""

    def _conn(self, row=("NVDA", None)):
        cur = mock.MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchall.return_value = [row]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        return conn

    def test_opened_connection_is_closed(self):
        from db import earnings

        for fn, row in ((earnings.load_earnings_map, ("NVDA", None)),
                        (earnings.load_earnings_details_map, ("NVDA", None, "AMC"))):
            conn = self._conn(row)
            with mock.patch.object(earnings, "_get_conn", return_value=conn):
                fn(["NVDA"])
            conn.close.assert_called_once()

    def test_injected_connection_is_left_open(self):
        from db import earnings

        conn = self._conn()
        earnings.load_earnings_map(["NVDA"], conn=conn)
        conn.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
