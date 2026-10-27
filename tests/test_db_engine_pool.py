"""Tests for warm per-thread Neon connection reuse in db.engine."""
import sys
import types
import unittest
from unittest import mock

from db import engine


class FakeConn:
    """Like psycopg: rollback() with no open transaction never touches the socket, so
    a connection the server dropped only fails on the next query."""

    def __init__(self):
        self.rollbacks = 0
        self.pings = 0
        self.closed = False
        self.server_gone = False

    def rollback(self):
        if self.closed:
            raise RuntimeError("connection is closed")
        self.rollbacks += 1

    def execute(self, sql):
        if self.server_gone:
            raise RuntimeError("server closed the connection unexpectedly")
        self.pings += 1

    def close(self):
        self.closed = True


def _fake_psycopg(conns):
    mod = types.ModuleType("psycopg")
    mod.rows = types.SimpleNamespace(dict_row=object())

    def connect(url, row_factory=None, **kwargs):  # accepts connect_timeout (Run 30)
        return conns.pop(0)

    mod.connect = connect
    return mod


class WarmConnTests(unittest.TestCase):
    def setUp(self):
        engine._pool_local.conn = None

    def _env(self, ping_s="30"):
        return mock.patch.dict(
            "os.environ", {"NEON_DATABASE_URL": "postgres://x", "AI_SCANNER_DB_POOL": "1",
                           "DB_WARM_PING_S": ping_s}
        )

    def test_connection_reused_and_close_ends_the_transaction(self):
        real = FakeConn()
        with self._env(), mock.patch.dict(sys.modules, {"psycopg": _fake_psycopg([real])}):
            c1 = engine.get_neon_conn()
            c1.close()  # keeps the socket warm but ends the transaction (no idle-in-transaction locks)
            self.assertFalse(real.closed)
            self.assertEqual(real.rollbacks, 1)
            c2 = engine.get_neon_conn()
        # Second checkout validated the same underlying connection via rollback.
        self.assertEqual(real.rollbacks, 2)
        self.assertIs(
            object.__getattribute__(c1, "_conn"), object.__getattribute__(c2, "_conn")
        )

    def test_dead_connection_replaced(self):
        dead, fresh = FakeConn(), FakeConn()
        with self._env(), mock.patch.dict(
            sys.modules, {"psycopg": _fake_psycopg([dead, fresh])}
        ):
            engine.get_neon_conn()
            dead.closed = True  # simulate Neon idle-timeout kill
            c2 = engine.get_neon_conn()
        self.assertIs(object.__getattribute__(c2, "_conn"), fresh)

    def test_idle_connection_dropped_by_server_is_replaced(self):
        # Neon suspended the idle compute: rollback() still succeeds, only a query fails.
        dropped, fresh = FakeConn(), FakeConn()
        with self._env(), mock.patch.dict(
            sys.modules, {"psycopg": _fake_psycopg([dropped, fresh])}
        ), mock.patch.object(engine._time, "monotonic", side_effect=[1000.0, 1100.0]):
            engine.get_neon_conn()
            dropped.server_gone = True
            c2 = engine.get_neon_conn()
        self.assertTrue(dropped.closed)
        self.assertIs(object.__getattribute__(c2, "_conn"), fresh)

    def test_recently_used_connection_is_not_pinged(self):
        real = FakeConn()
        with self._env(), mock.patch.dict(
            sys.modules, {"psycopg": _fake_psycopg([real])}
        ), mock.patch.object(engine._time, "monotonic", side_effect=[1000.0, 1005.0, 1040.0]):
            engine.get_neon_conn()
            engine.get_neon_conn()  # 5 s later: no extra round trip
            self.assertEqual(real.pings, 0)
            engine.get_neon_conn()  # 35 s after last use: pinged once
        self.assertEqual(real.pings, 1)

    def test_pool_disabled_returns_raw_connection(self):
        real = FakeConn()
        with mock.patch.dict(
            "os.environ", {"NEON_DATABASE_URL": "postgres://x", "AI_SCANNER_DB_POOL": "0"}
        ), mock.patch.dict(sys.modules, {"psycopg": _fake_psycopg([real])}):
            conn = engine.get_neon_conn()
        self.assertIs(conn, real)  # raw psycopg conn, close() is real

    def test_attribute_delegation(self):
        real = FakeConn()
        real.custom = "value"
        with self._env(), mock.patch.dict(sys.modules, {"psycopg": _fake_psycopg([real])}):
            conn = engine.get_neon_conn()
        self.assertEqual(conn.custom, "value")


if __name__ == "__main__":
    unittest.main()


class WarmConnContextManagerTests(unittest.TestCase):
    def setUp(self):
        engine._pool_local.conn = None

    def _conn(self, real):
        with mock.patch.dict(
            "os.environ", {"NEON_DATABASE_URL": "postgres://x", "AI_SCANNER_DB_POOL": "1"}
        ), mock.patch.dict(sys.modules, {"psycopg": _fake_psycopg([real])}):
            return engine.get_neon_conn()

    def test_with_block_commits_and_keeps_connection_open(self):
        real = FakeConn()
        real.commits = 0
        real.commit = lambda: setattr(real, "commits", real.commits + 1)
        conn = self._conn(real)
        with conn:
            pass
        self.assertEqual(real.commits, 1)
        self.assertFalse(real.closed)  # psycopg's own __exit__ would close it

    def test_with_block_rolls_back_on_exception(self):
        real = FakeConn()
        conn = self._conn(real)
        before = real.rollbacks
        with self.assertRaises(ValueError):
            with conn:
                raise ValueError("boom")
        self.assertEqual(real.rollbacks, before + 1)
        self.assertFalse(real.closed)
