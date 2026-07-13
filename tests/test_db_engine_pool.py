"""Tests for warm per-thread Neon connection reuse in db.engine."""
import sys
import types
import unittest
from unittest import mock

from db import engine


class FakeConn:
    def __init__(self):
        self.rollbacks = 0
        self.closed = False

    def rollback(self):
        if self.closed:
            raise RuntimeError("connection is dead")
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _fake_psycopg(conns):
    mod = types.ModuleType("psycopg")
    mod.rows = types.SimpleNamespace(dict_row=object())

    def connect(url, row_factory=None):
        return conns.pop(0)

    mod.connect = connect
    return mod


class WarmConnTests(unittest.TestCase):
    def setUp(self):
        engine._pool_local.conn = None

    def _env(self):
        return mock.patch.dict(
            "os.environ", {"NEON_DATABASE_URL": "postgres://x", "AI_SCANNER_DB_POOL": "1"}
        )

    def test_connection_reused_and_close_is_noop(self):
        real = FakeConn()
        with self._env(), mock.patch.dict(sys.modules, {"psycopg": _fake_psycopg([real])}):
            c1 = engine.get_neon_conn()
            c1.close()  # no-op: keeps the underlying connection warm
            self.assertFalse(real.closed)
            c2 = engine.get_neon_conn()
        # Second checkout validated the same underlying connection via rollback.
        self.assertEqual(real.rollbacks, 1)
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
