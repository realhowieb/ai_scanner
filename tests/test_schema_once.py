"""P1-44: schema setup runs once per process; warm connections have a connect timeout."""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from db import engine

ROOT = Path(__file__).resolve().parents[1]


class _Info:
    def __init__(self, host="db.example", port=5432, dbname="app"):
        self.host, self.port, self.dbname = host, port, dbname


class _PgConn:
    def __init__(self, **info):
        self.info = _Info(**info)


class SchemaOnceTests(unittest.TestCase):
    def setUp(self):
        engine._schema_done.clear()
        self.calls = []

        @engine.schema_once
        def ensure(conn, flag=None):
            self.calls.append(flag)

        self.ensure = ensure

    def test_runs_once_per_database(self):
        self.ensure(_PgConn(), 1)
        self.ensure(_PgConn(), 2)
        self.ensure(_PgConn(dbname="branch"), 3)  # another database runs it again
        self.assertEqual(self.calls, [1, 3])

    def test_connections_without_info_always_run(self):
        for conn in (object(), mock.MagicMock()):  # SQLite / test fakes
            self.ensure(conn)
            self.ensure(conn)
        self.assertEqual(len(self.calls), 4)

    def test_failure_runs_again_next_time(self):
        attempts = []

        @engine.schema_once
        def flaky(conn):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("lock timeout")

        with self.assertRaises(RuntimeError):
            flaky(_PgConn())
        flaky(_PgConn())
        flaky(_PgConn())
        self.assertEqual(len(attempts), 2)

    def test_env_switch_restores_every_call(self):
        with mock.patch.dict("os.environ", {"AI_SCANNER_SCHEMA_ONCE": "0"}):
            self.ensure(_PgConn())
            self.ensure(_PgConn())
        self.assertEqual(len(self.calls), 2)


class WarmConnectTimeoutTests(unittest.TestCase):
    def test_warm_connection_has_a_connect_timeout(self):
        captured = {}
        mod = types.ModuleType("psycopg")
        mod.rows = types.SimpleNamespace(dict_row=object())

        def connect(url, row_factory=None, **kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        mod.connect = connect
        engine._pool_local.conn = None
        env = {"NEON_DATABASE_URL": "postgres://x", "AI_SCANNER_DB_POOL": "1", "DB_CONNECT_TIMEOUT": "7"}
        with mock.patch.dict("os.environ", env), mock.patch.dict(sys.modules, {"psycopg": mod}):
            engine.get_neon_conn()
        engine._pool_local.conn = None
        self.assertEqual(captured.get("connect_timeout"), 7)


class HotPathCoverageTests(unittest.TestCase):
    """Every request-path ensure-schema helper must be wrapped."""

    REQUIRED = {
        "ui/auth_sessions.py": ["ensure_auth_sessions_schema"],
        "db/schema.py": [
            "ensure_neon_users_schema", "ensure_neon_login_attempts_schema",
            "ensure_neon_watchlists_schema", "ensure_neon_runs_schema", "ensure_neon_scan_errors_schema",
        ],
        "db/alerts.py": ["_ensure_alerts_schema"],
        "db/user_settings.py": ["_create_user_settings_schema"],
        "db/ai_usage.py": ["_ensure_schema"],
        "db/trades.py": ["_ensure_schema"],
        "db/paper_trading.py": ["_ensure_schema"],
        "db/intelligence_alerts.py": ["_ensure_schema", "_ensure_quality_schema"],
    }

    def test_hot_paths_are_wrapped(self):
        for rel, names in self.REQUIRED.items():
            tree = ast.parse((ROOT / rel).read_text())
            decorated = {
                n.name for n in tree.body
                if isinstance(n, ast.FunctionDef) and any(ast.unparse(d) == "schema_once" for d in n.decorator_list)
            }
            for name in names:
                self.assertIn(name, decorated, f"{rel}:{name}")


if __name__ == "__main__":
    unittest.main()
