"""API runtime limits: request thread cap and the per-query database time limit."""
import os
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.settings import Settings
    DEPS = True
except Exception:  # pragma: no cover
    DEPS = False


@unittest.skipUnless(DEPS, "needs FastAPI")
class ThreadCapTests(unittest.TestCase):
    def _limit_after_start(self, env):
        import anyio.to_thread

        seen = {}
        with mock.patch.dict(os.environ, env):
            app = create_app(Settings(jwt_secret="x" * 40, access_ttl_s=900, refresh_ttl_s=86400, cors_origins=()))

            @app.get("/_threads")
            async def _threads():
                seen["n"] = anyio.to_thread.current_default_thread_limiter().total_tokens
                return {}

            with TestClient(app) as c:
                c.get("/_threads")
        return seen["n"]

    def test_default_cap_is_24(self):
        self.assertEqual(self._limit_after_start({"HSF_API_THREADS": ""}), 24)

    def test_cap_follows_env(self):
        self.assertEqual(self._limit_after_start({"HSF_API_THREADS": "8"}), 8)


class StatementTimeoutTests(unittest.TestCase):
    def test_off_unless_set(self):
        from db import engine

        with mock.patch.dict(os.environ, {"HSF_DB_STATEMENT_TIMEOUT_MS": ""}):
            self.assertEqual(engine.statement_timeout_ms(), 0)
        with mock.patch.dict(os.environ, {"HSF_DB_STATEMENT_TIMEOUT_MS": "30000"}):
            self.assertEqual(engine.statement_timeout_ms(), 30000)
        with mock.patch.dict(os.environ, {"HSF_DB_STATEMENT_TIMEOUT_MS": "junk"}):
            self.assertEqual(engine.statement_timeout_ms(), 0)

    def test_new_transaction_carries_the_limit_in_one_round_trip(self):
        from db import engine

        conn = mock.Mock()
        with mock.patch.dict(os.environ, {"HSF_DB_STATEMENT_TIMEOUT_MS": "30000"}):
            engine._fresh_transaction(conn)
        conn.execute.assert_called_once_with("ROLLBACK; BEGIN; SET LOCAL statement_timeout = 30000")
        conn.rollback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
