"""save_daily_snapshot must be idempotent per (name, username, UTC day)."""
from __future__ import annotations

import unittest
from unittest import mock

from db import runs as R


class _Cursor:
    def __init__(self, fetchone_result):
        self._fetchone = fetchone_result
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone

    def close(self):
        return None


class _Conn:
    def __init__(self, fetchone_result):
        self.cur = _Cursor(fetchone_result)

    def cursor(self, *a, **k):
        return self.cur

    def commit(self):
        return None

    def close(self):
        return None


class SaveDailySnapshotDedupTests(unittest.TestCase):
    def _call(self, fetchone_result, **kw):
        conn = _Conn(fetchone_result)
        with mock.patch.object(R, "get_neon_conn", return_value=conn), \
             mock.patch.object(R, "ensure_neon_runs_schema", lambda c: None), \
             mock.patch.object(R, "save_run") as save_run:
            R.save_daily_snapshot("SP500", "[]", username="u@x.com", **kw)
        return conn, save_run

    def test_existing_snapshot_today_is_updated_not_inserted(self):
        # UPDATE ... RETURNING id found a row → no insert.
        conn, save_run = self._call(fetchone_result=(42,))
        save_run.assert_not_called()
        sql, _params = conn.cur.executed[0]
        self.assertIn("UPDATE runs", sql)

    def test_no_snapshot_today_falls_back_to_insert(self):
        # UPDATE affected nothing → insert via save_run as a snapshot.
        _conn, save_run = self._call(fetchone_result=None)
        save_run.assert_called_once()
        kwargs = save_run.call_args.kwargs
        self.assertEqual(kwargs["label"], "daily_snapshot")
        self.assertTrue(kwargs["is_snapshot"])

    def test_null_username_uses_is_null_clause(self):
        conn = _Conn((7,))
        with mock.patch.object(R, "get_neon_conn", return_value=conn), \
             mock.patch.object(R, "ensure_neon_runs_schema", lambda c: None), \
             mock.patch.object(R, "save_run"):
            R.save_daily_snapshot("SP500", "[]", username=None)
        sql, _params = conn.cur.executed[0]
        self.assertIn("username IS NULL", sql)

    def test_db_error_falls_back_to_insert(self):
        with mock.patch.object(R, "get_neon_conn", side_effect=RuntimeError("down")), \
             mock.patch.object(R, "save_run") as save_run:
            R.save_daily_snapshot("SP500", "[]", username="u")
        save_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
