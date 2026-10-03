"""P2-41: maturation saves each symbol's outcomes in one transaction.

Same write rule as before (INSERT ... ON CONFLICT DO NOTHING, first write wins,
never rewritten); a failed batch rolls back and is retried row by row so an
error is never counted as already written.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from db import hsf_observations as store
from scripts import mature_observations as worker
from tests.test_maturation_time_budget import NOW, TICKERS, _fetch, _observations

PG_URL = os.environ.get("HSF_TEST_PG_URL")  # optional: a throwaway local Postgres


def _outcome(oid, horizon, ret=1.0):
    return {"observation_id": oid, "horizon": horizon, "schema_version": "t", "return_pct": ret}


def _sqlite():
    path = Path(tempfile.mkdtemp()) / "obs.sqlite"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


class BatchStoreSqliteTests(unittest.TestCase):
    def setUp(self):
        self.conn = _sqlite()

    def records(self):
        cur = self.conn.execute("SELECT observation_id, horizon, record FROM hsf_observation_outcomes")
        return {(r[0], r[1]): json.loads(r[2]) for r in cur.fetchall()}

    def test_new_rows_then_repeats_are_already_written(self):
        batch = [_outcome("o1", "+5m"), _outcome("o1", "+15m"), _outcome("o2", "+5m")]
        self.assertEqual(store.save_outcomes_batch(batch, conn=self.conn), [True, True, True])
        self.assertEqual(store.save_outcomes_batch(batch, conn=self.conn), [False, False, False])
        self.assertEqual(len(self.records()), 3)

    def test_first_write_wins_and_is_never_rewritten(self):
        store.save_outcomes_batch([_outcome("o1", "+5m", ret=1.0)], conn=self.conn)
        res = store.save_outcomes_batch([_outcome("o1", "+5m", ret=9.9), _outcome("o1", "+30m")], conn=self.conn)
        self.assertEqual(res, [False, True])
        self.assertEqual(self.records()[("o1", "+5m")]["return_pct"], 1.0)

    def test_repeat_inside_one_batch_counts_once(self):
        res = store.save_outcomes_batch([_outcome("o1", "+5m"), _outcome("o1", "+5m", ret=2.0)], conn=self.conn)
        self.assertEqual(res, [True, False])
        self.assertEqual(self.records()[("o1", "+5m")]["return_pct"], 1.0)

    def test_rows_without_a_key_are_skipped(self):
        res = store.save_outcomes_batch([{"horizon": "+5m"}, _outcome("o1", "+5m")], conn=self.conn)
        self.assertEqual(res, [False, True])


class _FailingCursor:
    def execute(self, *_a, **_k):
        raise RuntimeError("connection lost")

    def close(self):
        pass


class _FailingConn:
    """Looks like Postgres (not sqlite3) and fails on the INSERT."""

    def __init__(self):
        self.rolled_back = self.committed = False

    def cursor(self):
        return _FailingCursor()

    def rollback(self):
        self.rolled_back = True

    def commit(self):
        self.committed = True


class BatchErrorTests(unittest.TestCase):
    def test_error_rolls_back_and_raises(self):
        conn = _FailingConn()
        with self.assertRaises(Exception):
            store.save_outcomes_batch([_outcome("o1", "+5m")], conn=conn)
        self.assertTrue(conn.rolled_back)
        self.assertFalse(conn.committed)


class WorkerBatchingTests(unittest.TestCase):
    def run_worker(self, save_fn):
        return worker.mature_observations(_observations(), now=NOW, fetch_bars_batch=_fetch, save_fn=save_fn)

    def test_one_batch_per_symbol(self):
        calls = []

        class Saver:
            def __call__(self, o):  # per-row path must not be used
                raise AssertionError("per-row save used")

            def save_many(self, outcomes):
                calls.append([o["observation_id"] for o in outcomes])
                return [True] * len(outcomes)

        r = self.run_worker(Saver())
        self.assertEqual(len(calls), len(TICKERS))  # one transaction per symbol
        self.assertEqual(r["attached"], sum(len(c) for c in calls))
        self.assertGreater(r["attached"], 0)

    def test_failed_batch_falls_back_row_by_row(self):
        rows = []

        class Saver:
            def __call__(self, o):
                rows.append(o)
                return True

            def save_many(self, outcomes):
                raise RuntimeError("batch failed")

        r = self.run_worker(Saver())
        self.assertEqual(r["attached"], len(rows))
        self.assertGreater(len(rows), 0)
        self.assertNotIn("DATABASE_ERROR", r["failures"])

    def test_row_errors_after_a_failed_batch_are_database_errors_not_already(self):
        class Saver:
            def __call__(self, o):
                raise RuntimeError("down")

            def save_many(self, outcomes):
                raise RuntimeError("down")

        r = self.run_worker(Saver())
        self.assertEqual(r["attached"], 0)
        self.assertGreater(r["failures"].get("DATABASE_ERROR", 0), 0)
        self.assertEqual(sum(h["already"] for h in r["horizons"].values()), 0)

    def test_run_saver_end_to_end_on_sqlite(self):
        conn = _sqlite()
        saver = worker._RunSaver(connect=lambda: conn)
        first = self.run_worker(saver)
        second = self.run_worker(saver)
        self.assertGreater(first["attached"], 0)
        self.assertEqual(second["attached"], 0)
        n = conn.execute("SELECT count(*) FROM hsf_observation_outcomes").fetchone()[0]
        self.assertEqual(n, first["attached"])


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class BatchStorePostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        self.conn = psycopg.connect(PG_URL, row_factory=psycopg.rows.dict_row)
        self.conn.execute("DROP TABLE IF EXISTS hsf_observation_outcomes")
        self.conn.commit()
        store._ensure_schema.__wrapped__(self.conn, False)

    def tearDown(self):
        self.conn.close()

    def test_postgres_batch_one_commit_and_first_write_wins(self):
        batch = [_outcome("o1", "+5m"), _outcome("o1", "+15m"), _outcome("o1", "+5m", ret=7.0)]
        self.assertEqual(store.save_outcomes_batch(batch, conn=self.conn), [True, True, False])
        res = store.save_outcomes_batch([_outcome("o1", "+5m", ret=5.0), _outcome("o2", "+5m")], conn=self.conn)
        self.assertEqual(res, [False, True])
        rec = self.conn.execute("SELECT record FROM hsf_observation_outcomes "
                                "WHERE observation_id='o1' AND horizon='+5m'").fetchone()["record"]
        self.assertEqual(rec["return_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
