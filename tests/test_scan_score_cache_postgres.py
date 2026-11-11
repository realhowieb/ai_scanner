"""Durable score projections on an explicitly configured throwaway Postgres."""
import importlib.util
import json
import os
import unittest
import uuid
from unittest.mock import patch

PG_URL = os.getenv('HSF_TEST_BILLING_PG_URL')
ENABLED = bool(PG_URL) and importlib.util.find_spec('psycopg2') is not None


@unittest.skipUnless(ENABLED, 'needs explicitly configured throwaway Postgres')
class ScanScoreCachePostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg2
        from psycopg2 import sql
        self.driver, self.sql = psycopg2, sql
        self.schema = 'score_cache_test_' + uuid.uuid4().hex
        with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        self.connect = lambda: psycopg2.connect(PG_URL, options=f'-csearch_path={self.schema}')
        self.env = patch.dict('os.environ', {'AI_SCANNER_SCHEMA_ONCE': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute('CREATE TABLE runs (id BIGINT PRIMARY KEY, username TEXT,label TEXT, '
                        'created_at TIMESTAMP,row_count INTEGER,results_json TEXT)')
            raw = json.dumps([{'Ticker': 'AAA', 'BreakoutScore': 8, 'PreBreakoutProb%': 65,
                               'ModelProvenance': 'must-not-transfer', 'Spark10D': [1, 2, 3]}])
            cur.execute("INSERT INTO runs VALUES(1,'cron','US_MARKET','2026-11-11 14:00:00',1,%s),"
                        "(2,'private','US_MARKET','2026-11-11 14:00:00',1,%s)", (raw, raw))

    def drop_schema(self):
        with self.driver.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(self.sql.SQL('DROP SCHEMA {} CASCADE').format(self.sql.Identifier(self.schema)))

    def test_slim_public_batch_round_trip_versions_and_cache_bound(self):
        from db import scan_score_cache as store
        with patch.object(store, 'get_neon_conn', side_effect=self.connect):
            source = store.load_score_rows([1, 2])
            self.assertEqual(set(source), {1})
            self.assertEqual(source[1]['rows'], [{'Ticker': 'AAA', 'BreakoutScore': 8, 'PreBreakoutProb%': 65}])
            self.assertEqual(source[1]['stamp'], '2026-11-11 14:00:00+00:00|1')
            self.assertEqual(store.load_scores([1], 'v1'), {})
            scores = {'AAA': {'score': 77, 'signals': ['prebreakout']}}
            store.save_scores([{'run_id': 1, 'stamp': source[1]['stamp'], 'scores': scores}], 'v1')
            self.assertEqual(store.load_scores([1], 'v1')[1], (source[1]['stamp'], scores))
            self.assertEqual(store.load_scores([1], 'v2'), {})
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute("UPDATE runs SET created_at='2026-11-11 15:00:00' WHERE id=1")
            self.assertNotEqual(store.load_score_rows([1])[1]['stamp'], source[1]['stamp'])
            store.save_scores([{'run_id': i, 'stamp': 's', 'scores': {}} for i in range(10, 120)], 'v1')
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute('SELECT count(*) FROM scan_score_cache')
                self.assertEqual(cur.fetchone()[0], 96)
