"""Forward quarantine and immutable prediction-only collection on isolated Postgres."""
import datetime as dt
import importlib.util
import json
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

PG_URL=os.getenv('HSF_TEST_BILLING_PG_URL')
ENABLED=bool(PG_URL) and importlib.util.find_spec('psycopg2') is not None


@unittest.skipUnless(ENABLED,'needs explicitly configured throwaway Postgres')
class ForwardPostgresTests(unittest.TestCase):
    def test_registration_collection_and_development_quarantine(self):
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extras import RealDictCursor

        from db import ml_forward_cohorts as store

        schema='forward_test_'+uuid.uuid4().hex
        protocol=json.loads((Path(__file__).resolve().parents[1]/'docs/ml_forward_cohort_2026-11-16.json').read_text())
        before=dt.datetime(2026,11,13,tzinfo=dt.timezone.utc)
        during=dt.datetime(2026,11,16,17,tzinfo=dt.timezone.utc)
        with psycopg2.connect(PG_URL) as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        try:
            def connect():
                return psycopg2.connect(PG_URL,options=f'-csearch_path={schema}',cursor_factory=RealDictCursor)
            with connect() as conn,conn.cursor() as cur:
                cur.execute('CREATE TABLE signal_outcomes(id BIGINT,ticker TEXT,fired_at TIMESTAMPTZ,source TEXT,signal_type TEXT,user_id TEXT,'
                            'setup_score FLOAT,ai_confidence FLOAT,prebreakout_prob FLOAT,indicators JSONB,raw_signal JSONB)')
            with patch.object(store,'get_neon_conn',side_effect=connect):
                store.register(protocol,now=before)
                store.register(protocol,now=during)  # identical retry is idempotent
                with self.assertRaisesRegex(ValueError,'immutable'):
                    store.register({**protocol,'model_policy':'changed'},now=before)
                self.assertEqual(store.collect(protocol,now=before)['status'],'WAITING_FOR_START')
                with connect() as conn,conn.cursor() as cur:
                    cur.execute("INSERT INTO signal_outcomes VALUES (1,'AAA','2026-11-16 15:00Z','opportunity','hsf_scanner',NULL,5,20,30,'{}','{\"hsf_score\":12,\"return_5d\":999}'),"
                                "(2,'AAA','2026-11-16 16:00Z','opportunity','hsf_scanner',NULL,5,20,30,'{}','{}'),"
                                "(3,'BBB','2026-12-16 15:00Z','opportunity','hsf_scanner',NULL,5,20,30,'{}','{}'),"
                                "(4,'CCC','2026-11-16 15:00Z','opportunity','hsf_scanner','private',5,20,30,'{}','{}'),"
                                "(5,'DDD','2026-11-16 15:00Z','opportunity','hsf_opportunity',NULL,5,20,30,'{}','{}'),"
                                "(6,'OLD','2026-10-01 15:00Z','opportunity','hsf_scanner',NULL,5,20,30,'{}','{}')")
                with self.assertRaisesRegex(ValueError,'quarantined'):
                    store.assert_development_allowed([1])  # before collector ran
                store.assert_development_allowed([6])
                self.assertEqual(store.development_ids(before-dt.timedelta(days=60),during),[6])
                result=store.collect(protocol,now=during)
                self.assertEqual((result['observations'],result['signal_days'],result['observed_days']),(2,1,1))
                self.assertFalse(result['outcomes_read'])
                with connect() as conn,conn.cursor() as cur:
                    cur.execute("UPDATE signal_outcomes SET raw_signal='{\"hsf_score\":99}' WHERE id=1")
                self.assertEqual(store.collect(protocol,now=during)['added'],0)
                with connect() as conn,conn.cursor() as cur:
                    cur.execute('SELECT snapshot FROM ml_forward_members WHERE observation_id=1')
                    snapshot=cur.fetchone()['snapshot']
                    self.assertEqual(snapshot['raw_signal']['hsf_score'],12)
                    self.assertNotIn('return_5d',snapshot['raw_signal'])
                with self.assertRaisesRegex(ValueError,'retrospective'):
                    store.register({**protocol,'cohort_id':'late'},now=during)
        finally:
            with psycopg2.connect(PG_URL) as conn,conn.cursor() as cur:
                cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
