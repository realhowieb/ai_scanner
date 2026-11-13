"""PreBreakout evidence queries on isolated Postgres with production-style dictionary rows."""
import importlib.util
import json
import os
import unittest
import uuid
from unittest.mock import patch

PG_URL = os.getenv("HSF_TEST_BILLING_PG_URL")
ENABLED = bool(PG_URL) and importlib.util.find_spec("psycopg2") is not None


@unittest.skipUnless(ENABLED, "needs explicitly configured throwaway Postgres")
class PrebreakoutPostgresTests(unittest.TestCase):
    def test_dictionary_rows_missing_captures_and_exact_snapshot_inputs(self):
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extras import RealDictCursor

        from db import prebreakout_admin as store

        schema = "pb_admin_test_" + uuid.uuid4().hex
        with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            def connect():
                return psycopg2.connect(PG_URL, options=f"-csearch_path={schema}", cursor_factory=RealDictCursor)
            with connect() as conn, conn.cursor() as cur:
                cur.execute("CREATE TABLE prebreakout_models (id BIGINT,model_version TEXT,trained_at TIMESTAMPTZ,"
                            "created_at TIMESTAMPTZ,metadata JSONB,feature_names JSONB,is_active BOOLEAN,auc FLOAT)")
                meta = {"dataset_version": "ds", "target_rule": "future setup then economic outcome", "calibration": [],
                        "calibration_map": {"method": "isotonic", "x": [0, 1]}}
                cur.execute("INSERT INTO prebreakout_models VALUES (1,'pb-v1',NOW(),NOW(),%s,%s,TRUE,0.62)",
                            (json.dumps(meta), json.dumps(["rsi", "vol"])))
                cur.execute("CREATE TABLE signal_outcomes (id BIGINT,ticker TEXT,fired_at TIMESTAMPTZ,source TEXT,"
                            "signal_type TEXT,prebreakout_prob FLOAT,setup_score FLOAT,raw_signal JSONB,indicators JSONB)")
                raw = {"hsf_score": 72, "models": {"prebreakout": {"status": "captured", "loaded_artifact": {"version": "pb-v1"},
                       "feature_names": ["rsi", "vol"], "feature_values": [32, 0], "default_mask": [False, True]}}}
                cur.execute("INSERT INTO signal_outcomes VALUES (1,'TEST',NOW(),'opportunity','hsf_scanner',"
                            "45,12,%s,'{}'),(2,'MISSING',NOW(),'opportunity','hsf_scanner',0,'0','{}','{}')", (json.dumps(raw),))
            with patch.object(store, "_conn", side_effect=connect):
                result = store.summary()
                self.assertEqual(result["model"]["target_rule"], meta["target_rule"])
                self.assertEqual(result["model"]["calibration_method"], "isotonic")
                self.assertEqual(result["model"]["features"], 2)
                scores = {r["ticker"]: r for r in result["latest"]["rows"]}
                self.assertEqual(scores["TEST"]["probability"], 45)
                self.assertIsNone(scores["MISSING"]["probability"])
                at = result["latest"]["scanned_at"]
                saved = store.inputs("test", at)
                self.assertEqual(saved["inputs"][1], {"name": "vol", "value": 0, "defaulted": True})
                self.assertIsNone(store.inputs("test", "2000-01-01T00:00:00Z"))
                self.assertIsNone(store.inputs("missing", at))
                self.assertIn("cannot verify", result["live_validation"])
        finally:
            with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
