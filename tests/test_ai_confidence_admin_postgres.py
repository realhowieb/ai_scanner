"""Run the admin confidence SQL on throwaway Postgres with dictionary rows."""
import importlib.util
import json
import os
import unittest
import uuid
from unittest.mock import patch

PG_URL = os.getenv("HSF_TEST_BILLING_PG_URL")
ENABLED = bool(PG_URL) and importlib.util.find_spec("psycopg2") is not None


@unittest.skipUnless(ENABLED, "needs explicitly configured throwaway Postgres")
class ConfidencePostgresTests(unittest.TestCase):
    def test_summary_and_explanation_preserve_all_dictionary_columns(self):
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extras import RealDictCursor

        from db import ai_confidence_admin as store

        schema = "confidence_test_" + uuid.uuid4().hex
        with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            def connect():
                return psycopg2.connect(PG_URL, options=f"-csearch_path={schema}", cursor_factory=RealDictCursor)

            with connect() as conn, conn.cursor() as cur:
                cur.execute("CREATE TABLE ai_confidence_models (id BIGINT, model_version TEXT, trained_at TIMESTAMPTZ, "
                            "created_at TIMESTAMPTZ, metadata JSONB, feature_names JSONB, is_active BOOLEAN)")
                meta = {"dataset_version": "ds1", "target_rule": "target", "calibration": [],
                        "calibration_map": {"method": "isotonic", "x": [0, 1]}}
                cur.execute("INSERT INTO ai_confidence_models VALUES (1,'model-v1',NOW(),NOW(),%s,%s,TRUE)",
                            (json.dumps(meta), json.dumps(["rsi", "rvol"])))
                cur.execute("CREATE TABLE signal_outcomes (id BIGINT,ticker TEXT,fired_at TIMESTAMPTZ, "
                            "source TEXT,signal_type TEXT,ai_confidence FLOAT,setup_score FLOAT,raw_signal JSONB, "
                            "indicators JSONB,mfe_5d FLOAT,mae_5d FLOAT)")
                raw = {"hsf_score": 72, "models": {"ai_confidence": {"loaded_artifact": {"version": "model-v1"},
                       "feature_names": ["rsi", "rvol"], "feature_values": [32, 2]}}}
                indicators = {"primary_setup": "rebound", "chg_pct": 1.5}
                cur.execute("INSERT INTO signal_outcomes VALUES (1,'TEST',NOW(),'opportunity','hsf_scanner',"
                            "45,12,%s,%s,0.05,-0.01)", (json.dumps(raw), json.dumps(indicators)))
            with patch.object(store, "get_neon_conn", side_effect=connect):
                result = store.summary()
                self.assertEqual(result["model"]["dataset_version"], "ds1")
                self.assertEqual(result["model"]["target_rule"], "target")
                self.assertEqual(result["model"]["calibration_points"], 2)
                self.assertEqual(result["model"]["features"], 2)
                score = result["latest"]["rows"][0]
                self.assertEqual((score["ticker"], score["hsf_score"], score["setup"], score["chg_pct"]),
                                 ("TEST", 72, "rebound", 1.5))
                self.assertEqual(result["live"]["buckets"][0]["hits"], 1)
                self.assertEqual(result["live"]["buckets"][0]["misses"], 0)
                self.assertEqual(store.ticker_features("test")["features"], {"rsi": 32, "rvol": 2})
        finally:
            with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
