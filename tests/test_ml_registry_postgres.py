"""ML v4 model registry on real Postgres: idempotent registration, lifecycle,
one champion per target + horizon (code and database), rollback, read-only reads.
Uses only an explicit throwaway URL (the billing Postgres job's
HSF_TEST_BILLING_PG_URL); each test gets its own schema."""
import importlib.util
import os
import unittest
import uuid

from analytics import ml_v4_lifecycle as L

URL = os.getenv("HSF_TEST_BILLING_PG_URL", "")
DRIVER = "psycopg2" if importlib.util.find_spec("psycopg2") else ("psycopg" if importlib.util.find_spec("psycopg") else None)
EVAL = {"evaluation": {"evaluation_version": "hsf-ml-v4-eval-1"}}
GATES = {"promotion_gates_passed": True}


def _connect(schema):
    if DRIVER == "psycopg2":
        import psycopg2

        return psycopg2.connect(URL, options=f"-csearch_path={schema}")
    import psycopg

    return psycopg.connect(URL, options=f"-csearch_path={schema}")


def _manifest(n, provider="xgboost", label="F_clean_path_5d", horizon=5):
    return {"model_version": f"ml-v4-{provider}-production-f5d-{n:012d}", "provider": provider,
            "feature_set": "production", "feature_schema_version": "fs1", "label_schema_version": 1,
            "target": {"label": label, "horizon_days": horizon}, "dataset_version": "hsf-ml-test-v1",
            "dataset_fingerprint": "fp", "artifact_sha256": "a" * 64, "artifact_bytes": 123,
            "artifact_file": "x.joblib", "params": {"max_depth": 3}, "seed": 20261103, "train_rows": 100,
            "trained_at": "2026-11-03T12:00:00+00:00", "git_sha": "abc"}


@unittest.skipUnless(URL and DRIVER, "set HSF_TEST_BILLING_PG_URL to a throwaway Postgres")
class RegistryPostgresTest(unittest.TestCase):
    def setUp(self):
        from db import ml_model_registry as R

        self.R = R
        self.schema = "hsf_ml_registry_test_" + uuid.uuid4().hex
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'CREATE SCHEMA "{self.schema}"')
        conn.close()
        self.addCleanup(self._drop)
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))
        R.set_connection_factory(lambda: _connect(self.schema))
        self.addCleanup(lambda: R.set_connection_factory(None))

    def _drop(self):
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        conn.close()

    def _champion(self, n, **kw):
        mid = self.R.register_model(_manifest(n, **kw), actor="ci", artifact_uri=f"github-release://t/{n}")["id"]
        self.R.transition(mid, "validated", actor="howard", reason="walk-forward ok", evidence=EVAL)
        self.R.transition(mid, "shadow", actor="howard", reason="shadow start")
        return mid, self.R.transition(mid, "champion", actor="howard", reason="gates passed", evidence=GATES,
                                      approved=True)

    def test_reads_before_tables_exist_create_nothing(self):
        self.assertEqual(self.R.list_models(), [])
        self.assertIsNone(self.R.get_model(1))
        self.assertEqual(self.R.model_events(1), [])
        conn = _connect(self.schema)
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('ml_models')")
        self.assertIsNone(cur.fetchone()[0])
        conn.close()

    def test_register_is_idempotent_and_logged(self):
        first = self.R.register_model(_manifest(1), actor="ci", artifact_uri="github-release://t/a.joblib")
        again = self.R.register_model({**_manifest(1), "artifact_sha256": "b" * 64}, actor="ci")
        self.assertTrue(first["created"])
        self.assertEqual(again, {"id": first["id"], "created": False, "status": "candidate"})
        m = self.R.get_model(first["id"])
        self.assertEqual((m["status"], m["artifact_sha256"], m["target_label"], m["horizon_days"]),
                         ("candidate", "a" * 64, "F_clean_path_5d", 5))
        self.assertEqual(m["training_config"]["params"], {"max_depth": 3})
        self.assertEqual(m["artifact_uri"], "github-release://t/a.joblib")
        ev = self.R.model_events(first["id"])
        self.assertEqual([(e["from_status"], e["to_status"]) for e in ev], [(None, "candidate")])
        with self.assertRaises(L.LifecycleError):
            self.R.register_model(_manifest(2), actor="")

    def test_refused_transition_changes_nothing(self):
        mid = self.R.register_model(_manifest(1), actor="ci")["id"]
        for to, kw in (("champion", dict(approved=True, evidence=GATES)), ("validated", {})):
            with self.assertRaises(L.LifecycleError):
                self.R.transition(mid, to, actor="howard", reason="x", **kw)
        self.assertEqual(self.R.get_model(mid)["status"], "candidate")
        self.assertEqual(len(self.R.model_events(mid)), 1)
        with self.assertRaises(L.LifecycleError):
            self.R.transition(999999, "retired", actor="howard", reason="x")

    def test_attach_evaluation_only_before_shadow(self):
        mid = self.R.register_model(_manifest(1), actor="ci")["id"]
        self.R.attach_evaluation(mid, metrics={"evaluation_version": "e1", "auc": 0.61},
                                 calibration={"method": "sigmoid"}, actor="ci")
        m = self.R.get_model(mid)
        self.assertEqual((m["metrics"]["auc"], m["calibration"]["method"]), (0.61, "sigmoid"))
        self.R.transition(mid, "validated", actor="howard", reason="ok", evidence=EVAL)
        self.R.transition(mid, "shadow", actor="howard", reason="ok")
        with self.assertRaises(L.LifecycleError):
            self.R.attach_evaluation(mid, metrics={}, calibration=None, actor="ci")

    def test_new_champion_retires_the_old_one(self):
        a, _ = self._champion(1)
        b, res = self._champion(2, provider="lightgbm")
        self.assertEqual(res["superseded"], a)
        self.assertEqual(self.R.get_model(a)["status"], "retired")
        self.assertEqual([m["id"] for m in self.R.list_models(status="champion")], [b])
        # another target keeps its own champion
        c, res = self._champion(3, label="C_excess_gt_0", horizon=5)
        self.assertIsNone(res["superseded"])
        self.assertEqual(len(self.R.list_models(status="champion")), 2)
        self.assertEqual(self.R.list_models(status="champion", target_label="F_clean_path_5d")[0]["id"], b)

    def test_database_rejects_a_second_champion(self):
        a, _ = self._champion(1)
        b = self.R.register_model(_manifest(2), actor="ci")["id"]
        conn = _connect(self.schema)
        cur = conn.cursor()
        with self.assertRaises(Exception):
            cur.execute("UPDATE ml_models SET status = 'champion' WHERE id = %s", (b,))
        conn.rollback()
        conn.close()

    def test_rollback_restores_previous_champion(self):
        a, _ = self._champion(1)
        b, _ = self._champion(2, provider="catboost")
        res = self.R.rollback("F_clean_path_5d", 5, actor="howard", reason="live drift")
        self.assertEqual(res, {"restored": a, "retired": b})
        self.assertEqual(self.R.get_model(a)["status"], "champion")
        self.assertEqual(self.R.get_model(b)["status"], "retired")
        last = self.R.model_events(a)[-1]
        self.assertEqual((last["from_status"], last["to_status"], last["evidence"]), ("retired", "champion",
                                                                                   {"rolled_back_from": b}))
        # rolling back again returns to b (the most recent former champion)
        self.assertEqual(self.R.rollback("F_clean_path_5d", 5, actor="howard", reason="undo")["restored"], b)

    def test_rollback_without_former_champion(self):
        self._champion(1)
        with self.assertRaises(L.LifecycleError):
            self.R.rollback("F_clean_path_5d", 5, actor="howard", reason="x")
        with self.assertRaises(L.LifecycleError):
            self.R.rollback("C_excess_gt_0", 1, actor="howard", reason="x")


@unittest.skipUnless(URL and DRIVER, "set HSF_TEST_BILLING_PG_URL to a throwaway Postgres")
class ShadowPredictionsPostgresTest(unittest.TestCase):
    def setUp(self):
        from db import ml_shadow_predictions as D

        self.D = D
        self.schema = "hsf_ml_shadow_test_" + uuid.uuid4().hex
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'CREATE SCHEMA "{self.schema}"')
        conn.close()
        self.addCleanup(self._drop)
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))
        D.set_connection_factory(lambda: _connect(self.schema))
        self.addCleanup(lambda: D.set_connection_factory(None))

    def _drop(self):
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        conn.close()

    def _pred(self, model_id, oid, p=0.4, scored_at="2026-11-03T21:40:00+00:00"):
        return {"model_id": model_id, "model_version": f"v{model_id}", "observation_id": oid, "ticker": "AAA",
                "entry_day": "2026-11-03", "observed_at": "2026-11-03T15:00:00+00:00",
                "target_label": "F_clean_path_5d", "horizon_days": 5, "raw_probability": p,
                "calibrated_probability": None, "feature_schema_version": "fs1", "scored_at": scored_at,
                "scoring_ms": 0.3}

    def test_reads_before_table_exists(self):
        import datetime as dt

        self.assertEqual(self.D.predictions([1], since=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)), [])
        self.assertEqual(self.D.unlabeled("F_clean_path_5d", 5), [])

    def test_insert_is_idempotent_labels_fill_once_and_prune(self):
        import datetime as dt

        self.assertEqual(self.D.insert_predictions([self._pred(1, 10), self._pred(2, 10), self._pred(1, 11)]), 3)
        self.assertEqual(self.D.insert_predictions([self._pred(1, 10, p=0.9)]), 0)  # never overwritten
        self.assertEqual(sorted(self.D.unlabeled("F_clean_path_5d", 5)), [10, 11])
        self.assertEqual(self.D.fill_labels("F_clean_path_5d", 5, [(10, 1)]), 2)  # both models
        self.assertEqual(self.D.fill_labels("F_clean_path_5d", 5, [(10, 0)]), 0)  # filled once
        self.assertEqual(self.D.unlabeled("F_clean_path_5d", 5), [11])
        since = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        rows = self.D.predictions([1], since=since)  # no signal_outcomes here: served column falls back
        self.assertEqual([(r["observation_id"], r["label"], r["raw_probability"]) for r in rows],
                         [(10, 1, 0.4), (11, None, 0.4)])
        self.assertIsNone(rows[0]["ai_confidence"])
        self.D.insert_predictions([self._pred(3, 12, scored_at="2026-01-01T00:00:00+00:00")])
        self.assertEqual(self.D.prune(days=30), 1)
        self.assertEqual(self.D.predictions([3], since=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)), [])

    def test_served_ai_confidence_is_joined(self):
        import datetime as dt

        self.D.insert_predictions([self._pred(1, 10)])
        conn = _connect(self.schema)
        cur = conn.cursor()
        cur.execute("CREATE TABLE signal_outcomes (id BIGINT PRIMARY KEY, ai_confidence DOUBLE PRECISION)")
        cur.execute("INSERT INTO signal_outcomes VALUES (10, 62.5)")
        conn.commit()
        conn.close()
        rows = self.D.predictions([1], since=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(rows[0]["ai_confidence"], 62.5)


@unittest.skipUnless(URL and DRIVER, "set HSF_TEST_BILLING_PG_URL to a throwaway Postgres")
class ReadinessAuditPostgresTest(unittest.TestCase):
    def setUp(self):
        self.schema = "hsf_ml_audit_test_" + uuid.uuid4().hex
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'CREATE SCHEMA "{self.schema}"')
        conn.close()
        self.addCleanup(self._drop)
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))

    def _drop(self):
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        conn.close()

    def test_reports_missing_tables_without_creating_them_then_counts(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import ml_v4_readiness_audit as audit

        from db import ml_model_registry as R

        conn = _connect(self.schema)
        empty = audit.collect(conn)
        self.assertEqual(set(empty["tables"].values()), {False})
        self.assertNotIn("registry", empty)
        R.set_connection_factory(lambda: _connect(self.schema))
        self.addCleanup(lambda: R.set_connection_factory(None))
        R.register_model(_manifest(1), actor="ci")
        conn.rollback()
        out = audit.collect(conn)
        conn.close()
        self.assertEqual(out["registry"], {"by_status": {"candidate": 1}, "champions": [], "events": 1})
        self.assertTrue(out["tables"]["ml_models"])
