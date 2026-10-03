import datetime as dt
import unittest
from pathlib import Path
from unittest import mock

from analytics import admin_research as ar
from db import admin_analytics

ROOT = Path(__file__).resolve().parents[1]


def observation(
    oid="o1", *, context="scheduled:us_market", score=None, cohort="CANDIDATE",
    outcome=True, raw_return=0.02, directional_return=0.02,
):
    row = {
        "observation_id": oid,
        "symbol": "XYZ",
        "timestamp": "2026-10-01T14:00:00+00:00",
        "context": context,
        "session": "REGULAR",
        "research_cohort": cohort,
        "research_metadata": {"scoring_version": "1"},
        "scanners": [{"name": "breakout", "triggered": True}],
    }
    if score is not None:
        row["hsf_score"] = score
    if outcome:
        row["outcomes"] = {
            "+15m": {
                "data_status": "MATURED",
                "raw_return": raw_return,
                "directional_return": directional_return,
                "mfe": 0.03,
                "mae": -0.01,
                "evaluation_time": "2026-10-01T14:15:00+00:00",
            }
        }
    return row


class EvidenceGuardrailTests(unittest.TestCase):
    def test_default_threshold_boundaries(self):
        for n, expected in ((0, False), (1, False), (29, False), (30, True)):
            rows = ar.flatten_research([observation(str(i)) for i in range(n)])
            self.assertEqual(ar.performance_summary(rows)["sufficient"], expected, n)

    def test_null_outcomes_are_not_zero_returns(self):
        rows = ar.flatten_research([observation(outcome=False)])
        summary = ar.performance_summary(rows, min_n=1)
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["average_return"])
        self.assertIsNone(summary["positive_rate"])

    def test_score_buckets_include_exact_boundaries(self):
        expected = {
            49.99: "<50", 50: "50-59", 60: "60-69", 70: "70-79",
            80: "80-89", 90: "90-100", 100: "90-100",
        }
        self.assertEqual({value: ar.score_bucket(value) for value in expected}, expected)

    def test_breakout_score_is_not_substituted_for_hsf_score(self):
        row = observation(score=None)
        row["BreakoutScore"] = 99
        row["scanners"][0]["score"] = 99
        self.assertIsNone(ar.hsf_score(row))

    def test_grouped_signal_horizon_aggregation(self):
        rows = ar.flatten_research([
            observation("a", directional_return=0.02),
            observation("b", directional_return=-0.01),
        ])
        groups = ar.grouped_performance(rows, ("signal", "horizon"), min_n=2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["n"], 2)
        self.assertAlmostEqual(groups[0]["positive_rate"], 0.5)
        self.assertTrue(groups[0]["sufficient"])

    def test_scanner_and_stair_stepper_are_isolated(self):
        scanner = observation("scanner")
        stair = observation("stair", context=ar.STAIR_STEPPER_CONTEXT)
        rows = ar.flatten_research([scanner, stair])
        self.assertEqual({row["observation_id"] for row in rows}, {"scanner"})
        self.assertTrue(ar.is_stair_stepper(stair))
        self.assertFalse(ar.is_scanner_research(stair))

    def test_evidence_funnel_excludes_stair_stepper(self):
        scanner = observation("scanner")
        pending = observation("pending", outcome=False)
        stair = observation("stair", context=ar.STAIR_STEPPER_CONTEXT)
        funnel = ar.evidence_funnel([scanner, pending, stair])
        self.assertEqual(funnel["captured"], 2)
        self.assertEqual(funnel["matured"], 1)
        self.assertEqual(funnel["pending"], 1)
        self.assertEqual(funnel["included_in_research"], 1)

    def test_plan_normalization_uses_basic_not_free(self):
        self.assertEqual(ar.normalize_plan("free"), "basic")
        self.assertEqual(ar.normalize_plan("PRO"), "pro")
        self.assertEqual(ar.normalize_plan("unknown"), "basic")


class _Cursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=()):
        self.connection.sql.append(sql)
        self.rows = self.connection.rows_for(sql)

    def fetchall(self):
        return self.rows

    def close(self):
        return None


class _Connection:
    def __init__(self):
        self.sql = []
        self.closed = False

    def cursor(self):
        return _Cursor(self)

    def rollback(self):
        return None

    def close(self):
        self.closed = True

    @staticmethod
    def rows_for(sql):
        if "FROM users" in sql:
            return [{"tier": "pro", "is_active": True, "created_at": dt.datetime.now(dt.timezone.utc),
                     "stripe_subscription_id": "sub_1"}]
        if "FROM hsf_observations" in sql:
            return [{"observation_id": "o1", "record": observation(), "horizon": "+15m",
                     "outcome_record": observation()["outcomes"]["+15m"]}]
        return []


class ReadOnlyLoaderTests(unittest.TestCase):
    def test_loader_is_bounded_and_never_runs_ddl(self):
        conn = _Connection()
        with mock.patch.object(admin_analytics, "get_neon_conn", return_value=conn):
            result = admin_analytics.load_admin_data(days=30, research_limit=500)
        self.assertTrue(result["available"])
        self.assertTrue(conn.closed)
        self.assertEqual(len(result["observations"]), 1)
        statements = "\n".join(conn.sql).upper()
        self.assertNotIn("CREATE TABLE", statements)
        self.assertNotIn("ALTER TABLE", statements)
        self.assertIn("LIMIT %S", statements)

    def test_empty_database_state_is_nonfatal(self):
        with mock.patch.object(admin_analytics, "get_neon_conn", return_value=None):
            result = admin_analytics.load_admin_data()
        self.assertFalse(result["available"])
        self.assertEqual(result["errors"], {"database": "unavailable"})

    def test_admin_ui_checks_authorization_before_loading(self):
        source = (ROOT / "ui" / "admin_analytics.py").read_text()
        guard = source.index('if not bool(st.session_state.get("is_admin"))')
        load = source.index("load_admin_data_cached(days, include_research)")
        self.assertLess(guard, load)

    def test_non_research_admin_views_skip_large_observation_load(self):
        conn = _Connection()
        with mock.patch.object(admin_analytics, "get_neon_conn", return_value=conn):
            result = admin_analytics.load_admin_data(days=30, include_research=False)
        self.assertTrue(result["available"])
        research_queries = [sql for sql in conn.sql if "FROM hsf_observations" in sql]
        self.assertEqual(research_queries, [])
        self.assertEqual(result["observations"], [])


if __name__ == "__main__":
    unittest.main()
