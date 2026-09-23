"""Run 46 — observation integrity, completeness & research-export tests."""
import unittest

from analytics import observation_integrity as oi
from analytics.hsf_observation import build_observation, build_outcome


def _obs(symbol="NVDA", ts="2026-08-25T14:00:00+00:00", scan_ts="2026-08-25T14:03:00+00:00",
         health="HEALTHY", **over):
    o = build_observation(
        symbol=symbol, timestamp=ts, context="scheduled:us_market",
        session="morning", scan_timestamp=scan_ts,
        market={"price": 131.0, "volume": 5_000_000},
        indicators={"rvol": 2.4, "vs_vwap_pct": 0.8, "adx": 28, "gap_pct": 0.4, "chg_pct": 1.5},
        scanners=[{"name": "breakout", "triggered": True, "direction": "long"}],
        models={"prebreakout": {"version": "prebreakout-xgb-v16", "probability": 0.27}},
        market_context={"source": "scheduled", "scan_id": "run1", "coverage_health": health},
    )
    o.update(over)
    return o


class SchemaProvenanceTests(unittest.TestCase):
    def test_schema_inventory_groups(self):
        groups = {f["group"] for f in oi.schema_inventory()}
        self.assertTrue({"IDENTITY", "MARKET_STATE", "TECHNICAL", "ML",
                         "PROVENANCE", "OUTCOMES"} <= groups)

    def test_provenance_present(self):
        o = _obs()
        self.assertIn("schema", o["versions"])
        self.assertEqual(o["versions"]["prebreakout_model"], "prebreakout-xgb-v16")
        self.assertEqual(o["schema_version"], "hsf-obs-1.0")


class ValidityTests(unittest.TestCase):
    def test_valid_observation_no_issues(self):
        self.assertEqual(oi.validate_observation(_obs()), [])

    def test_invalid_price_volume_rvol_rsi(self):
        o = _obs(market={"price": -1, "volume": -5}, indicators={"rvol": -2, "rsi": 150})
        issues = oi.validate_observation(o)
        self.assertIn("invalid_price", issues)
        self.assertIn("invalid_volume", issues)
        self.assertIn("invalid_rvol", issues)
        self.assertIn("invalid_rsi", issues)

    def test_nonfinite_flagged(self):
        o = _obs(indicators={"rvol": float("inf")})
        self.assertTrue(any("nonfinite" in i for i in oi.validate_observation(o)))

    def test_invalid_probability(self):
        # accepts [0,1] fraction or [0,100] percent; 150 is outside both.
        o = _obs(models={"prebreakout": {"probability": 150.0}})
        self.assertIn("invalid_probability", oi.validate_observation(o))
        self.assertNotIn("invalid_probability",
                         oi.validate_observation(_obs(models={"prebreakout": {"probability": 27.0}})))

    def test_invalid_direction(self):
        o = _obs(scanners=[{"name": "x", "triggered": True, "direction": "sideways"}])
        self.assertTrue(any("invalid_direction" in i for i in oi.validate_observation(o)))

    def test_malformed_ticker_and_missing_identity(self):
        self.assertIn("malformed_ticker", oi.validate_observation(_obs(symbol="TOOOOLONGXX")))
        o = _obs()
        o["observation_id"] = ""
        self.assertTrue(any("missing_identity" in i for i in oi.validate_observation(o)))


class PointInTimeTests(unittest.TestCase):
    def test_clean_observation_no_violations(self):
        self.assertEqual(oi.check_point_in_time(_obs()), [])

    def test_outcome_marker_in_features_flagged(self):
        o = _obs()
        o["indicators"]["raw_return"] = 0.02  # leakage: outcome inside features
        self.assertTrue(any("outcome_marker" in v for v in oi.check_point_in_time(o)))

    def test_separate_outcomes_do_not_violate(self):
        o = _obs()
        oc = build_outcome(observation_id=o["observation_id"], symbol="NVDA",
                           observation_timestamp=o["scan_timestamp"], horizon="+15m",
                           evaluation_time="2026-08-25T14:18:00+00:00", raw_return=0.01)
        o["outcomes"] = {"+15m": oc}
        self.assertEqual(oi.check_point_in_time(o), [])  # separate namespace, future eval

    def test_lookahead_outcome_flagged(self):
        o = _obs()
        o["outcomes"] = {"+15m": {"data_status": "MATURED",
                                  "evaluation_time": "2026-08-25T13:00:00+00:00"}}  # before
        self.assertTrue(any("lookahead" in v for v in oi.check_point_in_time(o)))


class DuplicateTests(unittest.TestCase):
    def test_exact_duplicates(self):
        o = _obs()
        rep = oi.duplicate_analysis([o, dict(o)])
        self.assertEqual(rep["exact_duplicates"], 1)
        self.assertEqual(rep["conflicting_duplicates"], 0)

    def test_conflicting_duplicates(self):
        a = _obs()
        b = dict(a)
        b["market"] = {"price": 999.0, "volume": 1}  # same id, different values
        rep = oi.duplicate_analysis([a, b])
        self.assertEqual(rep["conflicting_duplicates"], 1)
        self.assertIn(a["observation_id"], rep["conflicting_ids"])

    def test_unique_counts(self):
        rep = oi.duplicate_analysis([_obs("A"), _obs("B"), _obs("C")])
        self.assertEqual(rep["unique_logical"], 3)
        self.assertEqual(rep["exact_duplicates"], 0)


class CompletenessTests(unittest.TestCase):
    def test_population_rates(self):
        # NVDA has price/volume/rvol but no rsi/ema9 → those missing
        rep = oi.completeness_report([_obs(), _obs("AMD")])
        self.assertEqual(rep["n"], 2)
        self.assertEqual(rep["fields"]["symbol"]["pct"], 1.0)
        self.assertEqual(rep["fields"]["market.price"]["pct"], 1.0)
        self.assertEqual(rep["fields"]["indicators.rsi"]["pct"], 0.0)  # not captured

    def test_capture_rate(self):
        self.assertEqual(oi.capture_rate(100, 11631), round(100 / 11631, 4))
        self.assertIsNone(oi.capture_rate(100, 0))


class DatasetHealthTests(unittest.TestCase):
    def test_healthy_dataset(self):
        rep = oi.dataset_health_report([_obs("A"), _obs("B")], successfully_evaluated=11631)
        self.assertEqual(rep["dataset_health"], "HEALTHY")
        self.assertEqual(rep["unique_tickers"], 2)
        self.assertEqual(rep["scan_runs"], 1)
        self.assertIsNotNone(rep["capture_rate"])

    def test_degraded_on_conflicts(self):
        a = _obs()
        b = dict(a); b["market"] = {"price": 5.0}
        rep = oi.dataset_health_report([a, b])
        self.assertEqual(rep["dataset_health"], "DEGRADED")

    def test_failed_on_pit_violation(self):
        o = _obs()
        o["indicators"]["future_high"] = 200  # leakage
        rep = oi.dataset_health_report([o])
        self.assertEqual(rep["dataset_health"], "FAILED")
        self.assertGreaterEqual(rep["point_in_time_violations"], 1)

    def test_scan_health_linkage(self):
        rep = oi.dataset_health_report([_obs(health="HEALTHY"), _obs("X", health="DEGRADED")])
        self.assertIn("HEALTHY", rep["scan_health_distribution"])
        self.assertIn("DEGRADED", rep["scan_health_distribution"])


class ResearchExportTests(unittest.TestCase):
    def test_deterministic_ordering_and_columns(self):
        rows = oi.research_export([_obs("ZZZ"), _obs("AAA")])
        self.assertEqual([r["symbol"] for r in rows], ["AAA", "ZZZ"])  # sorted
        self.assertIn("market_price", rows[0])
        self.assertIn("ind_rvol", rows[0])
        self.assertIn("prebreakout_probability", rows[0])

    def test_no_outcome_fields_by_default(self):
        o = _obs()
        o["outcomes"] = {"+15m": {"raw_return": 0.02, "mfe": 0.03}}
        rows = oi.research_export([o])
        blob = str(rows[0]).lower()
        self.assertNotIn("raw_return", blob)
        self.assertNotIn("outcomes", blob)
        self.assertNotIn("mfe", blob)

    def test_healthy_only_filter(self):
        rows = oi.research_export([_obs("A", health="HEALTHY"),
                                   _obs("B", health="DEGRADED")], healthy_only=True)
        self.assertEqual([r["symbol"] for r in rows], ["A"])

    def test_date_range_filter(self):
        rows = oi.research_export(
            [_obs("A", ts="2026-08-25T14:00:00+00:00"),
             _obs("B", ts="2026-08-26T14:00:00+00:00")],
            start="2026-08-26", end="2026-08-26")
        self.assertEqual([r["symbol"] for r in rows], ["B"])

    def test_determinism(self):
        obs = [_obs("A"), _obs("B")]
        self.assertEqual(oi.research_export(obs), oi.research_export(obs))


if __name__ == "__main__":
    unittest.main()
