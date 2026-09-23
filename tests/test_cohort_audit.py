"""Run 53 — read-only research cohort integrity auditor tests."""
import unittest

from analytics.hsf_observation import build_observation
from scripts.audit_research_cohorts import audit_cohorts


def _obs(sym, cohort, ts="2026-09-23T14:00:00+00:00", scan_id="s1",
         price=100.0, volume=1e6, explicit=True, missing=False):
    o = build_observation(
        symbol=sym, timestamp=ts, context="scheduled:us_market",
        market={"price": price, "volume": volume},
        indicators={} if missing else {"gap_pct": 1.0, "rvol": 2.0},
        scanners=[{"name": "breakout", "direction": "long"}],
        market_context={"scan_id": scan_id}, scan_timestamp=ts)
    if explicit:
        o["research_cohort"] = cohort
        o["market_context"]["research_cohort"] = cohort
    return o


class CohortAuditTests(unittest.TestCase):
    def test_per_cohort_counts_and_separation(self):
        obs = [
            _obs("AAA", "CANDIDATE"),
            _obs("BBB", "NEAR_MISS"),
            _obs("CCC", "CONTROL", missing=True),
            _obs("DDD", "CONTROL", scan_id="s2", missing=True),
            _obs("LEG", "CANDIDATE", explicit=False),  # legacy untagged → inferred
        ]
        rep = audit_cohorts(obs, {})
        self.assertEqual(rep["total_observations"], 5)
        cand = rep["cohorts"]["CANDIDATE"]
        self.assertEqual(cand["observations"], 2)          # AAA + legacy LEG
        self.assertEqual(cand["explicitly_tagged"], 1)
        self.assertEqual(cand["legacy_inferred"], 1)       # Task 9: not silent
        ctrl = rep["cohorts"]["CONTROL"]
        self.assertEqual(ctrl["observations"], 2)
        self.assertEqual(ctrl["distinct_runs"], 2)         # s1 + s2
        self.assertEqual(ctrl["missing_field_observations"], 2)  # controls sparse by design
        gap = next(row for row in ctrl["field_missingness"] if row["field"] == "gap_pct")
        self.assertEqual(gap["classification"], "OPTIONAL")
        self.assertFalse(gap["required_for_run54"])

    def test_missing_fields_are_reported_by_field_not_treated_as_one_defect(self):
        modern = _obs("AAA", "CANDIDATE", missing=True)
        legacy = _obs("LEG", "CANDIDATE", explicit=False, missing=True)
        rep = audit_cohorts([modern, legacy], {})
        rows = {row["field"]: row for row in rep["cohorts"]["CANDIDATE"]["field_missingness"]}
        self.assertEqual(rows["research_cohort"]["missing_count"], 1)
        self.assertEqual(rows["research_cohort"]["classification"], "LEGACY_SCHEMA")
        self.assertEqual(rows["adx"]["classification"], "OPTIONAL")
        required = rep["cohorts"]["CANDIDATE"]["modern_required_field_missingness"]
        self.assertEqual(sum(row["missing_count"] for row in required), 0)

    def test_modern_horizon_direction_contract_excludes_incomplete_outcomes(self):
        obs = _obs("AAA", "CANDIDATE")
        oid = obs["observation_id"]
        complete = {
            "horizon": "+5m", "evaluation_time": "2026-09-23T14:05:00+00:00",
            "data_status": "MATURED", "raw_return": 0.01,
            "directional_return": 0.01, "mfe": 0.02, "mae": -0.005,
        }
        incomplete = {**complete, "horizon": "+15m", "mfe": None}
        rep = audit_cohorts([obs], {oid: [complete, incomplete]})
        maturity = rep["cohorts"]["CANDIDATE"]["modern_maturity_by_horizon_direction"]
        self.assertEqual(maturity["+5m"]["LONG"]["analysis_eligible"], 1)
        self.assertEqual(maturity["+15m"]["LONG"]["analysis_eligible"], 0)
        missing = rep["cohorts"]["CANDIDATE"]["matured_outcome_field_missingness"]
        self.assertEqual(next(row for row in missing if row["field"] == "mfe")["missing_count"], 1)

    def test_matured_unmatured_and_pit_violation(self):
        obs = [_obs("AAA", "CANDIDATE"), _obs("BBB", "CANDIDATE")]
        outcomes = {
            obs[0]["observation_id"]: [
                {"horizon": "+5m", "evaluation_time": "2026-09-23T14:05:00+00:00"}],
            obs[1]["observation_id"]: [
                # evaluation BEFORE the observation → point-in-time violation
                {"horizon": "+5m", "evaluation_time": "2026-09-23T13:00:00+00:00"}],
        }
        rep = audit_cohorts(obs, outcomes)
        cand = rep["cohorts"]["CANDIDATE"]
        self.assertEqual(cand["matured"], 2)
        self.assertEqual(cand["unmatured"], 0)
        self.assertEqual(cand["point_in_time_violations"], 1)

    def test_invalid_values_and_conflicting_duplicates(self):
        good = _obs("AAA", "CANDIDATE")
        bad = _obs("BBB", "CANDIDATE", price=-5.0)      # negative price = invalid
        dup = dict(good)
        dup["symbol"] = "ZZZ"                            # same id, different content
        rep = audit_cohorts([good, bad, dup], {})
        self.assertEqual(rep["cohorts"]["CANDIDATE"]["invalid_values"], 1)
        self.assertEqual(rep["conflicting_duplicates"], 1)

    def test_empty_is_safe(self):
        rep = audit_cohorts([], {})
        self.assertEqual(rep["total_observations"], 0)
        self.assertEqual(rep["cohorts"]["CONTROL"]["observations"], 0)

    def test_candidate_control_overlap_within_run_detected(self):
        # Same symbol tagged CANDIDATE and CONTROL for the same scan_run_id is a
        # serious separation violation (Part 2).
        obs = [
            _obs("AAA", "CANDIDATE", scan_id="run1"),
            _obs("AAA", "CONTROL", scan_id="run1"),   # overlap on AAA/run1
            _obs("BBB", "CANDIDATE", scan_id="run1"),
            _obs("AAA", "CONTROL", scan_id="run2"),    # different run → not an overlap
        ]
        rep = audit_cohorts(obs, {})
        self.assertEqual(rep["cohort_overlap_within_run"]["candidate_control"], 1)

    def test_no_overlap_when_clean(self):
        obs = [_obs("AAA", "CANDIDATE"), _obs("BBB", "NEAR_MISS"), _obs("CCC", "CONTROL")]
        rep = audit_cohorts(obs, {})
        self.assertEqual(rep["cohort_overlap_within_run"]["any"], 0)


class Run54ReadinessTests(unittest.TestCase):
    def test_no_live_audit_is_conditional(self):
        from scripts.run54_readiness import assess
        rep = assess(None)
        self.assertEqual(rep["run54_ready"], "CONDITIONAL")
        self.assertFalse(rep["have_live_audit"])
        self.assertEqual(rep["gates"]["long_live_direction"]["status"], "CONDITIONAL")

    def test_clean_audit_still_conditional_on_short_and_deploy(self):
        from scripts.run54_readiness import assess
        audit = {"total_observations": 300, "conflicting_duplicates": 0,
                 "cohort_overlap_within_run": {"candidate_control": 0},
                 "cohorts": {"CANDIDATE": {"explicitly_tagged": 100, "legacy_inferred": 0,
                                           "point_in_time_violations": 0}}}
        rep = assess(audit)
        # An old aggregate-only audit cannot prove a complete modern comparison.
        self.assertEqual(rep["run54_ready"], "NO")
        self.assertEqual(rep["gates"]["pit_integrity"]["status"], "PASS")
        self.assertEqual(rep["gates"]["cohort_separation"]["status"], "PASS")

    def test_pit_or_overlap_fails_hard(self):
        from scripts.run54_readiness import assess
        audit = {"total_observations": 300, "conflicting_duplicates": 5,
                 "cohort_overlap_within_run": {"candidate_control": 2},
                 "cohorts": {"CANDIDATE": {"explicitly_tagged": 100, "legacy_inferred": 0,
                                           "point_in_time_violations": 3}}}
        rep = assess(audit)
        self.assertEqual(rep["run54_ready"], "NO")
        self.assertEqual(rep["gates"]["pit_integrity"]["status"], "FAIL")

    def test_scoped_modern_comparison_can_pass_sample_gate(self):
        from scripts.run54_readiness import assess
        cell = {"observations": 35, "analysis_eligible": 35}
        cohort = {
            "explicitly_tagged": 35, "legacy_inferred": 0,
            "point_in_time_violations": 0,
            "modern_required_field_missingness": [],
            "modern_maturity_by_horizon_direction": {"+5m": {"LONG": cell}},
            "direction_examples": {"LONG": [{"symbol": "AAA"}]},
        }
        audit = {
            "total_observations": 70, "conflicting_duplicates": 0,
            "cohort_overlap_within_run": {"candidate_control": 0},
            "cohorts": {"CANDIDATE": cohort, "CONTROL": cohort},
        }
        rep = assess(audit, {"maturation_backlog_status": "BACKLOG_DRAINING"})
        self.assertEqual(rep["gates"]["adequate_scoped_sample"]["status"], "PASS")
        self.assertEqual(rep["gates"]["maturation_usable"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
