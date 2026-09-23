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
        self.assertEqual(rep["gates"]["long_direction"]["status"], "PASS")

    def test_clean_audit_still_conditional_on_short_and_deploy(self):
        from scripts.run54_readiness import assess
        audit = {"total_observations": 300, "conflicting_duplicates": 0,
                 "cohort_overlap_within_run": {"candidate_control": 0},
                 "cohorts": {"CANDIDATE": {"explicitly_tagged": 100, "legacy_inferred": 0,
                                           "point_in_time_violations": 0}}}
        rep = assess(audit)
        # LONG/outcome/legacy gates pass, but SHORT-live + maturation-deploy hold it CONDITIONAL.
        self.assertEqual(rep["run54_ready"], "CONDITIONAL")
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
        self.assertEqual(rep["gates"]["no_conflicting_duplicates"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
