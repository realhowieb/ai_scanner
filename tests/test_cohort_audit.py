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


if __name__ == "__main__":
    unittest.main()
