"""Run 47 — research cohorts (candidate/near-miss/control) + outcome integrity."""
import unittest

from analytics import observation_integrity as oi
from analytics import research_cohorts as rc


def _row(ticker, score=8.0, **kw):
    base = {"Ticker": ticker, "BreakoutScore": score, "Last": 100.0, "Volume": 1e6,
            "PctChange": 1.5, "GapPct": 0.6, "Trend20D%": 3.0, "Trend10D%": 2.0,
            "VolRel20": 2.4, "DollarVol20": 5e8, "Volatility20D%": 3.0, "IsBreakout": True}
    base.update(kw)
    return base


class CohortDefinitionTests(unittest.TestCase):
    def test_near_miss_deterministic_full_feature(self):
        rows = [_row(f"NM{i}") for i in range(10)]
        a = rc.build_near_miss_observations(rows, universe="US_MARKET",
                                            scan_timestamp="2026-09-22T14:00:00+00:00",
                                            scan_id="run1", n=5)
        b = rc.build_near_miss_observations(rows, universe="US_MARKET",
                                            scan_timestamp="2026-09-22T14:00:00+00:00",
                                            scan_id="run1", n=5)
        self.assertEqual(a, b)                    # deterministic
        self.assertEqual(len(a), 5)               # bounded by n
        self.assertEqual(a[0]["research_cohort"], "NEAR_MISS")
        self.assertEqual(a[0]["selection_reason"], "rank_below_cutoff")
        self.assertIn("gap_pct", a[0]["indicators"])   # full features like candidates

    def test_control_selection_deterministic_and_no_future(self):
        evaluated = [f"S{i}" for i in range(500)]
        a = rc.select_control_symbols(evaluated, scan_run_id="run1", exclude=["S0"], n=100)
        b = rc.select_control_symbols(evaluated, scan_run_id="run1", exclude=["S0"], n=100)
        self.assertEqual(a, b)                    # reproducible
        self.assertEqual(len(a), 100)
        self.assertNotIn("S0", a)                 # excludes candidates
        # different scan_run_id → different sample (seeded by scan_run_id+symbol)
        c = rc.select_control_symbols(evaluated, scan_run_id="run2", n=100)
        self.assertNotEqual(a, c)

    def test_control_records_compact(self):
        obs = rc.build_control_observations(
            ["AAA", "BBB"], {"AAA": {"price": 12.3, "volume": 1e5}},
            universe="US_MARKET", scan_timestamp="2026-09-22T14:00:00+00:00", scan_id="r1")
        self.assertEqual(obs[0]["research_cohort"], "CONTROL")
        self.assertEqual(obs[0]["selection_reason"], "deterministic_sample")
        self.assertEqual(obs[0]["market"]["price"], 12.3)
        self.assertEqual(obs[1]["market"], {})    # no snapshot → empty, not fabricated
        self.assertEqual(obs[0]["scanners"], [])  # controls carry no scanner triggers


class BoundsTests(unittest.TestCase):
    def test_hard_cap_prevents_full_market(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"RESEARCH_NEAR_MISS_N": "99999",
                                     "RESEARCH_CONTROL_N": "99999"}):
            self.assertEqual(rc.near_miss_n(), rc.HARD_CAP)
            self.assertEqual(rc.control_n(), rc.HARD_CAP)

    def test_control_n_bounded(self):
        big = [f"S{i}" for i in range(5000)]
        self.assertEqual(len(rc.select_control_symbols(big, scan_run_id="r", n=99999)),
                         rc.HARD_CAP)

    def test_defaults(self):
        self.assertEqual(rc.near_miss_n(), rc.DEFAULT_NEAR_MISS_N)
        self.assertEqual(rc.control_n(), rc.DEFAULT_CONTROL_N)


class CohortBalanceTests(unittest.TestCase):
    def test_balance_counts(self):
        obs = (rc.build_near_miss_observations([_row("A"), _row("B")], universe="U",
               scan_timestamp="t", scan_id="r")
               + rc.build_control_observations(["C"], {}, universe="U",
                 scan_timestamp="t", scan_id="r"))
        bal = rc.cohort_balance([{"research_cohort": "CANDIDATE"}] * 3 + obs, evaluated=11631)
        self.assertEqual(bal["candidate"], 3)
        self.assertEqual(bal["near_miss"], 2)
        self.assertEqual(bal["control"], 1)
        self.assertEqual(bal["research_rows"], 6)
        self.assertEqual(bal["evaluated"], 11631)

    def test_legacy_untagged_is_candidate(self):
        self.assertEqual(rc.cohort_of({"symbol": "X"}), "CANDIDATE")


class PointInTimeAllCohortsTests(unittest.TestCase):
    def test_no_leakage_in_any_cohort(self):
        nm = rc.build_near_miss_observations([_row("A")], universe="U",
             scan_timestamp="2026-09-22T14:00:00+00:00", scan_id="r")[0]
        ctrl = rc.build_control_observations(["B"], {"B": {"price": 5.0}}, universe="U",
             scan_timestamp="2026-09-22T14:00:00+00:00", scan_id="r")[0]
        self.assertEqual(oi.check_point_in_time(nm), [])
        self.assertEqual(oi.check_point_in_time(ctrl), [])


class ExportTests(unittest.TestCase):
    def _obs(self, cohort, sym="NVDA", health="HEALTHY"):
        o = rc.build_control_observations([sym], {sym: {"price": 100.0, "volume": 1e6}},
            universe="US_MARKET", scan_timestamp="2026-09-22T14:00:00+00:00",
            scan_id="r1", coverage_health=health)[0]
        o["research_cohort"] = cohort
        o["market_context"]["research_cohort"] = cohort
        o["market_context"]["coverage_health"] = health
        return o

    def test_cohort_filter(self):
        obs = [self._obs("CANDIDATE", "AAA"), self._obs("CONTROL", "BBB"),
               self._obs("NEAR_MISS", "CCC")]
        rows = oi.research_export(obs, cohorts=["CANDIDATE", "NEAR_MISS"])
        self.assertEqual({r["symbol"] for r in rows}, {"AAA", "CCC"})
        self.assertIn("research_cohort", rows[0])

    def test_outcomes_excluded_by_default_included_on_request(self):
        o = self._obs("CANDIDATE")
        o["outcomes"] = {"+15m": {"raw_return": 0.02, "data_status": "MATURED"}}
        default = oi.research_export([o])
        self.assertNotIn("outcomes", default[0])
        self.assertNotIn("raw_return", str(default[0]).lower())
        explicit = oi.research_export([o], include_outcomes=True)
        self.assertIn("outcomes", explicit[0])
        self.assertEqual(explicit[0]["outcomes"]["+15m"]["raw_return"], 0.02)

    def test_healthy_only(self):
        obs = [self._obs("CANDIDATE", "A", "HEALTHY"), self._obs("CANDIDATE", "B", "DEGRADED")]
        self.assertEqual([r["symbol"] for r in oi.research_export(obs, healthy_only=True)], ["A"])


class OutcomeValidationTests(unittest.TestCase):
    def test_outcome_before_observation_flagged(self):
        oc = {"data_status": "MATURED", "evaluation_time": "2026-09-22T13:00:00+00:00"}
        issues = oi.validate_outcome(oc, observation_timestamp="2026-09-22T14:00:00+00:00")
        self.assertIn("outcome_before_observation", issues)

    def test_nonfinite_flagged(self):
        self.assertTrue(any("nonfinite" in i for i in
                            oi.validate_outcome({"raw_return": float("inf")})))

    def test_valid_outcome_ok(self):
        oc = {"data_status": "MATURED", "evaluation_time": "2026-09-22T14:15:00+00:00",
              "raw_return": 0.01, "mfe": 0.02, "mae": -0.005}
        self.assertEqual(oi.validate_outcome(oc, observation_timestamp="2026-09-22T14:00:00+00:00"), [])


class ResearchReportTests(unittest.TestCase):
    def test_report_cohorts_and_health(self):
        obs = ([{"research_cohort": "CANDIDATE", "market_context": {"scan_id": "r"},
                 "timestamp": "2026-09-22T14:00:00+00:00"}]
               + rc.build_near_miss_observations([_row("A")], universe="U",
                 scan_timestamp="2026-09-22T14:00:00+00:00", scan_id="r")
               + rc.build_control_observations(["C"], {}, universe="U",
                 scan_timestamp="2026-09-22T14:00:00+00:00", scan_id="r"))
        rep = oi.research_dataset_report(obs)
        self.assertEqual(rep["cohorts"]["NEAR_MISS"]["count"], 1)
        self.assertEqual(rep["cohorts"]["CONTROL"]["count"], 1)
        self.assertIn("+5m", rep["cohorts"]["NEAR_MISS"]["outcome_maturity"])
        self.assertEqual(rep["point_in_time_violations"], 0)

    def test_missing_cohort_is_degraded(self):
        # only candidates → missing NEAR_MISS/CONTROL → DEGRADED (obvious)
        rep = oi.research_dataset_report([{"research_cohort": "CANDIDATE",
                                          "timestamp": "t", "market_context": {}}])
        self.assertEqual(rep["dataset_health"], "DEGRADED")


class EngineSinkTests(unittest.TestCase):
    def test_research_sink_param_exists(self):
        import inspect

        from scan.engine import run_breakout_scan
        self.assertIn("research_sink", inspect.signature(run_breakout_scan).parameters)


if __name__ == "__main__":
    unittest.main()
