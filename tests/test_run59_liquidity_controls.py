"""Run 59 (owner decision 2026-09-30, Option A): liquidity-comparable controls
and a restarted forward epoch. See docs/RUN59_LIQUIDITY_CONTROLS.md."""
import datetime as dt
import unittest
from pathlib import Path

import pandas as pd

from analytics import forward_readiness as fr
from analytics import maturation_parity as mp
from analytics import research_cohorts as rc
from scan import breakout

ROOT = Path(__file__).resolve().parents[1]


def frame(closes, vols):
    idx = pd.date_range("2026-09-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": vols}, index=idx)


class DollarVolumeParityTests(unittest.TestCase):
    """Controls must be judged by exactly the candidates' liquidity rule."""

    def candidate_rule(self, df):
        m = breakout._compute_basic_metrics(breakout._prep_symbol_df(df))
        close, avg, today = m["Close"], m["VolAvg20"], m["Volume"]
        if avg and avg > 0:
            return close * avg
        return close * today if today and today > 0 else None

    def test_matches_scan_breakout_with_and_without_20_days(self):
        cases = [frame([10.0 + i * 0.1 for i in range(30)], [1e6 + i * 1e4 for i in range(30)]),
                 frame([50.0] * 21, [2e5] * 20 + [9e6]),
                 frame([3.0] * 10, [5e5] * 10)]                     # < 21 bars -> today's volume
        for df in cases:
            self.assertAlmostEqual(rc.dollar_vol20(df), self.candidate_rule(df), places=4)

    def test_uncomputable(self):
        self.assertIsNone(rc.dollar_vol20(frame([5.0] * 5, [0] * 5)))
        self.assertIsNone(rc.dollar_vol20(pd.DataFrame({"Close": []})))


class EligibilityTests(unittest.TestCase):
    SNAP = {"LIQ": {"price": 20.0, "dollar_vol20": 9e6}, "THIN": {"price": 20.0, "dollar_vol20": 1e5},
            "CHEAP": {"price": 0.5, "dollar_vol20": 9e6}, "RICH": {"price": 5000.0, "dollar_vol20": 9e9},
            "NODV": {"price": 20.0, "dollar_vol20": None}, "EDGE": {"price": 1.0, "dollar_vol20": 5e6}}

    def test_same_floor_and_price_range_as_candidates(self):
        out = rc.liquidity_eligible(list(self.SNAP) + ["MISSING"], self.SNAP,
                                    min_dollar_vol=5e6, min_price=1.0, max_price=1000.0)
        self.assertEqual(out, ["LIQ", "EDGE"])

    def test_sample_is_seeded_from_the_eligible_pool(self):
        pool = rc.liquidity_eligible(list(self.SNAP), self.SNAP, min_dollar_vol=5e6, min_price=1.0, max_price=1000.0)
        picked = rc.select_control_symbols(pool, scan_run_id="r1", exclude=["EDGE"], n=100)
        self.assertEqual(picked, ["LIQ"])

    def test_new_controls_are_tagged_and_legacy_ones_unchanged(self):
        new = rc.build_control_observations(["LIQ"], self.SNAP, universe="US_MARKET", scan_timestamp="2026-10-05T13:35:00+00:00",
                                            control_design=rc.CONTROL_DESIGN)[0]
        old = rc.build_control_observations(["LIQ"], self.SNAP, universe="US_MARKET", scan_timestamp="2026-10-05T13:35:00+00:00")[0]
        self.assertEqual(new["market_context"]["control_design"], "run59b_liquidity_matched_disjoint_v2")
        self.assertEqual(new["selection_reason"], "liquidity_matched_sample")
        self.assertNotIn("control_design", old["market_context"])
        self.assertEqual(old["selection_reason"], "deterministic_sample")
        self.assertEqual(new["observation_id"], old["observation_id"])   # ids never depend on the design


class WiringTests(unittest.TestCase):
    def test_cron_draws_controls_from_the_liquidity_pool(self):
        src = (ROOT / "scheduler" / "cron_runner.py").read_text()
        self.assertIn("pool = liquidity_eligible(", src)
        self.assertIn("select_control_symbols(pool, scan_run_id=scan_id,", src)
        self.assertIn("exclude=list(candidate_symbols or []) + near_miss_symbols)", src)
        self.assertIn("control_design=CONTROL_DESIGN", src)

    def test_engine_records_floor_and_dollar_volume_without_changing_output(self):
        src = (ROOT / "scan" / "engine.py").read_text()
        self.assertIn('snap[k]["dollar_vol20"] = dollar_vol20(v)', src)
        self.assertIn('research_sink["control_floor"] = {"min_dollar_vol"', src)
        self.assertIn("df = df.head(top_n)  # production output unchanged", src)


class EpochTests(unittest.TestCase):
    def rec(self, oid, cohort, ts, design=None):
        mc = {"scan_id": "r", "research_cohort": cohort}
        if design:
            mc["control_design"] = design
        return {"observation_id": oid, "research_cohort": cohort, "scan_timestamp": ts, "timestamp": ts,
                "market_context": mc}

    def test_epoch_restarted_and_old_epoch_recorded(self):
        self.assertEqual(fr.FORWARD_EPOCH["forward_epoch_start_timestamp"], "2026-10-05T12:00:00+00:00")
        self.assertEqual(fr.FORWARD_EPOCH["control_design"], rc.CONTROL_DESIGN)
        self.assertEqual(fr.PREVIOUS_EPOCHS[0]["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")
        self.assertEqual((fr.PREVIOUS_EPOCHS[1]["forward_epoch_start_timestamp"], fr.PREVIOUS_EPOCHS[1]["control_design"]),
                         ("2026-10-01T12:00:00+00:00", rc.RUN59_CONTROL_DESIGN))

    def test_gates_are_unchanged(self):
        self.assertEqual(fr.GATES["E_maturation_parity"], {"max_gap_pp": 10.0, "preferred_gap_pp": 5.0})
        self.assertEqual(fr.GATES["A_trading_days"], {"min": 10, "preferred": 20})

    def test_old_design_controls_in_the_epoch_are_never_pooled(self):
        ts, before = "2026-10-05T13:35:00+00:00", "2026-10-02T13:35:00+00:00"
        obs = [self.rec("c1", "CANDIDATE", ts), self.rec("k_new", "CONTROL", ts, rc.CONTROL_DESIGN),
               self.rec("k_old", "CONTROL", ts), self.rec("k_v1", "CONTROL", ts, rc.RUN59_CONTROL_DESIGN),
               self.rec("k_pre", "CONTROL", before, rc.CONTROL_DESIGN)]
        sel = fr.select_forward(obs)
        self.assertEqual(sorted(o["observation_id"] for o in sel["forward"]), ["c1", "k_new"])
        self.assertEqual((sel["legacy_control_design_excluded"], sel["pre_epoch_excluded"]), (2, 1))
        self.assertEqual(sorted(o["observation_id"] for o in mp.population(obs, "forward")), ["c1", "k_new"])

    def test_near_misses_are_never_drawn_as_controls(self):
        pool = ["AAA", "BBB", "CCC", "NM1", "NM2", "CAND"]
        picked = rc.select_control_symbols(pool, scan_run_id="r1", exclude=["CAND", "NM1", "NM2"], n=100)
        self.assertEqual(picked, ["AAA", "BBB", "CCC"])

    def test_run59b_decision_is_pre_registered(self):
        doc = (ROOT / "docs" / "RUN59B_DISJOINT_CONTROLS.md").read_text()
        for must in ("2026-10-05T12:00:00+00:00", "run59b_liquidity_matched_disjoint_v2", "near-miss overlaps=22",
                     "not decided from effectiveness results", "Gates A–H are unchanged"):
            self.assertIn(must, doc)

    def test_decision_is_pre_registered(self):
        doc = (ROOT / "docs" / "RUN59_LIQUIDITY_CONTROLS.md").read_text()
        for must in ("Option A", "2026-10-01T12:00:00+00:00", "run59_liquidity_matched_v1",
                     "not decided from effectiveness results", "Gates A–H are unchanged"):
            self.assertIn(must, doc)


if __name__ == "__main__":
    unittest.main()
