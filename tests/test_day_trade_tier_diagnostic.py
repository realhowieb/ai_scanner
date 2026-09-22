"""Read-only DT tier diagnostics and historical observation plumbing."""
import json
import unittest

from analytics.day_trade_tier_diagnostic import diagnose_row, summarize
from scripts.diagnose_dt_tiers import build_diagnostic, render_markdown
from scripts.validate_day_trade_score import build_observation


def features(**overrides):
    row = {"chg_pct": 1.5, "gap_pct": 0.8, "rvol": 2.5,
           "vs_vwap_pct": 0.8, "adx": 30, "supertrend_direction": "green", "ewo": 3.0}
    row.update(overrides)
    return row


class TierDiagnosticTests(unittest.TestCase):
    def test_strong_gates_match_production(self):
        row = diagnose_row(features())
        self.assertEqual(row["quality"], "strong")
        self.assertTrue(all(row["gates"].values()))
        self.assertEqual(row["score"], row["score_components"]["calculated_score"])
        self.assertEqual(row["directional_signals"], 5)

    def test_conflict_gate_sensitivity_does_not_change_classifier(self):
        row = diagnose_row(features(chg_pct=-2, gap_pct=0.8, ewo=3,
                                    supertrend_direction="green", vs_vwap_pct=0.8,
                                    adx=40, rvol=4))
        report = summarize([row])
        self.assertEqual(report["gate_sensitivity"]["current"], int(all(row["gates"].values())))
        self.assertEqual(report["gate_sensitivity"]["without_conflict_gate"],
                         int(all(row["gates"][key] for key in row["gates"] if key != "conflict_gate")))
        self.assertIn("Momentum disagreement", report["conflict_frequency"])

    def test_v2_decomposition_uses_rejected_candidate_ceilings(self):
        row = diagnose_row(features(adx=40, rvol=3), profile="rejected_v2")
        self.assertEqual(row["score"], 65.3)  # a6820a9 reference result
        self.assertEqual(row["quality"], "strong")
        self.assertLess(row["score_components"]["subscores"]["adx"], 1)
        self.assertLess(row["score_components"]["subscores"]["rvol"], 1)
        self.assertEqual(row["directional_signals"], 5)
        self.assertEqual(row["score"], row["score_components"]["calculated_score"])

    def test_v2_replay_uses_candidate_score_not_production_score(self):
        original = features(adx=40, rvol=3)
        observation = build_observation(timestamp="T", ticker="ABC", features=original,
                                        prices_after=[100] + [101] * 61)
        report = build_diagnostic([observation], profile="rejected_v2")
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["score_or_direction_mismatches"], 0)
        self.assertEqual(report["directional_n"], 1)
        self.assertNotEqual(diagnose_row(original, profile="rejected_v2")["score"], observation["score"])

    def test_rejected_v2_has_reachable_strong_and_weak_tiers(self):
        strong = diagnose_row(features(adx=40, rvol=3), profile="rejected_v2")
        weak = diagnose_row(features(chg_pct=-2, gap_pct=1, rvol=0.8,
                                     vs_vwap_pct=0.4, adx=14,
                                     supertrend_direction="red", ewo=-2),
                            profile="rejected_v2")
        self.assertEqual(strong["quality"], "strong")
        self.assertEqual(weak["quality"], "weak")
        self.assertTrue(weak["weak_gates"]["low_rvol"])

    def test_high_raw_score_can_still_be_developing(self):
        row = diagnose_row(features(chg_pct=-5, gap_pct=-5, rvol=5,
                                    vs_vwap_pct=2, adx=55,
                                    supertrend_direction="green", ewo=1),
                           profile="rejected_v2")
        self.assertGreater(row["score_components"]["raw_score"], 70)
        self.assertEqual(row["quality"], "developing")
        self.assertFalse(row["gates"]["agreement_gate"])
        self.assertFalse(row["gates"]["conflict_gate"])
        self.assertEqual(row["conflict_count"], 2)

    def test_blocker_counts_and_weak_precedence_are_reported(self):
        rows = [diagnose_row(features(adx=40, rvol=3), profile="rejected_v2"),
                diagnose_row(features(chg_pct=-5, gap_pct=-5, rvol=5,
                                      vs_vwap_pct=2, adx=55,
                                      supertrend_direction="green", ewo=1),
                             profile="rejected_v2")]
        report = summarize(rows, profile="rejected_v2")
        self.assertEqual(report["directional_n"], 2)
        self.assertEqual(report["quality"]["strong"], 1)
        self.assertEqual(report["quality"]["developing"], 1)
        self.assertEqual(report["high_raw_blockers"]["n"], 1)
        self.assertEqual(report["high_raw_blockers"]["individual"]["conflict_gate"], 1)

    def test_report_preserves_per_observation_evidence_and_strict_json(self):
        observation = build_observation(timestamp="T", ticker="ABC", features=features(),
                                        prices_after=[100] + [101] * 61)
        report = build_diagnostic([observation], profile="rejected_v2")
        recorded = report["observations"][0]
        self.assertIn("raw_score", recorded["score_components"])
        self.assertIn("weak_gates", recorded)
        self.assertIn("confirmation_count", recorded)
        self.assertIn("Weak Tier Reachability", render_markdown(report))
        json.dumps(report, allow_nan=False)

    def test_observation_keeps_indicator_inputs_separate_from_outcomes(self):
        original = features()
        observation = build_observation(timestamp="T", ticker="ABC", features=original,
                                        prices_after=[100] + [101] * 61)
        self.assertEqual(observation["diagnostic_inputs"], original)
        self.assertNotIn("return_15m", observation["diagnostic_inputs"])
        report = build_diagnostic([observation], profile="production_v1")
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["directional_n"], 1)

    def test_missing_inputs_do_not_fabricate_gate_counts(self):
        report = build_diagnostic([{"score": 77.1, "direction": "bullish"}], profile="rejected_v2")
        self.assertEqual(report["status"], "INCOMPLETE_OR_MISMATCHED_INPUTS")
        self.assertNotIn("gate_pass", report)
        self.assertIn("can be inferred", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
