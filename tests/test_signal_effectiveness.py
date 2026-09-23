"""Run 48 — signal effectiveness framework tests (deterministic)."""
import unittest

from analytics import signal_effectiveness as se


def _obs(cohort="CANDIDATE", direction="long", ret=None, mfe=0.02, mae=-0.01,
         rvol=2.0, prebreak=0.7, health="HEALTHY", regime="bullish", rank=None,
         horizons=("+60m",)):
    outcomes = {}
    for h in horizons:
        if ret is not None:
            outcomes[h] = {"data_status": "MATURED", "raw_return": ret,
                           "mfe": mfe, "mae": mae}
    return {
        "research_cohort": cohort,
        "market_context": {"research_cohort": cohort, "coverage_health": health,
                           "market_regime": regime},
        "session": "morning", "rank": rank,
        "scanners": [{"name": "breakout", "triggered": True, "direction": direction}],
        "indicators": {"rvol": rvol, "adx": 25, "vs_vwap_pct": 0.5},
        "models": {"prebreakout": {"probability": prebreak}},
        "market": {"price": 100.0, "volume": 1e6},
        "outcomes": outcomes,
    }


class RecordTests(unittest.TestCase):
    def test_directional_return_long_short(self):
        r_long = se.observation_to_record(_obs(direction="long", ret=0.02))
        r_short = se.observation_to_record(_obs(direction="short", ret=0.02))
        self.assertEqual(r_long["outcomes"]["+60m"]["directional_return"], 0.02)
        self.assertEqual(r_short["outcomes"]["+60m"]["directional_return"], -0.02)

    def test_pending_outcome_excluded(self):
        o = _obs(ret=0.02)
        o["outcomes"]["+60m"]["data_status"] = "PENDING"
        rec = se.observation_to_record(o)
        self.assertEqual(rec["outcomes"], {})  # non-matured dropped


class EvidenceLevelTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(se.evidence_level(10), "INSUFFICIENT")
        self.assertEqual(se.evidence_level(50), "PRELIMINARY")
        self.assertEqual(se.evidence_level(200), "MODERATE")
        self.assertEqual(se.evidence_level(1000), "STRONG")


class BootstrapTests(unittest.TestCase):
    def test_deterministic_with_seed(self):
        vals = [0.01 * (i % 7 - 3) for i in range(60)]
        a = se.bootstrap_ci(vals, seed=42)
        b = se.bootstrap_ci(vals, seed=42)
        self.assertEqual(a, b)                        # reproducible
        self.assertTrue(a["sufficient"])
        self.assertIsNotNone(a["ci"][0])

    def test_insufficient_no_ci(self):
        r = se.bootstrap_ci([0.01, 0.02, 0.03])
        self.assertFalse(r["sufficient"])
        self.assertEqual(r["ci"], (None, None))


class HorizonStatsTests(unittest.TestCase):
    def test_win_rate_and_median(self):
        recs = [se.observation_to_record(_obs(ret=0.02)) for _ in range(20)] \
             + [se.observation_to_record(_obs(ret=-0.01)) for _ in range(20)]
        s = se.horizon_stats(recs, "+60m")
        self.assertEqual(s["n"], 40)
        self.assertEqual(s["win_rate"], 0.5)
        self.assertEqual(s["evidence"], "PRELIMINARY")

    def test_empty_group(self):
        self.assertEqual(se.horizon_stats([], "+60m")["evidence"], "INSUFFICIENT")

    def test_mfe_mae_aggregation(self):
        recs = [se.observation_to_record(_obs(ret=0.01, mfe=0.03, mae=-0.02)) for _ in range(30)]
        s = se.horizon_stats(recs, "+60m")
        self.assertAlmostEqual(s["median_mfe"], 0.03)
        self.assertAlmostEqual(s["median_mae"], -0.02)


class CohortLiftTests(unittest.TestCase):
    def _records(self, cand_ret, ctrl_ret, nm_ret, n=40):
        recs = []
        for _ in range(n):
            recs.append(se.observation_to_record(_obs("CANDIDATE", ret=cand_ret)))
            recs.append(se.observation_to_record(_obs("CONTROL", ret=ctrl_ret)))
            recs.append(se.observation_to_record(_obs("NEAR_MISS", ret=nm_ret)))
        return recs

    def test_candidate_lift_positive(self):
        lift = se.selection_lift(self._records(0.02, 0.0, 0.01), "+60m")
        self.assertGreater(lift["candidate_vs_control"]["estimate"], 0)
        self.assertGreater(lift["candidate_vs_near_miss"]["estimate"], 0)

    def test_lift_insufficient_small_sample(self):
        recs = self._records(0.02, 0.0, 0.01, n=5)  # <MIN_SAMPLE
        self.assertEqual(se.selection_lift(recs, "+60m")["candidate_vs_control"]["evidence"],
                         "INSUFFICIENT")

    def test_cohort_performance_groups(self):
        perf = se.cohort_performance(self._records(0.02, 0.0, 0.01))
        self.assertIn("CANDIDATE", perf)
        self.assertIn("CONTROL", perf)
        self.assertEqual(perf["CANDIDATE"]["+60m"]["n"], 40)


class SignalEffTests(unittest.TestCase):
    def test_monotonic_signal(self):
        # higher rvol → higher return (monotonic positive)
        recs = []
        for i in range(80):
            rvol = 1 + (i // 20)  # 1..4
            recs.append(se.observation_to_record(_obs(rvol=rvol, ret=0.005 * (i // 20))))
        eff = se.signal_effectiveness(recs, "rvol", "+60m", n_bins=4)
        self.assertGreaterEqual(eff["n"], 30)
        self.assertTrue(eff["monotonic"])

    def test_missing_feature_insufficient(self):
        recs = [se.observation_to_record(_obs(ret=0.01)) for _ in range(5)]
        self.assertEqual(se.signal_effectiveness(recs, "ema9", "+60m")["evidence"],
                         "INSUFFICIENT")


class GroupingTests(unittest.TestCase):
    def test_direction_and_regime_grouping(self):
        recs = [se.observation_to_record(_obs(direction="long", regime="bullish", ret=0.02)) for _ in range(30)] \
             + [se.observation_to_record(_obs(direction="short", regime="bearish", ret=0.01)) for _ in range(30)]
        bydir = se.group_analysis(recs, "direction", "+60m")
        self.assertIn("long", bydir)
        self.assertIn("short", bydir)
        byreg = se.group_analysis(recs, "regime", "+60m")
        self.assertIn("bullish", byreg)


class RankTests(unittest.TestCase):
    def test_rank_buckets(self):
        recs = [se.observation_to_record(_obs(rank=i, ret=0.02)) for i in range(40)]
        rb = se.rank_bucket_analysis(recs, "+60m", buckets=4)
        self.assertEqual(len(rb["buckets"]), 4)

    def test_rank_insufficient(self):
        self.assertEqual(se.rank_bucket_analysis([], "+60m")["evidence"], "INSUFFICIENT")


class ScorecardReportTests(unittest.TestCase):
    def test_scorecard_and_recommendations(self):
        recs = [se.observation_to_record(_obs(rvol=1 + i // 20, ret=0.005 * (i // 20)))
                for i in range(80)]
        sc = se.signal_scorecard(recs)
        self.assertIn("rvol", sc)
        self.assertIn(sc["rvol"]["status"],
                      ("STRONG POSITIVE", "POSITIVE", "MIXED", "NEUTRAL",
                       "NEGATIVE", "INSUFFICIENT DATA"))
        recms = se.build_recommendations(sc, {})
        self.assertTrue(all("recommended_action" in r for r in recms))

    def test_insufficient_dataset_verdict(self):
        # a handful of records → INSUFFICIENT LIVE DATA (honest)
        obs = [_obs("CANDIDATE", ret=0.02), _obs("CONTROL", ret=0.0)]
        rep = se.build_analysis_report(obs)
        self.assertEqual(rep["effectiveness_verdict"], "INSUFFICIENT LIVE DATA")
        self.assertFalse(rep["dataset"]["sufficient_for_conclusions"])

    def test_healthy_only_filter(self):
        obs = [_obs("CANDIDATE", ret=0.02, health="HEALTHY"),
               _obs("CANDIDATE", ret=0.02, health="DEGRADED")]
        rep = se.build_analysis_report(obs, healthy_only=True)
        self.assertEqual(rep["dataset"]["total_records"], 1)  # degraded excluded

    def test_manifest_separates_prespecified_exploratory(self):
        rep = se.build_analysis_report([_obs(ret=0.01)])
        self.assertIn("pre_specified", rep["manifest"])
        self.assertIn("exploratory", rep["manifest"])


if __name__ == "__main__":
    unittest.main()
