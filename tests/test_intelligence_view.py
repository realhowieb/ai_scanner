"""Run 49 — canonical IntelligenceView consolidation tests (presentation only)."""
import unittest

from analytics import intelligence_view as iv


def _obs(symbol="NVDA", direction="long", health="HEALTHY", **over):
    o = {
        "observation_id": "o1", "symbol": symbol, "timestamp": "2026-09-22T14:00:00+00:00",
        "scan_timestamp": "2026-09-22T14:03:00+00:00", "session": "morning",
        "market": {"price": 131.0, "volume": 5_000_000},
        "indicators": {"rvol": 3.1, "vs_vwap_pct": 0.8, "adx": 28, "chg_pct": 1.5,
                       "gap_pct": 0.4, "ema9": 130.0, "ema21": 128.0},
        "models": {"prebreakout": {"version": "prebreakout-xgb-v16", "probability": 0.74},
                   "ai_confidence": {"version": "ai-confidence-xgb-v1", "confidence": 0.68}},
        "scanners": [{"name": "prebreakout", "triggered": True, "direction": direction},
                     {"name": "momentum", "triggered": True, "direction": direction}],
        "market_context": {"coverage_health": health, "scan_id": "run1", "market_regime": "bullish"},
        "data_quality": {"feature_completeness": 1.0, "fallback_used": False, "stale": False},
        "versions": {"schema": "hsf-obs-1.0", "prebreakout_model": "prebreakout-xgb-v16"},
    }
    o.update(over)
    return o


class ViewTests(unittest.TestCase):
    def test_creation_and_terminology(self):
        v = iv.build_intelligence_view(_obs())
        self.assertEqual(v["schema_version"], iv.SCHEMA_VERSION)
        self.assertEqual(v["direction"], "LONG")  # bullish → LONG (terminology)
        self.assertIn(v["lifecycle"], ("NEW", "ACTIVE", "STRENGTHENING", "WEAKENING", "RESOLVED"))

    def test_scores_surfaced_exactly_not_combined(self):
        v = iv.build_intelligence_view(_obs(), opportunity_score=87)
        self.assertEqual(v["scores"]["opportunity_score"], 87.0)
        self.assertEqual(v["scores"]["prebreakout"], 0.74)     # exact
        self.assertEqual(v["scores"]["ml_probability"], 0.68)  # exact
        self.assertIn(v["scores"]["alert_priority"], ("HIGH", "MEDIUM", "LOW"))
        self.assertNotIn("hsf_score", v["scores"])  # no composite

    def test_unknown_is_none_not_zero(self):
        o = _obs(models={}, market={"price": 100.0})
        v = iv.build_intelligence_view(o)  # no scores provided
        self.assertIsNone(v["scores"]["opportunity_score"])
        self.assertIsNone(v["scores"]["prebreakout"])   # None, not 0
        self.assertIsNone(v["scores"]["ml_probability"])

    def test_deterministic_serialization(self):
        self.assertEqual(iv.to_dict(iv.build_intelligence_view(_obs())),
                         iv.to_dict(iv.build_intelligence_view(_obs())))


class FactorTests(unittest.TestCase):
    def test_factors_trace_to_real_values(self):
        f = iv.build_factors(_obs())
        by_code = {x["code"]: x for x in f["supporting"]}
        self.assertIn("RVOL_EXPANSION", by_code)
        self.assertEqual(by_code["RVOL_EXPANSION"]["value"], 3.1)   # real value
        self.assertEqual(by_code["RVOL_EXPANSION"]["source"], "indicators.rvol")
        self.assertIn("ABOVE_VWAP", by_code)
        self.assertIn("EMA_ALIGNMENT", by_code)
        self.assertIn("PREBREAKOUT_MODEL", by_code)

    def test_no_unsupported_factor_when_absent(self):
        o = _obs(indicators={"vs_vwap_pct": 0.5}, models={})  # no rvol/adx/ema/prebreak
        f = iv.build_factors(o)
        codes = {x["code"] for x in f["supporting"] + f["caution"]}
        self.assertNotIn("RVOL_EXPANSION", codes)
        self.assertNotIn("PREBREAKOUT_MODEL", codes)
        self.assertNotIn("EMA_ALIGNMENT", codes)

    def test_caution_factors_from_data(self):
        o = _obs(indicators={"vs_vwap_pct": -0.5, "rvol": 0.6, "adx": 12, "gap_pct": 6.0})
        codes = {x["code"] for x in iv.build_factors(o)["caution"]}
        self.assertIn("BELOW_VWAP", codes)
        self.assertIn("LOW_PARTICIPATION", codes)
        self.assertIn("WEAK_TREND", codes)
        self.assertIn("LARGE_GAP", codes)

    def test_confirmations_conflicts_consistent(self):
        v = iv.build_intelligence_view(_obs())
        self.assertEqual(v["confirmations"], v["supporting_factors"])
        self.assertEqual(v["conflicts"], v["caution_factors"])


class ScanHealthEvidenceTests(unittest.TestCase):
    def test_scan_health_not_prediction_confidence(self):
        v = iv.build_intelligence_view(_obs(health="DEGRADED"))
        self.assertEqual(v["scan_health"]["coverage_health"], "DEGRADED")
        self.assertIn("not the reliability", v["scan_health"]["note"].lower())

    def test_research_evidence_default_unvalidated(self):
        v = iv.build_intelligence_view(_obs())
        self.assertEqual(v["research_evidence"]["level"], "UNVALIDATED")
        self.assertEqual(v["research_evidence"]["note"], "no production recommendation")

    def test_bad_evidence_level_falls_back(self):
        v = iv.build_intelligence_view(_obs(), research_evidence="FIXTURE_STRONG")
        self.assertEqual(v["research_evidence"]["level"], "UNVALIDATED")  # never trust arbitrary


class StalenessProvenanceTests(unittest.TestCase):
    def test_staleness_and_provenance(self):
        v = iv.build_intelligence_view(_obs())
        self.assertEqual(v["staleness"]["scan_run_id"], "run1")
        self.assertEqual(v["staleness"]["market_data_as_of"], "2026-09-22T14:03:00+00:00")
        self.assertEqual(v["provenance"]["prebreakout_model"], "prebreakout-xgb-v16")

    def test_no_pii(self):
        blob = str(iv.build_intelligence_view(_obs())).lower()
        for pii in ("email", "@", "password", "user_id", "billing"):
            self.assertNotIn(pii, blob)


class Run48ReadinessTests(unittest.TestCase):
    def test_insufficient_when_no_outcomes(self):
        obs = [_obs("A"), _obs("B")]  # HEALTHY but no matured outcomes
        r = iv.run48_readiness(obs)
        self.assertEqual(r["evidence_level"], "INSUFFICIENT")
        self.assertEqual(r["recommendation"], "CONTINUE_ACCUMULATING")
        self.assertEqual(r["promotion_gate"], "no production recommendation")

    def test_counts_reported(self):
        r = iv.run48_readiness([_obs("A"), _obs("B", timestamp="2026-09-23T14:00:00+00:00")])
        self.assertEqual(r["trading_days"], 2)
        self.assertIn("+60m", r["matured_paired_by_horizon"])


class ConsistencyTests(unittest.TestCase):
    def test_shares_opportunity_view_semantics(self):
        # IntelligenceView direction/lifecycle come from the SAME engine that
        # Market Brief / Watchlist / Replay use → consistent classifications.
        from analytics import opportunity_view as ov
        o = _obs()
        base = ov.build_opportunity_view(o)
        v = iv.build_intelligence_view(o)
        self.assertEqual(v["lifecycle"], base["lifecycle_state"])
        self.assertEqual(v["scores"]["alert_priority"], base["alert_priority"])
        self.assertEqual(v["agreement"]["count"], base["scanner_count"])


if __name__ == "__main__":
    unittest.main()
