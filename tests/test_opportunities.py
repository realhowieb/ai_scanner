import unittest

from ui import opportunities as op


class OpportunityScoreTests(unittest.TestCase):
    def _data(self):
        return {
            "gappers": [{"ticker": "DELL", "chg_pct": 4.2, "gap_pct": 3.1},
                        {"ticker": "CPRT", "chg_pct": -2.3, "gap_pct": 2.0}],
            "golden": ["NTAP"],
            "top_setups": [("NTAP", 55), ("DELL", 52), ("HPE", 48)],
            "picks": [{"symbol": "NTAP", "prob": 71}],
            "gainers": [("NTAP", 5.1), ("DELL", 4.2)],
            "losers": [("CPRT", -2.3)],
        }

    def test_ranks_by_score_and_confluence(self):
        opps = op.build_opportunities(self._data(), top_n=5)
        self.assertTrue(opps)
        # Sorted descending.
        scores = [o["score"] for o in opps]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # NTAP has the most confluence -> highest, STRONG.
        self.assertEqual(opps[0]["ticker"], "NTAP")
        self.assertEqual(opps[0]["status"], "STRONG")
        self.assertGreaterEqual(opps[0]["n_signals"], 3)

    def test_score_is_bounded_and_missing_data_never_penalized(self):
        for n in range(0, 8):
            s = op.build_opportunity_score(
                n_signals=n, breakout_score=None, prob=None, chg_pct=None, fading=False
            )
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 100)
        # Absurd inputs still clamp to 0..100.
        s = op.build_opportunity_score(
            n_signals=99, breakout_score=9999, prob=999, chg_pct=999, fading=False
        )
        self.assertLessEqual(s, 100)

    def test_fading_demotes_to_caution(self):
        s_plain = op.build_opportunity_score(
            n_signals=3, breakout_score=55, prob=None, chg_pct=4.0, fading=False
        )
        s_fade = op.build_opportunity_score(
            n_signals=3, breakout_score=55, prob=None, chg_pct=4.0, fading=True
        )
        self.assertLess(s_fade, s_plain)
        self.assertEqual(op._status(90, fading=True), "CAUTION")

    def test_explanation_uses_only_real_fields(self):
        opps = op.build_opportunities(self._data(), top_n=5)
        ntap = next(o for o in opps if o["ticker"] == "NTAP")
        ex = op.build_opportunity_explanation(ntap, earnings_today=["NTAP"])
        joined = " ".join(ex["reasons"])
        self.assertIn("golden cross", joined.lower())
        self.assertIn("71%", joined)  # real model prob
        self.assertIn("Earnings reporting today", ex["risks"])
        # No fabricated metrics we didn't pass.
        self.assertNotIn("RVOL", joined)
        self.assertNotIn("resistance", joined.lower())

    def test_empty_and_malformed_inputs_are_safe(self):
        self.assertEqual(op.build_opportunities({}), [])
        self.assertEqual(op.build_opportunities(None), [])
        self.assertEqual(op.build_opportunities({"top_setups": [("BAD",)], "picks": [{}]}), [])

    def test_requires_confluence_or_model_score(self):
        # A single gainer with no model score and no confluence isn't an opportunity.
        opps = op.build_opportunities({"gainers": [("XYZ", 3.0)]})
        self.assertEqual(opps, [])
        # But a single breakout WITH a model score qualifies.
        opps = op.build_opportunities({"top_setups": [("XYZ", 50)]})
        self.assertEqual([o["ticker"] for o in opps], ["XYZ"])


class MarketRegimeTests(unittest.TestCase):
    def test_strong_bullish(self):
        r = op.classify_market_regime(spy_chg=0.84, qqq_chg=0.88, breadth=(142, 27),
                                      sectors=[("Tech", 1.8), ("Semis", 1.5)])
        self.assertEqual(r["regime"], "TRENDING BULLISH")
        self.assertTrue(r["interpretation"])

    def test_strong_bearish(self):
        r = op.classify_market_regime(spy_chg=-0.7, qqq_chg=-0.8, breadth=(27, 142),
                                      sectors=[("Utils", -0.7)])
        self.assertEqual(r["regime"], "TRENDING BEARISH")

    def test_mixed(self):
        r = op.classify_market_regime(spy_chg=0.05, qqq_chg=-0.05, breadth=(100, 100), sectors=[])
        self.assertEqual(r["regime"], "MIXED / CHOPPY")

    def test_missing_and_zero_breadth_and_partial_sectors(self):
        self.assertIsNone(op.classify_market_regime(spy_chg=None, qqq_chg=None, breadth=None)["regime"])
        # Zero breadth doesn't divide-by-zero; index-only -> RISK-ON.
        self.assertEqual(op.classify_market_regime(spy_chg=0.3, qqq_chg=0.3, breadth=(0, 0))["regime"], "RISK-ON")
        # Sectors present but no percentages -> ignored, still classifies.
        r = op.classify_market_regime(spy_chg=0.3, qqq_chg=0.3, breadth=None,
                                      sectors=[("Tech", None)])
        self.assertEqual(r["regime"], "RISK-ON")

    def test_high_volatility_divergence(self):
        r = op.classify_market_regime(spy_chg=1.4, qqq_chg=1.3, breadth=(30, 170), sectors=None)
        self.assertEqual(r["regime"], "HIGH VOLATILITY")


class SnapshotComparisonTests(unittest.TestCase):
    def _cur(self):
        return [
            {"ticker": "NTAP", "score": 92, "status": "STRONG", "fading": False, "n_signals": 4},
            {"ticker": "HPE", "score": 81, "status": "WATCH", "fading": False, "n_signals": 2},
            {"ticker": "CDW", "score": 76, "status": "WATCH", "fading": False, "n_signals": 2},
        ]

    def _prev(self):
        return [
            {"ticker": "NTAP", "score": 81, "status": "WATCH"},
            {"ticker": "CDW", "score": 83, "status": "WATCH"},
            {"ticker": "CPRT", "score": 60, "status": "WATCH"},
        ]

    def test_movement_states_and_badges(self):
        comp = op.compare_opportunities(self._cur(), self._prev())
        by = {c["ticker"]: c for c in comp}
        self.assertEqual(by["NTAP"]["movement_state"], "RISING")
        self.assertEqual(op.movement_badge(by["NTAP"]), "▲ +11")
        self.assertEqual(by["HPE"]["movement_state"], "NEW")
        self.assertEqual(op.movement_badge(by["HPE"]), "NEW")
        self.assertEqual(by["CDW"]["movement_state"], "FALLING")
        self.assertEqual(op.movement_badge(by["CDW"]), "▼ -7")

    def test_status_transition_detected(self):
        comp = op.compare_opportunities(self._cur(), self._prev())
        ntap = next(c for c in comp if c["ticker"] == "NTAP")
        self.assertEqual(ntap["status_transition"], ("WATCH", "STRONG"))

    def test_no_previous_snapshot_is_no_baseline_not_new(self):
        # With no baseline, nothing is genuinely NEW — it's NO_BASELINE.
        comp = op.compare_opportunities(self._cur(), None)
        self.assertTrue(all(c["movement_state"] == "NO_BASELINE" for c in comp))
        self.assertTrue(all(c["score_delta"] is None for c in comp))
        self.assertIsNone(op.summarize_changes(comp, None))

    def test_malformed_previous_records_are_safe(self):
        comp = op.compare_opportunities(self._cur(), [{"score": 50}, {}, {"ticker": None, "score": 9}])
        # No ticker match -> all NEW, no crash.
        self.assertTrue(all(c["movement_state"] == "NEW" for c in comp))

    def test_duplicate_previous_ticker_first_wins(self):
        prev = [{"ticker": "NTAP", "score": 81, "status": "WATCH"},
                {"ticker": "NTAP", "score": 10, "status": "CAUTION"}]
        comp = op.compare_opportunities(self._cur(), prev)
        ntap = next(c for c in comp if c["ticker"] == "NTAP")
        self.assertEqual(ntap["previous_score"], 81)

    def test_summary_and_dropped(self):
        comp = op.compare_opportunities(self._cur(), self._prev())
        s = op.summarize_changes(comp, self._prev())
        self.assertEqual(len(s["new"]), 1)
        self.assertEqual(len(s["strengthened"]), 1)
        self.assertEqual(len(s["weakened"]), 1)
        self.assertEqual(len(s["upgrades"]), 1)
        self.assertIn("CPRT", s["dropped"])
        self.assertEqual(s["biggest_mover"]["ticker"], "NTAP")

    def test_status_transitions_both_directions(self):
        cur = [{"ticker": "A", "score": 60, "status": "WATCH", "fading": False, "n_signals": 2},
               {"ticker": "B", "score": 45, "status": "CAUTION", "fading": True, "n_signals": 2}]
        prev = [{"ticker": "A", "score": 50, "status": "CAUTION"},
                {"ticker": "B", "score": 80, "status": "STRONG"}]
        comp = op.compare_opportunities(cur, prev)
        s = op.summarize_changes(comp, prev)
        self.assertEqual(len(s["upgrades"]), 1)    # A: CAUTION -> WATCH
        self.assertEqual(len(s["downgrades"]), 1)  # B: STRONG -> CAUTION

    def test_watch_next_prioritizes_upgrades_then_fading(self):
        comp = op.compare_opportunities(self._cur(), self._prev())
        wn = op.select_watch_next(comp, limit=3)
        self.assertEqual(wn[0]["ticker"], "NTAP")
        self.assertIn("STRONG", wn[0]["headline"])
        # Deterministic conditions, no invented price/level.
        for it in wn:
            self.assertNotIn("$", it["detail"])

    def test_watch_next_empty_when_nothing_notable(self):
        cur = [{"ticker": "Z", "score": 55, "status": "WATCH", "fading": False, "n_signals": 2}]
        comp = op.compare_opportunities(cur, [{"ticker": "Z", "score": 55, "status": "WATCH"}])
        self.assertEqual(op.select_watch_next(comp), [])


class MovementSemanticsTests(unittest.TestCase):
    """Run 20C — trustworthy movement semantics."""

    def _cur(self, score=80, status="STRONG", ver="1.0"):
        return [{"ticker": "NVDA", "score": score, "status": status, "score_version": ver,
                 "fading": False, "n_signals": 3}]

    def _prev(self, score=77, status="WATCH", ver="1.0"):
        return [{"ticker": "NVDA", "score": score, "status": status, "score_version": ver}]

    def test_no_baseline_distinct_from_new(self):
        self.assertEqual(op.compare_opportunities(self._cur(), None)[0]["movement_state"], "NO_BASELINE")
        self.assertEqual(op.compare_opportunities(self._cur(), [])[0]["movement_state"], "NO_BASELINE")

    def test_genuinely_new_when_absent_from_baseline(self):
        prev = [{"ticker": "AMD", "score": 60, "status": "WATCH", "score_version": "1.0"}]
        self.assertEqual(op.compare_opportunities(self._cur(), prev)[0]["movement_state"], "NEW")

    def test_same_version_allows_delta(self):
        c = op.compare_opportunities(self._cur(80), self._prev(77))[0]
        self.assertEqual(c["score_delta"], 3)
        self.assertEqual(c["movement_state"], "RISING")

    def test_version_mismatch_blocks_delta(self):
        c = op.compare_opportunities(self._cur(86, ver="1.1"), self._prev(72, ver="1.0"))[0]
        self.assertEqual(c["movement_state"], "VERSION_CHANGED")
        self.assertIsNone(c["score_delta"])
        self.assertIsNone(c["status_transition"])
        self.assertIsNone(c["previous_score"])

    def test_missing_previous_version_treated_compatible(self):
        prev = [{"ticker": "NVDA", "score": 77, "status": "WATCH"}]  # no score_version
        c = op.compare_opportunities(self._cur(80), prev)[0]
        self.assertEqual(c["movement_state"], "RISING")  # compatible -> delta computed

    def test_threshold_two_unchanged_three_rising_minus_three_falling(self):
        self.assertEqual(op.compare_opportunities(self._cur(79), self._prev(77))[0]["movement_state"], "UNCHANGED")
        self.assertEqual(op.compare_opportunities(self._cur(80), self._prev(77))[0]["movement_state"], "RISING")
        self.assertEqual(op.compare_opportunities(self._cur(74), self._prev(77))[0]["movement_state"], "FALLING")

    def test_status_transitions_both_directions(self):
        up = op.compare_opportunities(self._cur(85, "STRONG"), self._prev(78, "WATCH"))[0]
        self.assertEqual(up["status_transition"], ("WATCH", "STRONG"))
        down = op.compare_opportunities(self._cur(70, "WATCH"), self._prev(80, "STRONG"))[0]
        self.assertEqual(down["status_transition"], ("STRONG", "WATCH"))

    def test_duplicate_previous_ticker_first_wins(self):
        prev = [{"ticker": "NVDA", "score": 77, "status": "WATCH", "score_version": "1.0"},
                {"ticker": "NVDA", "score": 10, "status": "CAUTION", "score_version": "1.0"}]
        self.assertEqual(op.compare_opportunities(self._cur(80), prev)[0]["previous_score"], 77)

    def test_malformed_previous_payload_is_safe(self):
        comp = op.compare_opportunities(self._cur(80), [{"score": 5}, {}, {"ticker": None}])
        self.assertEqual(comp[0]["movement_state"], "NEW")  # baseline exists, no match

    def test_dropped_means_gone_from_ranking(self):
        cur = [{"ticker": "NVDA", "score": 80, "status": "STRONG", "fading": False, "n_signals": 3}]
        prev = [{"ticker": "NVDA", "score": 77, "status": "WATCH", "score_version": "1.0"},
                {"ticker": "CPRT", "score": 60, "status": "WATCH", "score_version": "1.0"}]
        s = op.summarize_changes(op.compare_opportunities(cur, prev), prev)
        self.assertIn("CPRT", s["dropped"])  # dropped from ranking, NOT a price claim

    def test_movement_badge_states(self):
        self.assertEqual(op.movement_badge({"movement_state": "NO_BASELINE"}), "—")
        self.assertEqual(op.movement_badge({"movement_state": "VERSION_CHANGED"}), "—")
        self.assertEqual(op.movement_badge({"movement_state": "NEW"}), "NEW")
        self.assertEqual(op.movement_badge({"movement_state": "RISING", "score_delta": 11}), "▲ +11")


if __name__ == "__main__":
    unittest.main()
