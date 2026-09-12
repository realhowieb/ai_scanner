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


if __name__ == "__main__":
    unittest.main()
