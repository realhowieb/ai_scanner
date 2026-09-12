import unittest

from ui import opportunities as op
from ui import results_intelligence as ri


def _row(t, **kw):
    r = {"Ticker": t}
    r.update(kw)
    return r


class ConsolidationTests(unittest.TestCase):
    def _rows(self):
        return [
            _row("NVDA", IsBreakout=True, EMACross="Golden", **{"PreBreakoutProb%": 78},
                 PctChange=2.4, BreakoutScore=74, Last=120.0),
            _row("NVDA", IsBreakout=True, PctChange=2.4, BreakoutScore=70),  # dup -> merge
            _row("AMD", IsBreakout=True, **{"PreBreakoutProb%": 55}, PctChange=1.0, BreakoutScore=60),
            _row("HPE", IsBreakout=True, PctChange=3.1, GapPct=2.5, BreakoutScore=48),
            _row("ZZZ", IsBreakout=False, PctChange=0.1),  # no signal/model -> excluded
        ]

    def test_reuses_canonical_score_and_version(self):
        opps = ri.consolidate_scanner_results(self._rows())
        self.assertEqual(opps[0]["score_version"], op.HSF_SCORE_VERSION)
        for o in opps:
            # Score exactly matches the canonical breakdown for its inputs.
            expect = op.build_opportunity_score(
                n_signals=o["n_signals"], breakout_score=o["breakout_score"],
                prob=o["prob"], chg_pct=o["chg_pct"], fading=o["fading"])
            self.assertEqual(o["score"], expect)

    def test_duplicate_ticker_consolidated_and_signals_merged(self):
        opps = ri.consolidate_scanner_results(self._rows())
        tickers = [o["ticker"] for o in opps]
        self.assertEqual(len(tickers), len(set(tickers)))  # no dup rows
        nvda = next(o for o in opps if o["ticker"] == "NVDA")
        self.assertGreaterEqual(nvda["n_signals"], 3)  # breakout+golden+prebreakout+gainer

    def test_non_qualifying_excluded(self):
        opps = ri.consolidate_scanner_results(self._rows())
        self.assertNotIn("ZZZ", [o["ticker"] for o in opps])

    def test_deterministic_ranking_is_stable(self):
        rows = [_row("BBB", IsBreakout=True, EMACross="Golden", PctChange=3.0, BreakoutScore=60),
                _row("AAA", IsBreakout=True, EMACross="Golden", PctChange=3.0, BreakoutScore=60)]
        a = ri.consolidate_scanner_results(rows)
        b = ri.consolidate_scanner_results(list(reversed(rows)))
        self.assertEqual([o["ticker"] for o in a], [o["ticker"] for o in b])  # tie -> alpha, stable

    def test_empty_and_malformed_safe(self):
        self.assertEqual(ri.consolidate_scanner_results([]), [])
        self.assertEqual(ri.consolidate_scanner_results([{"foo": 1}, None, {"Ticker": ""}]), [])

    def test_missing_price_or_chg_is_safe(self):
        opps = ri.consolidate_scanner_results([_row("X", IsBreakout=True, BreakoutScore=55)])
        self.assertEqual(opps[0]["ticker"], "X")  # scores off model alone, no crash


class MovementViewTests(unittest.TestCase):
    def _opps(self):
        return ri.consolidate_scanner_results([
            _row("NVDA", IsBreakout=True, EMACross="Golden", **{"PreBreakoutProb%": 78}, PctChange=2.4, BreakoutScore=74),
            _row("AMD", IsBreakout=True, **{"PreBreakoutProb%": 55}, PctChange=1.0, BreakoutScore=60),
            _row("HPE", IsBreakout=True, PctChange=3.1, GapPct=2.5, BreakoutScore=48),
        ])

    def test_movement_and_transitions(self):
        prev = [{"ticker": "NVDA", "score": 80, "status": "WATCH", "score_version": "1.0"}]
        comp = ri.enrich_movement(self._opps(), prev)
        by = {c["ticker"]: c for c in comp}
        self.assertEqual(by["NVDA"]["movement_state"], "RISING")
        self.assertEqual(by["HPE"]["movement_state"], "NEW")

    def test_score_version_mismatch_suppresses_delta(self):
        prev = [{"ticker": "AMD", "score": 40, "status": "CAUTION", "score_version": "0.9"}]
        comp = ri.enrich_movement(self._opps(), prev)
        amd = next(c for c in comp if c["ticker"] == "AMD")
        self.assertEqual(amd["movement_state"], "VERSION_CHANGED")
        self.assertIsNone(amd["score_delta"])
        self.assertIsNone(amd["status_transition"])

    def test_no_previous_all_new_but_summary_hides_counts(self):
        comp = ri.enrich_movement(self._opps(), None)
        self.assertTrue(all(c["movement_state"] == "NEW" for c in comp))
        s = ri.summarize_results(comp, total_matches=10, has_previous=False)
        self.assertNotIn("new", s)   # no valid previous -> don't surface NEW counts
        self.assertNotIn("rising", s)

    def test_classify_views(self):
        prev = [{"ticker": "NVDA", "score": 80, "status": "WATCH", "score_version": "1.0"}]
        comp = ri.enrich_movement(self._opps(), prev)
        v = ri.classify_views(comp)
        self.assertEqual(v["top"][0]["ticker"], "NVDA")
        self.assertIn("HPE", [c["ticker"] for c in v["new"]])

    def test_fading_view_uses_real_flags_only(self):
        comp = ri.enrich_movement(ri.consolidate_scanner_results([
            _row("FAD", IsBreakout=True, EMACross="Golden", PctChange=-3.0, BreakoutScore=55),
        ]), None)
        v = ri.classify_views(comp)
        self.assertIn("FAD", [c["ticker"] for c in v["fading"]])

    def test_summary_counts(self):
        prev = [{"ticker": "NVDA", "score": 70, "status": "WATCH", "score_version": "1.0"}]
        comp = ri.enrich_movement(self._opps(), prev)
        s = ri.summarize_results(comp, total_matches=20, has_previous=True, regime="RISK-ON")
        self.assertEqual(s["total_matches"], 20)
        self.assertEqual(s["opportunities"], 3)
        self.assertEqual(s["regime"], "RISK-ON")
        self.assertIn("new", s)


class ExplanationGroundingTests(unittest.TestCase):
    def test_explanation_only_real_fields(self):
        opps = ri.consolidate_scanner_results([
            _row("NVDA", IsBreakout=True, EMACross="Golden", **{"PreBreakoutProb%": 78}, PctChange=2.4, BreakoutScore=74)])
        ex = op.build_opportunity_explanation(opps[0], earnings_today=[])
        joined = " ".join(ex["reasons"]).lower()
        self.assertIn("golden cross", joined)
        self.assertNotIn("rvol", joined)
        self.assertNotIn("resistance", joined)
        self.assertNotIn("$", " ".join(ex["reasons"]))


if __name__ == "__main__":
    unittest.main()
