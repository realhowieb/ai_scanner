"""Run 41 — Market Brief 2.0 composition tests (deterministic, Streamlit-free)."""
import unittest

from analytics import market_brief_view as mb


def _opp(ticker="NVDA", signals=("prebreakout", "breakout", "gapper"),
         prob=78.0, chg=2.1, gap=1.2, fading=False):
    return {"ticker": ticker, "signals": list(signals), "prob": prob,
            "chg_pct": chg, "gap_pct": gap, "fading": fading, "primary_setup": "PreBreakout"}


class AdapterTests(unittest.TestCase):
    def test_opportunity_to_observation(self):
        obs = mb.observation_from_opportunity(_opp())
        self.assertEqual(obs["symbol"], "NVDA")
        names = {s["name"] for s in obs["scanners"]}
        self.assertEqual(names, {"prebreakout", "breakout", "gap_up"})
        self.assertEqual(obs["models"]["prebreakout"]["probability"], 78.0)
        self.assertEqual(obs["indicators"]["chg_pct"], 2.1)
        self.assertNotIn("adx", obs["indicators"])  # not in snapshot → absent

    def test_missing_fields_safe(self):
        obs = mb.observation_from_opportunity({"ticker": "X", "signals": []})
        self.assertEqual(obs["scanners"], [])
        self.assertEqual(obs["models"], {})


class TopOpportunityTests(unittest.TestCase):
    def test_ranked_and_top_n(self):
        opps = [_opp("AAA"), _opp("BBB", signals=("gainer",), prob=None),
                _opp("CCC", signals=("prebreakout", "gapper"), prob=45)]
        views = mb.build_top_opportunity_views(opps, top_n=2)
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["symbol"], "AAA")  # HIGH (3 scanners + 78%)

    def test_fading_adds_risk(self):
        views = mb.build_top_opportunity_views([_opp("ZZZ", signals=("gainer", "gapper"),
                                                     prob=None, fading=True)])
        self.assertIn("Fading / reversal risk", views[0]["risk_reasons"])

    def test_watchlist_flag(self):
        views = mb.build_top_opportunity_views([_opp("NVDA")], watchlist=["NVDA"])
        self.assertTrue(views[0]["is_watchlist"])

    def test_change_detection_via_prior(self):
        prior = {"NVDA": _opp("NVDA", signals=("prebreakout",), prob=61)}
        views = mb.build_top_opportunity_views([_opp("NVDA", signals=("prebreakout", "gapper"),
                                                     prob=76)], prior_by_ticker=prior)
        self.assertTrue(any("61% → 76%" in c for c in views[0]["changes_since_prior"]))


class StateDiffTests(unittest.TestCase):
    def test_summarize(self):
        views = mb.build_top_opportunity_views([_opp("AAA"), _opp("BBB", prob=30,
                                                signals=("gainer", "gapper"))])
        s = mb.summarize_brief_state(views, breadth=(61, 39), regime="Bullish")
        self.assertEqual(s["total_opportunities"], 2)
        self.assertEqual(s["breadth_pct"], 61)
        self.assertEqual(s["regime"], "Bullish")

    def test_diff_no_prior(self):
        self.assertEqual(mb.diff_brief_state({"high_priority": 3}, None), [])

    def test_diff_transitions(self):
        cur = {"high_priority": 9, "regime": "Bullish", "breadth_pct": 61,
               "tickers": ["AMD", "NVDA"]}
        prior = {"high_priority": 4, "regime": "Mixed", "breadth_pct": 52,
                 "tickers": ["NVDA"]}
        changes = mb.diff_brief_state(cur, prior)
        self.assertTrue(any("4 → 9" in c for c in changes))
        self.assertTrue(any("Mixed → Bullish" in c for c in changes))
        self.assertTrue(any("52% → 61%" in c for c in changes))
        self.assertTrue(any("NEW opportunity: AMD" in c for c in changes))

    def test_small_breadth_move_ignored(self):
        changes = mb.diff_brief_state({"breadth_pct": 53, "tickers": []},
                                      {"breadth_pct": 52, "tickers": []})
        self.assertEqual(changes, [])


class WatchlistPulseTests(unittest.TestCase):
    def test_watchlist_events_only_meaningful(self):
        views = mb.build_top_opportunity_views([_opp("NVDA"), _opp("XYZ")],
                                               watchlist=["NVDA"])
        events = mb.build_watchlist_events(views)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["symbol"], "NVDA")

    def test_market_pulse_supported_only(self):
        views = mb.build_top_opportunity_views([_opp("AAA"), _opp("BBB")])
        pulse = mb.market_pulse(views, sectors=[("Tech", 1.2)])
        self.assertTrue(any("Sector leader: Tech" in p for p in pulse))
        self.assertTrue(any("bullish" in p for p in pulse))

    def test_empty_pulse_when_no_data(self):
        self.assertEqual(mb.market_pulse([]), [])


class DeterminismTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(mb.build_top_opportunity_views([_opp()]),
                         mb.build_top_opportunity_views([_opp()]))


if __name__ == "__main__":
    unittest.main()
