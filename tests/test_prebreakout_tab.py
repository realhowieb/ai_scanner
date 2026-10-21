"""PreBreakout tab and scan alerts list the model's strongest names in each scan."""
import json
import unittest
from unittest import mock

from tests.test_alert_rules_engine import observation, opp


def scored(score, rank, raw, prob, prob_rank, **kw):
    return {**opp(score, rank, **kw), "prob_raw": raw, "prob": prob, "prob_rank": prob_rank}


class PicksTests(unittest.TestCase):
    def test_strongest_raw_scores_first_and_unscored_never(self):
        from ui.results_intelligence import PREBREAKOUT_TOP_N, prebreakout_picks

        self.assertEqual(PREBREAKOUT_TOP_N, 10)
        opps = [{"ticker": f"T{i}", "prob_raw": i / 100, "prob": 13.1 + i / 10} for i in range(15)]
        opps.append({"ticker": "NONE", "prob_raw": None, "prob": None})
        self.assertEqual([o["ticker"] for o in prebreakout_picks(opps)], [f"T{i}" for i in range(14, 4, -1)])
        self.assertEqual(len(prebreakout_picks(opps[:3])), 3)
        self.assertEqual(prebreakout_picks([{"prob": None}]), [])

    def test_no_raw_score_falls_back_to_rank_then_percent(self):
        from ui.results_intelligence import prebreakout_picks

        opps = [{"t": "A", "prob": 20.0, "prob_rank": 3}, {"t": "B", "prob": 31.0, "prob_rank": 1},
                {"t": "C", "prob": 25.0}]
        self.assertEqual([o["t"] for o in prebreakout_picks(opps, 2)], ["B", "A"])


class LatestScanTests(unittest.TestCase):
    def test_tab_keeps_top_model_names_in_hsf_order(self):
        from api import scans

        # HSF order A..L; the model likes L, K, ... best. Every % is below the old 50% cut.
        opps = [{"ticker": chr(65 + i), "score": 90 - i, "signals": [], "prob_raw": i / 100,
                 "prob": 20.0 + i, "prob_rank": 12 - i} for i in range(12)]
        run = {"id": 1, "created_at": None}
        with mock.patch.object(scans, "market_runs", return_value=[run]), \
                mock.patch.object(scans, "run_opportunities", return_value=opps), \
                mock.patch.object(scans, "scan_freshness", return_value={"stale": False}), \
                mock.patch.object(scans, "run_freshness", return_value={}), \
                mock.patch.object(scans, "_iso", return_value=None):
            r = scans.latest_scan({"can_early_breakout": True}, "premium", limit=50, offset=0, signal="prebreakout")
        self.assertEqual((r["total"], r["scan_total"]), (10, 12))
        self.assertEqual([s["ticker"] for s in r["setups"]], list("CDEFGHIJKL"))


class ScanAlertTests(unittest.TestCase):
    def test_prebreakout_scan_alert_matches_the_tab(self):
        from api.alert_rules import scan_matches

        ranked = {f"T{i:02d}": scored(70, i + 1, raw=(20 - i) / 100, prob=30.0, prob_rank=i + 1) for i in range(12)}
        rule = {"value": json.dumps({"setup": "prebreakout"}), "threshold": 0}
        obs = observation(1, ranked)
        self.assertEqual(scan_matches(obs, rule, premium=True), [f"T{i:02d}" for i in range(10)])
        self.assertEqual(scan_matches(obs, rule, premium=False), [])


if __name__ == "__main__":
    unittest.main()
