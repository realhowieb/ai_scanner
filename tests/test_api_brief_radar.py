"""The Brief's Opportunity Radar reads the Scanner's HSF Scores for the latest saved run."""
import unittest
from unittest import mock

# Imported before any patch: api.scans binds api.today.market_runs at import, so a first
# import under a patch would keep the mock for every later test.
import api.scans  # noqa: F401


def _opp(ticker, score, status="WATCH"):
    return {"ticker": ticker, "score": score, "status": status, "score_version": "1.0",
            "primary_setup": "Golden Cross", "signals": ["golden_cross"], "n_signals": 1}


class ScanRadarTest(unittest.TestCase):
    def test_top_names_use_scanner_scores_with_movement(self):
        from api.market import _scan_radar

        runs = {1: [_opp("CCC", 77), _opp("MANH", 66), _opp("PENG", 58)],
                0: [_opp("MANH", 60), _opp("XP", 50)]}
        with mock.patch("api.today.market_runs", return_value=[{"id": 1}, {"id": 0}]), \
                mock.patch("api.scans.run_opportunities", side_effect=lambda rid: runs[rid]):
            radar, has_previous = _scan_radar()
        self.assertTrue(has_previous)
        self.assertEqual([(o["ticker"], o["score"], o["rank"]) for o in radar],
                         [("CCC", 77, 1), ("MANH", 66, 2), ("PENG", 58, 3)])
        self.assertEqual(radar[0]["movement_state"], "NEW")
        self.assertEqual((radar[1]["score_delta"], radar[1]["previous_rank"]), (6, 1))

    def test_no_saved_run_keeps_the_brief_list(self):
        from api.market import _scan_radar

        with mock.patch("api.today.market_runs", return_value=[]):
            self.assertIsNone(_scan_radar())
        with mock.patch("api.today.market_runs", side_effect=RuntimeError("db down")):
            self.assertIsNone(_scan_radar())


if __name__ == "__main__":
    unittest.main()


class SparklineAndScoreHistoryTest(unittest.TestCase):
    def setUp(self):
        api.scans.spark_cache.clear()
        api.scans.score_cache.clear()

    def test_daily_closes_one_batch_last_21(self):
        import pandas as pd

        df = pd.DataFrame({"Close": [float(i) for i in range(30)]})
        with mock.patch("data.prices.fetch_price_data_batch", return_value=({"AAA": df}, [])) as fetch:
            out = api.scans.daily_closes(["aaa", "AAA", "ZZZ"])
            api.scans.daily_closes(["AAA", "ZZZ"])  # same set: served from the cache
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], ["AAA", "ZZZ"])
        self.assertEqual(out["series"]["AAA"], [float(i) for i in range(9, 30)])
        self.assertNotIn("ZZZ", out["series"])

    def test_score_history_uses_scanner_scores_gaps_and_added_signals(self):
        runs = [{"id": 3, "created_at": "2026-10-12T19:38:00+00:00"},
                {"id": 2, "created_at": "2026-10-12T16:38:00+00:00"},
                {"id": 1, "created_at": "2026-10-12T13:38:00+00:00"}]
        per_run = {1: {"CCC": {"score": 50, "signals": ["gapper"]}},
                   2: {},
                   3: {"CCC": {"score": 77, "signals": ["gapper", "golden_cross", "prebreakout"]}}}
        with mock.patch("api.scans.market_runs", return_value=runs), \
                mock.patch("api.scans._run_ticker_scores", side_effect=lambda rid: per_run[rid]):
            pro = api.scans.score_history("ccc", {"can_early_breakout": False})
            prem = api.scans.score_history("CCC", {"can_early_breakout": True})
        self.assertEqual([p["score"] for p in pro["points"]], [50, None, 77])
        self.assertEqual(pro["points"][2]["signals_added"], ["gapper", "golden_cross"])  # not ranked in the scan before
        self.assertNotIn("prebreakout", pro["points"][2]["signals"])
        self.assertIn("prebreakout", prem["points"][2]["signals_added"])
