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
