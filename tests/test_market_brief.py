"""Market brief reuses the morning-digest functions so email + UI never drift."""
from __future__ import annotations

import unittest
from unittest import mock


class MarketBriefTests(unittest.TestCase):
    def test_compute_assembles_from_digest(self):
        import scheduler.morning_digest as md
        import ui.market_brief as mb

        with mock.patch.object(md, "_latest_snapshot_df", return_value=object()), \
             mock.patch.object(md, "_market_gappers",
                               return_value=[{"ticker": "QCOM", "last": 174.08,
                                              "chg_pct": 3.15, "gap_pct": 6.93}]), \
             mock.patch.object(md, "_todays_setups",
                               return_value=(["CRWV", "VST"], [("EA", 51.0)])), \
             mock.patch.object(md, "_prebreakout_picks",
                               return_value=[{"symbol": "CNTA", "prob": 92.1}]), \
             mock.patch.object(md, "_earnings_days_map", return_value={}), \
             mock.patch.object(md, "_flag_earnings_rows", return_value=None):
            data = mb._compute_brief()

        self.assertEqual(data["gappers"][0]["ticker"], "QCOM")
        self.assertEqual(data["golden"], ["CRWV", "VST"])
        self.assertEqual(data["top_setups"], [("EA", 51.0)])
        self.assertEqual(data["picks"][0]["symbol"], "CNTA")

    def test_earnings_flag_appended_to_picks(self):
        import scheduler.morning_digest as md
        import ui.market_brief as mb

        with mock.patch.object(md, "_latest_snapshot_df", return_value=object()), \
             mock.patch.object(md, "_market_gappers", return_value=[]), \
             mock.patch.object(md, "_todays_setups", return_value=([], [])), \
             mock.patch.object(md, "_prebreakout_picks",
                               return_value=[{"symbol": "CNTA", "prob": 92.1}]), \
             mock.patch.object(md, "_earnings_days_map", return_value={"CNTA": 2}), \
             mock.patch.object(md, "_flag_earnings_rows", return_value=None):
            data = mb._compute_brief()
        self.assertIn("⚠️E2d", data["picks"][0]["symbol"])

    def test_none_without_snapshot(self):
        import scheduler.morning_digest as md
        import ui.market_brief as mb

        with mock.patch.object(md, "_latest_snapshot_df", return_value=None):
            self.assertIsNone(mb._compute_brief())

    def test_brief_page_is_login_gated(self):
        from pathlib import Path

        src = Path("pages/brief.py").read_text()
        self.assertIn("render_market_brief", src)
        self.assertIn('st.session_state.get("username")', src)


if __name__ == "__main__":
    unittest.main()
