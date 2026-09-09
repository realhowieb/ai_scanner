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


class MarketBriefExtrasTests(unittest.TestCase):
    def test_base_ticker_strips_earnings_flag(self):
        import ui.market_brief as mb

        self.assertEqual(mb._base_ticker("CNTA ⚠️E2d"), "CNTA")
        self.assertEqual(mb._base_ticker("qcom"), "QCOM")
        self.assertEqual(mb._base_ticker(None), "")

    def test_yesterday_performance_marks_picks_to_now(self):
        import datetime as dt

        import scheduler.morning_digest as md
        import ui.market_brief as mb

        class _DF:
            columns = ["Ticker", "Last"]

            def __len__(self):
                return 1

            def iterrows(self):
                return iter([(0, {"Ticker": "EA", "Last": 100.0})])

        runs = [
            {"id": 2, "created_at": dt.datetime(2026, 9, 9, 12, tzinfo=dt.timezone.utc)},
            {"id": 1, "created_at": dt.datetime(2026, 9, 8, 12, tzinfo=dt.timezone.utc)},
        ]
        with mock.patch("db.runs.list_snapshot_runs", return_value=runs), \
             mock.patch("db.runs.load_many_run_results", return_value={1: "[]"}), \
             mock.patch("ui.app_runtime.normalize_results_to_df", return_value=_DF()), \
             mock.patch.object(md, "_todays_setups", return_value=([], [("EA", 51.0)])), \
             mock.patch("market_data.get_latest_quotes", return_value={"EA": {"last": 105.0}}):
            y = mb._yesterday_performance()

        self.assertEqual(y["date"], dt.date(2026, 9, 8))
        self.assertEqual(y["rows"][0][0], "EA")
        self.assertAlmostEqual(y["rows"][0][1], 5.0)      # 100 → 105 = +5%



class EveningContentTests(unittest.TestCase):
    def test_compute_includes_evening_wrap_content(self):
        import scheduler.evening_wrap as ew
        import scheduler.morning_digest as md
        import ui.market_brief as mb

        with mock.patch.object(md, "_latest_snapshot_df", return_value=object()), \
             mock.patch.object(md, "_market_gappers", return_value=[]), \
             mock.patch.object(md, "_todays_setups", return_value=([], [])), \
             mock.patch.object(md, "_prebreakout_picks", return_value=[]), \
             mock.patch.object(md, "_earnings_days_map", return_value={}), \
             mock.patch.object(md, "_flag_earnings_rows", return_value=None), \
             mock.patch.object(md, "_earnings_today", return_value=set()), \
             mock.patch.object(ew, "_market_close_context",
                               return_value=[("S&P 500 (SPY)", 660.0, 0.3)]), \
             mock.patch.object(ew, "_day_movers",
                               return_value=([("DLR", 11.3)], [("HIG", -1.2)])):
            data = mb._compute_brief()

        self.assertEqual(data["market_close"][0][0], "S&P 500 (SPY)")
        self.assertEqual(data["gainers"], [("DLR", 11.3)])
        self.assertEqual(data["losers"], [("HIG", -1.2)])


if __name__ == "__main__":
    unittest.main()
