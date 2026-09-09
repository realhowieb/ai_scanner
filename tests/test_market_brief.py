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


class StandoutsTests(unittest.TestCase):
    def test_confluence_across_lists(self):
        import ui.market_brief as mb

        data = {
            "gappers": [{"ticker": "CRWV"}, {"ticker": "QCOM"}],
            "golden": ["CRWV", "VST"],
            "top_setups": [("CRWV", 39.4), ("EA", 51.0)],
            "picks": [{"symbol": "CNTA"}],
            "gainers": [("EA", 9.9)],
            "losers": [("QCOM ⚠️E1d", -2.0)],
        }
        out = dict(mb._standouts(data))
        self.assertEqual(out["CRWV"], ["gapper", "golden cross", "breakout"])  # 3 lists, first
        self.assertIn("loser", out["QCOM"])          # earnings flag stripped, still matched
        self.assertNotIn("VST", out)                 # only 1 list → not a standout
        self.assertNotIn("CNTA", out)                # only 1 list

    def test_no_standouts_when_no_overlap(self):
        import ui.market_brief as mb

        data = {"gappers": [{"ticker": "AAA"}], "golden": ["BBB"], "picks": []}
        self.assertEqual(mb._standouts(data), [])


class SummaryAndPositionsTests(unittest.TestCase):
    def test_market_summary_synthesizes(self):
        import ui.market_brief as mb

        data = {
            "market_close": [("S&P 500 (SPY)", 660.0, 0.6)],
            "breadth": (312, 188), "sectors": [("Tech", 1.2), ("Energy", -0.8)],
            "gappers": [{"ticker": "CRWV"}], "golden": ["CRWV"],
            "top_setups": [("CRWV", 39)], "picks": [], "gainers": [], "losers": [],
            "earnings_today": ["EA", "QCOM"],
        }
        s = mb._market_summary(data)
        self.assertIn("Risk-on", s)
        self.assertIn("breadth 312/188", s)
        self.assertIn("Tech leading", s)
        self.assertIn("1 standout", s)

    def test_open_positions_marks_to_now(self):
        from unittest import mock

        import ui.market_brief as mb

        trades = [
            {"ticker": "AAPL", "entry_price": 100.0, "closed_at": None},
            {"ticker": "OLD", "entry_price": 50.0, "closed_at": "2026-01-01"},
        ]
        with mock.patch("db.trades.list_trades", return_value=trades),              mock.patch("market_data.get_latest_quotes",
                        return_value={"AAPL": {"last": 102.0}}):
            pos = mb._open_positions("u@x.com")
        self.assertEqual(len(pos), 1)                 # closed trade excluded
        self.assertEqual(pos[0][0], "AAPL")
        self.assertAlmostEqual(pos[0][1], 2.0)        # 100 → 102 = +2%


if __name__ == "__main__":
    unittest.main()
