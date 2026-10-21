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



class LatestQuotesTest(unittest.TestCase):
    def setUp(self):
        api.scans.quote_cache.clear()

    @staticmethod
    def _snap(price, trade_t, day_c, day_t, prev_c):
        return {"latestTrade": {"p": price, "t": trade_t}, "dailyBar": {"c": day_c, "t": day_t},
                "prevDailyBar": {"c": prev_c, "t": "2026-10-16T04:00:00Z"}}

    def test_one_call_per_set_change_vs_previous_close(self):
        snaps = {"AAA": self._snap(110.0, "2026-10-19T19:59:00Z", 110.0, "2026-10-19T04:00:00Z", 100.0)}
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snaps) as fetch, \
                mock.patch("market_data._get_alpaca_data_feed", return_value="iex"):
            out = api.scans.latest_quotes(["aaa", "ZZZ"])
            api.scans.latest_quotes(["ZZZ", "AAA"])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], ["AAA", "ZZZ"])
        self.assertEqual(out["quotes"]["AAA"]["chg_pct"], 10.0)
        self.assertEqual(out["quotes"]["AAA"]["at"], "2026-10-19T19:59:00Z")
        self.assertNotIn("ZZZ", out["quotes"])
        self.assertEqual(out["feed"], "iex")

    def test_premarket_trade_is_measured_from_the_last_finished_session(self):
        # 8:30 ET on Oct 20; the daily bar is still Oct 19's (close 110).
        snaps = {"AAA": self._snap(121.0, "2026-10-20T12:30:00Z", 110.0, "2026-10-19T04:00:00Z", 100.0)}
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snaps):
            out = api.scans.latest_quotes(["AAA"])
        self.assertEqual(out["quotes"]["AAA"]["prev_close"], 110.0)
        self.assertEqual(out["quotes"]["AAA"]["chg_pct"], 10.0)

    def test_late_evening_trade_stays_on_its_new_york_day(self):
        # 7:30 PM ET in winter is the next day in UTC; still the Dec 1 session.
        snaps = {"AAA": self._snap(105.0, "2026-12-02T00:30:00Z", 104.0, "2026-12-01T05:00:00Z", 100.0)}
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snaps):
            out = api.scans.latest_quotes(["AAA"])
        self.assertEqual(out["quotes"]["AAA"]["prev_close"], 100.0)

    def test_empty_answer_is_not_cached(self):
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value={}) as fetch:
            self.assertEqual(api.scans.latest_quotes(["AAA"])["quotes"], {})
            api.scans.latest_quotes(["AAA"])
        self.assertEqual(fetch.call_count, 2)

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

    def test_daily_bar_value_keeps_its_actual_session_when_newest_close_is_missing(self):
        import pandas as pd

        df = pd.DataFrame({"Close": [10.0, 11.0, float("nan")]},
                          index=pd.date_range("2026-10-13", periods=3))
        with mock.patch("data.prices.fetch_price_data_batch", return_value=({"AAA": df}, [])) as fetch:
            out = api.scans.daily_closes(["AAA"])
            cached = api.scans.daily_closes(["AAA"])
        self.assertEqual(out["last_bar"]["AAA"], {"close": 11.0, "session": "2026-10-14"})
        self.assertEqual(out, cached)
        self.assertEqual(fetch.call_count, 1)

    def test_daily_bar_never_invents_a_date_for_undated_data(self):
        import pandas as pd

        with mock.patch("data.prices.fetch_price_data_batch", return_value=({"AAA": pd.DataFrame({"Close": [10.0, 11.0]})}, [])):
            out = api.scans.daily_closes(["AAA"])
        self.assertEqual(out["series"]["AAA"], [10.0, 11.0])
        self.assertEqual(out["last_bar"], {})


class BriefFreshnessTest(unittest.TestCase):
    def test_uses_the_matching_runs_market_data_time(self):
        from api.market import _brief_freshness

        run = {"id": 7, "created_at": "2026-10-21T13:39:55+00:00"}
        info = {"state": "fresh", "market_data_at": "2026-10-21T13:36:12+00:00"}
        with mock.patch("api.today.market_runs", return_value=[run]), \
                mock.patch("api.today.run_freshness", return_value=info) as rf:
            self.assertEqual(_brief_freshness("2026-10-21T13:39:55Z"), info)
        rf.assert_called_once_with(run)

    def test_no_matching_run_leaves_market_data_unknown(self):
        from api.market import _brief_freshness

        with mock.patch("api.today.market_runs", return_value=[{"id": 1, "created_at": "2026-10-20T20:39:00Z"}]):
            info = _brief_freshness("2026-10-21T13:39:55Z")
        self.assertIsNone(info["market_data_at"])
        with mock.patch("api.today.market_runs", side_effect=RuntimeError("db down")):
            self.assertEqual(_brief_freshness(None)["state"], "unavailable")
