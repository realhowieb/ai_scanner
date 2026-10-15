"""Pre-market coverage: Day Trader rows before the open measure today's trades
against yesterday's close (not yesterday's session), Movers ranks today's
pre-market gappers, and /v1/premarket/gappers serves the whole-market list."""
import datetime as dt
import unittest
from unittest import mock

import market_data as md
from tests.test_api_paid_features import PaidApiTestCase

UTC = dt.timezone.utc
WED = dt.date(2026, 10, 14)
WED_8_ET = dt.datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
WED_3_ET = dt.datetime(2026, 10, 14, 7, 0, tzinfo=UTC)
WED_OPEN = dt.datetime(2026, 10, 14, 13, 30, tzinfo=UTC)
SAT_8_ET = dt.datetime(2026, 10, 17, 12, 0, tzinfo=UTC)

YDAY_BAR = "2026-10-13T04:00:00Z"
PREV_BAR = "2026-10-12T04:00:00Z"


def snap(price, trade_t, *, close=100.0, prev_close=80.0, daily_t=YDAY_BAR):
    """Pre-market snapshot: dailyBar is yesterday (close 100, up 25% on the day)."""
    return {"latestTrade": {"p": price, "t": trade_t},
            "dailyBar": {"t": daily_t, "o": 82.0, "c": close, "vw": 95.0, "v": 9_000_000},
            "prevDailyBar": {"t": PREV_BAR, "c": prev_close, "v": 1_000_000}}


TODAY_TRADE = "2026-10-14T11:58:12.123456789Z"     # 7:58 ET today, nanosecond precision
YDAY_TRADE = "2026-10-13T19:59:34.804444297Z"      # yesterday's last regular print


class SessionDateTests(unittest.TestCase):
    def test_window(self):
        self.assertEqual(md.premarket_session_date(WED_8_ET), WED)
        self.assertEqual(md.premarket_session_date(dt.datetime(2026, 10, 14, 8, 0, tzinfo=UTC)), WED)  # 4:00 ET
        self.assertIsNone(md.premarket_session_date(WED_3_ET))
        self.assertIsNone(md.premarket_session_date(WED_OPEN))
        self.assertIsNone(md.premarket_session_date(SAT_8_ET))

    def test_et_date_handles_nanoseconds(self):
        self.assertEqual(md._et_date(YDAY_TRADE), dt.date(2026, 10, 13))
        self.assertIsNone(md._et_date(None))
        self.assertIsNone(md._et_date("garbage"))


class DayTraderPremarketTests(unittest.TestCase):
    def rows(self, snaps, now, pm=None):
        with mock.patch.object(md, "fetch_alpaca_snapshots", return_value=snaps), \
                mock.patch.object(md, "fetch_premarket_volume", return_value=pm or {}), \
                mock.patch.object(md, "_get_alpaca_data_feed", return_value="iex"):
            return {r["ticker"]: r for r in md.build_day_trader_metrics(list(snaps), with_rvol=False, now=now)}

    def test_premarket_measures_today_vs_yesterdays_close(self):
        r = self.rows({"GAP": snap(111.0, TODAY_TRADE)}, WED_8_ET)["GAP"]
        self.assertEqual(r["previous_close"], 100.0)           # yesterday's close, not the day before
        self.assertEqual(r["chg_pct"], 11.0)
        self.assertTrue(r["traded_today"])
        self.assertEqual(r["session"], "premarket")
        for k in ("open", "gap_pct", "change_dollar", "rvol", "close_today"):
            self.assertIsNone(r[k], k)

    def test_no_trade_today_has_no_change(self):
        """Yesterday's +25% (100 vs 80) must not show as today's move."""
        r = self.rows({"OLD": snap(100.0, YDAY_TRADE)}, WED_8_ET)["OLD"]
        self.assertIsNone(r["chg_pct"])
        self.assertFalse(r["traded_today"])
        self.assertIsNone(r["volume"])

    def test_premarket_row_carries_yesterdays_session_move(self):
        r = self.rows({"OLD": snap(100.0, YDAY_TRADE)}, WED_8_ET)["OLD"]
        self.assertEqual(r["prior_session_chg_pct"], 25.0)     # yesterday 100 vs 80

    def test_display_path_uses_premarket_volume_and_vwap(self):
        pm = {"GAP": {"volume": 50_000, "dollar_volume": 5.5e6, "vwap": 110.0, "high": 112.0, "low": 105.0}}
        snaps = {"GAP": snap(111.0, TODAY_TRADE)}
        with mock.patch.object(md, "fetch_alpaca_snapshots", return_value=snaps), \
                mock.patch.object(md, "fetch_premarket_volume", return_value=pm), \
                mock.patch.object(md, "fetch_avg_daily_volume", return_value={"GAP": 1e6}), \
                mock.patch.object(md, "fetch_ema_crosses", return_value={}), \
                mock.patch.object(md, "fetch_daily_range_metrics", return_value={}):
            r = md.build_day_trader_metrics(["GAP"], now=WED_8_ET)[0]
        self.assertEqual((r["volume"], r["vwap"], r["vs_vwap_pct"]), (50_000, 110.0, 0.91))
        self.assertIsNone(r["rvol"])                           # pre-market volume isn't a session's

    def test_regular_session_unchanged(self):
        today_bar = "2026-10-14T04:00:00Z"
        r = self.rows({"X": snap(105.0, "2026-10-14T15:00:00Z", daily_t=today_bar)}, dt.datetime(2026, 10, 14, 15, 0, tzinfo=UTC))["X"]
        self.assertEqual((r["previous_close"], r["chg_pct"], r["gap_pct"]), (80.0, 31.25, 2.5))
        self.assertNotIn("session", r)

    def test_undated_bars_never_take_the_premarket_path(self):
        s = {"latestTrade": {"p": 10.0}, "dailyBar": {"o": 9.0, "c": 10.0, "vw": 9.5, "v": 1e6},
             "prevDailyBar": {"c": 8.0}}
        r = self.rows({"U": s}, WED_8_ET)["U"]
        self.assertEqual(r["chg_pct"], 25.0)


class WaitingScoreTests(unittest.TestCase):
    def test_no_trade_today_gets_no_score(self):
        """Yesterday's Supertrend + EWO alone used to score 100 "Strong" before the first trade."""
        from analytics.day_trade_display import enrich_row

        base = {"ticker": "XP", "last": 31.38, "supertrend_direction": 1, "ewo": 2.0, "chg_pct": None}
        waiting = enrich_row({**base, "traded_today": False}, "premarket")
        self.assertEqual((waiting["day_trade_score"], waiting["dt_quality"], waiting["dt_direction"]),
                         (None, "insufficient", "neutral"))
        self.assertTrue(waiting["traded_today"] is False)
        traded = enrich_row({**base, "traded_today": True, "chg_pct": 3.0, "vs_vwap_pct": 1.0}, "premarket")
        self.assertIsNotNone(traded["day_trade_score"])

class GapperTests(unittest.TestCase):
    SNAPS = {
        "UP": snap(115.0, TODAY_TRADE),                  # +15%
        "DOWN": snap(92.0, TODAY_TRADE),                 # -8%
        "THIN": snap(130.0, TODAY_TRADE),                # +30% but little volume
        "FLAT": snap(101.0, TODAY_TRADE),                # +1%: below the 2% floor
        "STALE": snap(100.0, YDAY_TRADE),                # no trade today (yesterday +25%)
        "PENNY": snap(1.5, TODAY_TRADE, close=1.0),      # under $2
    }
    VOLS = {"UP": {"volume": 10_000, "dollar_volume": 1.15e6, "vwap": 114.0, "high": 116.0, "low": 110.0},
            "DOWN": {"volume": 2_000, "dollar_volume": 184_000, "vwap": 92.0, "high": 95.0, "low": 91.0},
            "THIN": {"volume": 100, "dollar_volume": 13_000, "vwap": 130.0, "high": 130.0, "low": 129.0}}

    def gappers(self, vols, now=WED_8_ET, feed="iex"):
        with mock.patch.object(md, "fetch_alpaca_snapshots", return_value=self.SNAPS), \
                mock.patch.object(md, "fetch_premarket_volume", return_value=vols) as fv, \
                mock.patch.object(md, "_get_alpaca_data_feed", return_value=feed):
            out = md.build_premarket_gappers(list(self.SNAPS), now=now)
        return out, fv

    def test_ranks_by_move_and_applies_floors(self):
        out, fv = self.gappers(self.VOLS)
        self.assertEqual([r["ticker"] for r in out], ["UP", "DOWN"])
        self.assertEqual((out[0]["pct"], out[0]["previous_close"], out[0]["pm_volume"]), (15.0, 100.0, 10_000))
        self.assertEqual(sorted(fv.call_args.args[0]), ["DOWN", "THIN", "UP"])   # only real movers get bars

    def test_sip_floor_is_higher(self):
        out, _ = self.gappers(self.VOLS, feed="sip")
        self.assertEqual([r["ticker"] for r in out], ["UP"])

    def test_volume_fetch_failure_keeps_moves(self):
        out, _ = self.gappers(None)
        self.assertEqual([r["ticker"] for r in out], ["THIN", "UP", "DOWN"])
        self.assertIsNone(out[0]["pm_volume"])

    def test_empty_outside_premarket(self):
        out, fv = self.gappers(self.VOLS, now=WED_OPEN)
        self.assertEqual(out, [])
        fv.assert_not_called()


class PremarketVolumeTests(unittest.TestCase):
    def setUp(self):
        md._pm_bars_cache.clear()
        self.addCleanup(md._pm_bars_cache.clear)

    def test_sums_minute_bars_since_4am(self):
        bars = {"UP": [{"c": 10.0, "h": 10.5, "l": 9.8, "v": 100}, {"c": 11.0, "h": 11.2, "l": 10.9, "v": 300}]}
        with mock.patch("data.price_alpaca.fetch_minute_bars_multi", return_value=bars) as f:
            out = md.fetch_premarket_volume(["up", "DOWN"], WED)
            md.fetch_premarket_volume(["UP", "DOWN"], WED)
        self.assertEqual(f.call_args.args[1], "2026-10-14T08:00:00Z")      # 4:00 ET
        self.assertEqual(f.call_count, 1)                                   # cached
        self.assertEqual(out["UP"], {"volume": 400.0, "dollar_volume": 4300.0, "vwap": 10.75, "high": 11.2, "low": 9.8})
        self.assertNotIn("DOWN", out)

    def test_provider_error_is_none(self):
        with mock.patch("data.price_alpaca.fetch_minute_bars_multi", side_effect=RuntimeError("429")):
            self.assertIsNone(md.fetch_premarket_volume(["UP"], WED))

class MoversPremarketTests(unittest.TestCase):
    def test_movers_rank_gappers_before_the_open(self):
        import ui.day_trader as d

        with mock.patch("market_data.premarket_session_date", return_value=WED), \
                mock.patch("market_data.build_premarket_gappers", return_value=[{"ticker": "UP"}, {"ticker": "DOWN"}]) as g, \
                mock.patch("market_data.build_day_trader_metrics") as m:
            self.assertEqual(d._top_movers_symbols(universe=["UP", "DOWN"]), ["UP", "DOWN"])
        m.assert_not_called()
        self.assertEqual(g.call_args.kwargs["limit"], 40)


class GappersApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        mock.patch("ui.day_trader._movers_universe", return_value=["UP"]).start()
        mock.patch("market_data._get_alpaca_data_feed", return_value="iex").start()

    def test_pro_gets_gappers_free_refused(self):
        row = {"ticker": "UP", "last": 115.0, "previous_close": 100.0, "pct": 15.0, "pm_volume": 10_000,
               "pm_dollar_volume": 1150000, "pm_vwap": 114.0, "pm_high": 116.0, "pm_low": 110.0,
               "trade_ts": TODAY_TRADE, "market_data_feed": "iex"}
        with mock.patch("market_data.premarket_session_date", return_value=WED), \
                mock.patch("market_data.build_premarket_gappers", return_value=[row]) as g:
            r = self.get("pro@example.com", "/v1/premarket/gappers")
            self.get("pro@example.com", "/v1/premarket/gappers")
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertEqual((b["active"], b["feed"], b["rows"][0]["ticker"], b["rows"][0]["pct"]), (True, "iex", "UP", 15.0))
        self.assertEqual(g.call_count, 1)                                   # shared 3-minute cache
        self.assertEqual(self.get("free@example.com", "/v1/premarket/gappers").status_code, 403)

    def test_inactive_outside_premarket(self):
        with mock.patch("market_data.premarket_session_date", return_value=None), \
                mock.patch("market_data.build_premarket_gappers") as g:
            b = self.get("pro@example.com", "/v1/premarket/gappers").json()
        self.assertEqual((b["active"], b["rows"]), (False, []))
        g.assert_not_called()


if __name__ == "__main__":
    unittest.main()
