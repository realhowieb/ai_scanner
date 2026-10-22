"""Day Trader speed: per-symbol caches for the daily enrichments, sparklines served
stale while reloading, and the market-hours keep-warm pass."""
import datetime as dt
import unittest
from unittest import mock

import market_data as md


class PerSymbolCacheTest(unittest.TestCase):
    def setUp(self):
        md.clear_daily_enrich_cache()
        self.addCleanup(md.clear_daily_enrich_cache)

    def test_only_new_symbols_are_fetched(self):
        calls = []

        def fetch(syms):
            calls.append(list(syms))
            return {s: f"v-{s}" for s in syms}

        self.assertEqual(md._per_symbol_cached("k", ["aaa", "BBB"], fetch), {"AAA": "v-AAA", "BBB": "v-BBB"})
        self.assertEqual(md._per_symbol_cached("k", ["BBB", "CCC"], fetch), {"BBB": "v-BBB", "CCC": "v-CCC"})
        self.assertEqual(calls, [["AAA", "BBB"], ["CCC"]])

    def test_none_is_cached_and_absent_symbols_retry(self):
        calls = []

        def fetch(syms):
            calls.append(list(syms))
            return {"AAA": None} if "AAA" in syms else {"BBB": 1.0}

        self.assertEqual(md._per_symbol_cached("k", ["AAA", "BBB"], fetch), {})
        # AAA answered "nothing to show" (cached); BBB wasn't returned, so it is
        # retried only after the short miss TTL, not on the very next call.
        self.assertEqual(md._per_symbol_cached("k", ["AAA", "BBB"], fetch), {})
        self.assertEqual(len(calls), 1)
        with mock.patch.object(md.time, "monotonic", return_value=md.time.monotonic() + md.DAILY_ENRICH_MISS_TTL_S + 1):
            self.assertEqual(md._per_symbol_cached("k", ["AAA", "BBB"], fetch), {"BBB": 1.0})
        self.assertEqual(calls[-1], ["BBB"])

    def test_provider_down_caches_nothing(self):
        answers = [{}, {"AAA": 2.0}]
        fetch = mock.Mock(side_effect=lambda syms: answers.pop(0))
        self.assertEqual(md._per_symbol_cached("k", ["AAA"], fetch), {})
        self.assertEqual(md._per_symbol_cached("k", ["AAA"], fetch), {"AAA": 2.0})

    def test_entries_expire(self):
        fetch = mock.Mock(side_effect=lambda syms: {s: 1 for s in syms})
        md._per_symbol_cached("k", ["AAA"], fetch)
        later = md.time.monotonic() + md.DAILY_ENRICH_TTL_S + 1
        with mock.patch.object(md.time, "monotonic", return_value=later):
            md._per_symbol_cached("k", ["AAA"], fetch)
        self.assertEqual(fetch.call_count, 2)

    def test_range_metrics_and_ema_reuse_daily_downloads(self):
        frames = mock.Mock(side_effect=lambda syms, period: {s: object() for s in syms})
        with mock.patch.object(md, "_daily_frames", frames), \
                mock.patch.object(md, "_range_metrics", return_value={"atr_pct": 2.0}), \
                mock.patch.object(md, "ema_cross_label", side_effect=lambda f: None):
            for _ in range(3):
                self.assertEqual(md.fetch_daily_range_metrics(["AAA", "BBB"]),
                                 {"AAA": {"atr_pct": 2.0}, "BBB": {"atr_pct": 2.0}})
                self.assertEqual(md.fetch_ema_crosses(["AAA", "BBB"]), {})
        # One 150d and one 90d download in total, not one per call.
        self.assertEqual(sorted(c.args[1] for c in frames.call_args_list), ["150d", "90d"])

    def test_avg_volume_cached_per_symbol(self):
        import pandas as pd

        download = mock.Mock(side_effect=lambda syms, **kw: {s: pd.DataFrame({"Volume": [100.0] * 25}) for s in syms})
        with mock.patch("data.price_alpaca.download_multi_alpaca", download):
            self.assertEqual(md.fetch_avg_daily_volume(["AAA"], feed="iex"), {"AAA": 100.0})
            self.assertEqual(md.fetch_avg_daily_volume(["AAA", "BBB"], feed="iex"), {"AAA": 100.0, "BBB": 100.0})
        self.assertEqual([c.args[0] for c in download.call_args_list], [["AAA"], ["BBB"]])


class SparklinesStaleTest(unittest.TestCase):
    def setUp(self):
        from api import today

        today._cache.clear()
        self.addCleanup(today._cache.clear)

    def test_expired_sparklines_served_while_reloading(self):
        from api import market, today

        bars = {"AAA": [{"t": "2026-10-22T14:00:00Z", "c": 1.0}]}
        fetch = mock.Mock(return_value=bars)
        with mock.patch("ui.stair_stepper.fetch_recent_minute_bars", fetch), \
                mock.patch("analytics.day_trade_display.sparkline", return_value=[1.0]):
            first = market.day_trader_sparklines(["AAA"])
            self.assertEqual(first["series"], {"AAA": [1.0]})
            later = today.time.monotonic() + market.DT_MOVERS_TTL_S + 5
            with mock.patch.object(today.time, "monotonic", return_value=later), \
                    mock.patch.object(today.threading, "Thread") as thread:
                again = market.day_trader_sparklines(["AAA"])
            self.assertEqual(again["series"], {"AAA": [1.0]})
            thread.assert_called_once()  # one background reload, the visitor didn't wait
        self.assertEqual(fetch.call_count, 1)


class KeepWarmTest(unittest.TestCase):
    def _at(self, hour, minute, day=22):  # 2026-10-22 is a Thursday; times are ET (UTC-4)
        return dt.datetime(2026, 10, day, hour, minute, tzinfo=dt.timezone.utc) + dt.timedelta(hours=4)

    def test_warms_only_in_market_hours(self):
        from api import market

        with mock.patch.object(market, "day_trader") as dtr, mock.patch.dict("os.environ", {}, clear=False):
            self.assertTrue(market.keep_day_trader_warm_once(self._at(11, 0)))
            self.assertTrue(market.keep_day_trader_warm_once(self._at(8, 0)))   # pre-market
            self.assertFalse(market.keep_day_trader_warm_once(self._at(22, 0)))  # night
            self.assertFalse(market.keep_day_trader_warm_once(self._at(11, 0, day=24)))  # Saturday
        self.assertEqual(dtr.call_count, 2)
        dtr.assert_called_with("movers")

    def test_switches_off(self):
        from api import market

        for env in ({"HSF_DT_KEEP_WARM": "0"}, {"HSF_WARM_DAY_TRADER": "0"}):
            with mock.patch.object(market, "day_trader") as dtr, mock.patch.dict("os.environ", env):
                self.assertFalse(market.keep_day_trader_warm_once(self._at(11, 0)))
            dtr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
