"""Stock Intelligence chart bars: top up a stale scan cache from the price provider."""
import datetime as dt
import unittest
from unittest import mock

import pandas as pd

from api import scans


def _frame(days):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    n = len(days)
    return pd.DataFrame({"Open": [1.0] * n, "High": [2.0] * n, "Low": [0.5] * n,
                         "Close": [1.5] * n, "Volume": [100] * n}, index=idx)


# Monday 2026-10-12, 20:30 UTC = 4:30 PM ET: Monday's session has closed.
MON_CLOSED = dt.datetime(2026, 10, 12, 20, 30, tzinfo=dt.timezone.utc)


class LastSessionDayTests(unittest.TestCase):
    def test_after_close_counts_today(self):
        self.assertEqual(scans._last_session_day(MON_CLOSED), dt.date(2026, 10, 12))

    def test_before_close_monday_is_friday(self):
        self.assertEqual(scans._last_session_day(dt.datetime(2026, 10, 12, 15, 0, tzinfo=dt.timezone.utc)),
                         dt.date(2026, 10, 9))

    def test_weekend_is_friday(self):
        self.assertEqual(scans._last_session_day(dt.datetime(2026, 10, 11, 18, 0, tzinfo=dt.timezone.utc)),
                         dt.date(2026, 10, 9))


class DailyBarsTests(unittest.TestCase):
    def setUp(self):
        scans.stock_cache.clear()
        p = mock.patch("api.scans._last_session_day", return_value=dt.date(2026, 10, 12))
        p.start()
        self.addCleanup(p.stop)

    def _snapshot(self, days):
        return mock.patch("db.prices.get_price_data_snapshot",
                          return_value=({"ABC": _frame(days)} if days else {}, set()))

    def test_fresh_cache_skips_provider(self):
        with self._snapshot(["2026-10-09", "2026-10-12"]), \
                mock.patch("data.prices.fetch_price_data_batch") as fetch:
            out = scans.daily_bars("ABC")
        fetch.assert_not_called()
        self.assertEqual(out["as_of"], "2026-10-12")

    def test_stale_cache_uses_provider_bars(self):
        with self._snapshot(["2026-10-06", "2026-10-07"]), \
                mock.patch("data.prices.fetch_price_data_batch",
                           return_value=({"ABC": _frame(["2026-10-09", "2026-10-12", "2026-10-13"])}, [])):
            out = scans.daily_bars("ABC")
        # The 13th (a session not yet closed) is dropped.
        self.assertEqual([b["date"] for b in out["bars"]], ["2026-10-09", "2026-10-12"])
        self.assertEqual(out["as_of"], "2026-10-12")

    def test_provider_failure_keeps_cached_bars(self):
        with self._snapshot(["2026-10-06", "2026-10-07"]), \
                mock.patch("data.prices.fetch_price_data_batch", side_effect=RuntimeError("down")):
            out = scans.daily_bars("ABC")
        self.assertEqual(out["as_of"], "2026-10-07")

    def test_no_cache_uses_provider(self):
        with self._snapshot([]), \
                mock.patch("data.prices.fetch_price_data_batch",
                           return_value=({"ABC": _frame(["2026-10-09", "2026-10-12"])}, [])):
            out = scans.daily_bars("ABC")
        self.assertEqual(len(out["bars"]), 2)

    def test_provider_reused_within_ttl(self):
        with self._snapshot(["2026-10-07"]), \
                mock.patch("data.prices.fetch_price_data_batch",
                           return_value=({"ABC": _frame(["2026-10-12"])}, [])) as fetch:
            scans.daily_bars("ABC")
            scans.daily_bars("ABC")
        self.assertEqual(fetch.call_count, 1)


if __name__ == "__main__":
    unittest.main()


class EmaLineTests(unittest.TestCase):
    def test_emas_match_the_scanner_golden_cross(self):
        from scan.indicators import ema_cross_detail

        # Falling then rising closes: the 9 EMA crosses above the 21 somewhere in the rise.
        closes = [100 - i for i in range(40)] + [61 + 2 * i for i in range(20)]
        days = pd.bdate_range("2026-06-01", periods=len(closes))
        df = pd.DataFrame({"Close": closes}, index=days)
        bars = scans._frame_bars(df, 500, dt.date(2026, 12, 31))
        self.assertIsNone(bars[0]["ema9"])
        cross = next(i for i in range(1, len(bars))
                     if bars[i - 1]["ema9"] is not None
                     and bars[i - 1]["ema9"] <= bars[i - 1]["ema21"] and bars[i]["ema9"] > bars[i]["ema21"])
        detail = ema_cross_detail(df.iloc[:cross + 1])
        self.assertEqual(detail["direction"], "bullish")
        self.assertAlmostEqual(detail["ema_fast"], bars[cross]["ema9"], places=3)
        self.assertIsNone(ema_cross_detail(df.iloc[:cross]))
