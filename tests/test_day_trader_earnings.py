"""Day Trader rows carry an earnings tag for names reporting today through 3 days out."""
import datetime as dt
import unittest
from unittest import mock

from api import market


class DayTraderEarningsTest(unittest.TestCase):
    def test_map_keys_days_and_time(self):
        today = dt.date(2026, 11, 10)
        calls = []

        def fetch(**kw):
            calls.append(kw)
            return [
                {"symbol": "NVDA", "earnings_date": dt.date(2026, 11, 10), "earnings_time": "AMC"},
                {"symbol": "BRK.B", "earnings_date": dt.datetime(2026, 11, 12, 0, 0), "earnings_time": "unknown"},
                {"symbol": "NVDA", "earnings_date": dt.date(2026, 11, 13), "earnings_time": "bmo"},
                {"symbol": "", "earnings_date": dt.date(2026, 11, 10)},
                {"symbol": "XYZ", "earnings_date": None},
            ]

        out = market.dt_earnings_map(today, fetch=fetch)
        self.assertEqual(calls, [{"days_ahead": market.DT_EARNINGS_DAYS, "start": today}])
        self.assertEqual(out["NVDA"], {"earnings_date": "2026-11-10", "earnings_time": "amc", "earnings_days": 0})
        self.assertEqual(out["BRK-B"], {"earnings_date": "2026-11-12", "earnings_time": None, "earnings_days": 2})
        self.assertEqual(set(out), {"NVDA", "BRK-B"})

    def test_db_outage_gives_empty(self):
        self.assertEqual(market.dt_earnings_map(dt.date(2026, 11, 10), fetch=mock.Mock(side_effect=RuntimeError)), {})

    def test_rows_get_tagged(self):
        rows = [{"ticker": "NVDA", "last": 100.0}, {"ticker": "BRK.B", "last": 400.0}, {"ticker": "AAPL", "last": 1.0}]
        earn = {"NVDA": {"earnings_date": "2026-11-10", "earnings_time": "amc", "earnings_days": 0},
                "BRK-B": {"earnings_date": "2026-11-11", "earnings_time": None, "earnings_days": 1}}
        from api import today as api_today

        def cached(key, loader, ttl_s=0, stale_s=0):
            if key == "dt_state":
                return "open"
            if isinstance(key, tuple) and key[0] == "dt_rows":
                return rows
            return loader()

        with mock.patch.object(api_today, "_cached", side_effect=cached), \
                mock.patch.object(market, "_dt_symbols", return_value=["NVDA", "BRK.B", "AAPL"]), \
                mock.patch.object(market, "_dt_earnings", return_value=earn), \
                mock.patch("analytics.day_trade_display.enrich_row", side_effect=lambda r, s, n: dict(r)), \
                mock.patch("analytics.day_trade_display.rank_key", side_effect=lambda r: r["ticker"]):
            out = {r["ticker"]: r for r in market.day_trader("custom", custom=["NVDA"])["rows"]}
        self.assertEqual(out["NVDA"]["earnings_days"], 0)
        self.assertEqual(out["NVDA"]["earnings_time"], "amc")
        self.assertEqual(out["BRK.B"]["earnings_days"], 1)
        self.assertNotIn("earnings_days", out["AAPL"])
        self.assertNotIn("earnings_days", rows[0])  # the shared cached rows stay untouched


if __name__ == "__main__":
    unittest.main()
