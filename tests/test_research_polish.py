"""Research polish (2026-11-01): a quote's daily-bar fallback is labelled, saved Scanner views
keep the price and relative-volume filters, and plan copy matches what the web app has."""
import importlib.util
import unittest
from unittest import mock

HAS_API = all(importlib.util.find_spec(m) for m in ("fastapi", "pydantic", "pandas"))


@unittest.skipUnless(HAS_API, "needs fastapi, pydantic and pandas")
class QuoteSourceTests(unittest.TestCase):
    def _quotes(self, snaps):
        from api import scans

        scans.quote_cache.clear()
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snaps), \
                mock.patch("market_data._get_alpaca_data_feed", return_value="iex"):
            return scans.latest_quotes(list(snaps))["quotes"]

    def test_trade_is_labelled_trade(self):
        q = self._quotes({"AAA": {"latestTrade": {"p": 10.5, "t": "2026-10-30T19:59:58Z"},
                                  "dailyBar": {"c": 10.4, "o": 10.0, "t": "2026-10-30T04:00:00Z"},
                                  "prevDailyBar": {"c": 10.0, "t": "2026-10-29T04:00:00Z"}}})["AAA"]
        self.assertEqual((q["source"], q["last"], q["at"]), ("trade", 10.5, "2026-10-30T19:59:58Z"))

    def test_missing_trade_falls_back_to_a_labelled_daily_close(self):
        q = self._quotes({"BBB": {"dailyBar": {"c": 20.0, "o": 19.0, "t": "2026-10-30T04:00:00Z"},
                                  "prevDailyBar": {"c": 19.5, "t": "2026-10-29T04:00:00Z"}}})["BBB"]
        self.assertEqual((q["source"], q["last"], q["at"]), ("daily_bar", 20.0, "2026-10-30T04:00:00Z"))

    def test_model_carries_the_source(self):
        from api.models import LatestQuote

        self.assertEqual(LatestQuote(last=1.0).source, "trade")
        self.assertEqual(LatestQuote(last=1.0, source="daily_bar").source, "daily_bar")


@unittest.skipUnless(HAS_API, "needs fastapi, pydantic and pandas")
class SavedViewTests(unittest.TestCase):
    def test_price_and_rvol_round_trip_and_default_for_old_views(self):
        from pydantic import ValidationError

        from api.workspace import ScannerView

        v = ScannerView(name="Cheap movers", price="u5", rvol=3)
        self.assertEqual((v.price, v.rvol), ("u5", 3))
        old = ScannerView.model_validate({"name": "Old", "signal": "breakout", "min": 60})
        self.assertEqual((old.price, old.rvol), ("", 0))
        with self.assertRaises(ValidationError):
            ScannerView(name="Bad", price="cheap")
        with self.assertRaises(ValidationError):
            ScannerView(name="Bad", rvol=5)


class PlanCopyTests(unittest.TestCase):
    def test_rows_name_web_features_not_retired_screens(self):
        from ui.pricing import ROWS

        labels = [label for label, _ in ROWS]
        for retired in ("Market lenses, card view and saved screens", "My Stocks watchlists", "CSV export & interactive results table"):
            self.assertNotIn(retired, labels)
        self.assertIn(("CSV export of Scanner results", "can_export_csv"), ROWS)
        self.assertIn("Scanner filters and saved views", labels)


if __name__ == "__main__":
    unittest.main()
