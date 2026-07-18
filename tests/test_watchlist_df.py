"""build_watchlist_df should use batched Alpaca quotes, not per-ticker yfinance."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock

_DEPS = (
    importlib.util.find_spec("streamlit") is not None
    and importlib.util.find_spec("pandas") is not None
)


@unittest.skipUnless(_DEPS, "watchlists needs streamlit + pandas")
class BuildWatchlistDfTests(unittest.TestCase):
    def test_alpaca_path_computes_change_and_skips_yfinance(self):
        import ui.watchlists as w

        quotes = {
            "AAPL": {"last": 180.0, "prev_close": 176.0, "volume": 1_000},
            "MSFT": {"last": 400.0, "prev_close": 405.0, "volume": 2_000},
        }
        # A yfinance module that explodes if touched — proves we didn't fall back.
        booby = mock.MagicMock()
        booby.Ticker.side_effect = AssertionError("yfinance must not be used")
        with mock.patch("market_data.get_latest_quotes", return_value=quotes), \
             mock.patch.dict(sys.modules, {"yfinance": booby}):
            df = w.build_watchlist_df(["aapl", "msft"])

        self.assertEqual(list(df["Symbol"]), ["AAPL", "MSFT"])
        self.assertEqual(list(df.columns), list(w._WATCHLIST_DF_COLUMNS))
        self.assertAlmostEqual(df.iloc[0]["Change"], 4.0)
        self.assertAlmostEqual(df.iloc[1]["% Change"], (400.0 - 405.0) / 405.0 * 100.0)

    def test_zero_prev_close_leaves_change_none(self):
        import ui.watchlists as w

        quotes = {"XYZ": {"last": 10.0, "prev_close": 0, "volume": 1}}
        with mock.patch("market_data.get_latest_quotes", return_value=quotes):
            df = w.build_watchlist_df(["XYZ"])
        self.assertIsNone(df.iloc[0]["Change"])
        self.assertIsNone(df.iloc[0]["% Change"])
        self.assertEqual(df.iloc[0]["Last"], 10.0)

    def test_falls_back_when_alpaca_empty(self):
        import ui.watchlists as w

        # Alpaca returns nothing; yfinance import fails → empty-but-shaped frame.
        booby = None
        with mock.patch("market_data.get_latest_quotes", return_value={}), \
             mock.patch.dict(sys.modules, {"yfinance": booby}):
            df = w.build_watchlist_df(["AAA", "BBB"])
        self.assertEqual(list(df["Symbol"]), ["AAA", "BBB"])
        self.assertTrue(df["Last"].isna().all())


if __name__ == "__main__":
    unittest.main()
