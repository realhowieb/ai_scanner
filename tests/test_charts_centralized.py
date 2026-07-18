"""Charts + single-ticker quote should be Alpaca-first, yfinance fallback only."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_DEPS = (
    importlib.util.find_spec("streamlit") is not None
    and importlib.util.find_spec("pandas") is not None
)


@unittest.skipUnless(_DEPS, "charts/single_ticker need streamlit + pandas")
class CentralizedDataSourceTests(unittest.TestCase):
    def _ohlc(self):
        import pandas as pd

        return pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.9], "Close": [1.5]},
            index=pd.to_datetime(["2026-07-01"]),
        )

    def test_charts_prefer_alpaca_and_skip_yfinance(self):
        import ui.charts as c

        with mock.patch("data.price_alpaca.download_multi_alpaca",
                        return_value={"AAPL": self._ohlc()}), \
             mock.patch.object(c, "_get_yf",
                               side_effect=AssertionError("yfinance must not run")):
            out = c._fetch_unadjusted_ohlc("aapl", period="6mo")
        self.assertFalse(out.empty)
        self.assertIn("Close", out.columns)

    def test_charts_fall_back_to_yfinance_when_alpaca_empty(self):
        import ui.charts as c

        fake_yf = mock.MagicMock()
        fake_yf.download.return_value = self._ohlc()
        with mock.patch("data.price_alpaca.download_multi_alpaca", return_value={}), \
             mock.patch.object(c, "_get_yf", return_value=fake_yf):
            out = c._fetch_unadjusted_ohlc("aapl")
        self.assertFalse(out.empty)
        self.assertTrue(fake_yf.download.called)

    def test_get_live_quote_prefers_alpaca(self):
        import ui.single_ticker as s

        with mock.patch("market_data.get_latest_quotes",
                        return_value={"NVDA": {"last": 212.4}}):
            self.assertEqual(s.get_live_quote("nvda"), 212.4)

    def test_get_live_quote_none_when_no_data(self):
        import ui.single_ticker as s

        with mock.patch("market_data.get_latest_quotes", return_value={}), \
             mock.patch.dict("sys.modules", {"yfinance": None}):
            self.assertIsNone(s.get_live_quote("zzz"))


if __name__ == "__main__":
    unittest.main()
