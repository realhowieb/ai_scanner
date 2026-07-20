import unittest


class TickerStripTests(unittest.TestCase):
    def test_normalize_tickers_dedupes_uppercases_and_limits(self):
        from ui.ticker_strip import normalize_tickers

        self.assertEqual(
            normalize_tickers([" aapl ", "MSFT", "aapl", "", None, "nvda"], limit=2),
            ["AAPL", "MSFT"],
        )

    def test_build_ticker_strip_html_has_scroll_and_escapes_symbols(self):
        from ui.ticker_strip import build_ticker_strip_html

        html = build_ticker_strip_html(["AAPL", "<BAD>"], label="Alerts")
        self.assertIn("symbol-tape-scroll", html)
        self.assertIn("AAPL", html)
        self.assertIn("&lt;BAD&gt;", html)
        self.assertNotIn("<BAD>", html)

    def test_alerts_and_day_trader_render_symbol_tape(self):
        from pathlib import Path

        alerts_source = Path("ui/alerts.py").read_text()
        day_trader_source = Path("ui/day_trader.py").read_text()
        self.assertIn("render_ticker_strip(watch_tickers", alerts_source)
        self.assertIn("render_ticker_strip(watch_tickers", day_trader_source)


if __name__ == "__main__":
    unittest.main()
