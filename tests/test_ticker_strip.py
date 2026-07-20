import unittest


class TickerStripTests(unittest.TestCase):
    def test_normalize_tickers_dedupes_uppercases_and_limits(self):
        from ui.ticker_strip import normalize_tickers

        self.assertEqual(
            normalize_tickers([" aapl ", "MSFT", "aapl", "", None, "nvda"], limit=2),
            ["AAPL", "MSFT"],
        )

    def test_build_ticker_strip_html_has_scroll_and_escapes_symbols(self):
        from ui.ticker_strip import TAPE_GROUP_COUNT, build_ticker_strip_html

        html = build_ticker_strip_html(["AAPL", "<BAD>"], label="Alerts")
        self.assertIn("symbol-tape-scroll", html)
        self.assertIn(f"calc(-100% / {TAPE_GROUP_COUNT})", html)
        self.assertEqual(html.count("class='symbol-tape__group'"), TAPE_GROUP_COUNT)
        self.assertIn("AAPL", html)
        self.assertIn("&lt;BAD&gt;", html)
        self.assertNotIn("<BAD>", html)

    def test_build_ticker_strip_html_colors_known_changes(self):
        from ui.ticker_strip import build_ticker_strip_html, ticker_change_map

        changes = ticker_change_map(
            [
                {"ticker": "AAPL", "chg_pct": 1.25},
                {"Symbol": "MSFT", "% Change": "-0.5%"},
                {"Ticker": "FLAT", "Chg %": 0},
            ]
        )
        html = build_ticker_strip_html(["AAPL", "MSFT", "FLAT"], changes=changes)
        self.assertIn("symbol-tape__item--up", html)
        self.assertIn("symbol-tape__item--down", html)
        self.assertIn("+1.25%", html)
        self.assertIn("-0.50%", html)

    def test_alerts_and_day_trader_render_symbol_tape(self):
        from pathlib import Path

        alerts_source = Path("ui/alerts.py").read_text()
        day_trader_source = Path("ui/day_trader.py").read_text()
        self.assertIn("render_ticker_strip(", alerts_source)
        self.assertIn("watch_tickers", alerts_source)
        self.assertIn("render_ticker_strip(", day_trader_source)
        self.assertIn("watch_tickers", day_trader_source)
        self.assertIn("active_watchlist_quote_rows", alerts_source)
        self.assertIn("dt_rows", day_trader_source)


if __name__ == "__main__":
    unittest.main()
