import unittest


class AlertTypesTests(unittest.TestCase):
    def test_ema_cross_is_allowed_alert_type(self):
        from db.alerts import ALERT_TYPES

        self.assertIn("ema_cross", ALERT_TYPES)

    def test_ema_cross_is_visible_in_alert_ui_source(self):
        from pathlib import Path

        source = Path("ui/alerts.py").read_text()
        self.assertIn("EMA Cross", source)
        self.assertIn("Golden Cross (bullish)", source)
        self.assertIn("Death Cross (bearish)", source)

    def test_breakout_history_preview_is_opt_in(self):
        from pathlib import Path

        source = Path("ui/alerts.py").read_text()
        self.assertIn("Show threshold history", source)
        self.assertIn("expanded=_has_alert_prefill()", source)
        self.assertLess(
            source.index("Show threshold history"),
            source.index("render_breakout_threshold_insight(float(thr))"),
        )

    def test_alert_page_does_not_load_watchlists_before_render(self):
        from pathlib import Path

        source = Path("pages/alerts.py").read_text()
        self.assertIn("_session_watch_tickers()", source)
        self.assertNotIn("from db.watchlists import", source)
        self.assertNotIn("list_watchlists(", source)
        self.assertNotIn("get_watchlist_tickers(", source)


if __name__ == "__main__":
    unittest.main()
