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


if __name__ == "__main__":
    unittest.main()
