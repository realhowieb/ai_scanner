import unittest

pd = None
try:
    import pandas as pd
except ImportError:
    pass

if pd is not None:
    from ui.smart_alerts import build_smart_alert_suggestions
else:
    build_smart_alert_suggestions = None  # type: ignore[assignment]


@unittest.skipIf(pd is None, "pandas not installed")
class SmartAlertSuggestionTests(unittest.TestCase):
    def test_builds_breakout_rvol_and_ema_suggestions(self):
        df = pd.DataFrame(
            [
                {
                    "Ticker": "AMD",
                    "BreakoutScore": 12.5,
                    "RVOL": 2.3,
                    "EMA Cross": "Golden Cross",
                },
                {
                    "Ticker": "TSLA",
                    "BreakoutScore": 7.0,
                    "RVOL": 0.9,
                    "EMA Cross": "—",
                },
            ]
        )

        suggestions = build_smart_alert_suggestions(df, max_suggestions=5)
        labels = [s.label for s in suggestions]
        types = [s.alert_type for s in suggestions]

        self.assertIn("AMD: breakout score alert", labels)
        self.assertIn("AMD: RVOL alert", labels)
        self.assertIn("AMD: Golden Cross alert", labels)
        self.assertIn("breakout", types)
        self.assertIn("rvol", types)
        self.assertIn("ema_cross", types)

    def test_prebreakout_probability_can_drive_watch_alert(self):
        df = pd.DataFrame(
            [
                {
                    "Ticker": "NVDA",
                    "BreakoutScore": 5.0,
                    "PreBreakoutProb": 0.72,
                }
            ]
        )

        suggestions = build_smart_alert_suggestions(df)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].ticker, "NVDA")
        self.assertEqual(suggestions[0].alert_type, "breakout")
        self.assertIn("PreBreakout probability is 72.0%", suggestions[0].reason)

    def test_empty_or_weak_rows_return_no_suggestions(self):
        df = pd.DataFrame([{"Ticker": "XYZ", "BreakoutScore": 1.0, "RVOL": 0.5}])

        self.assertEqual(build_smart_alert_suggestions(df), [])


if __name__ == "__main__":
    unittest.main()
