import datetime as dt
import unittest

try:
    import pandas as pd

    _PANDAS = True
except Exception:
    _PANDAS = False


@unittest.skipUnless(_PANDAS, "alert_runner evaluation requires pandas")
class EvaluateTest(unittest.TestCase):
    def _df(self):
        return pd.DataFrame(
            [
                {"Ticker": "AAPL", "BreakoutScore": 9.2, "Last": 250.0},
                {"Ticker": "MSFT", "BreakoutScore": 4.0, "Last": 370.0},
                {"Ticker": "NVDA", "BreakoutScore": 8.5, "Last": 190.0},
            ]
        )

    def test_breakout_threshold(self):
        from scheduler.alert_runner import _evaluate

        alert = {"alert_type": "breakout", "threshold": 8.0}
        lines = _evaluate(alert, self._df(), set())
        joined = " ".join(lines)
        self.assertIn("AAPL", joined)
        self.assertIn("NVDA", joined)
        self.assertNotIn("MSFT", joined)  # 4.0 < 8.0

    def test_breakout_watchlist_only(self):
        from scheduler.alert_runner import _evaluate

        alert = {"alert_type": "breakout", "threshold": 8.0, "watchlist_only": True}
        lines = _evaluate(alert, self._df(), {"AAPL"})
        self.assertEqual(len(lines), 1)
        self.assertIn("AAPL", lines[0])

    def test_watchlist_membership(self):
        from scheduler.alert_runner import _evaluate

        alert = {"alert_type": "watchlist"}
        lines = _evaluate(alert, self._df(), {"NVDA", "ZZZZ"})
        self.assertEqual(len(lines), 1)
        self.assertIn("NVDA", lines[0])

    def test_price_above_and_below(self):
        from scheduler.alert_runner import _evaluate

        df = self._df()
        above = _evaluate(
            {"alert_type": "price", "ticker": "AAPL", "direction": "above", "threshold": 200.0},
            df,
            set(),
        )
        self.assertEqual(len(above), 1)
        below = _evaluate(
            {"alert_type": "price", "ticker": "AAPL", "direction": "below", "threshold": 200.0},
            df,
            set(),
        )
        self.assertEqual(below, [])

    def test_price_unknown_ticker(self):
        from scheduler.alert_runner import _evaluate

        lines = _evaluate(
            {"alert_type": "price", "ticker": "ZZZZ", "direction": "above", "threshold": 1.0},
            self._df(),
            set(),
        )
        self.assertEqual(lines, [])

    def test_symbol_column_fallback(self):
        from scheduler.alert_runner import _evaluate

        df = pd.DataFrame([{"Symbol": "TSLA", "BreakoutScore": 9.0, "Last": 380.0}])
        lines = _evaluate({"alert_type": "breakout", "threshold": 8.0}, df, set())
        self.assertEqual(len(lines), 1)
        self.assertIn("TSLA", lines[0])


class ThrottleTest(unittest.TestCase):
    def test_none_never_throttled(self):
        from scheduler.alert_runner import _throttled

        self.assertFalse(_throttled(None, 12))

    def test_recent_is_throttled(self):
        from scheduler.alert_runner import _throttled

        recent = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        self.assertTrue(_throttled(recent, 12))

    def test_old_is_not_throttled(self):
        from scheduler.alert_runner import _throttled

        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
        self.assertFalse(_throttled(old, 12))

    def test_naive_datetime_handled(self):
        from scheduler.alert_runner import _throttled

        naive = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).replace(
            tzinfo=None
        )
        self.assertTrue(_throttled(naive, 12))


if __name__ == "__main__":
    unittest.main()
