"""Tests for the real-time price-alert worker's pure logic."""
import datetime as dt
import unittest

from billing_service.realtime_alerts import crossed, market_session_open


def _utc(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc)


class CrossedTests(unittest.TestCase):
    def test_above(self):
        self.assertTrue(crossed("above", 401.0, 400.0))
        self.assertTrue(crossed("above", 400.0, 400.0))
        self.assertFalse(crossed("above", 399.99, 400.0))

    def test_below(self):
        self.assertTrue(crossed("below", 399.0, 400.0))
        self.assertFalse(crossed("below", 400.01, 400.0))

    def test_unknown_direction_defaults_to_above(self):
        self.assertTrue(crossed("sideways", 401.0, 400.0))


class MarketSessionTests(unittest.TestCase):
    # July = EDT (UTC-4). Monday 2026-07-13.
    def test_extended_hours_open(self):
        self.assertTrue(market_session_open(_utc("2026-07-13T08:05:00")))   # 4:05a ET premarket
        self.assertTrue(market_session_open(_utc("2026-07-13T14:00:00")))   # 10:00a ET
        self.assertTrue(market_session_open(_utc("2026-07-13T23:59:00")))   # 7:59p ET afterhours

    def test_overnight_closed(self):
        self.assertFalse(market_session_open(_utc("2026-07-14T00:05:00")))  # 8:05p ET Mon
        self.assertFalse(market_session_open(_utc("2026-07-13T07:55:00")))  # 3:55a ET

    def test_weekend_closed(self):
        self.assertFalse(market_session_open(_utc("2026-07-11T14:00:00")))  # Saturday


if __name__ == "__main__":
    unittest.main()


class EvaluateAlertTests(unittest.TestCase):
    def _alert(self, atype, threshold, direction="above"):
        return {
            "ticker": "TSLA",
            "alert_type": atype,
            "threshold": threshold,
            "direction": direction,
        }

    def test_price_alert_fires(self):
        from billing_service.realtime_alerts import evaluate_alert

        msg = evaluate_alert(self._alert("price", 400.0), {"last": 401.0})
        self.assertIn("above your 400.00", msg)
        self.assertIsNone(evaluate_alert(self._alert("price", 400.0), {"last": 399.0}))

    def test_move_alert_fires_both_directions(self):
        from billing_service.realtime_alerts import evaluate_alert

        up = evaluate_alert(self._alert("move", 5.0), {"last": 106.0, "prev_close": 100.0})
        self.assertIn("+6.0%", up)
        down = evaluate_alert(self._alert("move", 5.0), {"last": 94.0, "prev_close": 100.0})
        self.assertIn("-6.0%", down)
        self.assertIsNone(
            evaluate_alert(self._alert("move", 5.0), {"last": 103.0, "prev_close": 100.0})
        )

    def test_move_without_prev_close_is_silent(self):
        from billing_service.realtime_alerts import evaluate_alert

        self.assertIsNone(evaluate_alert(self._alert("move", 5.0), {"last": 106.0}))

    def test_rvol_alert_uses_cached_average(self):
        from billing_service import realtime_alerts as ra
        import datetime as dt

        ra._AVG_VOL["TSLA"] = (dt.datetime.now(dt.timezone.utc).date(), 1_000_000.0)
        msg = ra.evaluate_alert(
            self._alert("rvol", 2.0), {"last": 100.0, "volume": 2_500_000.0}
        )
        self.assertIn("2.5x", msg)
        self.assertIsNone(
            ra.evaluate_alert(self._alert("rvol", 2.0), {"last": 100.0, "volume": 500_000.0})
        )
