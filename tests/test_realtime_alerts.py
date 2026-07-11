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
