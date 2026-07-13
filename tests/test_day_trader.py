"""Tests for the Day Trader panel's pure helpers."""
import datetime as dt
import unittest

from ui.day_trader import _parse_symbols, detect_moves, market_state


def _utc(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc)


class MarketStateTests(unittest.TestCase):
    # July = EDT (UTC-4). Monday 2026-07-13.
    def test_states_through_the_day(self):
        self.assertEqual(market_state(_utc("2026-07-13T07:00:00")), "closed")     # 3:00a ET
        self.assertEqual(market_state(_utc("2026-07-13T08:30:00")), "premarket")  # 4:30a ET
        self.assertEqual(market_state(_utc("2026-07-13T13:29:00")), "premarket")  # 9:29a ET
        self.assertEqual(market_state(_utc("2026-07-13T13:30:00")), "open")       # 9:30a ET
        self.assertEqual(market_state(_utc("2026-07-13T19:59:00")), "open")       # 3:59p ET
        self.assertEqual(market_state(_utc("2026-07-13T20:00:00")), "afterhours") # 4:00p ET
        self.assertEqual(market_state(_utc("2026-07-13T23:59:00")), "afterhours") # 7:59p ET
        self.assertEqual(market_state(_utc("2026-07-14T00:00:00")), "closed")     # 8:00p ET

    def test_weekend_closed(self):
        self.assertEqual(market_state(_utc("2026-07-11T14:00:00")), "closed")  # Saturday


class DetectMovesTests(unittest.TestCase):
    def test_flags_moves_past_threshold_both_directions(self):
        moves = detect_moves(
            {"AAA": 100.0, "BBB": 50.0, "CCC": 10.0},
            {"AAA": 102.5, "BBB": 48.0, "CCC": 10.05},
            2.0,
        )
        self.assertIn(("AAA", 2.5), moves)
        self.assertIn(("BBB", -4.0), moves)
        self.assertEqual(len(moves), 2)  # CCC's +0.5% is below threshold

    def test_missing_baseline_or_price_ignored(self):
        self.assertEqual(detect_moves({}, {"AAA": 100.0}, 1.0), [])
        self.assertEqual(detect_moves({"AAA": 0.0}, {"AAA": 100.0}, 1.0), [])
        self.assertEqual(detect_moves({"AAA": 100.0}, {"AAA": None}, 1.0), [])


class ParseSymbolsTests(unittest.TestCase):
    def test_parse_dedupes_and_uppercases(self):
        self.assertEqual(_parse_symbols("aapl, TSLA,aapl\nnvda"), ["AAPL", "TSLA", "NVDA"])


if __name__ == "__main__":
    unittest.main()
