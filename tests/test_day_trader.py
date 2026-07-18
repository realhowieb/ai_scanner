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


class MarketStateClockTests(unittest.TestCase):
    def test_clock_open_overrides_time_guess(self):
        # Early-close day afternoon: time logic says open, exchange says closed.
        self.assertEqual(
            market_state(_utc("2026-11-27T19:30:00"), clock_is_open=False), "closed"
        )
        # Exchange open wins outright.
        self.assertEqual(
            market_state(_utc("2026-07-13T14:00:00"), clock_is_open=True), "open"
        )

    def test_holiday_midday_closed_with_clock(self):
        # July 3 2026 (observed July 4th) midday.
        self.assertEqual(
            market_state(_utc("2026-07-03T15:00:00"), clock_is_open=False), "closed"
        )

    def test_extended_hours_unaffected_by_clock_false(self):
        # Clock says regular session closed at 5pm ET — that's just after-hours.
        self.assertEqual(
            market_state(_utc("2026-07-13T21:00:00"), clock_is_open=False), "afterhours"
        )

    def test_winter_dst(self):
        # January = EST (UTC-5): 14:35 UTC is 9:35a ET -> open.
        self.assertEqual(market_state(_utc("2026-01-12T14:35:00")), "open")
        # 21:05 UTC = 4:05p EST -> afterhours.
        self.assertEqual(market_state(_utc("2026-01-12T21:05:00")), "afterhours")

    def test_naive_datetime_treated_as_utc(self):
        naive = dt.datetime(2026, 7, 13, 14, 0, 0)  # 10:00a ET Monday
        self.assertEqual(market_state(naive), "open")


class AfterHoursPctTests(unittest.TestCase):
    def test_basic_move(self):
        from ui.day_trader import after_hours_pct

        self.assertEqual(after_hours_pct(101.0, 100.0), 1.0)
        self.assertEqual(after_hours_pct(98.5, 100.0), -1.5)

    def test_no_ah_trade_or_missing_close_hidden(self):
        from ui.day_trader import after_hours_pct

        self.assertIsNone(after_hours_pct(100.0, 100.0))  # equal = no AH print
        self.assertIsNone(after_hours_pct(None, 100.0))
        self.assertIsNone(after_hours_pct(100.0, None))
        self.assertIsNone(after_hours_pct(100.0, 0.0))


class EmaCrossDisplayTests(unittest.TestCase):
    def test_formats_ema_cross_for_day_trader_table(self):
        from ui.day_trader import _ema_cross_display

        self.assertEqual(_ema_cross_display("Golden"), "Golden Cross")
        self.assertEqual(_ema_cross_display("Death"), "Death Cross")
        self.assertEqual(_ema_cross_display(None), "—")


class ParseValidationTests(unittest.TestCase):
    def test_rejects_junk_and_caps_count(self):
        raw = "AAPL, not a ticker!!, BRK.B, BRK-B, x" + ", FAKE" * 300
        out = _parse_symbols(raw)
        self.assertIn("AAPL", out)
        self.assertIn("BRK.B", out)
        self.assertIn("BRK-B", out)
        self.assertNotIn("NOT A TICKER!!", out)
        self.assertLessEqual(len(out), 200)

    def test_rejects_overlong(self):
        self.assertEqual(_parse_symbols("ABCDEFGHIJK"), [])


class DetectMovesEdgeTests(unittest.TestCase):
    def test_zero_threshold_flags_any_change(self):
        moves = detect_moves({"AAA": 100.0}, {"AAA": 100.01}, 0.0)
        self.assertEqual(len(moves), 1)

    def test_negative_threshold_behaves_like_zero(self):
        moves = detect_moves({"AAA": 100.0}, {"AAA": 100.0}, -5.0)
        self.assertEqual(moves, [("AAA", 0.0)])


class EmaCrossLabelTests(unittest.TestCase):
    def test_detects_bullish_and_bearish_cross_labels(self):
        import pandas as pd

        from market_data import ema_cross_label

        bullish = pd.DataFrame({"Close": [10.0] * 25 + [20.0]})
        bearish = pd.DataFrame({"Close": [20.0] * 25 + [10.0]})

        self.assertEqual(ema_cross_label(bullish), "Golden")
        self.assertEqual(ema_cross_label(bearish), "Death")

    def test_missing_close_has_no_cross_label(self):
        import pandas as pd

        from market_data import ema_cross_label

        self.assertIsNone(ema_cross_label(pd.DataFrame({"Open": [10.0] * 30})))
