"""Tests for the Day Trader panel's pure helpers."""
import datetime as dt
import importlib.util
import unittest

from ui.day_trader import _parse_symbols, detect_moves, market_state

_PANDAS = importlib.util.find_spec("pandas") is not None


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

    def test_day_trader_page_does_not_load_watchlists_before_render(self):
        from pathlib import Path

        source = Path("pages/day_trader.py").read_text()
        self.assertIn("_session_watch_tickers()", source)
        self.assertNotIn("from db.watchlists import", source)
        self.assertNotIn("list_watchlists(", source)
        self.assertNotIn("get_watchlist_tickers(", source)

    def test_day_trader_panel_renders_header_before_market_data_import(self):
        from pathlib import Path

        source = Path("ui/day_trader.py").read_text()
        # Scope the check to the render function — helper functions (e.g. the
        # movers screen) may lazily import market_data earlier in the file, but
        # those only execute when called, after the header renders.
        render = source[source.index("def render_day_trader_panel"):]
        header_idx = render.index('st.markdown("## ⚡ Day Trader — live")')
        table_import_idx = render.index("from market_data import build_day_trader_metrics")
        self.assertLess(header_idx, table_import_idx)


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


@unittest.skipUnless(_PANDAS, "ema_cross_label needs pandas")
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


@unittest.skipUnless(_PANDAS, "top movers screen needs pandas/market_data")
class DayTradeMoversTests(unittest.TestCase):
    def test_score_leads_with_momentum_and_vwap_alignment(self):
        from ui.day_trader import day_trade_score

        up_aligned = {"chg_pct": 6.0, "gap_pct": 4.0, "rvol": 3.0, "vs_vwap_pct": 2.0}
        up_misaligned = {"chg_pct": 6.0, "gap_pct": 4.0, "rvol": 3.0, "vs_vwap_pct": -2.0}
        flat = {"chg_pct": 0.1, "gap_pct": 0.0, "rvol": 1.0, "vs_vwap_pct": 0.0}
        self.assertGreater(day_trade_score(up_aligned), day_trade_score(up_misaligned))
        self.assertGreater(day_trade_score(up_misaligned), day_trade_score(flat))

    def test_score_is_nan_safe(self):
        from ui.day_trader import day_trade_score

        self.assertEqual(
            day_trade_score({"chg_pct": None, "gap_pct": float("nan"),
                             "rvol": None, "vs_vwap_pct": None}),
            0.0,
        )

    def test_top_movers_ranks_by_intraday_score(self):
        from unittest import mock

        import ui.day_trader as dt

        # last*volume must clear the liquidity floor ($1M) to be considered.
        rows = [
            {"ticker": "MOVE", "chg_pct": 8.0, "gap_pct": 5.0, "vs_vwap_pct": 3.0,
             "last": 100.0, "volume": 5_000_000},
            {"ticker": "MILD", "chg_pct": 1.0, "gap_pct": 0.5, "vs_vwap_pct": 0.4,
             "last": 50.0, "volume": 2_000_000},
            {"ticker": "DEAD", "chg_pct": 0.0, "gap_pct": 0.0, "vs_vwap_pct": 0.0,
             "last": 20.0, "volume": 3_000_000},  # liquid but flat (score 0) -> excluded
            {"ticker": "THIN", "chg_pct": 9.0, "gap_pct": 6.0, "vs_vwap_pct": 4.0,
             "last": 3.0, "volume": 10_000},  # big move but illiquid ($30k) -> excluded
        ]
        with mock.patch.object(dt, "_movers_universe",
                               return_value=["MOVE", "MILD", "DEAD", "THIN"]), \
             mock.patch("market_data.build_day_trader_metrics", return_value=rows):
            out = dt._top_movers_symbols(limit=10)
        # Ranked by score desc; flat and illiquid names dropped.
        self.assertEqual(out, ["MOVE", "MILD"])


if __name__ == "__main__":
    unittest.main()
