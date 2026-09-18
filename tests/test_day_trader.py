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


class DayTraderFormattingTests(unittest.TestCase):
    def test_change_dollar_formatting(self):
        from ui.day_trader import format_change_dollar

        self.assertEqual(format_change_dollar(0.75), "+$0.75")
        self.assertEqual(format_change_dollar(-0.4), "-$0.40")
        self.assertEqual(format_change_dollar(float("nan")), "—")
        self.assertEqual(format_change_dollar(None), "—")

    def test_volume_millions_formatting(self):
        from ui.day_trader import format_volume_millions

        self.assertEqual(format_volume_millions(850_000), "0.85M")
        self.assertEqual(format_volume_millions(12_400_000), "12.40M")
        self.assertEqual(format_volume_millions(float("nan")), "—")

    def test_volume_column_label_discloses_iex_partial_feed(self):
        from ui.day_trader import _volume_column_label

        self.assertEqual(
            _volume_column_label([{"volume_source": "alpaca_iex"}]),
            "Volume (IEX M)",
        )
        self.assertEqual(
            _volume_column_label([{"volume_source": "alpaca_sip"}]),
            "Volume (M)",
        )

    def test_vwap_distance_formatting_shows_above_below(self):
        from ui.day_trader import format_vwap_distance

        self.assertEqual(format_vwap_distance(0.42), "above")
        self.assertEqual(format_vwap_distance(-1.15), "below")
        self.assertEqual(format_vwap_distance(0), "at VWAP")
        self.assertEqual(format_vwap_distance(None), "—")

    def test_supertrend_display(self):
        from ui.day_trader import format_supertrend_direction

        self.assertEqual(format_supertrend_direction("green"), "🟢 Green")
        self.assertEqual(format_supertrend_direction("red"), "🔴 Red")
        self.assertEqual(format_supertrend_direction(None), "—")

    def test_primary_table_column_order(self):
        from ui.day_trader import DAY_TRADER_TABLE_COLUMNS

        self.assertEqual(
            DAY_TRADER_TABLE_COLUMNS,
            [
                "Ticker",
                "Open",
                "Last",
                "Change $",
                "Gap %",
                "ADX",
                "VWAP",
                "vs VWAP",
                "RVOL",
                "Volume (M)",
                "SuperTrend (13,2)",
                "EWO",
                "Direction",
                "DT Score",
                "Setup",
            ],
        )

    @unittest.skipUnless(_PANDAS, "display table shaping needs pandas")
    def test_primary_table_keeps_indicator_columns_when_missing(self):
        import pandas as pd

        from ui.day_trader import DAY_TRADER_TABLE_COLUMNS, _ensure_day_trader_table_columns

        df = pd.DataFrame(
            [{"Ticker": "CRWD", "Open": 236.38, "Last": 245.44, "Volume (M)": 0.33}]
        )
        shaped = _ensure_day_trader_table_columns(df)

        self.assertEqual(list(shaped.columns), DAY_TRADER_TABLE_COLUMNS)
        self.assertTrue(pd.isna(shaped.loc[0, "ADX"]))
        self.assertEqual(shaped.loc[0, "SuperTrend (13,2)"], "—")
        self.assertTrue(pd.isna(shaped.loc[0, "EWO"]))

    @unittest.skipUnless(_PANDAS, "display table styling needs pandas")
    def test_missing_indicator_cells_do_not_break_styling(self):
        import pandas as pd

        from ui.day_trader import _ensure_day_trader_table_columns, _styled

        df = pd.DataFrame(
            [{"Ticker": "CRWD", "Open": 236.38, "Last": 245.44, "Volume (M)": 0.33}]
        )
        shaped = _ensure_day_trader_table_columns(df)
        styled = _styled(shaped, moved_now=set())

        self.assertTrue(hasattr(styled, "to_html"))
        self.assertIn("CRWD", styled.to_html())

    def test_day_trader_formula_audit_helper(self):
        from market_data import calculate_day_trader_row_audit

        audit = calculate_day_trader_row_audit(
            {
                "ticker": "DELL",
                "open": 572.0,
                "previous_close": 563.32,
                "last": 588.42,
                "change_dollar": 16.42,
                "gap_pct": 1.54,
                "vwap": 584.04,
                "open_source": "alpaca_iex",
                "previous_close_source": "alpaca_iex",
                "vwap_source": "alpaca_iex",
            }
        )

        self.assertEqual(audit["expected_change_dollar"], 16.42)
        self.assertEqual(audit["expected_gap_pct"], 1.54)
        self.assertTrue(audit["gap_pass"])
        self.assertTrue(audit["change_pass"])
        self.assertEqual(audit["expected_vwap_state"], "above")

    def test_day_trader_formula_audit_neutral_vwap_state(self):
        from market_data import calculate_day_trader_row_audit

        audit = calculate_day_trader_row_audit(
            {
                "ticker": "FLAT",
                "open": 100.0,
                "previous_close": 100.0,
                "last": 101.0,
                "change_dollar": 1.0,
                "gap_pct": 0.0,
                "vwap": 101.0,
            }
        )

        self.assertEqual(audit["expected_vwap_state"], "at VWAP")

    def test_metric_timeframe_matrix_documents_daily_indicators(self):
        from market_data import day_trader_metric_timeframes

        rows = {row["metric"]: row for row in day_trader_metric_timeframes()}

        self.assertEqual(rows["ADX"]["timeframe"], "1d")
        self.assertEqual(rows["SuperTrend"]["source"], "daily OHLC bars, SuperTrend(13,2)")
        self.assertEqual(rows["EWO"]["source"], "daily close bars, SMA(5)-SMA(35)")
        self.assertEqual(rows["VWAP"]["source"], "Alpaca dailyBar.vw")


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


class SymbolSourceTests(unittest.TestCase):
    def test_top_movers_respects_passed_universe(self):
        """SP500/NASDAQ movers scope the screen to the given universe rather than
        the full S&P 500 + NASDAQ union."""
        from unittest import mock

        import ui.day_trader as d

        captured = {}

        def fake_metrics(universe, **kw):
            captured["universe"] = list(universe)
            return [{"ticker": "AAA", "last": 10.0, "volume": 1_000_000,
                     "chg_pct": 5.0, "gap_pct": 1.0, "rvol": 2.0, "vs_vwap_pct": 1.0}]

        with mock.patch("market_data.build_day_trader_metrics", side_effect=fake_metrics):
            out = d._top_movers_symbols(universe=["AAA", "BBB"])
        self.assertEqual(captured["universe"], ["AAA", "BBB"])
        self.assertIn("AAA", out)

    def test_session_scan_picks_selected_by_label(self):
        """Premarket/postmarket sources load the most recent run whose label
        matches the session, not any other scan."""
        from unittest import mock

        import ui.day_trader as d

        runs = [
            {"id": 2, "label": "regular"},
            {"id": 1, "label": "premarket"},
        ]

        class _DF:
            columns = ["Ticker"]

            def __len__(self):
                return 2

            def __getitem__(self, k):
                class _Col:
                    def head(self, n):
                        class _H:
                            def tolist(self_inner):
                                return ["GAPR", "MOVR"]
                        return _H()
                return _Col()

        import sys
        import types

        fake_runtime = types.ModuleType("ui.app_runtime")
        fake_runtime.normalize_results_to_df = lambda raw: _DF()

        with mock.patch("db.runs.list_runs", return_value=runs), \
             mock.patch("db.runs.load_run_results", return_value="[]") as load, \
             mock.patch.dict(sys.modules, {"ui.app_runtime": fake_runtime}):
            out = d._session_scan_symbols("premarket")
        # loaded the premarket run (id=1), not the regular one (id=2)
        load.assert_called_once_with(1)
        self.assertEqual(out, ["GAPR", "MOVR"])

    def test_stale_delisted_tickers_dropped_from_movers(self):
        """A delisted name (old last trade, e.g. QMMM) must not appear as a mover."""
        import datetime as dt
        from unittest import mock

        import ui.day_trader as d

        fresh = dt.datetime.now(dt.timezone.utc).isoformat()
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
        metrics = [
            {"ticker": "LIVE", "last": 10.0, "volume": 5_000_000, "chg_pct": 8.0,
             "gap_pct": 2.0, "rvol": 2.0, "vs_vwap_pct": 1.0, "trade_ts": fresh},
            {"ticker": "QMMM", "last": 116.0, "volume": 5_000_000, "chg_pct": 17.0,
             "gap_pct": 5.0, "rvol": 0.0, "vs_vwap_pct": 1.0, "trade_ts": old},
        ]
        with mock.patch("market_data.build_day_trader_metrics", return_value=metrics):
            out = d._top_movers_symbols(universe=["LIVE", "QMMM"])
        self.assertIn("LIVE", out)
        self.assertNotIn("QMMM", out)          # stale → filtered

    def test_is_stale_bounds(self):
        import datetime as dt

        import ui.day_trader as d

        recent = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=20)).isoformat()
        self.assertFalse(d._is_stale(recent))
        self.assertTrue(d._is_stale(old))
        self.assertFalse(d._is_stale(None))     # unknown → never over-filter
        self.assertFalse(d._is_stale("garbage"))

    def test_session_scan_none_when_label_absent(self):
        from unittest import mock

        import ui.day_trader as d

        with mock.patch("db.runs.list_runs", return_value=[{"id": 2, "label": "regular"}]):
            self.assertEqual(d._session_scan_symbols("postmarket"), [])


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


@unittest.skipUnless(_PANDAS, "range metrics need pandas")
class RangeMetricsTests(unittest.TestCase):
    def _df(self, closes):
        import pandas as pd
        c = pd.Series([float(x) for x in closes])
        return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c})

    def test_donchian_and_bollinger_shapes(self):
        from scan.indicators import bollinger, donchian

        df = self._df(range(100, 160))
        u, low_b = donchian(df, 20)
        mid, bu, bl = bollinger(df["Close"], 20, 2)
        self.assertGreater(float(u.iloc[-1]), float(low_b.iloc[-1]))
        self.assertGreater(float(bu.iloc[-1]), float(bl.iloc[-1]))
        self.assertAlmostEqual(float(mid.iloc[-1]), (float(bu.iloc[-1]) + float(bl.iloc[-1])) / 2, places=4)

    def test_range_metrics_uptrend(self):
        from market_data import _range_metrics

        m = _range_metrics(self._df(range(100, 160)))  # steady uptrend
        self.assertIsNotNone(m)
        self.assertGreater(m["donchian_pos"], 70)          # near the 20d high
        self.assertEqual(m["donchian_breakout"], "up")     # fresh 20d high
        self.assertIn("atr_pct", m)
        self.assertIsInstance(m["bb_squeeze"], bool)

    def test_range_metrics_insufficient_bars(self):
        from market_data import _range_metrics

        self.assertIsNone(_range_metrics(self._df(range(100, 110))))
        self.assertIsNone(_range_metrics(None))

    def test_range_metrics_includes_professional_indicators(self):
        from market_data import _range_metrics

        df = self._df(range(100, 170))
        df.attrs["source"] = "alpaca_multi"
        df.attrs["feed"] = "iex"
        m = _range_metrics(df)
        self.assertIsNotNone(m)
        self.assertIn("adx", m)
        self.assertIn("supertrend", m)
        self.assertIn("supertrend_direction", m)
        self.assertIn("ewo", m)
        self.assertIsNotNone(m["adx"])
        self.assertIn(m["supertrend_direction"], ("green", "red"))
        self.assertIsNotNone(m["ewo"])
        self.assertIsNotNone(m["ewo_pct"])
        self.assertEqual(m["adx_period"], 14)
        self.assertEqual(m["adx_timeframe"], "1d")
        self.assertEqual(m["supertrend_period"], 13)
        self.assertEqual(m["supertrend_multiplier"], 2.0)
        self.assertEqual(m["supertrend_timeframe"], "1d")
        self.assertEqual(m["ewo_fast"], 5)
        self.assertEqual(m["ewo_slow"], 35)
        self.assertEqual(m["ewo_timeframe"], "1d")
        self.assertEqual(m["ewo_source"], "alpaca_multi")
        self.assertEqual(m["ewo_feed"], "iex")

    def test_build_day_trader_metrics_adds_open_and_change_dollar(self):
        from unittest import mock

        from market_data import build_day_trader_metrics

        snapshots = {
            "TEST": {
                "latestTrade": {"p": 10.75, "t": "2026-09-17T14:30:00Z"},
                "dailyBar": {"o": 10.0, "c": 10.5, "vw": 10.2, "v": 1_000_000},
                "prevDailyBar": {"c": 9.5},
            }
        }
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snapshots), \
             mock.patch("market_data._get_alpaca_data_feed", return_value="iex"), \
             mock.patch("market_data.fetch_avg_daily_volume", return_value={"TEST": 500_000}), \
             mock.patch("market_data.fetch_ema_crosses", return_value={}), \
             mock.patch("market_data.fetch_daily_range_metrics", return_value={}):
            rows = build_day_trader_metrics(["TEST"])

        self.assertEqual(rows[0]["open"], 10.0)
        self.assertEqual(rows[0]["last"], 10.75)
        self.assertEqual(rows[0]["previous_close"], 9.5)
        self.assertEqual(rows[0]["change_dollar"], 0.75)
        self.assertEqual(rows[0]["expected_change_dollar"], 0.75)
        self.assertEqual(rows[0]["expected_gap_pct"], 5.26)
        self.assertTrue(rows[0]["gap_pass"])
        self.assertTrue(rows[0]["change_pass"])
        self.assertEqual(rows[0]["volume"], 1_000_000)
        self.assertEqual(rows[0]["volume_source"], "alpaca_iex")
        self.assertEqual(rows[0]["open_source"], "alpaca_iex")
        self.assertEqual(rows[0]["open_timeframe"], "snapshot_daily_bar")
        self.assertEqual(rows[0]["prev_close_source"], "alpaca_iex")
        self.assertEqual(rows[0]["previous_close_source"], "alpaca_iex")
        self.assertEqual(
            rows[0]["previous_close_timeframe"],
            "previous_completed_regular_session_daily_bar",
        )
        self.assertEqual(rows[0]["vwap_source"], "alpaca_iex")
        self.assertEqual(rows[0]["vwap_formula"], "provider_supplied_daily_bar_vw")
        self.assertEqual(rows[0]["rvol_source"], "alpaca_iex_current_vs_20d_alpaca_iex_avg")

    def test_build_day_trader_metrics_missing_market_data_is_safe(self):
        from unittest import mock

        from market_data import build_day_trader_metrics

        snapshots = {
            "TEST": {
                "latestTrade": {"p": 10.75},
                "dailyBar": {"c": 10.5, "vw": 10.2, "v": 1_000_000},
                "prevDailyBar": {"c": 9.5},
            }
        }
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snapshots), \
             mock.patch("market_data._get_alpaca_data_feed", return_value="iex"), \
             mock.patch("market_data.fetch_avg_daily_volume", return_value={}), \
             mock.patch("market_data.fetch_ema_crosses", return_value={}), \
             mock.patch("market_data.fetch_daily_range_metrics", return_value={}):
            rows = build_day_trader_metrics(["TEST"])

        self.assertIsNone(rows[0]["open"])
        self.assertIsNone(rows[0]["change_dollar"])

    def test_gap_uses_daily_open_not_latest_trade(self):
        from unittest import mock

        from market_data import build_day_trader_metrics

        snapshots = {
            "TEST": {
                "latestTrade": {"p": 120.0},
                "dailyBar": {"o": 105.0, "c": 110.0, "vw": 112.0, "v": 1_000_000},
                "prevDailyBar": {"c": 100.0},
            }
        }
        with mock.patch("market_data.fetch_alpaca_snapshots", return_value=snapshots), \
             mock.patch("market_data._get_alpaca_data_feed", return_value="iex"), \
             mock.patch("market_data.fetch_avg_daily_volume", return_value={}), \
             mock.patch("market_data.fetch_ema_crosses", return_value={}), \
             mock.patch("market_data.fetch_daily_range_metrics", return_value={}):
            rows = build_day_trader_metrics(["TEST"])

        self.assertEqual(rows[0]["open"], 105.0)
        self.assertEqual(rows[0]["change_dollar"], 15.0)
        self.assertEqual(rows[0]["gap_pct"], 5.0)
        self.assertEqual(rows[0]["expected_gap_pct"], 5.0)
        self.assertEqual(rows[0]["expected_change_dollar"], 15.0)


@unittest.skipUnless(_PANDAS, "professional indicators need pandas")
class ProfessionalIndicatorTests(unittest.TestCase):
    def _df(self, closes):
        import pandas as pd

        c = pd.Series([float(x) for x in closes])
        return pd.DataFrame({"High": c * 1.02, "Low": c * 0.98, "Close": c})

    def test_ewo_calculation(self):
        from scan.indicators import ewo, ewo_pct

        osc = ewo(self._df(range(1, 41)))
        self.assertAlmostEqual(float(osc.iloc[-1]), 15.0, places=4)
        norm = ewo_pct(self._df(range(1, 41)))
        self.assertAlmostEqual(float(norm.iloc[-1]), (15.0 / 23.0) * 100.0, places=4)

    def test_adx_calculation(self):
        from scan.indicators import adx

        value = float(adx(self._df(range(100, 170))).iloc[-1])
        self.assertGreater(value, 20.0)
        self.assertLessEqual(value, 100.0)

    def test_adx_insufficient_bars_yields_nan(self):
        from scan.indicators import adx

        value = adx(self._df(range(100, 110))).iloc[-1]
        self.assertTrue(value != value)

    def test_supertrend_bullish_and_bearish_state(self):
        from scan.indicators import supertrend

        bullish = supertrend(self._df(range(100, 170)), 13, 2)
        bearish = supertrend(self._df(range(170, 100, -1)), 13, 2)
        self.assertEqual(bullish["direction"].iloc[-1], "green")
        self.assertEqual(bearish["direction"].iloc[-1], "red")

    def test_supertrend_insufficient_bars_yields_missing_state(self):
        from scan.indicators import supertrend

        st = supertrend(self._df(range(100, 110)), 13, 2)
        self.assertTrue(st["supertrend"].isna().all())
        self.assertTrue(st["direction"].isna().all())
