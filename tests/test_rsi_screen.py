"""RSI correctness and custom-scan screening contracts."""
import json
import unittest

import pandas as pd

from scan.rsi_screen import daily_rsi14, rsi_screen


def bars(up=0.5):
    closes = [100 + i * .5 for i in range(20)] + [109 - i for i in range(15)] + [95 + up * i for i in range(1, 5)]
    return pd.DataFrame({"Open": closes, "Close": closes, "High": [x + .1 for x in closes],
                         "Low": [x - .1 for x in closes], "Volume": [1000] * (len(closes)-1) + [1200]},
                        index=pd.date_range("2026-01-01", periods=len(closes)))


class RsiScreenTests(unittest.TestCase):
    def test_wilder_reference_and_flat_or_one_direction_markets(self):
        prices = pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28])
        self.assertAlmostEqual(daily_rsi14(prices).iloc[-1], 70.464135, delta=.000001)
        for prices, expected in [(range(1, 30), 100), (range(30, 1, -1), 0), ([10]*30, 50)]:
            assert daily_rsi14(pd.Series(prices)).iloc[-1] == expected


    def test_inclusive_bounds_and_missing_history_fail_closed(self):
        frame = bars()
        value, _, _ = rsi_screen(frame)
        assert rsi_screen(frame, min_rsi=value, max_rsi=value)[2]
        assert not rsi_screen(frame, max_rsi=value-.01)[2]
        for frame in (frame.iloc[:10], frame.assign(Close=float("nan"))):
            assert rsi_screen(frame) == (None, False, True)
            assert not rsi_screen(frame, min_rsi=0, max_rsi=100)[2]


    def test_rebound_requires_recovery_volume_and_up_close(self):
        assert rsi_screen(bars(), rebound=True, relative_volume=1)[1:]==(True, True)
        for frame, volume in [(bars(.3), 1.2), (bars(1.5), 1.2), (bars(), .99), (bars(), None)]:
            assert not rsi_screen(frame, rebound=True, relative_volume=volume)[2]
        frame = bars()
        frame.loc[frame.index[-1], "Close"] = frame.Close.iloc[-2] - .1
        assert not rsi_screen(frame, rebound=True, relative_volume=2)[2]
        assert daily_rsi14(bars().Close).iloc[:-1].equals(daily_rsi14(bars().Close.iloc[:-1]))


    def test_engine_filters_before_top_n(self):
        from scan.breakout import run_breakout_scan
        strong = bars().assign(Close=list(range(100, 139)), Open=list(range(100, 139)),
                               High=list(range(101, 140)), Low=list(range(99, 138)))
        result = run_breakout_scan(price_data={"STRONG": strong, "RECOVER": bars()}, spy_df=None,
                                  premarket=False, afterhours=False, unusual_volume=False, min_gap=0, min_price=1, max_price=1000,
                                  max_rsi=40, top_n=1, diagnostics=False)
        assert list(result.Ticker) == ["RECOVER"]
        assert result.RSI14.iloc[0] < 40


    def test_plan_preserves_bounds_and_rejects_conflicting_or_invalid_settings(self):
        from tests.test_api_scans import _plan
        plan = _plan({"universe": "sp500", "filters": {"min_rsi": 0, "max_rsi": 30}}, "premium")
        assert (plan["min_rsi"], plan["max_rsi"], plan["min_gap"]) == (0, 30, 0)
        for filters in ({"min_rsi": 40, "max_rsi": 30}, {"max_rsi": 101}, {"min_rsi": float("nan")}):
            with self.assertRaises(ValueError):
                _plan({"universe": "sp500", "filters": filters}, "premium")
        with self.assertRaisesRegex(ValueError, "Score all"):
            _plan({"universe": "ticker", "ticker": "AAPL", "score_all": True, "filters": {"rebound": True}}, "premium")


    def test_ai_builder_roundtrips_rsi_and_disables_score_all(self):
        from api.ai_premium import parse_builder
        result = parse_builder(json.dumps({"universe": "ticker", "ticker": "AAPL", "min_rsi": 28,
                                           "max_rsi": 32, "rebound": True, "score_all": True}), [])
        assert result["view"]["min_rsi"] == 28
        assert result["view"]["max_rsi"] == 32
        assert result["view"]["rebound"] is True
        assert result["view"]["score_all"] is False

    def test_momentum_cross_and_pullback_have_distinct_rules(self):
        from scan.rsi_screen import evaluate_screen

        base = [100 + i for i in range(70)] + [169 - 2*i for i in range(1, 8)]
        def frame(gain):
            return pd.DataFrame({"Close": base + [base[-1] + gain]}, index=pd.date_range("2026-01-01", periods=len(base)+1))
        recovery, passed = evaluate_screen(frame(3), momentum_recovery=True, relative_volume=1)
        self.assertTrue(passed)
        self.assertLessEqual(recovery["rsi_previous"], 50)
        self.assertGreater(recovery["rsi_14"], 50)
        pullback, passed = evaluate_screen(frame(1), uptrend_pullback=True, relative_volume=1)
        self.assertTrue(passed)
        self.assertGreater(pullback["close"], pullback["sma50"])
        self.assertFalse(evaluate_screen(frame(3), uptrend_pullback=True, relative_volume=1)[1])
        self.assertFalse(evaluate_screen(frame(1), momentum_recovery=True, relative_volume=1)[1])
        self.assertFalse(evaluate_screen(frame(1), uptrend_pullback=True, relative_volume=.99)[1])
        self.assertFalse(evaluate_screen(frame(1).iloc[-40:], uptrend_pullback=True, relative_volume=1)[1])
        # Cross above 50 must happen on this bar, not an earlier one.
        continued = pd.concat([frame(3), pd.DataFrame({"Close": [159]}, index=[pd.Timestamp("2026-03-20")])])
        self.assertFalse(evaluate_screen(continued, momentum_recovery=True, relative_volume=1)[1])

    def test_macd_cross_versus_above_signal_and_insufficient_history(self):
        from scan.rsi_screen import evaluate_screen

        closes = [100-i*.5 for i in range(50)]
        # A sufficiently large up close flips the histogram above zero on the last bar.
        frame = pd.DataFrame({"Close": closes + [100]}, index=pd.date_range("2026-01-01", periods=51))
        e, passed = evaluate_screen(frame, macd_confirmation="bullish_cross")
        self.assertTrue(passed)
        self.assertLessEqual(e["macd_previous_histogram"], 0)
        self.assertGreater(e["macd_histogram"], 0)
        next_frame = pd.concat([frame, pd.DataFrame({"Close": [101]}, index=[pd.Timestamp("2026-02-21")])])
        self.assertTrue(evaluate_screen(next_frame, macd_confirmation="above_signal")[1])
        self.assertFalse(evaluate_screen(next_frame, macd_confirmation="bullish_cross")[1])
        for mode in ("above_signal", "bullish_cross", "unsupported"):
            self.assertFalse(evaluate_screen(frame.iloc[-34:], macd_confirmation=mode)[1])
        # EMA values match the existing daily MACD implementation.
        from scan.indicators import macd
        lines, signals, histogram = macd(frame.Close)
        self.assertAlmostEqual(e["macd_line"], lines.iloc[-1])
        self.assertAlmostEqual(e["macd_signal"], signals.iloc[-1])
        self.assertAlmostEqual(e["macd_histogram"], histogram.iloc[-1])

    def test_new_plan_and_ai_fields_validate_and_disable_score_all(self):
        from api.ai_premium import parse_builder
        from tests.test_api_scans import _plan

        for filters in ({"momentum_recovery": True}, {"uptrend_pullback": True}, {"macd_confirmation": "above_signal"},
                        {"rebound": True, "macd_confirmation": "bullish_cross"}):
            plan = _plan({"universe": "sp500", "filters": filters}, "premium")
            for key, value in filters.items():
                self.assertEqual(plan[key], value)
            self.assertEqual(plan["min_gap"], 0)
            with self.assertRaisesRegex(ValueError, "Score all"):
                _plan({"universe": "ticker", "ticker": "BG", "score_all": True, "filters": filters}, "premium")
            view = parse_builder(json.dumps({"universe": "ticker", "ticker": "BG", "score_all": True, **filters}), [])["view"]
            self.assertFalse(view["score_all"])
            for key, value in filters.items():
                self.assertEqual(view[key], value)
        for invalid in ({"momentum_recovery": True, "rebound": True}, {"macd_confirmation": "guaranteed"}):
            with self.assertRaises(ValueError):
                _plan({"universe": "sp500", "filters": invalid}, "premium")
            with self.assertRaises(ValueError):
                parse_builder(json.dumps(invalid), [])

    def test_evidence_dates_json_and_chart_relative_volume(self):
        from api.models import IndicatorEvidence
        from api.scans import daily_indicator_evidence
        from scan.rsi_screen import indicator_evidence

        frame = bars()
        e = indicator_evidence(frame, relative_volume=1.2)
        self.assertEqual(e["bar_date"], frame.index[-1].date().isoformat())
        self.assertEqual(e["prior_5_rsi_low_date"], frame.index[-5].date().isoformat())
        self.assertEqual(IndicatorEvidence(**e).model_dump(), e)
        json.dumps(e, allow_nan=False)
        chart_bars = [{"date": d.date().isoformat(), "close": float(r.Close), "volume": float(r.Volume)} for d, r in frame.iterrows()]
        self.assertEqual(daily_indicator_evidence(chart_bars), e)
        chart_bars[-1]["volume"] = None
        self.assertIsNone(daily_indicator_evidence(chart_bars)["relative_volume"])
        self.assertFalse(daily_indicator_evidence(chart_bars)["rebound_candidate"])

    def test_custom_scan_passes_new_filters_and_keeps_snapshot_evidence(self):
        from unittest import mock

        from api.custom_scans import run_scan
        from scan.rsi_screen import indicator_evidence
        from tests.test_api_scans import _plan

        e = indicator_evidence(bars(), relative_volume=1.2)
        frame = pd.DataFrame([{"Ticker": "BG", "RSI14": e["rsi_14"], "ReboundCandidate": True,
                               "IndicatorEvidence": e, "Last": e["close"]}])
        params = _plan({"universe": "ticker", "ticker": "BG", "filters": {
            "rebound": True, "macd_confirmation": "above_signal"}}, "premium")
        captured = {}
        def execute(**kwargs):
            captured.update(kwargs)
            return kwargs["runner"](tickers=kwargs["tickers"])
        with mock.patch("scan.execution.run_manual_scan_execution", side_effect=execute), \
                mock.patch("scan.engine.run_breakout_scan", return_value=frame) as engine, \
                mock.patch("ui.market_scans.top_setups", return_value=[{"ticker": "BG", "score": 30, "signals": []}]), \
                mock.patch("db.runs.save_run") as save:
            result = run_scan(params, "user@example.com", lambda _: None)
        self.assertEqual(result["label"], "Search: BG")
        self.assertEqual(engine.call_args.kwargs["macd_confirmation"], "above_signal")
        self.assertTrue(engine.call_args.kwargs["rebound"])
        self.assertEqual(captured["min_gap"], 0)
        self.assertEqual(result["setups"][0]["indicator_evidence"], e)
        self.assertEqual(result["setups"][0]["primary_setup"], "Rebound candidate")
        saved_rows = json.loads(save.call_args.args[1])
        self.assertAlmostEqual(saved_rows[0]["IndicatorEvidence"]["rsi_14"], e["rsi_14"], places=7)
        self.assertEqual(saved_rows[0]["IndicatorEvidence"]["rule_version"], "daily-indicators-v1")
