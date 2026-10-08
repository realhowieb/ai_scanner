"""Live PreBreakout scoring builds inputs the way training does.

serving_skew_audit showed live scans scoring near chance (AUC 0.52) because
scan frames carry no Timestamp, daily OHLCV bars or SPY/QQQ context, so most
features were zero. score_prebreakout now enriches the frame first.
"""
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd

import ml_prebreakout as m


class _Model:
    def predict_proba(self, X):
        v = 1 / (1 + np.exp(-X["F1"].to_numpy(dtype=float)))
        return np.column_stack([1 - v, v])


_BUNDLE = {"model": _Model(), "features": ["F1"], "calibration_map": None}


def _features(df, benchmark_context=None, **_):
    out = df.copy()
    out["F1"] = pd.to_numeric(out.get("OHLC_Close", out["Last"]), errors="coerce")
    return out


class LiveFeatureTests(unittest.TestCase):
    def setUp(self):
        m._BENCHMARK_CACHE.clear()
        self.scan = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Last": [1.0, -1.0, 0.0]}, index=[7, 3, 5])

    def _score(self, enrich=None, bench=None, env=None):
        enrich = enrich or (lambda df, **_: df.assign(OHLC_Close=-df["Last"]).iloc[::-1])
        bench = bench or (lambda days: {"SPY": pd.DataFrame({"x": [1]}), "QQQ": pd.DataFrame()})
        with mock.patch.object(m, "load_prebreakout_model", return_value=_BUNDLE), \
                mock.patch.object(m, "add_historical_ohlcv_context", side_effect=enrich) as e, \
                mock.patch.object(m, "load_benchmark_regime_context", side_effect=bench) as b, \
                mock.patch.object(m, "add_prebreakout_features", side_effect=_features) as f, \
                mock.patch.dict(os.environ, env or {}, clear=False):
            out = m.score_prebreakout(self.scan.copy())
        return out, e, b, f

    def test_enriches_and_keeps_scores_on_their_rows(self):
        out, e, b, f = self._score()
        e.assert_called_once()
        self.assertIn("Timestamp", e.call_args.args[0].columns)  # scan time stamped
        self.assertIsNotNone(f.call_args.kwargs["benchmark_context"])
        # F1 = -Last after enrichment, even though enrichment reversed the rows.
        expected = 1 / (1 + np.exp(self.scan["Last"].to_numpy()))
        self.assertTrue(np.allclose(out["PreBreakoutProbRaw"].to_numpy(), expected))
        self.assertEqual(list(out.index), [7, 3, 5])

    def test_provider_failure_falls_back_to_the_scan_alone(self):
        def boom(*_, **__):
            raise RuntimeError("alpaca down")

        out, *_ = self._score(enrich=boom, bench=boom)
        expected = 1 / (1 + np.exp(-self.scan["Last"].to_numpy()))
        self.assertTrue(np.allclose(out["PreBreakoutProbRaw"].to_numpy(), expected))

    def test_env_flag_turns_enrichment_off(self):
        _, e, b, f = self._score(env={"PREBREAKOUT_LIVE_ENRICH": "0"})
        e.assert_not_called()
        b.assert_not_called()
        self.assertIsNone(f.call_args.kwargs["benchmark_context"])

    def test_benchmark_context_is_cached(self):
        self._score()
        _, _, b, _ = self._score()
        b.assert_not_called()


class LiveFeatureRealMergeTests(unittest.TestCase):
    """No feature mocks: only the price download and the model are stubbed.

    Regression: a scan stamped with a python datetime (us) merged against
    daily bars (ns) raised MergeError, the scan swallowed it, and every row
    showed no PreBreakout score.
    """

    def setUp(self):
        m._BENCHMARK_CACHE.clear()

    def test_scan_time_merges_with_daily_bars_and_spy_qqq(self):
        idx = pd.date_range("2026-07-01", periods=90, freq="B", tz="America/New_York")

        def download(symbols, **_):
            close = np.linspace(100.0, 130.0, len(idx))
            bars = pd.DataFrame(
                {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}, index=idx
            )
            return {s: bars for s in symbols}

        scan = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Last": [10.0, 20.0], "Trend10D%": [1.0, 2.0]}, index=[4, 9])
        bundle = {"model": _Model(), "features": ["F1"], "calibration_map": None}
        with mock.patch("data.price_alpaca.download_multi_alpaca", side_effect=download), \
                mock.patch.object(m, "_utc_now", return_value=datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)), \
                mock.patch.object(m, "load_prebreakout_model", return_value=bundle), \
                mock.patch.dict(os.environ, {"PREBREAKOUT_LIVE_ENRICH": "1"}):
            X = m._live_feature_frame(scan)
            out = m.score_prebreakout(scan.copy())
        self.assertTrue((X["Close"] == 130.0).all())  # last completed bar
        self.assertTrue(X["SPYTrend10D"].notna().all())
        self.assertTrue(out["PreBreakoutProbRaw"].notna().all())
        self.assertEqual(list(out.index), [4, 9])

    def test_enriched_feature_failure_still_scores(self):
        calls = []

        def features(df, benchmark_context=None, **_):
            calls.append(benchmark_context)
            if benchmark_context is not None:
                raise ValueError("incompatible merge keys")
            return df.assign(F1=df["Last"])

        bundle = {"model": _Model(), "features": ["F1"], "calibration_map": None}
        scan = pd.DataFrame({"Symbol": ["AAA"], "Last": [1.0]})
        with mock.patch.object(m, "load_prebreakout_model", return_value=bundle), \
                mock.patch.object(m, "add_historical_ohlcv_context", side_effect=lambda df, **_: df), \
                mock.patch.object(m, "load_benchmark_regime_context", return_value={"SPY": pd.DataFrame({"x": [1]})}), \
                mock.patch.object(m, "add_prebreakout_features", side_effect=features):
            out = m.score_prebreakout(scan)
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.isclose(out["PreBreakoutProbRaw"].iloc[0], 1 / (1 + np.exp(-1.0))))


if __name__ == "__main__":
    unittest.main()
