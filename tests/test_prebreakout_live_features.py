"""Live PreBreakout scoring builds inputs the way training does.

serving_skew_audit showed live scans scoring near chance (AUC 0.52) because
scan frames carry no Timestamp, daily OHLCV bars or SPY/QQQ context, so most
features were zero. score_prebreakout now enriches the frame first.
"""
import os
import unittest
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


if __name__ == "__main__":
    unittest.main()
