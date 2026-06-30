import importlib.util
import unittest
from unittest.mock import patch

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from scan.execution import run_manual_scan_execution


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for scan execution tests")
class ScanExecutionTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            [
                {"Ticker": "LOW", "GapPct": 0.5, "VolRel20": 1.0, "Close": 10.0},
                {"Ticker": "MID", "GapPct": 2.0, "VolRel20": 2.5, "Close": 20.0},
                {"Ticker": "HIGH", "GapPct": 4.0, "VolRel20": 3.0, "Close": 30.0},
            ]
        )

    def _run(self, runner, **overrides):
        kwargs = {
            "runner": runner,
            "tickers": ["LOW", "MID", "HIGH"],
            "premarket": False,
            "afterhours": False,
            "unusual_volume": False,
            "min_gap": 1.0,
            "min_price": 1.0,
            "max_price": 100.0,
            "top_n": 10,
            "profile": "regular",
            "apply_gap_filter": False,
        }
        kwargs.update(overrides)
        return run_manual_scan_execution(**kwargs)

    def test_calls_runner_with_progress_and_snapshot_when_supported(self):
        seen = {}

        def runner(**kwargs):
            seen.update(kwargs)
            return self._frame()

        result = self._run(
            runner,
            progress_cb=lambda *_args, **_kwargs: None,
            snapshot_id="snap-1",
        )

        self.assertEqual(list(result["Ticker"]), ["LOW", "MID", "HIGH"])
        self.assertIn("progress_cb", seen)
        self.assertEqual(seen["snapshot_id"], "snap-1")

    def test_falls_back_for_runner_without_progress_or_snapshot_args(self):
        def runner(
            *,
            tickers,
            premarket,
            afterhours,
            unusual_volume,
            min_gap,
            min_price,
            max_price,
            top_n,
            profile,
            diagnostics,
        ):
            self.assertEqual(tickers, ["LOW", "MID", "HIGH"])
            return self._frame()

        result = self._run(runner, progress_cb=lambda *_args, **_kwargs: None, snapshot_id="snap-1")

        self.assertEqual(list(result["Ticker"]), ["LOW", "MID", "HIGH"])

    def test_applies_gap_filter_when_enabled(self):
        result = self._run(
            lambda **_kwargs: self._frame(),
            apply_gap_filter=True,
            min_gap=2.0,
        )

        self.assertEqual(list(result["Ticker"]), ["MID", "HIGH"])

    def test_missing_gap_column_returns_empty_frame_when_gap_filter_enabled(self):
        result = self._run(
            lambda **_kwargs: self._frame().drop(columns=["GapPct"]),
            apply_gap_filter=True,
        )

        self.assertTrue(result.empty)

    def test_applies_strategy_filter_and_top_n(self):
        result = self._run(
            lambda **_kwargs: self._frame(),
            strategy="unusual_vol",
            top_n=1,
        )

        self.assertEqual(list(result["Ticker"]), ["HIGH"])

    def test_applies_extended_price_transform_for_extended_hours(self):
        def transform(frame):
            out = frame.copy()
            out["Close"] = out["Close"] + 1
            return out

        result = self._run(
            lambda **_kwargs: self._frame(),
            premarket=True,
            top_n=1,
            extended_price_transform=transform,
        )

        self.assertEqual(float(result.loc[0, "Close"]), 11.0)

    def test_scores_prebreakout_probabilities_for_manual_scans(self):
        def fake_score(frame):
            out = frame.copy()
            out["PreBreakoutProb"] = [0.1, 0.5, 0.9][: len(out)]
            out["PreBreakoutProb%"] = (out["PreBreakoutProb"] * 100).round(1)
            return out

        with patch("scan.execution.score_prebreakout", side_effect=fake_score) as scorer:
            result = self._run(lambda **_kwargs: self._frame())

        scorer.assert_called_once()
        self.assertIn("PreBreakoutProb", result.columns)
        self.assertIn("PreBreakoutProb%", result.columns)
        self.assertEqual(list(result["PreBreakoutProb%"]), [10.0, 50.0, 90.0])


if __name__ == "__main__":
    unittest.main()
