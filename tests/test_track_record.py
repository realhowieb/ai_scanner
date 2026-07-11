"""Unit tests for the signal track record compute (forward returns vs SPY)."""
import datetime as dt
import unittest
from unittest import mock

import pytest

pd = pytest.importorskip("pandas")

from analytics import track_record as tr  # noqa: E402


def _bars(start: str, closes):
    """Daily bars DataFrame with a DatetimeIndex, matching price_alpaca output."""
    idx = pd.date_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


class ForwardReturnTests(unittest.TestCase):
    def test_entry_is_first_bar_on_or_after_run_date(self):
        # Mon Jan 6 .. Fri Jan 10; run_date Tue Jan 7 → entry 102, exit +2 bars 104.
        bars = _bars("2025-01-06", [100.0, 102.0, 103.0, 104.0, 105.0])
        r = tr._forward_return(bars, dt.date(2025, 1, 7), horizon_days=2)
        self.assertAlmostEqual(r, (104.0 - 102.0) / 102.0)

    def test_run_date_on_weekend_uses_next_trading_day(self):
        bars = _bars("2025-01-06", [100.0, 102.0, 103.0, 104.0, 105.0])
        # Sat Jan 4 → first bar on/after is Mon Jan 6 (100) → exit 103.
        r = tr._forward_return(bars, dt.date(2025, 1, 4), horizon_days=2)
        self.assertAlmostEqual(r, (103.0 - 100.0) / 100.0)

    def test_incomplete_forward_window_returns_none(self):
        bars = _bars("2025-01-06", [100.0, 102.0, 103.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 1, 7), horizon_days=5))

    def test_run_date_after_all_bars_returns_none(self):
        bars = _bars("2025-01-06", [100.0, 102.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 6, 1), horizon_days=1))

    def test_zero_entry_price_returns_none(self):
        bars = _bars("2025-01-06", [0.0, 102.0, 103.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 1, 6), horizon_days=1))


class RankedSymbolsTests(unittest.TestCase):
    def _df(self):
        return pd.DataFrame(
            {
                "Ticker": ["LOW", "HIGH", "MID"],
                "BreakoutScore": [1.0, 9.0, 5.0],
            }
        )

    def test_breakout_ranking_sorts_by_score(self):
        self.assertEqual(tr._ranked_symbols(self._df(), "breakout", 2), ["HIGH", "MID"])

    def test_prebreakout_ranking_uses_model_probability(self):
        def fake_score(df):
            out = df.copy()
            # Reverse of BreakoutScore so the two rankings must differ.
            out["PreBreakoutProb%"] = [90.0, 10.0, 50.0]
            return out

        with mock.patch("ml_prebreakout.score_prebreakout", side_effect=fake_score):
            self.assertEqual(
                tr._ranked_symbols(self._df(), "prebreakout", 2), ["LOW", "MID"]
            )

    def test_prebreakout_ranking_empty_when_model_unavailable(self):
        with mock.patch(
            "ml_prebreakout.score_prebreakout", side_effect=RuntimeError("no model")
        ):
            self.assertEqual(tr._ranked_symbols(self._df(), "prebreakout", 2), [])


class ComputeTrackRecordTests(unittest.TestCase):
    def test_excess_vs_benchmark_math(self):
        run_date = dt.date(2025, 1, 6)
        snapshot_df = pd.DataFrame(
            {"Ticker": ["AAA", "BBB"], "BreakoutScore": [9.0, 8.0]}
        )
        bars = {
            # 1-day forward returns from Jan 6: AAA +10%, BBB -5%, SPY +2%.
            "AAA": _bars("2025-01-06", [100.0, 110.0, 111.0]),
            "BBB": _bars("2025-01-06", [100.0, 95.0, 96.0]),
            "SPY": _bars("2025-01-06", [100.0, 102.0, 103.0]),
        }
        with mock.patch.object(
            tr, "_eligible_snapshots", return_value=[(run_date, snapshot_df)]
        ), mock.patch("data.price_alpaca.download_multi_alpaca", return_value=bars):
            results = tr.compute_track_record(horizon_days=1)

        self.assertIn("breakout", results)
        summary = results["breakout"]
        # Excess: (+10% - 2%) and (-5% - 2%) → avg +1.5%, 1 of 2 beat SPY.
        self.assertAlmostEqual(summary["avg_return"], (0.08 - 0.07) / 2)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["sample_size"], 2)
        self.assertEqual(summary["benchmark"], "SPY")
        # prebreakout ranking absent (model unavailable) rather than fabricated.
        self.assertNotIn("prebreakout", results)

    def test_returns_none_without_benchmark_bars(self):
        run_date = dt.date(2025, 1, 6)
        snapshot_df = pd.DataFrame({"Ticker": ["AAA"], "BreakoutScore": [9.0]})
        bars = {"AAA": _bars("2025-01-06", [100.0, 110.0])}
        with mock.patch.object(
            tr, "_eligible_snapshots", return_value=[(run_date, snapshot_df)]
        ), mock.patch("data.price_alpaca.download_multi_alpaca", return_value=bars):
            self.assertIsNone(tr.compute_track_record(horizon_days=1))

    def test_returns_none_without_snapshots(self):
        with mock.patch.object(tr, "_eligible_snapshots", return_value=[]):
            self.assertIsNone(tr.compute_track_record(horizon_days=1))


if __name__ == "__main__":
    unittest.main()
