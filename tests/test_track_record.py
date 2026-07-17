"""Unit tests for the signal track record compute (forward returns vs SPY)."""
import datetime as dt
import unittest
from unittest import mock

# Pandas-gated via stdlib skip (not pytest.importorskip) so the core smoke job,
# which runs unittest discovery without pytest installed, can still collect.
try:
    import pandas as pd
except Exception:  # pragma: no cover - core envs without pandas
    pd = None

from analytics import track_record as tr  # noqa: E402

requires_pandas = unittest.skipIf(pd is None, "pandas not installed")


def _bars(start: str, closes):
    """Daily bars DataFrame with a DatetimeIndex, matching price_alpaca output."""
    idx = pd.date_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


@requires_pandas
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

    def test_open_entry_captures_signal_day_move(self):
        # Open-entry enters at the signal-day OPEN; close-entry misses that move.
        idx = pd.date_range(start="2025-01-06", periods=4, freq="B")
        bars = pd.DataFrame(
            {"Open": [100.0, 108.0, 109.0, 110.0], "Close": [107.0, 108.0, 109.0, 110.0]},
            index=idx,
        )
        rd = dt.date(2025, 1, 6)
        close_r = tr._forward_return(bars, rd, horizon_days=1, entry_mode="close")
        open_r = tr._forward_return(bars, rd, horizon_days=1, entry_mode="open")
        self.assertAlmostEqual(close_r, (108.0 - 107.0) / 107.0)  # close→close
        self.assertAlmostEqual(open_r, (108.0 - 100.0) / 100.0)   # open→close
        self.assertGreater(open_r, close_r)

    def test_open_entry_falls_back_to_close_when_no_open_column(self):
        bars = _bars("2025-01-06", [100.0, 102.0, 103.0])
        r = tr._forward_return(bars, dt.date(2025, 1, 6), horizon_days=1, entry_mode="open")
        self.assertAlmostEqual(r, (102.0 - 100.0) / 100.0)  # uses Close, no crash

    def test_incomplete_forward_window_returns_none(self):
        bars = _bars("2025-01-06", [100.0, 102.0, 103.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 1, 7), horizon_days=5))

    def test_run_date_after_all_bars_returns_none(self):
        bars = _bars("2025-01-06", [100.0, 102.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 6, 1), horizon_days=1))

    def test_zero_entry_price_returns_none(self):
        bars = _bars("2025-01-06", [0.0, 102.0, 103.0])
        self.assertIsNone(tr._forward_return(bars, dt.date(2025, 1, 6), horizon_days=1))


@requires_pandas
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

    def test_prebreakout_ranking_empty_on_all_zero_probabilities(self):
        # score_prebreakout emits all-zero probs when no trained model loads;
        # zeros are not a ranking and must not fall back to stored order.
        def zero_score(df):
            out = df.copy()
            out["PreBreakoutProb%"] = 0.0
            return out

        with mock.patch("ml_prebreakout.score_prebreakout", side_effect=zero_score):
            self.assertEqual(tr._ranked_symbols(self._df(), "prebreakout", 2), [])


@requires_pandas
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
        ), mock.patch(
            "data.price_alpaca.download_multi_alpaca", return_value=bars
        ), mock.patch(
            # Deterministic across environments: with ML libs installed but no
            # trained model, the real score_prebreakout returns zero probs.
            "ml_prebreakout.score_prebreakout",
            side_effect=RuntimeError("no model"),
        ):
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
