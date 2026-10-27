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
class EligibleSnapshotsTests(unittest.TestCase):
    """Dedup same-day snapshots and fence out pre-epoch runs."""

    def _run(self, runs):
        # _eligible_snapshots does local `from db.runs import ...` and
        # `from ui.app_runtime import ...`. Inject fakes via sys.modules so the
        # test doesn't depend on those (heavy) modules being importable in CI.
        import sys
        import types

        payloads = {r["id"]: [{"Ticker": "AAA", "BreakoutScore": 1.0}] for r in runs}

        fake_runs = types.ModuleType("db.runs")
        fake_runs.list_snapshot_runs = lambda **kw: runs
        fake_runs.load_many_run_results = lambda ids: payloads
        fake_ar = types.ModuleType("ui.app_runtime")
        fake_ar.normalize_results_to_df = lambda raw: pd.DataFrame(raw) if raw else None

        with mock.patch.dict(sys.modules, {"db.runs": fake_runs, "ui.app_runtime": fake_ar}):
            return tr._eligible_snapshots(horizon_days=5, lookback_days=45, max_snapshots=30)

    def test_same_day_snapshots_deduped_and_preepoch_fenced(self):
        # Build now-relative dates inside the 5D "complete window" + 45d lookback.
        # Use two WEEKDAYS so the weekend-snapshot filter doesn't drop them.
        now = dt.datetime.now(dt.timezone.utc)
        weekdays = [
            (now - dt.timedelta(days=delta)).date()
            for delta in range(12, 30)
            if (now - dt.timedelta(days=delta)).date().weekday() < 5
        ]
        nxt, base = weekdays[0], weekdays[3]  # nxt more recent than base; both weekdays
        pre = base - dt.timedelta(days=1)

        def at(d, h):
            return dt.datetime(d.year, d.month, d.day, h, tzinfo=dt.timezone.utc)

        runs = [
            # three snapshots on the SAME day → collapse to one
            {"id": 1, "created_at": at(base, 20)},
            {"id": 2, "created_at": at(base, 13)},
            {"id": 3, "created_at": at(base, 6)},
            {"id": 4, "created_at": at(nxt, 12)},   # distinct day → kept
            {"id": 5, "created_at": at(pre, 12)},   # pre-epoch → fenced
        ]
        with mock.patch.object(tr, "SCORE_EPOCH", base):
            out = self._run(runs)
        dates = sorted(d for d, _df in out)
        self.assertEqual(dates, [base, nxt])          # one per day, pre fenced
        self.assertTrue(all(d >= base for d in dates))

    def test_weekend_snapshots_are_skipped(self):
        # A weekend (forced) snapshot enters on the next session and duplicates
        # that day — it must be dropped so it can't double-count or paint a
        # spurious weekend heatmap cell.
        now = dt.datetime.now(dt.timezone.utc)
        fri = sat = None
        for delta in range(11, 40):  # inside the eligible window (~10–45d ago)
            d = (now - dt.timedelta(days=delta)).date()
            if d.weekday() == 4 and fri is None:
                fri = d
            if d.weekday() == 5 and sat is None:
                sat = d
        self.assertIsNotNone(fri)
        self.assertIsNotNone(sat)

        def at(d, h):
            return dt.datetime(d.year, d.month, d.day, h, tzinfo=dt.timezone.utc)

        runs = [
            {"id": 1, "created_at": at(fri, 12)},   # weekday → kept
            {"id": 2, "created_at": at(sat, 12)},   # weekend → skipped
        ]
        with mock.patch.object(tr, "SCORE_EPOCH", min(fri, sat) - dt.timedelta(days=1)):
            out = self._run(runs)
        dates = [d for d, _df in out]
        self.assertIn(fri, dates)
        self.assertNotIn(sat, dates)


class LoadDailyExcessTests(unittest.TestCase):
    """Read-time weekend filter cleans pre-fix rows without a data migration."""

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return self._rows

        def close(self):
            pass

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self, *a, **k):
            return LoadDailyExcessTests._Cur(self._rows)

        def close(self):
            pass

    def test_weekend_rows_are_dropped_at_read(self):
        from db import track_record as dbtr

        rows = [
            (dt.date(2026, 6, 26), -0.02),   # Friday   → kept
            (dt.date(2026, 6, 27), 0.05),    # Saturday → dropped
            (dt.date(2026, 6, 28), -0.10),   # Sunday   → dropped
            (dt.date(2026, 6, 29), 0.01),    # Monday   → kept
        ]
        with mock.patch.object(dbtr, "get_neon_conn", return_value=self._Conn(rows)):
            out = dbtr.load_daily_excess("prebreakout", 5, 120)
        days = [d for d, _v in out]
        self.assertEqual(days, [dt.date(2026, 6, 26), dt.date(2026, 6, 29)])


class TrackRecordMethodTests(unittest.TestCase):
    def test_new_results_are_stored_separately_from_legacy_rescoring(self):
        from db import track_record as dbtr

        conn = mock.MagicMock()
        with mock.patch.object(dbtr, "get_neon_conn", return_value=conn), \
                mock.patch.object(dbtr, "_ensure_schema"):
            dbtr.save_track_record(horizon_days=5, avg_return=0.01, median_return=0.01,
                                  win_rate=0.6, sample_size=30, runs_used=6, ranking="prebreakout")
            self.assertEqual(conn.cursor.return_value.execute.call_args.args[1][-1], dbtr.PREBREAKOUT_SNAPSHOT)
            dbtr.save_daily_excess("prebreakout", 5, [(dt.date(2026, 10, 1), 0.01, 5)])
            self.assertEqual(conn.cursor.return_value.execute.call_args.args[1][1], dbtr.PREBREAKOUT_SNAPSHOT)

    def test_reads_select_only_snapshot_method_and_keep_public_ranking_name(self):
        from db import track_record as dbtr

        conn = mock.MagicMock()
        cur = conn.cursor.return_value
        cur.fetchone.return_value = {"horizon_days": 5, "ranking": dbtr.PREBREAKOUT_SNAPSHOT}
        cur.fetchall.return_value = [cur.fetchone.return_value]
        with mock.patch.object(dbtr, "get_neon_conn", return_value=conn), \
                mock.patch.object(dbtr, "_ensure_schema"):
            single = dbtr.load_latest_track_record(5, "prebreakout")
            self.assertEqual(single["ranking"], "prebreakout")
            self.assertEqual(cur.execute.call_args.args[1][1], dbtr.PREBREAKOUT_SNAPSHOT)
            rows = dbtr.load_latest_track_records()
            self.assertEqual(rows[0]["ranking"], "prebreakout")
            self.assertIn("ranking != 'prebreakout'", cur.execute.call_args.args[0])
            cur.fetchall.return_value = []
            dbtr.load_daily_excess("prebreakout", 5, 120)
            self.assertEqual(cur.execute.call_args.args[1][0], dbtr.PREBREAKOUT_SNAPSHOT)


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

    def test_prebreakout_ranking_uses_saved_probability_without_loading_model(self):
        saved = self._df().assign(**{"PreBreakoutProb%": [90.0, 10.0, 50.0]})
        before = saved.copy(deep=True)
        with mock.patch("ml_prebreakout.score_prebreakout", side_effect=AssertionError("must not rescore")) as scorer:
            self.assertEqual(
                tr._ranked_symbols(saved, "prebreakout", 2), ["LOW", "MID"]
            )
            scorer.assert_not_called()
        pd.testing.assert_frame_equal(saved, before)

    def test_prebreakout_ranking_empty_without_saved_predictions(self):
        self.assertEqual(tr._ranked_symbols(self._df(), "prebreakout", 2), [])

    def test_prebreakout_ranking_empty_on_all_zero_probabilities(self):
        saved = self._df().assign(**{"PreBreakoutProb%": [0.0, 0.0, 0.0]})
        self.assertEqual(tr._ranked_symbols(saved, "prebreakout", 2), [])

    def test_saved_provenance_wins_over_display_and_rejects_unavailable(self):
        import json

        saved = self._df().assign(**{"PreBreakoutProb%": [95.0, 90.0, 80.0]})
        saved["ModelProvenance"] = [json.dumps({"prebreakout": v}) for v in [
            {"status": "captured", "calibrated_probability": 0.2},
            {"status": "unavailable"},
            {"status": "captured", "calibrated_probability": 0.9},
        ]]
        self.assertEqual(tr._ranked_symbols(saved, "prebreakout", 3), ["MID", "LOW"])

    def test_invalid_saved_probabilities_are_excluded(self):
        for values in ([float("nan"), float("inf"), -1], [101, "bad", None]):
            saved = self._df().assign(**{"PreBreakoutProb%": values})
            self.assertEqual(tr._ranked_symbols(saved, "prebreakout", 3), [])

    def test_saved_fractional_probabilities_are_supported(self):
        saved = self._df().assign(PreBreakoutProb=[0.9, 0.1, 0.5])
        self.assertEqual(tr._ranked_symbols(saved, "prebreakout", 2), ["LOW", "MID"])


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
            # No current model may fill in this snapshot's missing predictions.
            "ml_prebreakout.score_prebreakout",
            side_effect=AssertionError("must not reconstruct historical predictions"),
        ):
            results = tr.compute_track_record(horizon_days=1)

        self.assertIn("breakout", results)
        summary = results["breakout"]
        # Excess: (+10% - 2%) and (-5% - 2%) → avg +1.5%, 1 of 2 beat SPY.
        self.assertAlmostEqual(summary["avg_return"], (0.08 - 0.07) / 2)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["sample_size"], 2)
        self.assertEqual(summary["benchmark"], "SPY")
        # Prebreakout ranking absent (no saved predictions) rather than fabricated.
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
