"""Shared snapshot inputs agree across training subsets and live batches."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ml_prebreakout as ml


def history(symbol='AAA', offset=0):
    x = np.arange(35, dtype=float)
    return pd.DataFrame({
        'Symbol': symbol, 'Timestamp': pd.date_range('2026-08-03', periods=35, freq='B', tz='UTC'),
        'Last': 100+x+offset, 'Close': 100+x+offset, 'Open': 99+x+offset,
        'High': 102+x+offset, 'Low': 98+x+offset, 'Volume': 1e6+x*1000,
        'BreakoutScore': x/10, 'VolRel20': 1+x/100, 'GapPct': x/100,
        'Trend10D%': x/20, 'Trend20D%': x/30, 'BreakoutPos20D': x/50,
    })


class InputHistoryTests(unittest.TestCase):
    def test_live_last_row_matches_training_full_history(self):
        frame = history()
        expected = ml.snapshot_feature_frame(frame).iloc[-1]
        current = frame.tail(1).set_axis([91])
        actual = ml.snapshot_feature_frame(current, history=frame.iloc[:-1]).iloc[0]
        for col in ['BreakoutScoreDelta1D', 'VolRel20Delta3D', 'HigherLowCount10D',
                    'LowSlope10D', 'RangeCompression5D']:
            self.assertTrue(np.isfinite(actual[col]), col)
            self.assertAlmostEqual(actual[col], expected[col], msg=col)
        self.assertEqual(ml.snapshot_feature_frame(current, history=frame.iloc[:-1]).index.tolist(), [91])
        # Compare the complete generated row, including missingness, rather
        # than only the five formerly defaulted examples above.
        pd.testing.assert_series_equal(actual.reindex(expected.index), expected, check_names=False)


    def test_other_ticker_and_future_rows_cannot_change_inputs(self):
        frame = history()
        target = frame.iloc[[30]]
        expected = ml.snapshot_feature_frame(target, history=frame.iloc[:30])
        polluted = pd.concat([frame, history('BBB', 9000)], ignore_index=True)
        actual = ml.snapshot_feature_frame(target, history=polluted)
        for col in ['BreakoutScoreDelta1D', 'LowSlope10D', 'HigherLowCount10D']:
            self.assertAlmostEqual(actual[col].iloc[0], expected[col].iloc[0])

    def test_training_label_filter_does_not_recompute_history(self):
        prepared = ml.snapshot_feature_frame(history())
        prepared[ml.PREBREAKOUT_TARGET_COLUMN] = 0
        subset = prepared.iloc[[10, 20, 34]].copy()
        x, _ = ml.build_ml_dataset(subset, feature_cols=['BreakoutScoreDelta1D'])
        self.assertTrue(np.allclose(x['BreakoutScoreDelta1D'], 0.1))

    def test_duplicate_ticker_timestamp_is_rejected(self):
        frame = history().tail(1)
        with self.assertRaises(ValueError):
            ml.snapshot_feature_frame(pd.concat([frame, frame], ignore_index=True))

    def test_live_loads_history_once_and_enriches_in_one_batch(self):
        frame = history()
        with patch('db.prebreakout_history.load_feature_history', return_value=frame.iloc[:-1]) as load, \
             patch.object(ml, 'add_historical_ohlcv_context', side_effect=lambda f, **_: f) as enrich, \
             patch.object(ml, '_live_benchmark_context', return_value={}):
            actual = ml._live_feature_frame(frame.tail(1))
        load.assert_called_once()
        enrich.assert_called_once()
        self.assertEqual(len(enrich.call_args.args[0]), 35)
        self.assertAlmostEqual(actual['BreakoutScoreDelta1D'].iloc[0], 0.1)
        self.assertEqual(actual.attrs['provenance_availability']['scanner_history'], 'available')

    def test_history_failure_is_recorded_and_missing_values_remain_missing(self):
        with patch('db.prebreakout_history.load_feature_history', side_effect=RuntimeError('offline')), \
             patch.object(ml, 'add_historical_ohlcv_context', side_effect=lambda f, **_: f), \
             patch.object(ml, '_live_benchmark_context', return_value={}):
            actual = ml._live_feature_frame(history().tail(1))
        self.assertTrue(pd.isna(actual['BreakoutScoreDelta1D'].iloc[0]))
        self.assertEqual(actual.attrs['provenance_availability']['scanner_history'], 'unavailable')


class HistoryQueryTests(unittest.TestCase):
    def test_history_query_is_batched_public_bounded_and_read_only(self):
        from unittest.mock import MagicMock
        from db.prebreakout_history import load_feature_history
        conn = MagicMock()
        cur = conn.cursor.return_value
        cur.fetchall.return_value = [('AAA', pd.Timestamp('2026-08-01', tz='UTC'), {'Last': 10})]
        before = pd.Timestamp('2026-09-01', tz='UTC').to_pydatetime()
        with patch('db.prebreakout_history.get_neon_conn', return_value=conn):
            out = load_feature_history(['AAA', 'BBB', 'AAA'], before=before)
        cur.execute.assert_called_once()
        sql, params = cur.execute.call_args.args
        self.assertIn("username='cron'", sql)
        self.assertIn("label='daily_snapshot'", sql)
        self.assertIn('created_at <', sql)
        self.assertIn('LIMIT 128', sql)
        self.assertIn('rn <= 64', sql)
        self.assertEqual(params[-1], ['AAA', 'BBB'])
        self.assertEqual(params[0], before)
        self.assertNotIn('ModelProvenance', params[-2])
        self.assertEqual(out['Symbol'].tolist(), ['AAA'])
        conn.commit.assert_not_called()
        cur.close.assert_called_once()
        conn.close.assert_called_once()

    def test_training_history_filters_private_and_noncanonical_runs(self):
        now = ml._utc_now()
        runs = [
            {'id': 1, 'created_at': now, 'username': 'cron', 'label': 'daily_snapshot', 'is_snapshot': True},
            {'id': 2, 'created_at': now, 'username': 'person', 'label': 'daily_snapshot', 'is_snapshot': True},
            {'id': 3, 'created_at': now, 'username': 'cron', 'label': 'custom', 'is_snapshot': False},
        ]
        with patch.object(ml, 'list_runs', return_value=runs), \
             patch.object(ml, 'load_run_results', return_value=[{'Ticker': 'AAA', 'Last': 10}]) as load:
            result = ml.load_run_history(canonical_only=True)
        self.assertEqual(len(result), 1)
        load.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()
