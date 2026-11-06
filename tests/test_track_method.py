"""Timing, coverage and return semantics of the versioned ranking study."""
import datetime as dt
import unittest
from unittest.mock import MagicMock, patch

from analytics import track_method as m

try:
    import pandas as pd
except ImportError:
    pd = None

UTC = dt.timezone.utc


def at(value):
    return dt.datetime.fromisoformat(value).replace(tzinfo=UTC)


class WindowTests(unittest.TestCase):
    def test_post_close_moves_to_next_session_and_records_provenance(self):
        w = m.window(at('2026-11-05T22:00:00'), 1)
        self.assertEqual(w['entry_at'], '2026-11-06T14:30:00+00:00')
        self.assertEqual(w['exit_at'], '2026-11-06T21:00:00+00:00')
        self.assertGreater(dt.datetime.fromisoformat(w['entry_at']), at(w['signal_at']))

    def test_premarket_and_exact_open(self):
        self.assertEqual(m.window(at('2026-11-05T13:00:00'), 1)['entry_day'], dt.date(2026, 11, 5))
        self.assertEqual(m.window(at('2026-11-05T14:30:00'), 1)['entry_day'], dt.date(2026, 11, 6))

    def test_weekend_dst_holiday_and_early_close(self):
        self.assertEqual(m.window(at('2026-10-30T21:00:00'), 1)['entry_at'], '2026-11-02T14:30:00+00:00')
        self.assertEqual(m.window(at('2026-07-02T21:00:00'), 1)['entry_day'], dt.date(2026, 7, 6))
        self.assertEqual(m.window(at('2026-11-26T12:00:00'), 1)['exit_at'], '2026-11-27T18:00:00+00:00')
        # Same instant expressed in another zone must choose the same session.
        signal = dt.datetime.fromisoformat('2026-11-05T05:00:00-08:00')
        self.assertEqual(m.window(signal, 1), m.window(at('2026-11-05T13:00:00'), 1))

    def test_unknown_timestamp_or_calendar_fails_closed(self):
        self.assertEqual(m.window(dt.date(2026, 11, 5), 1)['reason'], 'SIGNAL_TIMESTAMP_UNAVAILABLE')
        self.assertEqual(m.window(at('2029-01-02T13:00:00'), 1)['reason'], 'CALENDAR_UNSUPPORTED')


@unittest.skipIf(pd is None, 'needs pandas')
class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.signal = at('2026-10-05T12:00:00')
        self.now = at('2026-11-06T23:00:00')

    def bars(self, close=100, open_=100):
        return pd.DataFrame({'Open': [open_], 'Close': [close]}, index=pd.to_datetime(['2026-10-05']))

    def evaluate(self, bars, selections=None):
        return m.evaluate(selections or [(self.signal, ['AAA'])], bars, 1, now=self.now)

    def test_negative_stock_can_beat_benchmark(self):
        s = self.evaluate({'AAA': self.bars(98), 'SPY': self.bars(95)})
        self.assertAlmostEqual(s['avg_absolute_return'], -.02)
        self.assertAlmostEqual(s['avg_excess_return'], .03)
        self.assertEqual((s['positive_return_rate'], s['benchmark_outperformance_rate']), (0, 1))
        self.assertEqual(s['avg_return'], s['avg_excess_return'])
        self.assertEqual(s['observations'][0]['entry_policy'], 'first_session_open_strictly_after_snapshot')

    def test_positive_stock_can_underperform_and_ties_are_not_wins(self):
        s = self.evaluate({'AAA': self.bars(103), 'SPY': self.bars(105)})
        self.assertEqual((s['positive_return_rate'], s['benchmark_outperformance_rate']), (1, 0))
        self.assertAlmostEqual(s['avg_excess_return'], -.02)
        s = self.evaluate({'AAA': self.bars(), 'SPY': self.bars()})
        self.assertEqual((s['positive_return_rate'], s['benchmark_outperformance_rate']), (0, 0))

    def test_missing_bars_exclusions_reconcile(self):
        s = self.evaluate({'AAA': self.bars(103), 'SPY': self.bars()}, [(self.signal, ['AAA', 'BBB'])])
        self.assertEqual((s['selected_count'], s['evaluated_count'], s['excluded_count']), (2, 1, 1))
        self.assertEqual(s['exclusions'], {'MISSING_SYMBOL_BARS': 1})
        self.assertEqual(s['coverage_pct'], 50)
        s = self.evaluate({'AAA': self.bars()})
        self.assertEqual(s['exclusions'], {'MISSING_BENCHMARK_BARS': 1})
        self.assertIsNone(s['avg_absolute_return'])

    def test_open_missing_nan_zero_and_invalid_never_becomes_close(self):
        for value in (None, float('nan'), 0, -1, float('inf'), 'bad'):
            s = self.evaluate({'AAA': self.bars(103, value), 'SPY': self.bars()})
            self.assertEqual(s['exclusions'], {'INVALID_ENTRY_PRICE': 1})
        s = self.evaluate({'AAA': self.bars().drop(columns='Open'), 'SPY': self.bars()})
        self.assertEqual(s['exclusions'], {'ENTRY_FIELD_UNAVAILABLE': 1})

    def test_missing_benchmark_open_and_invalid_exit_are_explicit(self):
        s = self.evaluate({'AAA': self.bars(), 'SPY': self.bars().drop(columns='Open')})
        self.assertEqual(s['exclusions'], {'BENCHMARK_ENTRY_FIELD_UNAVAILABLE': 1})
        s = self.evaluate({'AAA': self.bars(float('nan')), 'SPY': self.bars()})
        self.assertEqual(s['exclusions'], {'INVALID_EXIT_PRICE': 1})

    def test_incomplete_dates_cannot_shift_stock_vs_benchmark(self):
        spec = m.window(self.signal, 2)
        self.assertEqual(m.outcome(self.bars(), spec, now=self.now)['reason'], 'INCOMPLETE_FORWARD_WINDOW')
        self.assertEqual(m.outcome(self.bars(), m.window(self.now, 5), now=self.now)['reason'], 'INCOMPLETE_FORWARD_WINDOW')

    def test_repeated_ticker_is_two_observations_one_unique_ticker(self):
        s = self.evaluate({'AAA': self.bars(103), 'SPY': self.bars()}, [(self.signal, ['AAA']), (self.signal, ['AAA'])])
        self.assertEqual((s['observation_count'], s['unique_ticker_count'], s['signal_date_count']), (2, 1, 1))

    def test_old_method_rows_cannot_masquerade_as_new_and_reads_do_not_migrate(self):
        from db import track_record as store
        self.assertIsNone(store._public_summary({'ranking': 'breakout', 'avg_return': .5}))
        self.assertIsNone(store._public_summary({'ranking': m.METHOD+':breakout', 'details': {'methodology': 'old'}}))
        conn = MagicMock()
        conn.cursor.return_value.fetchall.return_value = []
        with patch.object(store, 'get_neon_conn', return_value=conn), patch.object(store, '_ensure_schema') as ddl:
            self.assertEqual(store.load_latest_track_records(), [])
        ddl.assert_not_called()
        self.assertEqual(conn.cursor.return_value.execute.call_args.args[1], (m.METHOD+':breakout', m.METHOD+':prebreakout'))

    def test_post_close_price_before_signal_cannot_enter_the_return(self):
        signal = at('2026-10-05T22:00:00')
        bars = pd.DataFrame({'Open': [100, 200], 'Close': [110, 190]}, index=pd.to_datetime(['2026-10-05', '2026-10-06']))
        result = m.evaluate([(signal, ['AAA'])], {'AAA': bars, 'SPY': bars.assign(Close=[100, 200])}, 1, now=self.now)
        self.assertAlmostEqual(result['avg_absolute_return'], -.05)
        self.assertEqual(result['observations'][0]['entry_at'], '2026-10-06T13:30:00+00:00')

    def test_adjacent_signal_dates_have_overlapping_five_session_windows(self):
        bars = pd.DataFrame({'Open': [100]*7, 'Close': [101]*7}, index=pd.bdate_range('2026-10-05', periods=7))
        signals = [(self.signal, ['AAA']), (at('2026-10-06T12:00:00'), ['AAA'])]
        result = m.evaluate(signals, {'AAA': bars, 'SPY': bars.assign(Close=100)}, 5, now=self.now)
        self.assertEqual((result['observation_count'], result['unique_ticker_count'], result['signal_date_count']), (2, 1, 2))
        self.assertGreater(result['observations'][0]['exit_at'], result['observations'][1]['entry_at'])

    def test_api_filters_old_method_and_versions_cache_keys(self):
        from api import history
        old = {'horizon_days': 5, 'ranking': 'breakout', 'sample_size': 100, 'avg_return': .9}
        with patch('db.track_record.load_latest_track_records', return_value=[old]):
            result = history._track_record()
        self.assertEqual(result['summaries'], [])
        self.assertEqual(result['methodology'], m.METHOD)
        with patch('api.today._cached', return_value={'summaries': [1]}) as cache:
            history.track_record()
        self.assertEqual(cache.call_args.args[0], ('track_record', m.METHOD))

    def test_source_query_is_restricted_to_public_scheduler_snapshots(self):
        from db import runs
        conn = MagicMock()
        conn.cursor.return_value.fetchall.return_value = []
        with patch.object(runs, 'get_neon_conn', return_value=conn), patch.object(runs, 'ensure_neon_runs_schema'):
            self.assertEqual(runs.list_snapshot_runs(public_only=True), [])
        query, params = conn.cursor.return_value.execute.call_args.args
        self.assertIn("username = 'cron' AND label = 'daily_snapshot'", query)
        self.assertTrue(params[1])
        self.assertIn('ORDER BY created_at DESC, id DESC', query)


    def test_invalid_bar_index_is_an_exclusion_not_a_crash(self):
        bars = self.bars()
        bars.index = pd.DatetimeIndex([pd.NaT])
        result = self.evaluate({'AAA': bars, 'SPY': self.bars()})
        self.assertEqual(result['exclusions'], {'INVALID_BAR_INDEX': 1})
