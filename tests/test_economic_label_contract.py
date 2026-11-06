"""Cached terminal returns must never substitute for a first-barrier path label."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ml_prebreakout as m


class EconomicLabelContract(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'Symbol': ['AAA'], 'Timestamp': [pd.Timestamp('2026-01-05', tz='UTC')],
                                   'Return_5D': [0.06]})
        self.bars = pd.DataFrame({'Close': [100, 99, 101, 102, 104, 106],
                                  'High': [101, 100, 102, 103, 105, 107],
                                  'Low': [99, 97, 100, 101, 103, 105]},
                                 index=pd.bdate_range('2026-01-05', periods=6, tz='UTC'))

    def test_cached_positive_cannot_override_earlier_stop(self):
        with patch.object(m, '_download_label_bars', return_value={'AAA': self.bars}):
            cached = m.add_forward_return_labels(self.frame)
            uncached = m.add_forward_return_labels(self.frame.drop(columns='Return_5D'))
        self.assertEqual(list(cached[m.TARGET_COLUMN]), [0])
        self.assertEqual(list(uncached[m.TARGET_COLUMN]), [0])
        terminal = m.add_forward_return_labels_variant(self.frame, target_kind='terminal_return')
        self.assertEqual(list(terminal[m.TARGET_COLUMN]), [1])
        self.assertNotEqual(cached.iloc[0][m.TARGET_COLUMN+'_definition'], terminal.iloc[0][m.TARGET_COLUMN+'_definition'])

    def test_missing_or_nonfinite_path_is_not_a_terminal_fallback(self):
        for bars in (self.bars.drop(columns='High'), self.bars.iloc[:3], self.bars.assign(Low=np.nan), self.bars.assign(Close=[100, np.nan, 101, 102, 104, 106])):
            with self.subTest(columns=list(bars), rows=len(bars)), patch.object(m, '_download_label_bars', return_value={'AAA': bars}):
                self.assertTrue(m.add_forward_return_labels(self.frame).empty)

    def test_same_bar_stop_wins_and_extrema_cover_full_horizon(self):
        bars = self.bars.copy()
        bars.iloc[1, bars.columns.get_loc('High')] = 106
        stats = m._forward_path_stats(bars, self.frame.Timestamp[0].date(), 5)
        self.assertFalse(stats['hit'])
        self.assertAlmostEqual(stats['mfe'], .07)
        self.assertAlmostEqual(stats['mae'], -.03)

    def test_another_tickers_breakout_cannot_label_this_ticker(self):
        frame = pd.DataFrame({'Symbol': ['AAA', 'AAA', 'BBB', 'BBB'],
                              'Timestamp': pd.date_range('2026-01-01', periods=4, tz='UTC'),
                              'IsBreakout': [0, 0, 1, 0]})
        out = m.add_future_breakout_label(frame, horizon_scans=3)
        self.assertEqual(list(out.loc[out.Symbol == 'AAA', 'FutureBreakout']), [0, 0])
        frame.loc[1, 'IsBreakout'] = 1
        self.assertEqual(m.add_future_breakout_label(frame).iloc[0].FutureBreakout, 1)
