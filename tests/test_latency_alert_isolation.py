"""Slow optional reads and rejected tickers must not poison shared requests."""
import datetime as dt
import sys
import unittest
from unittest import mock

import pandas as pd


class ScoreProjectionTests(unittest.TestCase):
    def setUp(self):
        from api import scans
        scans.score_cache.clear()
        self.scans = scans
        self.runs = [{'id': 1, 'created_at': dt.datetime(2026, 11, 11, 14, tzinfo=dt.timezone.utc), 'row_count': 2},
                     {'id': 2, 'created_at': dt.datetime(2026, 11, 11, 15, tzinfo=dt.timezone.utc), 'row_count': 2}]

    def test_cold_history_loads_payloads_in_one_batch_and_reuses_them(self):
        with mock.patch('db.scan_score_cache.load_scores', return_value={}), \
             mock.patch('db.scan_score_cache.save_scores') as save, \
             mock.patch('db.scan_score_cache.load_score_rows', return_value={1: [], 2: []}) as load:
            # Valid empty arrays are successful projections (no ranked tickers).
            load.return_value = {r['id']: {'stamp': f"{r['created_at']}|2", 'rows': [{'Ticker': 'AAA'}]} for r in self.runs}
            first = self.scans._score_window(self.runs)
            self.assertEqual(set(first), {1, 2})
            self.scans._score_window(self.runs)
        load.assert_called_once_with([1, 2])
        self.assertEqual(len(save.call_args.args[0]), 2)

    def test_durable_projection_survives_process_cache_clear_and_refresh_invalidates(self):
        stamp = f"{self.runs[0]['created_at']}|2"
        saved = {1: (stamp, {'AAA': {'score': 70, 'signals': []}})}
        with mock.patch('db.scan_score_cache.load_scores', return_value=saved), \
             mock.patch('db.scan_score_cache.save_scores'), \
             mock.patch('db.scan_score_cache.load_score_rows', return_value={1: {'stamp': f"{self.runs[0]['created_at']}|2", 'rows': [{'Ticker': 'AAA'}]}}) as load:
            self.scans._score_window(self.runs[:1])
            self.scans.score_cache.clear()
            self.scans._score_window(self.runs[:1])
            load.assert_not_called()
            changed = [{**self.runs[0], 'created_at': self.runs[1]['created_at']}]
            self.scans._score_window(changed)
            load.assert_called_once_with([1])

    def test_partial_read_preserves_good_runs_and_is_retried(self):
        with mock.patch('db.scan_score_cache.load_scores', return_value={}), \
             mock.patch('db.scan_score_cache.save_scores'), \
             mock.patch('db.scan_score_cache.load_score_rows', return_value={1: {'stamp': f"{self.runs[0]['created_at']}|2", 'rows': [{'Ticker': 'AAA'}]}}) as load:
            self.assertEqual(set(self.scans._score_window(self.runs)), {1})
            self.assertEqual(set(self.scans._score_window(self.runs)), {1})
            self.assertEqual(load.call_count, 2)


class AlertIsolationTests(unittest.TestCase):
    def test_bad_provider_symbol_does_not_hide_valid_ticker_prices(self):
        from billing_service.realtime_alerts import _latest_snapshots
        calls = []

        def get(url, **kwargs):
            syms = kwargs['params']['symbols'].split(',')
            calls.append(syms)
            if 'ZQ00' in syms:
                return mock.Mock(status_code=400, json=lambda: {'message': 'invalid symbol: ZQ00'})
            return mock.Mock(status_code=200, raise_for_status=lambda: None,
                             json=lambda: {s: {'latestTrade': {'p': 100}} for s in syms})
        with mock.patch.dict('os.environ', {'ALPACA_API_KEY_ID': 'k', 'ALPACA_API_SECRET_KEY': 's'}), \
             mock.patch.dict(sys.modules, {'httpx': mock.Mock(get=get)}):
            result = _latest_snapshots(['GPN', 'JNJ', 'ZQ00', 'MRNA', 'NVAX', 'BRK-B', 'bad ticker'])
        self.assertEqual(set(result), {'GPN', 'JNJ', 'MRNA', 'NVAX', 'BRK-B'})
        self.assertEqual(len(calls), 2)
        self.assertNotIn('ZQ00', calls[1])

    def test_unnamed_rejection_is_split_but_server_failure_is_not(self):
        from billing_service.realtime_alerts import _latest_snapshots

        def get(url, **kwargs):
            syms = kwargs['params']['symbols'].split(',')
            if 'BAD' in syms:
                return mock.Mock(status_code=422, json=lambda: {})
            return mock.Mock(status_code=200, raise_for_status=lambda: None,
                             json=lambda: {s: {'latestTrade': {'p': 1}} for s in syms})
        fetch = mock.Mock(side_effect=get)
        with mock.patch.dict('os.environ', {'ALPACA_API_KEY_ID': 'k', 'ALPACA_API_SECRET_KEY': 's'}), \
             mock.patch.dict(sys.modules, {'httpx': mock.Mock(get=fetch)}):
            self.assertEqual(set(_latest_snapshots(['AAA', 'BAD', 'BBB'])), {'AAA', 'BBB'})
            self.assertLessEqual(fetch.call_count, 24)
        fetch = mock.Mock(return_value=mock.Mock(status_code=429, raise_for_status=mock.Mock(side_effect=RuntimeError)))
        with mock.patch.dict('os.environ', {'ALPACA_API_KEY_ID': 'k', 'ALPACA_API_SECRET_KEY': 's'}), \
             mock.patch.dict(sys.modules, {'httpx': mock.Mock(get=fetch)}):
            self.assertEqual(_latest_snapshots(['AAA', 'BBB']), {})
            fetch.assert_called_once()


class SavedBriefPicksTests(unittest.TestCase):
    def test_saved_mode_never_imports_model_and_preserves_raw_tiebreak(self):
        from scheduler.morning_digest import _prebreakout_picks
        frame = pd.DataFrame({'Ticker': ['AAA', 'BBB', 'CCC'], 'PreBreakoutProb%': [13.1, 13.1, 0],
                              'PreBreakoutProbRaw': [.1, .2, 0], 'Last': [10, 20, 30]})
        original = frame.copy(deep=True)
        with mock.patch.dict(sys.modules, {'ml_prebreakout': None}):
            picks = _prebreakout_picks(frame, rescore=False)
        self.assertEqual([p['symbol'] for p in picks], ['BBB', 'AAA'])
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_saved_scores_does_not_trigger_model_loading(self):
        from scheduler.morning_digest import _prebreakout_picks
        with mock.patch.dict(sys.modules, {'ml_prebreakout': None}):
            self.assertEqual(_prebreakout_picks(pd.DataFrame({'Ticker': ['AAA']}), rescore=False), [])
