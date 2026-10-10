"""Selected cohort reads must not alter shared orphan/conflict evidence."""
import datetime as dt
import json
import sqlite3
import unittest
from unittest import mock

from analytics import forward_readiness, signal_evidence
from db import hsf_observations as store
from scripts import audit_research_cohorts as audit
from tests.test_cohort_audit import _obs


class CohortLoaderTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.addCleanup(self.conn.close)
        self.observations = [_obs('AAA', 'CANDIDATE'), _obs('BBB', 'CONTROL')]
        for row in self.observations:
            store.save_observation(row, conn=self.conn)
        self.selected_id = self.observations[0]['observation_id']
        forward_ts = (forward_readiness.epoch_start() + dt.timedelta(days=1)).isoformat()
        self.records = [
            (self.selected_id, '+5m', {'horizon': '+5m', 'data_status': 'MATURED', 'raw_return': .01,
                                     'evaluation_time': '2026-09-23T13:00:00+00:00'}),
            ('outside-window', '+5m', {'horizon': '+5m', 'raw_return': .02}),
            ('orphan', '+5m', {'horizon': '+5m', 'raw_return': .01,
                              'observation_timestamp': forward_ts}),
            # Different relational horizons but conflicting payload horizons:
            # diagnostics intentionally inspect record contents, not only keys.
            ('orphan', '+15m', {'horizon': '+5m', 'raw_return': .03,
                               'observation_timestamp': forward_ts}),
        ]
        self.conn.executemany('INSERT INTO hsf_observation_outcomes '
                              '(observation_id,horizon,schema_version,record) VALUES (?,?,?,?)',
                              [(oid, horizon, 'test', json.dumps(rec)) for oid, horizon, rec in self.records])
        self.conn.commit()

    def load(self, **kwargs):
        with mock.patch('db.engine.get_neon_conn', return_value=None), \
             mock.patch('db.engine.get_sqlite_conn', return_value=self.conn), \
             mock.patch.object(store, 'load_recent_observations', return_value=self.observations):
            return audit._load_live(**kwargs)

    def test_selected_report_equal_without_unmatched_downloads_or_history_loss(self):
        statements = []
        self.conn.set_trace_callback(statements.append)
        obs, all_outcomes = self.load()
        _, selected = self.load(selected_outcomes_only=True)
        self.assertEqual(dict(selected), {self.selected_id: all_outcomes[self.selected_id]})
        before, after = audit.audit_cohorts(obs, all_outcomes), audit.audit_cohorts(obs, selected)
        before.pop('generated_at')
        after.pop('generated_at')
        self.assertEqual(after, before)
        self.assertEqual(after['cohorts']['CANDIDATE']['point_in_time_violations'], 1)
        self.assertTrue(any('json_each(' in sql for sql in statements))
        self.assertEqual(self.conn.execute('SELECT count(*) FROM hsf_observation_outcomes').fetchone()[0], 4)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM hsf_observations').fetchone()[0], 2)

    def test_default_retains_orphan_and_conflicting_payload_evidence(self):
        obs, actual = self.load()
        expected = {}
        for oid, _, record in self.records:
            expected.setdefault(oid, []).append(record)
        self.assertEqual(dict(actual), expected)
        now = dt.datetime(2026, 10, 9, tzinfo=dt.timezone.utc)
        built = signal_evidence.build_records(obs, actual)
        report = signal_evidence.readiness(obs, actual, built, [], now=now)
        self.assertEqual(report, signal_evidence.readiness(obs, expected, built, [], now=now))
        self.assertEqual(report['orphan_outcomes'], 3)
        self.assertEqual(report['conflicting_outcomes'], 1)
        forward = forward_readiness.monitor(obs, actual, now=now)
        self.assertEqual(forward, forward_readiness.monitor(obs, expected, now=now))
        self.assertEqual(forward['data_quality']['orphan_forward_outcomes'], 2)

    def test_empty_selection_skips_connection_and_outcome_read(self):
        with mock.patch.object(store, 'load_recent_observations', return_value=[]), \
             mock.patch('db.engine.get_neon_conn', side_effect=AssertionError('no connection')):
            self.assertEqual(audit._load_live(selected_outcomes_only=True), ([], {}))

    def test_postgres_array_avoids_per_id_bind_parameters(self):
        conn = mock.Mock()
        conn.cursor.return_value.fetchall.return_value = []
        observations = [{'observation_id': str(i)} for i in range(70000)]
        with mock.patch.object(store, 'load_recent_observations', return_value=observations), \
             mock.patch('db.engine.get_neon_conn', return_value=conn):
            audit._load_live(selected_outcomes_only=True)
        query, params = conn.cursor.return_value.execute.call_args.args
        self.assertIn('ANY(%s)', query)
        self.assertEqual(len(params), 1)
        self.assertEqual(len(params[0]), 70000)

    def test_cli_opts_in_while_shared_loader_default_stays_full(self):
        import tempfile
        with tempfile.TemporaryDirectory() as out, \
             mock.patch('sys.argv', ['audit', '--out', out]), \
             mock.patch.object(audit, '_load_live', return_value=([], {})) as load, \
             mock.patch('builtins.print'):
            self.assertEqual(audit.main(), 0)
        load.assert_called_once_with(selected_outcomes_only=True)
