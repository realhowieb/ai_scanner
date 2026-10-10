"""Calendar boundaries, safe operational snapshots, authorization and bounded IO."""
import datetime as dt
import importlib.util
import json
import sqlite3
import types
import unittest
from unittest import mock

from analytics.data_freshness import describe
from analytics.operations import summary

NOW = dt.datetime(2026, 10, 9, 21, tzinfo=dt.timezone.utc)


def health():
    return {'generated_at': NOW.isoformat(), 'system_status': 'HEALTHY',
            'credentials': 'private-secret', 'subsystems': {
                'workflows': {'metrics': {'workflows': [{'workflow': 'scheduled-scans.yml',
                    'status': 'HEALTHY', 'last_success': NOW.isoformat(), 'consecutive_failures': 0,
                    'username': 'private-user', 'sql': 'private-sql'}]}},
                'maturation': {'metrics': {'last_success': NOW.isoformat(), 'ready_observations': 12,
                    'oldest_pending_age_min': 35, 'raw_outcomes': 'private-record'}}},
            'database_traffic': [{'workflow': 'scheduled-scans.yml', 'observed_at': NOW.isoformat(),
                'metrics': {'measurement': 'application_payload_estimate', 'calls': 2,
                            'db_to_client_payload_bytes': 100, 'client_to_db_payload_bytes': 20,
                            'sql': 'private-sql'}}]}


class FreshnessTests(unittest.TestCase):
    def test_weekend_and_holiday_preserve_last_session_freshness(self):
        for scan, now in [('2026-10-09T20:25:00Z', '2026-10-11T19:00:00Z'),
                          ('2026-01-16T20:25:00Z', '2026-01-19T19:00:00Z')]:
            out = describe(scan, market_data_at=scan, now=now)
            self.assertEqual(out['state'], 'fresh')
            self.assertIsNone(out['scan_completed_at'])

    def test_missed_slot_stale_even_when_market_closed(self):
        out = describe('2026-10-08T20:25:00Z', now='2026-10-10T17:00:00Z')
        self.assertEqual(out['state'], 'stale')

    def test_early_close_does_not_require_regular_scan_after_close(self):
        out = describe('2026-11-27T17:25:00Z', market_data_at='2026-11-27T17:25:00Z',
                       now='2026-11-27T21:00:00Z')
        self.assertEqual(out['state'], 'fresh')
        self.assertEqual(out['expected_scan_at'], '2026-11-27T16:35:00+00:00')

    def test_unknown_market_data_and_old_inputs_are_distinct_from_new_scan(self):
        current = '2026-10-09T20:25:00Z'
        out = describe(current, now=NOW)
        self.assertEqual((out['state'], out['scan_state'], out['market_data_state']), ('partial', 'fresh', 'unavailable'))
        self.assertIsNone(out['market_data_at'])
        out = describe(current, market_data_at='2026-10-08T19:40:00Z', now=NOW)
        self.assertEqual(out['state'], 'stale')
        self.assertEqual(out['scan_state'], 'fresh')

    def test_timezone_invalid_future_and_missing_values(self):
        naive = describe('2026-10-09T20:25:00', now=NOW)
        aware = describe('2026-10-09T16:25:00-04:00', now=NOW)
        self.assertEqual(naive, aware)
        for stamp in (None, 'bad', '2026-10-10T21:00:00Z'):
            self.assertEqual(describe(stamp, now=NOW)['state'], 'unavailable')
        self.assertEqual(describe('2030-01-02T20:25:00Z', now='2030-01-02T21:00:00Z')['state'], 'partial')


class OperationsTests(unittest.TestCase):
    def setUp(self):
        from api.today import clear_cache
        clear_cache()
        self.addCleanup(clear_cache)

    def test_allowlist_omits_all_sensitive_or_raw_data(self):
        report = summary(health(), now=NOW)
        encoded = json.dumps(report, allow_nan=False)
        for private in ('private-secret', 'private-user', 'private-sql', 'private-record'):
            self.assertNotIn(private, encoded)
        self.assertEqual(report['maturation']['pending_observations'], 12)
        self.assertFalse(report['neon_usage']['available'])
        self.assertIsNone(report['neon_usage']['public_network_transfer_bytes'])

    def test_absent_metrics_are_unknown_not_zero(self):
        report = summary(None, now=NOW)
        self.assertFalse(report['available'])
        self.assertTrue(report['stale'])
        self.assertIsNone(report['maturation']['pending_observations'])
        self.assertFalse(report['database_traffic']['available'])

    def test_cache_reuses_one_bounded_snapshot_and_retains_on_failure(self):
        from api.operations import get_summary
        from db import system_health
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        system_health.save_snapshot(health(), conn=conn)
        clock = [1000.]
        def load():
            return system_health.load_latest(conn=conn)
        statements = []
        conn.set_trace_callback(statements.append)
        with mock.patch('db.system_health.load_latest', side_effect=[load(), None, None]) as fetch, \
             mock.patch('api.operations.time.monotonic', side_effect=lambda: clock[0]):
            first = get_summary(NOW)
            self.assertEqual(first, get_summary(NOW))
            self.assertEqual(fetch.call_count, 1)
            clock[0] += 121
            failed = get_summary(NOW)
            self.assertTrue(failed['refresh_failed'])
            self.assertTrue(failed['stale'])
            self.assertEqual(failed['workflows'], first['workflows'])
            get_summary(NOW)
            self.assertEqual(fetch.call_count, 2)
            clock[0] += 3601
            self.assertFalse(get_summary(NOW)['available'])
        reads = [q for q in statements if q.startswith('SELECT')]
        self.assertEqual(len(reads), 1)
        self.assertIn('LIMIT 1', reads[0])

    def test_bounded_collector_only_downloads_latest_known_artifacts(self):
        from scripts.system_health import latest_database_traffic
        names = ('scheduled-scans.yml', 'mature-observations.yml', 'forward-evidence-readiness.yml', 'system-health.yml')
        workflows = {name: [{'id': i, 'created_at': NOW.isoformat(), 'conclusion': 'success'}] for i, name in enumerate(names, 1)}
        workflows['private-job'] = workflows[names[0]] * 100
        with mock.patch('scripts.system_health._gh', return_value=(mock.Mock(), 'repo')), \
             mock.patch('scripts.system_health._run_artifact_json', return_value=health()['database_traffic'][0]['metrics']) as fetch:
            result = latest_database_traffic(workflows)
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(len(result), 4)
        self.assertNotIn('private-sql', json.dumps(result))

    @unittest.skipUnless(importlib.util.find_spec('streamlit'), 'needs Streamlit UI dependency')
    def test_dashboard_blocks_non_admin_before_loading(self):
        from ui.system_health_view import render_system_health
        fake = types.SimpleNamespace(session_state={'is_admin': False}, info=mock.Mock())
        with mock.patch('ui.system_health_view.st', fake), \
             mock.patch('api.operations.load_snapshot', side_effect=AssertionError('must not load')):
            render_system_health()
        fake.info.assert_called_once()


try:
    from tests.test_api_v1 import DEPS, ApiTestCase
except ImportError:
    DEPS, ApiTestCase = False, unittest.TestCase


@unittest.skipUnless(DEPS, 'needs API dependencies')
class OperationsApiTests(ApiTestCase):
    def test_authentication_and_admin_gate_precede_summary_load(self):
        from api import today
        today.clear_cache()
        with mock.patch('api.operations.get_summary', return_value={'available': False}) as load:
            self.assertEqual(self.client.get('/v1/admin/operations').status_code, 401)
            token = self.login().json()['access_token']
            self.assertEqual(self.client.get('/v1/admin/operations', headers={'Authorization': 'Bearer ' + token}).status_code, 403)
            load.assert_not_called()
            token = self.login('boss@example.com').json()['access_token']
            response = self.client.get('/v1/admin/operations', headers={'Authorization': 'Bearer ' + token})
            self.assertEqual(response.status_code, 200)
            load.assert_called_once_with()
