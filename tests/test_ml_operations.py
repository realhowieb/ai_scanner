"""Operational views must distinguish history, capture failures and actual identity."""
import unittest
from unittest.mock import MagicMock, patch

from analytics.ml_readiness import feature_capture_health
from tests.research_fixtures import opp, ts


def test_capture_health_keeps_rolling_failures_and_separate_healthy_streak():
    rows = [opp(1, 'OLD', '2026-10-01T16:00:00'), opp(2, 'GOOD', '2026-10-06T16:00:00'),
            opp(3, 'GOOD2', '2026-10-07T16:00:00')]
    rows[0]['ai_confidence'] = None
    rows[0]['raw_signal']['models'] = {'ai_confidence': {'reason': 'model_unavailable'}}
    for row in rows[1:]:
        row['ai_confidence'] = 55
    records = [{'observation': {'observation_id': r['id']}, 'features': MagicMock(join={
        'status': 'MATCHED' if r['id'] > 1 else 'ONLY_LATER_SCANS'})} for r in rows]
    health = feature_capture_health(rows, records, ts('2026-10-09T17:00:00'))
    for check in health:
        assert (check['total'], check['captured'], check['missing']) == (3, 2, 1)
        assert (check['current_total'], check['current_captured']) == (2, 2)
        assert check['healthy_rows'] == 2
        assert check['last_failure_at'] == rows[0]['fired_at'].isoformat()
    assert health[1]['failure_reasons'] == {'model_unavailable': 1}


def test_empty_capture_is_unknown_not_perfect():
    for check in feature_capture_health([], [], ts('2026-10-09T17:00:00')):
        assert check['current_total'] == 0
        assert check['healthy_since'] is None
        assert check['healthy_rows'] == 0


def test_slim_failure_reason_and_late_failure_reset_streak():
    rows = [opp(1, 'A', '2026-10-06T16:00:00'), opp(2, 'B', '2026-10-07T16:00:00')]
    rows[0]['ai_confidence'] = 55
    rows[1]['ai_confidence'] = None
    rows[1]['raw_signal']['ai_confidence_capture_reason'] = 'inference_failed'
    health = feature_capture_health(rows, [], ts('2026-10-09T17:00:00'))[1]
    assert health['healthy_rows'] == 0
    assert health['failure_reasons'] == {'inference_failed': 1}


def test_production_evidence_accepts_dict_rows_without_reading_model_bytes():
    from db.ml_operations import production_confidence
    conn = MagicMock()
    cursor = conn.cursor.return_value
    cursor.fetchone.side_effect = [
        {'models': 'ai_confidence_models', 'signals': 'signal_outcomes'},
        {'id': 2, 'model_version': 'configured', 'trained_at': '2026-09-09', 'dataset': None},
        {'fired_at': '2026-11-05', 'sha': 'hash', 'inferred_at': '2026-11-05', 'version': 'actual', 'legacy': 'old'},
    ]
    with patch('db.ml_operations.get_neon_conn', return_value=conn):
        value = production_confidence()
    assert value['configured_model']['model_version'] == 'configured'
    assert value['latest_capture']['model_version'] == 'actual'
    assert value['latest_capture']['artifact_sha256'] == 'hash'
    for call in cursor.execute.call_args_list:
        assert call.args[0].startswith('SELECT ')
        assert 'model_bytes' not in call.args[0]
    conn.close.assert_called_once()




class PrebreakoutEvidenceTests(unittest.TestCase):
    def test_identity_is_slim_read_only_and_not_confidence(self):
        from db.ml_operations import production_prebreakout
        conn = MagicMock()
        cursor = conn.cursor.return_value
        cursor.fetchone.side_effect = [
            ('prebreakout_models', 'signal_outcomes'),
            (7, 'configured-pb', '2026-10-07', None),
            ('2026-11-06', 'pb-hash', '2026-11-06', 'actual-pb', 'legacy-pb'),
        ]
        with patch('db.ml_operations.get_neon_conn', return_value=conn):
            result = production_prebreakout()
        self.assertEqual(result['configured_model']['model_version'], 'configured-pb')
        self.assertEqual(result['latest_capture']['model_version'], 'actual-pb')
        self.assertEqual(result['latest_capture']['artifact_sha256'], 'pb-hash')
        queries = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(all(q.startswith('SELECT ') and 'model_bytes' not in q for q in queries))
        self.assertIn("'captured'", queries[-1])
        self.assertIn('prebreakout_prob > 0', queries[-1])
        self.assertNotIn('ai_confidence', ''.join(queries))
        conn.close.assert_called_once()

    def test_absent_tables_and_database_failure_are_distinct(self):
        from db.ml_model_registry import RegistryUnavailable
        from db.ml_operations import production_prebreakout
        conn = MagicMock()
        conn.cursor.return_value.fetchone.return_value = (None, None)
        with patch('db.ml_operations.get_neon_conn', return_value=conn):
            result = production_prebreakout()
        self.assertIsNone(result['configured_model'])
        self.assertIsNone(result['latest_capture'])
        with patch('db.ml_operations.get_neon_conn', return_value=None):
            with self.assertRaises(RegistryUnavailable):
                production_prebreakout()

    def test_query_failure_closes_resources(self):
        from db.ml_model_registry import RegistryUnavailable
        from db.ml_operations import production_prebreakout
        conn = MagicMock()
        conn.cursor.return_value.execute.side_effect = RuntimeError('db failure')
        with patch('db.ml_operations.get_neon_conn', return_value=conn):
            with self.assertRaises(RegistryUnavailable):
                production_prebreakout()
        conn.cursor.return_value.close.assert_called_once()
        conn.close.assert_called_once()
