"""Operational views must distinguish history, capture failures and actual identity."""
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
