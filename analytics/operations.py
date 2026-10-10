"""Strict operational allowlist shared by admin API/UI; no research payloads."""
import math

from analytics.data_freshness import timestamp
from analytics.system_health import STATUS_ORDER, WORKFLOW_SPECS

TRAFFIC_FIELDS = ('calls', 'errors', 'rows_fetched', 'db_to_client_payload_bytes',
                  'client_to_db_payload_bytes', 'connections_opened', 'connections_reused',
                  'cache_hits', 'cache_misses', 'execute_ms', 'fetch_ms', 'elapsed_ms')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value


def iso(value):
    parsed = timestamp(value)
    return parsed.isoformat() if parsed else None


def traffic_report(raw):
    if not isinstance(raw, dict) or raw.get('measurement') != 'application_payload_estimate':
        return None
    return {key: number(raw.get(key)) for key in TRAFFIC_FIELDS}


def summary(health, *, now):
    from ui.trust_banner import health_is_fresh

    health = health if isinstance(health, dict) else {}
    subs = health.get('subsystems') or {}
    generated = timestamp(health.get('generated_at'))
    workflows = []
    for row in ((subs.get('workflows') or {}).get('metrics') or {}).get('workflows') or []:
        if not isinstance(row, dict) or row.get('workflow') not in WORKFLOW_SPECS:
            continue
        workflows.append({'workflow': row['workflow'], 'label': WORKFLOW_SPECS[row['workflow']]['label'],
                          'status': row.get('status') if row.get('status') in
                          ('HEALTHY', 'UNKNOWN', 'NEVER_RUN', 'WORKFLOW_STALE', 'SCHEDULE_GAP', 'CONSECUTIVE_FAILURES') else 'UNKNOWN',
                          'last_run': iso(row.get('last_run')), 'last_success': iso(row.get('last_success')),
                          'last_failure': iso(row.get('last_failure')),
                          'consecutive_failures': number(row.get('consecutive_failures'))})
    mat = ((subs.get('maturation') or {}).get('metrics') or {})
    traffic = []
    for row in health.get('database_traffic') or []:
        if not isinstance(row, dict) or row.get('workflow') not in WORKFLOW_SPECS:
            continue
        metrics = traffic_report(row.get('metrics'))
        if metrics is not None:
            traffic.append({'workflow': row['workflow'], 'observed_at': iso(row.get('observed_at')), 'metrics': metrics})
    status = health.get('system_status')
    sample_totals = {key: sum(row['metrics'][key] for row in traffic)
                     if traffic and all(row['metrics'][key] is not None for row in traffic) else None
                     for key in TRAFFIC_FIELDS}
    return {'available': generated is not None, 'generated_at': iso(generated),
            'stale': not health_is_fresh(generated, now),
            'system_status': status if status in STATUS_ORDER else 'UNKNOWN',
            'workflows': workflows[:len(WORKFLOW_SPECS)],
            'maturation': {'last_success_at': iso(mat.get('last_success')),
                           'pending_observations': number(mat.get('ready_observations')),
                           'oldest_pending_age_minutes': number(mat.get('oldest_pending_age_min')),
                           'scope': 'ready observations in latest maturation report selection; not all pending or global backlog'},
            'database_traffic': {'available': bool(traffic), 'measurement': 'application_payload_estimate',
                                 'scope': 'latest completed instrumented run per workflow; not a time-window total',
                                 'sample_totals': sample_totals,
                                 'workflows': traffic[:len(WORKFLOW_SPECS)]},
            'neon_usage': {'available': False, 'public_network_transfer_bytes': None,
                           'compute_cu_hours': None, 'storage_bytes': None,
                           'note': 'Provider usage unavailable; application estimates are not billed usage'}}
