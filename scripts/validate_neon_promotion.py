"""Synthetic workloads on disposable localhost PostgreSQL, never app URLs.

Run with HSF_TEST_POSTGRES=1; expects localhost:55432, user/database postgres.
Prints aggregate estimates only. Timings are local observations, not Neon savings.
"""
import json
import logging
import os
import uuid
from unittest import mock

from db import hsf_observations, prices, traffic
from scripts.benchmark_db_transfer import benchmark, price_upload_model


def validate():
    if os.getenv('HSF_TEST_POSTGRES') != '1':
        raise SystemExit('Requires explicit disposable localhost PostgreSQL opt-in')
    import pandas as pd
    import psycopg
    from fastapi.testclient import TestClient
    from psycopg import sql
    from psycopg.rows import dict_row

    from api import main
    from api.settings import Settings
    from scripts.mature_observations import OBSERVATION_FIELDS
    from tests.test_maturation_parity import dataset

    conn = psycopg.connect('postgresql://postgres@127.0.0.1:55432/postgres',
                           row_factory=dict_row, **traffic.connect_options())
    schema = 'promotion_' + uuid.uuid4().hex
    try:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        observations, outcomes = dataset(runs=10)
        for row in observations:
            hsf_observations.save_observation(row, conn=conn)
        outcome_rows = [row for rows in outcomes.values() for row in rows]
        for row in outcome_rows:
            hsf_observations.save_outcome(row, conn=conn)
        conn.commit()
        with traffic.scope('validation.before_recent') as before:
            rows = conn.execute("SELECT o.record, oc.horizons FROM hsf_observations o LEFT JOIN "
                                "(SELECT observation_id, string_agg(horizon, ',') AS horizons "
                                "FROM hsf_observation_outcomes GROUP BY observation_id) oc "
                                "ON o.observation_id = oc.observation_id "
                                "ORDER BY o.timestamp DESC, o.observation_id DESC LIMIT 100").fetchall()
        expected = []
        for row in rows:
            record = row['record']
            for horizon in (row['horizons'] or '').split(','):
                if horizon:
                    record.setdefault('outcomes', {}).setdefault(horizon, True)
            expected.append(record)
        with traffic.scope('validation.after_recent') as after:
            actual = hsf_observations.load_recent_observations(limit=100, attach_outcomes=True, conn=conn)
        assert actual == expected
        with traffic.scope('validation.slim_recent') as slim_metrics:
            slim = hsf_observations.load_recent_observations(limit=100, attach_outcomes=True,
                                                           fields=OBSERVATION_FIELDS, conn=conn)
        assert slim == [{k: v for k, v in row.items()
                         if k in set(OBSERVATION_FIELDS) | {'outcomes'} and v is not None} for row in actual]
        frame = pd.DataFrame({'Open': [10., 11.], 'High': [12., 13.], 'Low': [9., 10.],
                              'Close': [11., 12.], 'Volume': [100000., 110000.]},
                             index=pd.date_range('2026-10-01', periods=2))
        with mock.patch.object(prices, 'get_neon_conn', return_value=conn):
            prices.upsert_price_data_snapshot({'AAA': frame})
            with traffic.scope('validation.price_cache') as cache_metrics:
                cached, missing = prices.get_price_data_snapshot(['AAA', 'BBB'])
        pd.testing.assert_frame_equal(cached['AAA'], frame, check_freq=False, check_dtype=False)
        assert missing == {'BBB'}

        # Actual API request middleware and sync worker propagation, local DB only.
        app = main.create_app(Settings(jwt_secret='local-validation-' + 'x' * 40,
                                       access_ttl_s=900, refresh_ttl_s=3600, cors_origins=()))
        @app.get('/_validation/local-db')
        def local_endpoint():
            with psycopg.connect('postgresql://postgres@127.0.0.1:55432/postgres',
                                 **traffic.connect_options()) as request_conn:
                traffic.count(connections_opened=1)
                return {'rows': request_conn.execute('SELECT %s::int', (7,)).fetchall()}
        messages = []
        with mock.patch.object(main.access_log, 'info', side_effect=messages.append):
            with TestClient(app) as client:
                response = client.get('/_validation/local-db')
        assert response.status_code == 200 and response.json() == {'rows': [[7]]}
        api_metrics = json.loads(messages[-1])['db_traffic']
        assert api_metrics['calls'] == 1 and api_metrics['rows_fetched'] == 1
        assert api_metrics['connections_opened'] == 1

        # Confirm configured warning behavior without modifying repository variables.
        with mock.patch.dict(os.environ, {'DB_TRAFFIC_WARN_BYTES': '1', 'GITHUB_ACTIONS': 'false'}), \
             mock.patch.object(traffic.log, 'log') as alert:
            with traffic.scope('validation.alert'):
                traffic.count(db_to_client_payload_bytes=2)
        assert json.loads(alert.call_args.args[1])['budget_exceeded']
        assert alert.call_args.args[0] == logging.WARNING
        counts = {table: conn.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(sql.Identifier(table))).fetchone()['n']
                  for table in ('hsf_observations', 'hsf_observation_outcomes')}
        assert counts == {'hsf_observations': len(observations), 'hsf_observation_outcomes': len(outcome_rows)}
        return {'measurement': 'local_synthetic_application_payload_estimates', 'postgres_version': conn.info.server_version,
                'recent_observations': {'outputs_equal': True, 'before': before.snapshot(), 'after': after.snapshot(),
                                        'slim': slim_metrics.snapshot(), 'slim_already_in_pr53': True},
                'history_counts_unchanged': counts, 'price_cache': cache_metrics.snapshot(),
                'api_request': api_metrics, 'warning_threshold_verified': True,
                'retry': benchmark(), 'price_upload_model': price_upload_model(),
                'provider_baseline': {'status': 'unavailable', 'reason': 'No configured Neon credentials or connector'},
                'limitations': ['Synthetic fixture sizes, not production workload samples',
                                'Single local timing sample; not a performance or billing forecast',
                                'Scan engine and readiness integration use deterministic mocked market/data boundaries in tests']}
    finally:
        conn.rollback()
        conn.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
        conn.commit()
        conn.close()


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
