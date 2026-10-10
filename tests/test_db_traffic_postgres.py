"""Opt-in disposable localhost PostgreSQL checks; never uses app database URLs.

HSF_TEST_POSTGRES=1 python -m pytest tests/test_db_traffic_postgres.py -q
Expects a disposable instance on localhost:55432, user/database postgres.
"""
import json
import os
import uuid

try:
    import pytest
except ImportError:
    from unittest import SkipTest
    raise SkipTest('requires pytest; run with requirements-dev.txt')

from db import traffic

pytestmark = pytest.mark.skipif(os.getenv('HSF_TEST_POSTGRES') != '1', reason='requires disposable localhost PostgreSQL')


@pytest.mark.parametrize('driver', ['psycopg', 'psycopg2'])
def test_real_driver_cursor_compatibility(driver):
    module = pytest.importorskip(driver)
    conn = module.connect('postgresql://postgres@127.0.0.1:55432/postgres', **traffic.connect_options(driver))
    try:
        with traffic.scope('test.local_postgres') as metrics:
            with conn.cursor() as cur:
                cur.execute('SELECT %s::text UNION ALL SELECT %s::text', ('é', 'abc'))
                assert cur.fetchone() == ('é',)
                assert cur.fetchmany() == [('abc',)]
                assert cur.fetchall() == []
                cur.executemany('SELECT %s::int', ((i,) for i in range(2)))
                cur.execute('SELECT generate_series(1,3)')
                assert list(cur) == [(1,), (2,), (3,)]
        assert metrics.calls == 4
        assert metrics.rows_fetched == 5
        assert metrics.db_to_client_payload_bytes == 8
    finally:
        conn.close()


def test_postgres_recent_horizons_and_slim_records_preserve_outputs():
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    from db import hsf_observations as store
    from scripts.mature_observations import OBSERVATION_FIELDS
    from tests.test_maturation_parity import dataset

    conn = psycopg.connect('postgresql://postgres@127.0.0.1:55432/postgres', row_factory=dict_row,
                           **traffic.connect_options())
    schema = 'hsf_audit_' + uuid.uuid4().hex
    try:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        observations, by_id = dataset(runs=2)
        outcomes = [o for rows in by_id.values() for o in rows]
        for record in observations:
            store.save_observation(record, conn=conn)
        for record in outcomes:
            store.save_outcome(record, conn=conn)
        for context in (None, conn.execute('SELECT context FROM hsf_observations LIMIT 1').fetchone()['context'], 'missing'):
            params = (4,) if context is None else (context, 4)
            where = '' if context is None else 'WHERE o.context = %s '
            baseline = conn.execute('SELECT o.record, oc.horizons FROM hsf_observations o LEFT JOIN '
                                    "(SELECT observation_id, string_agg(horizon, ',') AS horizons "
                                    'FROM hsf_observation_outcomes GROUP BY observation_id) oc '
                                    'ON o.observation_id = oc.observation_id ' + where +
                                    'ORDER BY o.timestamp DESC, o.observation_id DESC LIMIT %s', params).fetchall()
            expected = []
            for row in baseline:
                record = row['record']
                if row['horizons']:
                    record.setdefault('outcomes', {})
                    for horizon in row['horizons'].split(','):
                        record['outcomes'].setdefault(horizon, True)
                expected.append(record)
            with traffic.scope('test.local_maturation') as metrics:
                actual = store.load_recent_observations(limit=4, context=context, attach_outcomes=True, conn=conn)
            assert actual == expected
            assert metrics.calls == 1
            slim = store.load_recent_observations(limit=4, context=context, attach_outcomes=True,
                                                   fields=OBSERVATION_FIELDS, conn=conn)
            assert slim == [{k: v for k, v in o.items() if k in set(OBSERVATION_FIELDS) | {'outcomes'} and v is not None}
                            for o in actual]
        assert conn.execute('SELECT count(*) AS n FROM hsf_observations').fetchone()['n'] == len(observations)
        assert conn.execute('SELECT count(*) AS n FROM hsf_observation_outcomes').fetchone()['n'] == len(outcomes)
    finally:
        conn.rollback()
        conn.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
        conn.commit()
        conn.close()


def test_cohort_selected_outcomes_reduce_payload_without_changing_report():
    from unittest import mock

    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    from db import hsf_observations as store
    from scripts import audit_research_cohorts as audit
    from tests.test_maturation_parity import dataset

    conn = psycopg.connect('postgresql://postgres@127.0.0.1:55432/postgres', row_factory=dict_row,
                           **traffic.connect_options())
    schema = 'cohort_transfer_' + uuid.uuid4().hex
    try:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        # schema_once caches by database, not search_path; this test owns a new
        # schema in the same database as earlier tests and must initialize it.
        store._ensure_schema.__wrapped__(conn, False)
        observations, by_id = dataset(runs=2)
        outcomes = [row for rows in by_id.values() for row in rows]
        for record in observations:
            assert store.save_observation(record, conn=conn)
        for record in outcomes:
            assert store.save_outcome(record, conn=conn)
        assert store.save_outcome({'observation_id': 'true-orphan', 'horizon': '+5m', 'raw_return': .03}, conn=conn)
        # Selected matched records exercise real outcome retrieval; all history
        # outside this window must remain available to default diagnostic callers.
        selected = [row for row in observations if row['observation_id'] in by_id][:10]
        with mock.patch.object(store, 'load_recent_observations', return_value=selected), \
             mock.patch('db.engine.get_neon_conn', return_value=conn):
            with traffic.scope('test.cohort.before') as before:
                old_obs, all_outcomes = audit._load_live()
            with traffic.scope('test.cohort.after') as after:
                new_obs, selected_outcomes = audit._load_live(selected_outcomes_only=True)
        old_report = audit.audit_cohorts(old_obs, all_outcomes)
        new_report = audit.audit_cohorts(new_obs, selected_outcomes)
        old_report.pop('generated_at')
        new_report.pop('generated_at')
        assert old_report == new_report
        assert 'true-orphan' in all_outcomes and 'true-orphan' not in selected_outcomes
        assert set(selected_outcomes) == {row['observation_id'] for row in selected}
        assert before.calls == after.calls == 1
        assert before.rows_fetched == len(outcomes) + 1
        assert after.rows_fetched == sum(len(by_id[row['observation_id']]) for row in selected)
        assert after.db_to_client_payload_bytes < before.db_to_client_payload_bytes
        counts = {'observations': conn.execute('SELECT count(*) AS n FROM hsf_observations').fetchone()['n'],
                  'outcomes': conn.execute('SELECT count(*) AS n FROM hsf_observation_outcomes').fetchone()['n']}
        assert counts == {'observations': len(observations), 'outcomes': len(outcomes) + 1}
        print('COHORT_TRANSFER_BENCHMARK=' + json.dumps({'measurement': 'local_synthetic_application_payload_estimate',
              'selected_observations': len(selected), 'history_counts_unchanged': counts,
              'reports_equal_excluding_generated_at': True, 'default_retains_orphan_evidence': True,
              'before': before.snapshot(), 'after': after.snapshot()}))
    finally:
        conn.rollback()
        conn.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
        conn.commit()
        conn.close()
