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
