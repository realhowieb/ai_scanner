"""Privacy, cursor compatibility, bounded recovery and query equivalence."""
import asyncio
import datetime as dt
import json
import sqlite3
from unittest import mock

import pytest

from db import traffic


class DriverCursor:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.params = []
        self.arraysize = 2

    def execute(self, query, params=None, **kwargs):
        if query == 'fail':
            raise RuntimeError('secret error')
        return self

    def executemany(self, query, params_seq, **kwargs):
        self.params = list(params_seq)
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchmany(self, size=None):
        n = self.arraysize if size is None else size
        rows, self.rows = self.rows[:n], self.rows[n:]
        return rows

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows


class Cursor(traffic.TrafficCursorMixin, DriverCursor):
    pass


def test_fetch_modes_preserve_rows_and_count_once(caplog):
    caplog.set_level('INFO', logger='hsf.db_traffic')
    c = Cursor([('private-value',), ('é',), ('abc',), ('last',)])
    with traffic.scope('test') as m:
        assert c.execute('SELECT private-literal', ('password',)) is c
        assert c.fetchone() == ('private-value',)
        assert c.fetchmany() == [('é',), ('abc',)]
        assert iter(c) is c
        assert next(c) == ('last',)
        assert list(c) == []
        assert c.fetchall() == []
    assert m.calls == 1
    assert m.rows_fetched == 4
    assert m.db_to_client_payload_bytes == len('private-valueéabclast'.encode())
    assert m.client_to_db_payload_bytes == len('SELECT private-literalpassword')
    for secret in ('private-value', 'private-literal', 'password'):
        assert secret not in caplog.text


def test_bulk_generator_and_failure_are_counted_without_swallowing():
    c = Cursor()
    with traffic.scope('test') as m:
        assert c.executemany('INSERT', ((i,) for i in range(3))) is c
        with pytest.raises(RuntimeError):
            c.execute('fail')
    assert c.params == [(0,), (1,), (2,)]
    assert (m.calls, m.errors) == (4, 1)


def test_no_scope_preserves_default_fetchmany_and_disabled_cost():
    c = Cursor([(1,), (2,), (3,)])
    with mock.patch.object(traffic, 'size', side_effect=AssertionError):
        assert c.fetchmany() == [(1,), (2,)]
        assert c.execute('SELECT') is c


def test_async_request_isolation_and_thread_propagation():
    async def task(n):
        with traffic.scope('test') as m:
            await asyncio.to_thread(traffic.count, calls=n)
            await asyncio.sleep(0)
        return m.calls
    async def run():
        return await asyncio.gather(task(2), task(7))
    assert asyncio.run(run()) == [2, 7]
    assert traffic._current.get() is None


def test_warning_has_aggregates_only_even_on_failure(caplog, monkeypatch):
    caplog.set_level('INFO', logger='hsf.db_traffic')
    monkeypatch.setenv('DB_TRAFFIC_WARN_BYTES', '1')
    with pytest.raises(ValueError), traffic.scope('test'):
        traffic.count(db_to_client_payload_bytes=2)
        raise ValueError('sensitive')
    summary = json.loads(caplog.records[-1].message)
    assert summary['budget_exceeded'] is True
    assert 'sensitive' not in caplog.text


def test_readiness_retry_reuses_successful_rows_and_same_bounds():
    from scripts import ml_readiness as worker
    now = dt.datetime(2026, 10, 9, tzinfo=dt.timezone.utc)
    rows = [{'ticker': 'AAA'}]
    with mock.patch('db.research_datasets.fetch_readiness_rows', return_value=rows) as read, \
         mock.patch('db.research_datasets.fetch_scan_index', side_effect=[RuntimeError(), ['scan']]) as scans, \
         mock.patch.object(worker.time, 'sleep') as sleep:
        result = worker.load(now)
    assert result == {'rows': rows, 'scans': ['scan']}
    assert read.call_count == 1
    assert scans.call_count == 2
    assert scans.call_args_list[0] == scans.call_args_list[1]
    assert 2.5 <= sleep.call_args.args[0] <= 5


def test_readiness_retry_cap_does_not_sleep_after_final_failure():
    from scripts import ml_readiness as worker
    with mock.patch('db.research_datasets.fetch_readiness_rows', side_effect=RuntimeError()) as read, \
         mock.patch.object(worker.time, 'sleep') as sleep:
        with pytest.raises(SystemExit):
            worker.load(dt.datetime.now(dt.timezone.utc))
    assert read.call_count == 3
    assert sleep.call_count == 2


def test_recent_horizons_query_matches_old_query_and_keeps_history():
    from db import hsf_observations as store
    from tests.test_maturation_parity import dataset
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    observations, outcomes = dataset(runs=2)
    try:
        for row in observations:
            store.save_observation(row, conn=conn)
        outcome_rows = [row for rows in outcomes.values() for row in rows]
        for row in outcome_rows:
            store.save_outcome(row, conn=conn)
        statements = []
        conn.set_trace_callback(statements.append)
        actual = store.load_recent_observations(limit=4, attach_outcomes=True, conn=conn)
        sql = next(s for s in statements if s.startswith('WITH selected'))
        assert 'WHERE observation_id IN (SELECT observation_id FROM selected)' in sql
        old = conn.execute('SELECT o.record, oc.horizons FROM hsf_observations o LEFT JOIN '
                           '(SELECT observation_id, group_concat(horizon, \',\') AS horizons '
                           'FROM hsf_observation_outcomes GROUP BY observation_id) oc '
                           'ON o.observation_id = oc.observation_id ORDER BY o.timestamp DESC, o.observation_id DESC LIMIT 4').fetchall()
        expected = []
        for row in old:
            record = json.loads(row['record'])
            if row['horizons']:
                record.setdefault('outcomes', {})
                for h in row['horizons'].split(','):
                    record['outcomes'].setdefault(h, True)
            expected.append(record)
        assert actual == expected
        assert conn.execute('SELECT count(*) FROM hsf_observations').fetchone()[0] == len(observations)
        assert conn.execute('SELECT count(*) FROM hsf_observation_outcomes').fetchone()[0] == len(outcome_rows)
    finally:
        conn.close()


def test_event_projection_preserves_contract_when_schema_grows():
    from db import alert_rules as store
    conn = sqlite3.connect(':memory:')
    store.set_connection_factory(lambda: conn)
    try:
        store._ensure_schema(conn, True)
        conn.execute('ALTER TABLE hsf_alert_rule_events ADD COLUMN extra_payload TEXT')
        conn.execute("INSERT INTO hsf_alert_rule_events (user_id, rule_id, ticker, rule_type, operator, "
                     "message, observation_id, triggered_at, extra_payload) "
                     "VALUES ('u', 1, 'AAA', 'price', 'above', 'message', 'o', '2026-10-09', 'large')")
        rows = store.list_events('u', limit=10)
        assert len(rows) == 1
        assert 'extra_payload' not in rows[0]
        assert set(rows[0]) == {'id', *store.EVENT_FIELDS, 'delivery_attempts', 'last_delivery_error'}
    finally:
        store.set_connection_factory(None)
        conn.close()


def test_price_cache_hits_are_not_reuploaded_and_partial_hits_keep_universe():
    import pandas as pd

    from scan import engine
    from tests.test_scan_engine_fake_provider import ScanEngineFakeProviderTests

    fixture = ScanEngineFakeProviderTests()
    cached = {'AAA': fixture._ohlcv(), 'SPY': fixture._ohlcv(start_close=400)}
    fresh = {'BBB': fixture._ohlcv()}
    with mock.patch.object(engine, '_db_cache_allowed_for_run', return_value=True), \
         mock.patch.object(engine, '_db_load_price_cache', return_value=(cached, {'BBB'})), \
         mock.patch.object(engine, '_db_save_price_cache') as save, \
         mock.patch('scan.engine.st.session_state', {'show_scan_progress': False}), \
         mock.patch('data.prices.fetch_price_data_parallel', return_value=(fresh, [])) as fetch, \
         mock.patch('data.prices.fetch_price_data_batch', side_effect=AssertionError('no fallback')):
        result = engine.run_breakout_scan(['AAA', 'BBB'], use_cache=False, min_gap=0, min_price=1,
                                          max_price=100, unusual_volume=False, premarket=False, afterhours=False, top_n=10)
    assert fetch.call_args.args[0] == ['BBB']
    assert set(result['Ticker']) == {'AAA', 'BBB'}
    assert set(save.call_args.args[0]) == {'BBB'}
    pd.testing.assert_frame_equal(save.call_args.args[0]['BBB'], fresh['BBB'])


def test_all_fresh_price_cache_writes_no_payload_and_preserves_frames():
    from scan import engine
    from tests.test_scan_engine_fake_provider import ScanEngineFakeProviderTests

    cached = {'AAA': ScanEngineFakeProviderTests()._ohlcv(), 'SPY': ScanEngineFakeProviderTests()._ohlcv()}
    with mock.patch.object(engine, '_db_cache_allowed_for_run', return_value=True), \
         mock.patch.object(engine, '_db_load_price_cache', return_value=(cached, set())), \
         mock.patch('db.prices.upsert_price_data_snapshot') as upsert, \
         mock.patch('scan.engine.st.session_state', {'show_scan_progress': False}), \
         mock.patch('data.prices.fetch_price_data_parallel', side_effect=AssertionError('no fetch')):
        result = engine.run_breakout_scan(['AAA'], use_cache=False, min_gap=0, min_price=1,
                                          max_price=100, unusual_volume=False, premarket=False, afterhours=False, top_n=10)
    assert list(result['Ticker']) == ['AAA']
    assert not upsert.called


def test_standalone_billing_connection_has_no_scanner_dependency(monkeypatch):
    import builtins

    from billing_service import realtime_alerts

    original = builtins.__import__
    def without_scanner(name, *args, **kwargs):
        if name == 'db.traffic':
            raise ImportError('standalone root')
        return original(name, *args, **kwargs)
    monkeypatch.setenv('DATABASE_URL', 'postgresql://synthetic')
    with mock.patch('psycopg2.connect', return_value='connection') as connect, \
         mock.patch('builtins.__import__', side_effect=without_scanner):
        assert realtime_alerts._conn() == 'connection'
    connect.assert_called_once_with('postgresql://synthetic', connect_timeout=8)


def test_cache_families_do_not_mix_symbol_and_lookup_rates():
    with traffic.scope('test') as metrics:
        traffic.cache('api_results', hits=1, misses=1)
        traffic.cache('neon_price_frames', hits=8000, misses=1000)
    assert metrics.snapshot()['cache_activity'] == {
        'api_results': {'hits': 1, 'misses': 1, 'stale_hits': 0},
        'neon_price_frames': {'hits': 8000, 'misses': 1000, 'stale_hits': 0},
    }


def test_job_wrapper_keeps_arguments_exit_status_and_writes_on_failure(tmp_path, monkeypatch):
    import sys

    from scripts import db_traffic_job as job

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['wrapper', 'scripts.fake_job', '--limit', '123'])
    def module_run(module, run_name):
        assert module == 'scripts.fake_job'
        assert run_name == '__main__'
        assert sys.argv == ['scripts.fake_job', '--limit', '123']
        traffic.count(calls=2)
        raise SystemExit(7)
    with mock.patch.object(job.runpy, 'run_module', side_effect=module_run):
        with pytest.raises(SystemExit) as exc:
            job.main()
    assert exc.value.code == 7
    report = json.loads((tmp_path / 'artifacts/db_traffic/scripts_fake_job.json').read_text())
    assert report['calls'] == 2
    assert report['measurement'] == 'application_payload_estimate'


def test_workflow_yaml_and_wrapped_modules_remain_valid():
    import importlib.util
    from pathlib import Path

    import yaml

    for path in Path('.github/workflows').glob('*.yml'):
        workflow = yaml.safe_load(path.read_text())
        assert isinstance(workflow['jobs'], dict)
        for job in workflow['jobs'].values():
            for step in job.get('steps', []):
                for line in step.get('run', '').splitlines():
                    if 'python -m scripts.db_traffic_job' in line:
                        module = line.split('python -m scripts.db_traffic_job ', 1)[1].split()[0]
                        assert importlib.util.find_spec(module) is not None
