"""Versioned research persistence on an explicitly configured throwaway Postgres."""
import datetime as dt
import importlib.util
import os
import unittest
import uuid
from unittest.mock import patch

PG_URL = os.getenv('HSF_TEST_BILLING_PG_URL')
ENABLED = bool(PG_URL) and importlib.util.find_spec('psycopg2') is not None


@unittest.skipUnless(ENABLED, 'needs explicitly configured throwaway Postgres')
class TrackRecordPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg2
        from psycopg2 import sql

        self.driver, self.sql = psycopg2, sql
        self.schema = 'hsf_track_test_' + uuid.uuid4().hex
        with psycopg2.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        self.connect = lambda: psycopg2.connect(PG_URL, options=f'-csearch_path={self.schema}')
        self.env = patch.dict('os.environ', {'AI_SCANNER_SCHEMA_ONCE': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def drop_schema(self):
        with self.driver.connect(PG_URL) as conn, conn.cursor() as cur:
            cur.execute(self.sql.SQL('DROP SCHEMA {} CASCADE').format(self.sql.Identifier(self.schema)))

    def test_legacy_schema_upgrade_summary_and_daily_are_method_isolated(self):
        from analytics.track_method import METHOD
        from db import track_record as store

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute('CREATE TABLE signal_track_record (id BIGSERIAL PRIMARY KEY, horizon_days INTEGER NOT NULL, '
                        'avg_return DOUBLE PRECISION, median_return DOUBLE PRECISION, win_rate DOUBLE PRECISION, '
                        'sample_size INTEGER NOT NULL DEFAULT 0, runs_used INTEGER NOT NULL DEFAULT 0, '
                        'computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW())')
            cur.execute('INSERT INTO signal_track_record(horizon_days, avg_return) VALUES (5, .99)')
        details = {'methodology': METHOD, 'entry_policy': 'first_session_open_strictly_after_snapshot',
                   'avg_absolute_return': -.02, 'avg_excess_return': .03, 'selected_count': 2,
                   'evaluated_count': 1, 'excluded_count': 1, 'exclusions': {'MISSING_SYMBOL_BARS': 1},
                   'observations': [{'symbol': 'AAA', 'signal_at': '2026-10-05T20:01:00+00:00',
                                     'entry_at': '2026-10-06T13:30:00+00:00', 'exit_at': '2026-10-12T20:00:00+00:00'}]}
        day = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
        while day.weekday() >= 5:
            day -= dt.timedelta(days=1)
        with patch.object(store, 'get_neon_conn', side_effect=self.connect):
            self.assertEqual(store.load_latest_track_records(), [])
            self.assertTrue(store.save_track_record(horizon_days=5, avg_return=.03, median_return=.03,
                                                   win_rate=1, sample_size=1, runs_used=1,
                                                   ranking='breakout', benchmark='SPY', details=details))
            row = store.load_latest_track_record(5, 'breakout')
            self.assertEqual(row['methodology'], METHOD)
            self.assertEqual(row['avg_absolute_return'], -.02)
            self.assertEqual(row['observations'], details['observations'])
            store.save_daily_excess('breakout', 5, [(day, .99, 1)])  # old namespace remains hidden
            self.assertEqual(store.load_daily_excess('breakout', 5), [])
            store.save_daily_excess('breakout', 5, [(day, .03, 1)], methodology=METHOD)
            self.assertEqual(store.load_daily_excess('breakout', 5), [(day, .03)])

    def test_leaderboard_payload_and_versioned_entry_mode_round_trip(self):
        from db import signal_leaderboard as store

        row = {'signal': 'breakout', 'methodology': 'track-v2-next-open', 'display': 'Breakout',
               'avg_excess': .03, 'median_excess': .03, 'win_rate': 1, 'sample_size': 1,
               'runs_used': 1, 'benchmark': 'SPY', 'top_n': 5,
               'observations': [{'signal_at': '2026-10-05T20:01:00+00:00'}]}
        with patch.object(store, 'get_neon_conn', side_effect=self.connect):
            self.assertEqual(store.save_leaderboard(5, [row], entry_mode='open'), 1)
            result = store.load_leaderboard(5, entry_mode='open')
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['observations'], row['observations'])
            self.assertEqual(result[0]['entry_mode'], 'open')
            self.assertEqual(result[0]['methodology'], 'track-v2-next-open')
            self.assertEqual(store.load_leaderboard(5, entry_mode='close'), [])


    def test_real_snapshot_timestamp_column_is_normalized_to_aware_utc(self):
        from db import runs
        stamp = dt.datetime(2026, 11, 5, 20, 56)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE runs (id SERIAL PRIMARY KEY, name TEXT, label TEXT, username TEXT, row_count INT, "
                        "duration_sec REAL, is_snapshot BOOLEAN, created_at TIMESTAMP WITHOUT TIME ZONE)")
            cur.execute("INSERT INTO runs(name,label,username,is_snapshot,created_at) VALUES (%s,%s,%s,TRUE,%s)",
                        ('US_MARKET', 'daily_snapshot', 'cron', stamp))
        with patch.object(runs, 'get_neon_conn', side_effect=self.connect), patch.object(runs, 'ensure_neon_runs_schema'):
            rows = runs.list_snapshot_runs(days=5000, public_only=True)
        self.assertEqual(rows[0]['created_at'].tzinfo, dt.timezone.utc)
        self.assertEqual(rows[0]['created_at'], stamp.replace(tzinfo=dt.timezone.utc))
        # The helper remains strict about ambiguous naive inputs; the DB reader supplies the known timezone.
        from analytics.track_method import window
        self.assertNotIn('reason', window(rows[0]['created_at'], 1))
