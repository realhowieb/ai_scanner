"""Disposable, bounded projections of public scans; survives API deployments."""
import json

from db.engine import get_neon_conn, schema_once


@schema_once
def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS scan_score_cache (
            run_id BIGINT NOT NULL, method TEXT NOT NULL, stamp TEXT NOT NULL,
            scores JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, method))''')
    conn.commit()


def load_scores(run_ids, method):
    conn = get_neon_conn()
    if conn is None:
        return {}
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute('SELECT run_id, stamp, scores FROM scan_score_cache WHERE run_id=ANY(%s) AND method=%s',
                        (list(run_ids), method))
            out = {}
            for row in cur.fetchall():
                rid, stamp, scores = (row['run_id'], row['stamp'], row['scores']) if isinstance(row, dict) else row
                out[int(rid)] = (stamp, json.loads(scores) if isinstance(scores, str) else scores)
            return out
    finally:
        conn.close()


def save_scores(records, method):
    if not records:
        return
    conn = get_neon_conn()
    if conn is None:
        return
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute('''INSERT INTO scan_score_cache (run_id,method,stamp,scores)
                SELECT (x->>'run_id')::bigint, %s, x->>'stamp', x->'scores'
                FROM jsonb_array_elements(%s::jsonb) x
                ON CONFLICT (run_id,method) DO UPDATE SET stamp=EXCLUDED.stamp,
                scores=EXCLUDED.scores,updated_at=NOW()''', (method, json.dumps(records, allow_nan=False)))
            # Derived data only; retain at most 96 recent public scan projections.
            cur.execute('''DELETE FROM scan_score_cache WHERE (run_id,method) IN (
                SELECT run_id,method FROM scan_score_cache ORDER BY updated_at DESC,run_id DESC OFFSET 96)''')
        conn.commit()
    finally:
        conn.close()


SCORE_INPUT_FIELDS = ['Ticker', 'Symbol', 'ticker', 'GapPct', 'Gap %', 'PctChange', 'Chg %', 'chg_pct',
                      'PreBreakoutProb%', 'PreBreakoutProb', 'PreBreakoutProbRaw', 'BreakoutScore',
                      'EMACross', 'IsBreakout', 'Last', 'last', 'Price', 'VolRel20']


def load_score_rows(run_ids):
    """One slim source read, excluding model vectors/charts from API working memory.

    Return the actual version with each payload to detect a snapshot rewritten
    between the metadata read and this read. Public full-market scans only.
    """
    conn = get_neon_conn()
    if conn is None:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute('''SELECT id, created_at AT TIME ZONE 'UTC' AS created_at, row_count,
                (SELECT COALESCE(jsonb_agg((SELECT jsonb_object_agg(key,value)
                    FROM jsonb_each(item) WHERE key=ANY(%s))), '[]'::jsonb)
                 FROM jsonb_array_elements(results_json::jsonb) item
                 WHERE jsonb_typeof(item)='object') AS rows
                FROM runs WHERE id=ANY(%s) AND username='cron' AND label='US_MARKET' ''',
                        (SCORE_INPUT_FIELDS, list(run_ids)))
            out = {}
            for row in cur.fetchall():
                rid, created, count, rows = (row['id'], row['created_at'], row['row_count'], row['rows']) if isinstance(row, dict) else row
                out[int(rid)] = {'stamp': f'{created}|{count}', 'rows': rows}
            return out
    finally:
        conn.close()
