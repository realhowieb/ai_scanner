"""Bounded, read-only public scanner context for PreBreakout feature construction."""
import json

import pandas as pd

from db.engine import get_neon_conn

# Only scanner inputs, never account data, model vectors or future outcomes.
INPUT_COLUMNS = [
    'Symbol', 'Ticker', 'Last', 'Price', 'Close', 'Open', 'High', 'Low', 'Volume',
    'BreakoutScore', 'IsBreakout', 'GapPct', 'VolRel20', 'DollarVol20',
    'Trend10D%', 'Trend20D%', 'BreakoutPos20D', 'Spark10D',
    'EMA9', 'EMA21', 'High20', 'High20D', 'High50D', 'ATR14', 'Volatility20D%',
]


def load_feature_history(symbols, *, before, days_back=90):
    """One query, at most 64 observations per ticker from canonical public scans.

    Refreshed snapshots have an updated created_at; rows not strictly earlier
    than the requested prediction are excluded. Never read private custom scans.
    The same population is used by training and inference.
    """
    symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not symbols:
        return pd.DataFrame()
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError('PreBreakout scanner history unavailable')
    cur = None
    try:
        cur = conn.cursor()
        cur.execute('''
            WITH scans AS (
              SELECT id, created_at AT TIME ZONE 'UTC' AS observed_at, results_json
              FROM runs WHERE username='cron' AND label='daily_snapshot' AND is_snapshot=TRUE
                AND created_at < (%s::timestamptz AT TIME ZONE 'UTC')
                AND created_at >= ((%s::timestamptz - make_interval(days => %s)) AT TIME ZONE 'UTC')
              ORDER BY created_at DESC, id DESC LIMIT 128
            ), rows AS (
              SELECT id, observed_at,
                upper(coalesce(item->>'Symbol', item->>'Ticker')) AS symbol,
                (SELECT jsonb_object_agg(key,value) FROM jsonb_each(item)
                 WHERE key=ANY(%s)) AS inputs
              FROM scans CROSS JOIN LATERAL jsonb_array_elements(results_json::jsonb) item
              WHERE upper(coalesce(item->>'Symbol',item->>'Ticker'))=ANY(%s)
            ), ranked AS (
              SELECT *, row_number() OVER (PARTITION BY symbol ORDER BY observed_at DESC,id DESC) AS rn
              FROM rows
            ) SELECT symbol, observed_at, inputs FROM ranked WHERE rn <= 64
              ORDER BY observed_at, symbol
        ''', (before, before, int(days_back), INPUT_COLUMNS, symbols))
        records = []
        for row in cur.fetchall():
            symbol, observed_at, inputs = (row['symbol'], row['observed_at'], row['inputs']) if isinstance(row, dict) else row
            if isinstance(inputs, str):
                inputs = json.loads(inputs)
            records.append({**(inputs or {}), 'Symbol': symbol, 'Timestamp': observed_at})
        return pd.DataFrame(records)
    finally:
        if cur is not None:
            cur.close()
        conn.close()
