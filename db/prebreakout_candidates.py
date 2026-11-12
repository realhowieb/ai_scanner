"""PreBreakout candidates: the near-setup names each scheduled scan scored.

The PreBreakout model is trained on stocks that are NOT yet setups (no
breakout, BreakoutScore below 8, below the 20-day high). A scan's saved results
are its top setups, so none of those qualify. Each scheduled scan therefore
scores its near-setup names separately and saves them here, one row per scan:
the PreBreakout tab reads the latest row, and the weekly audit reads the stored
scores to measure live hit rate. Bounded: rows older than 180 days are pruned.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn, schema_once

RETENTION_DAYS = 180


@schema_once
def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS prebreakout_candidates (
            id BIGSERIAL PRIMARY KEY,
            universe TEXT NOT NULL,
            scan_id TEXT NOT NULL,
            scanned_at TIMESTAMPTZ NOT NULL,
            model_version TEXT,
            row_count INTEGER NOT NULL,
            rows JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (universe, scan_id))''')
        cur.execute('CREATE INDEX IF NOT EXISTS prebreakout_candidates_scanned_idx '
                    'ON prebreakout_candidates (universe, scanned_at DESC)')
    conn.commit()


def _row(r, key, i):
    return r[key] if isinstance(r, dict) else r[i]


def save_candidates(universe: str, scan_id: str, scanned_at, rows: List[Dict[str, Any]],
                    *, model_version: Optional[str] = None) -> bool:
    """Upsert one scan's scored candidates. Best-effort: False on any failure."""
    if not rows:
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        ensure_schema(conn)
        payload = json.dumps(rows, default=str, allow_nan=False)
        with conn.cursor() as cur:
            cur.execute('''INSERT INTO prebreakout_candidates
                (universe, scan_id, scanned_at, model_version, row_count, rows)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (universe, scan_id) DO UPDATE SET rows=EXCLUDED.rows,
                row_count=EXCLUDED.row_count, model_version=EXCLUDED.model_version''',
                        (universe, scan_id, scanned_at, model_version, len(rows), payload))
            cur.execute("DELETE FROM prebreakout_candidates WHERE scanned_at < NOW() - make_interval(days => %s)",
                        (RETENTION_DAYS,))
        conn.commit()
        return True
    finally:
        conn.close()


def load_latest(universe: str = "US_MARKET", *, before=None) -> Optional[Dict[str, Any]]:
    """The newest scan's candidates (optionally at or before ``before``), or None."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute('''SELECT scan_id, scanned_at, model_version, rows FROM prebreakout_candidates
                WHERE universe=%s AND (%s::timestamptz IS NULL OR scanned_at <= %s::timestamptz)
                ORDER BY scanned_at DESC LIMIT 1''', (universe, before, before))
            r = cur.fetchone()
        if not r:
            return None
        rows = _row(r, "rows", 3)
        return {"scan_id": _row(r, "scan_id", 0), "scanned_at": _row(r, "scanned_at", 1),
                "model_version": _row(r, "model_version", 2),
                "rows": json.loads(rows) if isinstance(rows, str) else (rows or [])}
    finally:
        conn.close()


def load_since(days: int, universe: str = "US_MARKET") -> List[Dict[str, Any]]:
    """Every stored scan in the last ``days`` days, oldest first (for audits)."""
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute('''SELECT scan_id, scanned_at, model_version, rows FROM prebreakout_candidates
                WHERE universe=%s AND scanned_at >= NOW() - make_interval(days => %s)
                ORDER BY scanned_at''', (universe, int(days)))
            out = []
            for r in cur.fetchall():
                rows = _row(r, "rows", 3)
                out.append({"scan_id": _row(r, "scan_id", 0), "scanned_at": _row(r, "scanned_at", 1),
                            "model_version": _row(r, "model_version", 2),
                            "rows": json.loads(rows) if isinstance(rows, str) else (rows or [])})
            return out
    finally:
        conn.close()
