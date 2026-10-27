"""Shared HSF scores for every upcoming earnings reporter (Premium auto-score).

The scheduled scan scores every name reporting from today through the next 7 days
once per slot (scheduler/earnings_autoscore.py) and saves the rows here. The
Earnings tab's "All reporters" view reads the newest row so Premium users see a
score on every reporter as the page loads, with no scoring work on the API.

Kept out of the `runs` table on purpose: research and ML readers walk every run,
and these screen-free scores are not market-scan observations.

Table:
  earnings_autoscore(id, created_at, session, window_start, window_end,
                     symbols, duration_s, setups JSONB)
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn, schema_once

KEEP_ROWS = 50   # about a week of slots; older rows are deleted on each save

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS earnings_autoscore (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session TEXT NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    symbols INTEGER NOT NULL,
    duration_s REAL,
    setups JSONB NOT NULL
)
"""


@schema_once
def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_SQL)
        cur.execute("CREATE INDEX IF NOT EXISTS earnings_autoscore_created ON earnings_autoscore (created_at DESC)")
    conn.commit()


def save(setups: List[Dict[str, Any]], *, session: str, window_start: dt.date, window_end: dt.date,
         symbols: int, duration_s: Optional[float] = None) -> Optional[int]:
    """Store one scoring pass (setups must already be strict JSON) and prune old passes."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO earnings_autoscore (session, window_start, window_end, symbols, duration_s, setups) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb) RETURNING id",
                (session, window_start, window_end, int(symbols), duration_s, json.dumps(setups, allow_nan=False)),
            )
            row = cur.fetchone()
            new_id = int(row["id"] if isinstance(row, dict) else row[0])
            cur.execute("DELETE FROM earnings_autoscore WHERE id NOT IN "
                        "(SELECT id FROM earnings_autoscore ORDER BY id DESC LIMIT %s)", (KEEP_ROWS,))
        conn.commit()
        return new_id
    finally:
        conn.close()


def latest_stamp(max_age_s: int) -> Optional[Dict[str, Any]]:
    """{id, created_at, session, window_start, window_end, symbols} of the newest pass
    younger than max_age_s, without its rows (cheap enough to check every minute)."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, created_at, session, window_start, window_end, symbols FROM earnings_autoscore "
                        "WHERE created_at > NOW() - make_interval(secs => %s) ORDER BY id DESC LIMIT 1",
                        (int(max_age_s),))
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def load_setups(autoscore_id: int) -> List[Dict[str, Any]]:
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT setups FROM earnings_autoscore WHERE id = %s", (int(autoscore_id),))
            row = cur.fetchone()
        conn.commit()
        if not row:
            return []
        data = row["setups"] if isinstance(row, dict) else row[0]
        if isinstance(data, str):
            data = json.loads(data)
        return data if isinstance(data, list) else []
    finally:
        conn.close()
