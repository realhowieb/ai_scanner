"""Shared HSF scores for watchlist stocks the market scan didn't rank (Premium auto-score).

After each scheduled scan, scheduler/watchlist_autoscore.py scores every ticker saved on
any watchlist that the scan's ranked setups don't include, and saves the rows here. The
API fills Premium watchlists' unranked rows from the newest pass (items[].auto), so the
Scanner's Watchlist tab and the Watchlists page show a score on every stock.

Kept out of `runs` for the same reason as db.earnings_autoscore: these screen-free
scores are not market-scan observations.

Table:
  watchlist_autoscore(id, created_at, session, symbols, duration_s, setups JSONB)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn, schema_once

KEEP_ROWS = 30   # a few days of slots; older rows are deleted on each save

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS watchlist_autoscore (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session TEXT NOT NULL,
    symbols INTEGER NOT NULL,
    duration_s REAL,
    setups JSONB NOT NULL
)
"""


@schema_once
def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_SQL)
        cur.execute("CREATE INDEX IF NOT EXISTS watchlist_autoscore_created ON watchlist_autoscore (created_at DESC)")
    conn.commit()


def watched_tickers(limit: int) -> List[str]:
    """Every ticker on any watchlist, most-watched first (then A-Z), at most `limit`."""
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT UPPER(TRIM(ticker)) AS t, COUNT(DISTINCT watchlist_id) AS n FROM watchlist_items "
                        "WHERE TRIM(COALESCE(ticker, '')) <> '' GROUP BY 1 ORDER BY n DESC, t ASC LIMIT %s",
                        (int(limit),))
            rows = cur.fetchall()
        conn.commit()
        return [str(r["t"] if isinstance(r, dict) else r[0]) for r in rows]
    finally:
        conn.close()


def save(setups: List[Dict[str, Any]], *, session: str, symbols: int,
         duration_s: Optional[float] = None) -> Optional[int]:
    """Store one scoring pass (setups must already be strict JSON) and prune old passes."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO watchlist_autoscore (session, symbols, duration_s, setups) "
                "VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
                (session, int(symbols), duration_s, json.dumps(setups, allow_nan=False)),
            )
            row = cur.fetchone()
            new_id = int(row["id"] if isinstance(row, dict) else row[0])
            cur.execute("DELETE FROM watchlist_autoscore WHERE id NOT IN "
                        "(SELECT id FROM watchlist_autoscore ORDER BY id DESC LIMIT %s)", (KEEP_ROWS,))
        conn.commit()
        return new_id
    finally:
        conn.close()


def latest_stamp(max_age_s: int) -> Optional[Dict[str, Any]]:
    """{id, created_at, session, symbols} of the newest pass younger than max_age_s."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, created_at, session, symbols FROM watchlist_autoscore "
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
            cur.execute("SELECT setups FROM watchlist_autoscore WHERE id = %s", (int(autoscore_id),))
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
