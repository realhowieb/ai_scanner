"""Tiny derived store of computed Top-Opportunity snapshots for movement.

Historical scanner snapshots hold raw results, not the computed HSF Opportunity
scores, so we persist a minimal (ticker, score, status) list per scan snapshot
here to power score-movement / status-transition comparison. One small JSONB row
per snapshot_time; idempotent upsert. Every function is non-fatal and returns a
safe default when the database is unavailable.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunity_snapshots (
            snapshot_time TIMESTAMPTZ PRIMARY KEY,
            opportunities JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.commit()
    cur.close()


def save_opportunity_snapshot(snapshot_time: Any, opportunities: List[Dict[str, Any]]) -> bool:
    """Upsert the computed opportunities for one scan snapshot. Idempotent —
    re-rendering the same snapshot overwrites the same row, not a new one."""
    if snapshot_time is None:
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO opportunity_snapshots (snapshot_time, opportunities)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (snapshot_time)
            DO UPDATE SET opportunities = EXCLUDED.opportunities
            """,
            (snapshot_time, json.dumps(opportunities or [])),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return False


def load_previous_opportunity_snapshot(before_time: Any) -> Optional[Dict[str, Any]]:
    """Most recent snapshot strictly before `before_time` (the prior scan).

    Returns {"snapshot_time", "opportunities": [...]} or None (no prior snapshot,
    malformed payload, or DB down). Never raises.
    """
    if before_time is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT snapshot_time, opportunities
            FROM opportunity_snapshots
            WHERE snapshot_time < %s
            ORDER BY snapshot_time DESC
            LIMIT 1
            """,
            (before_time,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None
    if not row:
        return None
    if isinstance(row, dict):
        ts, payload = row.get("snapshot_time"), row.get("opportunities")
    else:
        ts, payload = row[0], row[1]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = []
    if not isinstance(payload, list):
        payload = []
    return {"snapshot_time": ts, "opportunities": payload}
