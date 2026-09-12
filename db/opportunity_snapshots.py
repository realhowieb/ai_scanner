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
    # Backward-compatible: `context` tags which surface produced the snapshot so
    # movement only ever compares like-for-like (never Market Brief opps vs
    # Scanner opps). Old rows default to 'market_brief' — the only writer before
    # this column existed — so their history stays comparable, not orphaned.
    cur.execute("ALTER TABLE opportunity_snapshots ADD COLUMN IF NOT EXISTS "
                "context TEXT NOT NULL DEFAULT 'market_brief'")
    conn.commit()
    cur.close()


def _dedupe_by_ticker(opps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One canonical record per ticker (first wins), so a snapshot never carries
    duplicate ticker rows that would double-count movement."""
    seen, out = set(), []
    for o in (opps or []):
        t = str(o.get("ticker") or "").upper()
        if t and t not in seen:
            seen.add(t)
            out.append(o)
    return out


def save_opportunity_snapshot(
    snapshot_time: Any, opportunities: List[Dict[str, Any]], *, context: str = "market_brief"
) -> bool:
    """Upsert the computed opportunities for one snapshot. Idempotent per
    snapshot_time; dedupes tickers before storing."""
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
            INSERT INTO opportunity_snapshots (snapshot_time, opportunities, context)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (snapshot_time)
            DO UPDATE SET opportunities = EXCLUDED.opportunities, context = EXCLUDED.context
            """,
            (snapshot_time, json.dumps(_dedupe_by_ticker(opportunities)), str(context)),
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


def load_recent_snapshots(context: str = "market_brief", limit: int = 2) -> List[Dict[str, Any]]:
    """The most recent snapshots for a context, newest first — for computing the
    latest state transition (current vs previous). Never raises; [] if DB down."""
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT snapshot_time, opportunities, context FROM opportunity_snapshots "
            "WHERE context = %s ORDER BY snapshot_time DESC LIMIT %s",
            (str(context), int(limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    out = []
    for r in rows:
        ts, payload = (r.get("snapshot_time"), r.get("opportunities")) if isinstance(r, dict) else (r[0], r[1])
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = []
        out.append({"snapshot_time": ts, "opportunities": payload if isinstance(payload, list) else []})
    return out


def load_previous_opportunity_snapshot(
    before_time: Any, *, context: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Most recent snapshot STRICTLY before `before_time` — the prior comparable
    snapshot. When `context` is given, only same-context snapshots are returned
    (so movement never crosses surfaces). Returns {"snapshot_time",
    "opportunities", "context"} or None. Never raises.
    """
    if before_time is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        if context is None:
            cur.execute(
                "SELECT snapshot_time, opportunities, context FROM opportunity_snapshots "
                "WHERE snapshot_time < %s ORDER BY snapshot_time DESC LIMIT 1",
                (before_time,),
            )
        else:
            cur.execute(
                "SELECT snapshot_time, opportunities, context FROM opportunity_snapshots "
                "WHERE snapshot_time < %s AND context = %s ORDER BY snapshot_time DESC LIMIT 1",
                (before_time, str(context)),
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
    ctx = (row.get("context") if isinstance(row, dict) else (row[2] if len(row) > 2 else None))
    return {"snapshot_time": ts, "opportunities": payload, "context": ctx}
