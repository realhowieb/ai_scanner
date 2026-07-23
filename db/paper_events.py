"""Durable feed of a user's Alpaca paper-account order events.

The "real-time event engine" (poll-on-refresh): the app polls Alpaca for the
user's recent orders on each render and upserts them here, keyed by order id, so
the activity feed survives page reloads and accumulates history. Latest status
wins per order. Same plain cursor + commit + close pattern as the rest of db/.
"""
from __future__ import annotations

from typing import Any, Dict, List

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alpaca_paper_order_events (
            user_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            symbol TEXT,
            side TEXT,
            qty DOUBLE PRECISION,
            filled_qty DOUBLE PRECISION,
            order_type TEXT,
            status TEXT,
            filled_avg_price DOUBLE PRECISION,
            submitted_at TIMESTAMPTZ,
            filled_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, order_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_events_user "
        "ON alpaca_paper_order_events (user_id, updated_at DESC)"
    )
    conn.commit()
    cur.close()


def _get_conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available.")
    _ensure_schema(conn)
    return conn


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def sync_orders(user_id: str, orders: List[Dict[str, Any]]) -> int:
    """Upsert polled Alpaca orders into the durable feed. Returns rows written."""
    user = (user_id or "").strip().lower()
    if not user or not orders:
        return 0
    try:
        conn = _get_conn()
        cur = conn.cursor()
    except Exception:
        return 0
    written = 0
    for o in orders:
        oid = o.get("id")
        if not oid:
            continue
        try:
            cur.execute(
                """
                INSERT INTO alpaca_paper_order_events
                    (user_id, order_id, symbol, side, qty, filled_qty, order_type,
                     status, filled_avg_price, submitted_at, filled_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, order_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    filled_qty = EXCLUDED.filled_qty,
                    filled_avg_price = EXCLUDED.filled_avg_price,
                    filled_at = EXCLUDED.filled_at,
                    updated_at = NOW()
                """,
                (
                    user, str(oid), o.get("symbol"), o.get("side"),
                    _f(o.get("qty")), _f(o.get("filled_qty")), o.get("type"),
                    o.get("status"), _f(o.get("filled_avg_price")),
                    o.get("submitted_at") or None, o.get("filled_at") or None,
                ),
            )
            written += 1
        except Exception:
            continue
    try:
        conn.commit()
        cur.close()
    except Exception:
        pass
    return written


def list_events(user_id: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Most-recent order events for a user, newest first."""
    user = (user_id or "").strip().lower()
    if not user:
        return []
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT order_id, symbol, side, qty, filled_qty, order_type, status,
                   filled_avg_price, submitted_at, filled_at, updated_at
            FROM alpaca_paper_order_events
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user, int(limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception:
        return []
    cols = ("order_id", "symbol", "side", "qty", "filled_qty", "order_type",
            "status", "filled_avg_price", "submitted_at", "filled_at", "updated_at")
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]
