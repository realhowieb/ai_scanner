"""Durable paper-order intents; no provider secrets are stored here."""

import json

from db.engine import get_neon_conn, schema_once
from domain.paper_order_state import transition


@schema_once
def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS paper_order_requests (
        user_id TEXT NOT NULL, client_order_id TEXT NOT NULL, account_ref TEXT NOT NULL,
        ticker TEXT NOT NULL, qty INTEGER NOT NULL, context JSONB NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending', result JSONB, error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, client_order_id))""")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS paper_order_requests_pending ON paper_order_requests (user_id, state, updated_at)"
    )
    conn.commit()
    cur.close()


def _conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Paper order storage unavailable")
    try:
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _row(cur, row):
    if row is None:
        return None
    out = dict(row) if isinstance(row, dict) else dict(zip([d.name for d in cur.description], row))
    for key in ("context", "result"):
        if isinstance(out.get(key), str):
            out[key] = json.loads(out[key])
    return out


def reserve(user, client_id, account_ref, ticker, qty, context):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO paper_order_requests
            (user_id, client_order_id, account_ref, ticker, qty, context) VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (user_id, client_order_id) DO NOTHING""",
            (user, client_id, account_ref, ticker, qty, json.dumps(context, allow_nan=False)),
        )
        cur.execute("SELECT * FROM paper_order_requests WHERE user_id = %s AND client_order_id = %s", (user, client_id))
        row = _row(cur, cur.fetchone())
        conn.commit()
        cur.close()
        return row
    finally:
        conn.close()


def save(user, client_id, *, state, result=None, error=None):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM paper_order_requests WHERE user_id = %s AND client_order_id = %s FOR UPDATE",
                    (user, client_id))
        current = _row(cur, cur.fetchone())
        if current is None:
            raise RuntimeError("Missing paper order intent")
        state, result, error = transition(current, state, result, error)
        cur.execute(
            """UPDATE paper_order_requests SET state = %s, result = COALESCE(%s::jsonb, result),
            error = %s, updated_at = NOW() WHERE user_id = %s AND client_order_id = %s RETURNING *""",
            (state, json.dumps(result, allow_nan=False) if result else None, error, user, client_id),
        )
        saved = _row(cur, cur.fetchone())
        conn.commit()
        cur.close()
        return saved
    finally:
        conn.close()


def get(user, client_id):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM paper_order_requests WHERE user_id = %s AND client_order_id = %s", (user, client_id))
        row = _row(cur, cur.fetchone())
        cur.close()
        return row
    finally:
        conn.close()


def pending(user, account_ref, limit=5):
    # Lookups are read-only and checked against the connected broker account.
    # Include older key fingerprints so rotating keys doesn't orphan journal sync.
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM paper_order_requests WHERE user_id = %s
            AND state IN ('pending', 'submitted') ORDER BY updated_at ASC LIMIT %s""",
            (user, limit),
        )
        rows = [_row(cur, r) for r in cur.fetchall()]
        cur.close()
        return rows
    finally:
        conn.close()
