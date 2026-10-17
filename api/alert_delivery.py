"""Push delivery receipts. Accepted means accepted by the push service, not read.

Retries only target the failed device, use an atomic claim, and expire after four
hours. No token or endpoint is stored in this ledger or exposed in the API.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

from api.store import _conn
from db.engine import schema_once


@schema_once
def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("hsf_schema_push_deliveries",))
        cur.execute("""CREATE TABLE IF NOT EXISTS hsf_push_deliveries (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE, device_id BIGINT NOT NULL,
            message JSONB NOT NULL, status TEXT NOT NULL, attempts INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
        cur.execute("CREATE INDEX IF NOT EXISTS hsf_push_deliveries_user ON hsf_push_deliveries (user_id, created_at DESC)")
    conn.commit()


def _run(sql, args, fetch=False):
    conn = _conn()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(sql, args)
            rows = []
            if fetch:
                cols = [d.name for d in cur.description]
                rows = [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in cur.fetchall()]
        conn.commit()
        return rows
    finally:
        conn.close()


def begin(user, device_id, message):
    key = uuid.uuid4().hex
    _run("DELETE FROM hsf_push_deliveries WHERE user_id = %s AND created_at < NOW() - INTERVAL '7 days'", (user,))
    _run("INSERT INTO hsf_push_deliveries (id, user_id, device_id, message, status) VALUES (%s, %s, %s, %s::jsonb, 'sending')",
         (key, user, device_id, json.dumps(message)))
    return key


def finish(key, status):
    _run("UPDATE hsf_push_deliveries SET status = %s, attempts = attempts + 1 WHERE id = %s", (status, key))


def _public(row):
    when = row["created_at"]
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    message = row["message"]
    if isinstance(message, str):
        message = json.loads(message)
    return {"id": row["id"], "title": message.get("body", "HSF alert")[:240], "status": row["status"],
            "attempts": row["attempts"], "at": when.isoformat(),
            "retryable": row["status"] == "failed" and row["attempts"] < 3
            and (dt.datetime.now(dt.timezone.utc) - when).total_seconds() < 14400}


def recent(user):
    return [_public(r) for r in _run("SELECT * FROM hsf_push_deliveries WHERE user_id = %s "
                                    "ORDER BY created_at DESC LIMIT 50", (user,), True)]


def retry(user, key):
    from api import devices, store, webpush
    from api.workspace import is_quiet

    if not webpush.enabled() or not webpush.plan_allows(store.get_account(user)) or is_quiet(user):
        return None
    rows = _run("UPDATE hsf_push_deliveries SET status = 'retrying' WHERE id = %s AND user_id = %s "
                "AND status = 'failed' AND attempts < 3 AND created_at > NOW() - INTERVAL '4 hours' RETURNING *",
                (key, user), True)
    if not rows:
        return None
    row = rows[0]
    device = next((d for d in devices.devices_for_user(user) if d["id"] == row["device_id"]
                   and d.get("provider") == webpush.PROVIDER), None)
    status = "gone"
    if device:
        message = row["message"]
        if isinstance(message, str):
            message = json.loads(message)
        try:
            ok, code = webpush.send(device["token"], message)
            status = "accepted" if ok else "gone" if code in (404, 410) else "unconfirmed" if code is None else "failed"
        except Exception:
            status = "unconfirmed"
    finish(key, status)
    row.update(status=status, attempts=row["attempts"] + 1)
    return _public(row)
