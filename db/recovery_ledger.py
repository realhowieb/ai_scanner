"""Run 60 — append-only recovery ledger (`hsf_recovery_events`).

Every recovery decision that executes, is skipped, hits a cooldown or attempt
limit, escalates, or trips / resets the circuit breaker is appended here. There
is deliberately no update or delete function: the ledger is the audit trail for
autonomy. Separate from all research tables. Non-fatal on DB errors.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Mapping

from db.hsf_observations import _ph, _resolve_conn


def _ensure(c, is_sqlite: bool) -> None:
    cur = c.cursor()
    if is_sqlite:
        cur.execute("CREATE TABLE IF NOT EXISTS hsf_recovery_events ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, recovery_id TEXT NOT NULL, "
                    "incident_id TEXT, action TEXT, result TEXT NOT NULL, started_at TEXT NOT NULL, "
                    "record TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    else:
        cur.execute("CREATE TABLE IF NOT EXISTS hsf_recovery_events ("
                    "id SERIAL PRIMARY KEY, recovery_id TEXT NOT NULL, incident_id TEXT, action TEXT, "
                    "result TEXT NOT NULL, started_at TIMESTAMPTZ NOT NULL, record JSONB NOT NULL, "
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
    c.commit()
    cur.close()


def append_event(event: Mapping[str, Any], *, conn=None) -> bool:
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        raise RuntimeError("recovery ledger unavailable (no database)")
    try:
        _ensure(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cast = "" if is_sqlite else "::jsonb"
        cur.execute(f"INSERT INTO hsf_recovery_events (recovery_id, incident_id, action, result, started_at, record) "
                    f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}{cast})",
                    (event.get("recovery_id"), event.get("incident_id"), event.get("action"),
                     event.get("result"), event.get("started_at"), json.dumps(dict(event), default=str)))
        c.commit()
        cur.close()
        return True
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def list_recent(*, days: int = 7, conn=None) -> List[Dict[str, Any]]:
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        raise RuntimeError("recovery ledger unavailable (no database)")
    try:
        _ensure(c, is_sqlite)
        cur = c.cursor()
        since = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).isoformat()
        cur.execute(f"SELECT record FROM hsf_recovery_events WHERE started_at >= {_ph(is_sqlite)} "
                    f"ORDER BY started_at, id", (since,))
        rows = cur.fetchall() or []
        cur.close()
        out = []
        for r in rows:
            v = list(r.values())[0] if isinstance(r, dict) else r[0]
            out.append(v if isinstance(v, dict) else json.loads(v))
        return out
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass
