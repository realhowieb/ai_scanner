"""Run 59 — append-only store for system-health snapshots.

A separate operational table (`hsf_system_health`); it never touches research
observations or outcomes. The System Health workflow appends one snapshot per
run; the Streamlit admin view and the next health run read the latest one.
Non-fatal: every function returns None/False when the database is unavailable.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from db.hsf_observations import _ph, _resolve_conn


def _ensure(c, is_sqlite: bool) -> None:
    cur = c.cursor()
    if is_sqlite:
        cur.execute("CREATE TABLE IF NOT EXISTS hsf_system_health ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, generated_at TEXT NOT NULL, "
                    "system_status TEXT NOT NULL, record TEXT NOT NULL, "
                    "created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    else:
        cur.execute("CREATE TABLE IF NOT EXISTS hsf_system_health ("
                    "id SERIAL PRIMARY KEY, generated_at TIMESTAMPTZ NOT NULL, "
                    "system_status TEXT NOT NULL, record JSONB NOT NULL, "
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
    c.commit()
    cur.close()


def save_snapshot(report: Dict[str, Any], *, conn=None) -> bool:
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return False
    try:
        _ensure(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cast = "" if is_sqlite else "::jsonb"
        cur.execute(f"INSERT INTO hsf_system_health (generated_at, system_status, record) "
                    f"VALUES ({ph},{ph},{ph}{cast})",
                    (report.get("generated_at"), report.get("system_status"),
                     json.dumps(report, default=str)))
        c.commit()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def load_latest(*, conn=None) -> Optional[Dict[str, Any]]:
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return None
    try:
        _ensure(c, is_sqlite)
        cur = c.cursor()
        cur.execute("SELECT record FROM hsf_system_health ORDER BY generated_at DESC, id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        val = list(row.values())[0] if isinstance(row, dict) else row[0]
        return val if isinstance(val, dict) else json.loads(val)
    except Exception:
        return None
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass
