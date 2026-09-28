"""P1-42 — admin audit log: who changed what on which account, and when.

Written by the Admin Users actions (create user, update plan / active, mark
email verified, grant / revoke admin). Recording never raises, so a logging
problem can't block or undo the admin action itself. Never store passwords,
hashes or tokens in `detail`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .engine import get_neon_conn

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hsf_admin_events (
    id BIGSERIAL PRIMARY KEY,
    at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    detail TEXT
)
"""
_FORBIDDEN_KEYS = ("password", "hash", "token", "secret")


def _clean(detail: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (detail or {}).items()
            if not any(f in str(k).lower() for f in _FORBIDDEN_KEYS)}


def record_admin_event(actor: Any, action: str, target: Any,
                       detail: Optional[Dict[str, Any]] = None) -> bool:
    """Append one event. True when saved; False (never an exception) otherwise."""
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        cur.execute(
            "INSERT INTO hsf_admin_events (actor, action, target, detail) VALUES (%s, %s, %s, %s)",
            (str(actor or "unknown").strip().lower(), str(action), str(target or "").strip().lower(),
             json.dumps(_clean(detail), default=str, sort_keys=True)),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def recent_admin_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest first: [{at, actor, action, target, detail}]."""
    try:
        conn = get_neon_conn()
        if conn is None:
            return []
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        conn.commit()
        cur.execute("SELECT at, actor, action, target, detail FROM hsf_admin_events "
                    "ORDER BY at DESC, id DESC LIMIT %s", (max(1, int(limit)),))
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        return []
    out = []
    for r in rows:
        at, actor, action, target, detail = list(r.values()) if isinstance(r, dict) else list(r)
        try:
            parsed = json.loads(detail) if detail else {}
        except (TypeError, ValueError):
            parsed = {}
        out.append({"at": at, "actor": actor, "action": action, "target": target, "detail": parsed})
    return out


def describe(detail: Dict[str, Any]) -> str:
    """Short human text for the admin table, e.g. 'tier: basic → pro'."""
    parts = []
    for k, v in sorted((detail or {}).items()):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            parts.append(f"{k}: {v[0]} → {v[1]}")
        else:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)
