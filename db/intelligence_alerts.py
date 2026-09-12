"""Storage for HSF Intelligence Alerts (state-change notifications) + prefs.

Separate from the static user_alerts system — never touches it. Dedupe is
state-aware + cooldown-windowed (not a hard UNIQUE), so a persistent state won't
re-fire every cron run yet a genuine later re-transition can notify again. Every
function is per-user scoped and non-fatal (returns a safe default when the DB is
unavailable).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from db.engine import get_neon_conn

_DEFAULT_COOLDOWN_HOURS = 6


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hsf_intelligence_alerts (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT,
            fingerprint TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            copy TEXT,
            delivery_status TEXT NOT NULL DEFAULT 'DETECTED',
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            delivered_at TIMESTAMPTZ
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_ia_user_time "
                "ON hsf_intelligence_alerts (user_id, detected_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_ia_user_fp "
                "ON hsf_intelligence_alerts (user_id, fingerprint, detected_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hsf_alert_prefs (
            user_id TEXT PRIMARY KEY,
            prefs JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.commit()
    cur.close()


def recent_fingerprints(user_id: str, *, hours: int = _DEFAULT_COOLDOWN_HOURS) -> Set[str]:
    """Fingerprints already recorded for this user within the cooldown window."""
    if not user_id:
        return set()
    conn = get_neon_conn()
    if conn is None:
        return set()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT fingerprint FROM hsf_intelligence_alerts "
            "WHERE user_id = %s AND detected_at >= NOW() - make_interval(hours => %s)",
            (user_id, int(hours)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return set()
    return {(r["fingerprint"] if isinstance(r, dict) else r[0]) for r in rows}


def record_intelligence_alert(
    *, user_id: str, note: Dict[str, Any], copy: str, fingerprint: str,
    delivery_status: str = "DETECTED",
) -> Optional[int]:
    """Persist one detected intelligence alert. Returns row id or None."""
    if not user_id or not note.get("ticker"):
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO hsf_intelligence_alerts
              (user_id, ticker, event_type, severity, fingerprint, payload, copy, delivery_status)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (user_id, str(note.get("ticker")).upper(), note.get("event_type"),
             note.get("severity"), fingerprint,
             json.dumps(_json_safe(note)), copy, delivery_status),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return int(row[0] if not isinstance(row, dict) else row["id"]) if row else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def update_delivery_status(alert_id: int, status: str) -> bool:
    conn = get_neon_conn()
    if conn is None or not alert_id:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        delivered = ", delivered_at = NOW()" if status == "DELIVERED" else ""
        cur.execute(
            f"UPDATE hsf_intelligence_alerts SET delivery_status = %s{delivered} WHERE id = %s",
            (status, int(alert_id)),
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


def list_recent_intelligence_alerts(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT ticker, event_type, severity, copy, delivery_status, detected_at "
            "FROM hsf_intelligence_alerts WHERE user_id = %s "
            "ORDER BY detected_at DESC LIMIT %s",
            (user_id, int(limit)),
        )
        rows = cur.fetchall() or []
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]


def get_hsf_alert_prefs(user_id: str) -> Optional[Dict[str, Any]]:
    """User's intelligence-alert preferences, or None (caller applies defaults)."""
    if not user_id:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT prefs FROM hsf_alert_prefs WHERE user_id = %s", (user_id,))
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
    prefs = row["prefs"] if isinstance(row, dict) else row[0]
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except json.JSONDecodeError:
            prefs = {}
    return prefs if isinstance(prefs, dict) else {}


def set_hsf_alert_prefs(user_id: str, prefs: Dict[str, Any]) -> bool:
    if not user_id:
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO hsf_alert_prefs (user_id, prefs, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (user_id) DO UPDATE SET prefs = EXCLUDED.prefs, updated_at = NOW()",
            (user_id, json.dumps(prefs or {})),
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


def list_alert_pref_user_ids() -> List[str]:
    """Users who explicitly opted into HSF intelligence alerts (have a prefs
    row). The conservative V1 subscription set. [] if DB unavailable."""
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM hsf_alert_prefs")
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    return [(r["user_id"] if isinstance(r, dict) else r[0]) for r in rows]


def intelligence_alert_health() -> Dict[str, Any]:
    """Read-only health: totals + last evaluation. Actual values only."""
    conn = get_neon_conn()
    if conn is None:
        return {"available": False}
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(detected_at), "
                    "COUNT(*) FILTER (WHERE delivery_status='DELIVERED'), "
                    "COUNT(*) FILTER (WHERE delivery_status='FAILED') "
                    "FROM hsf_intelligence_alerts")
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"available": False}
    g = (lambda i: row[i]) if not isinstance(row, dict) else (lambda i: list(row.values())[i])
    return {"available": True, "total": g(0), "last_detected_at": g(1),
            "delivered": g(2), "failed": g(3)} if row else {"available": True, "total": 0}


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
