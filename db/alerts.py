"""Per-user price/scan alerts: storage + CRUD + fired-event log.

Alert types (alert_type column):
- 'breakout'  : fire when any ticker's BreakoutScore >= threshold in the latest
                snapshot (optionally limited to the user's watchlist).
- 'watchlist' : fire when any of the user's watchlist tickers appears in the
                latest scan results (i.e. it passed the scan).
- 'price'     : fire when ticker's Last crosses `direction` ('above'/'below')
                the `threshold` price.
- 'ema_cross' : fire when EMA 9 crosses EMA 21 for `ticker`; direction is
                'bullish' or 'bearish'.

Backed by Neon/PostgreSQL (psycopg, dict_row). Mirrors the proven connection
pattern in db/watchlists.py: plain cursor + conn.commit() + cur.close(). Rows
are normalized to plain dicts so callers don't depend on the cursor row factory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn

# 'move' (abs % change today >= threshold) and 'rvol' (today's volume vs 20d
# avg >= threshold) are evaluated by the real-time worker, not the cron.
ALERT_TYPES = ("breakout", "watchlist", "price", "move", "rvol", "ema_cross")


def _ensure_alerts_schema(conn) -> None:
    """Create the alert tables/indexes if they don't exist (idempotent)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_alerts (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            ticker TEXT,
            threshold DOUBLE PRECISION,
            direction TEXT,
            watchlist_only BOOLEAN NOT NULL DEFAULT FALSE,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            last_fired_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_alerts_user ON user_alerts(user_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_alerts_enabled ON user_alerts(enabled)"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_events (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            alert_id BIGINT,
            ticker TEXT,
            message TEXT NOT NULL,
            fired_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_alert_events_user "
        "ON alert_events(user_id, fired_at DESC)"
    )
    conn.commit()
    cur.close()


def _get_conn():
    """Return a fresh Neon connection with the alert schema ensured.

    Raises RuntimeError when Neon is not configured/reachable so the UI can
    fall back gracefully instead of silently receiving None.
    """
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")
    _ensure_alerts_schema(conn)
    return conn


def _row_to_alert(r: Any) -> Dict[str, Any]:
    """Normalize a row (dict_row or tuple) into a plain alert dict."""
    if isinstance(r, dict):
        return {
            "id": r.get("id"),
            "user_id": r.get("user_id"),
            "alert_type": r.get("alert_type"),
            "ticker": r.get("ticker"),
            "threshold": r.get("threshold"),
            "direction": r.get("direction"),
            "watchlist_only": bool(r.get("watchlist_only")),
            "enabled": bool(r.get("enabled")),
            "last_fired_at": r.get("last_fired_at"),
            "created_at": r.get("created_at"),
        }
    return {
        "id": r[0],
        "user_id": r[1],
        "alert_type": r[2],
        "ticker": r[3],
        "threshold": r[4],
        "direction": r[5],
        "watchlist_only": bool(r[6]),
        "enabled": bool(r[7]),
        "last_fired_at": r[8],
        "created_at": r[9] if len(r) > 9 else None,
    }


_SELECT_COLS = (
    "id, user_id, alert_type, ticker, threshold, direction, "
    "watchlist_only, enabled, last_fired_at, created_at"
)


def create_alert(
    user_id: str,
    alert_type: str,
    *,
    ticker: Optional[str] = None,
    threshold: Optional[float] = None,
    direction: Optional[str] = None,
    watchlist_only: bool = False,
) -> None:
    """Create an alert for the user. alert_type must be in ALERT_TYPES."""
    if alert_type not in ALERT_TYPES:
        raise ValueError(f"Unknown alert_type: {alert_type!r}")
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_alerts
            (user_id, alert_type, ticker, threshold, direction, watchlist_only)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            alert_type,
            (ticker or "").upper() or None,
            threshold,
            direction,
            bool(watchlist_only),
        ),
    )
    conn.commit()
    cur.close()


def list_alerts(user_id: str) -> List[Dict[str, Any]]:
    """Return all alerts owned by user_id, newest first."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        f"SELECT {_SELECT_COLS} FROM user_alerts WHERE user_id = %s "
        "ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    return [_row_to_alert(r) for r in rows]


def list_all_enabled_alerts() -> List[Dict[str, Any]]:
    """Return every enabled alert across all users (for the headless runner)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        f"SELECT {_SELECT_COLS} FROM user_alerts WHERE enabled = TRUE "
        "ORDER BY user_id ASC, created_at DESC, id DESC"
    )
    rows = cur.fetchall()
    cur.close()
    return [_row_to_alert(r) for r in rows]


def delete_alert(alert_id: int, user_id: str) -> None:
    """Delete an alert (ownership-checked)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM user_alerts WHERE id = %s AND user_id = %s",
        (alert_id, user_id),
    )
    conn.commit()
    cur.close()


def set_alert_enabled(alert_id: int, user_id: str, enabled: bool) -> None:
    """Enable/disable an alert (ownership-checked)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE user_alerts SET enabled = %s WHERE id = %s AND user_id = %s",
        (bool(enabled), alert_id, user_id),
    )
    conn.commit()
    cur.close()


def mark_alert_fired(alert_id: int) -> None:
    """Stamp last_fired_at = NOW() for throttling repeat sends."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE user_alerts SET last_fired_at = NOW() WHERE id = %s", (alert_id,))
    conn.commit()
    cur.close()


def record_alert_event(
    user_id: str, alert_id: Optional[int], ticker: Optional[str], message: str
) -> None:
    """Append a fired-event row for the in-app 'Recently triggered' feed."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alert_events (user_id, alert_id, ticker, message)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, alert_id, (ticker or "").upper() or None, message),
    )
    conn.commit()
    cur.close()


def list_recent_events(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the user's most recent fired alert events."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, alert_id, ticker, message, fired_at FROM alert_events "
        "WHERE user_id = %s ORDER BY fired_at DESC LIMIT %s",
        (user_id, int(limit)),
    )
    rows = cur.fetchall()
    cur.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(
                {
                    "id": r.get("id"),
                    "alert_id": r.get("alert_id"),
                    "ticker": r.get("ticker"),
                    "message": r.get("message"),
                    "fired_at": r.get("fired_at"),
                }
            )
        else:
            out.append(
                {
                    "id": r[0],
                    "alert_id": r[1],
                    "ticker": r[2],
                    "message": r[3],
                    "fired_at": r[4],
                }
            )
    return out
