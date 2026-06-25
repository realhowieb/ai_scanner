"""Per-user daily AI usage tracking for cost control.

Best-effort: if the DB is unavailable, calls are allowed (fail-open) so a DB
blip never blocks a paying user. The global AI_ENABLED switch is the hard stop.
"""
from __future__ import annotations

try:
    from db.engine import get_neon_conn
except Exception:
    get_neon_conn = None  # type: ignore[assignment]


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_usage (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            feature TEXT,
            called_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute("ALTER TABLE ai_usage ADD COLUMN IF NOT EXISTS feature TEXT")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_usage_user ON ai_usage (username, called_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_usage_feature ON ai_usage (feature, called_at)")
    conn.commit()
    cur.close()


def ai_calls_today(username: str) -> int:
    """Count this user's AI calls in the last 24h. Returns 0 on any failure."""
    user = (username or "").strip().lower()
    if get_neon_conn is None or not user:
        return 0
    try:
        conn = get_neon_conn()
        if conn is None:
            return 0
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM ai_usage WHERE username = %s AND called_at > NOW() - INTERVAL '24 hours'",
            (user,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return 0
        return int(row[0] if isinstance(row, (tuple, list)) else list(row.values())[0])
    except Exception:
        return 0


def record_ai_call(username: str, feature: str | None = None) -> None:
    """Log one AI call for this user. Best-effort; never raises."""
    user = (username or "").strip().lower()
    if get_neon_conn is None or not user:
        return
    try:
        conn = get_neon_conn()
        if conn is None:
            return
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ai_usage (username, feature) VALUES (%s, %s)",
            (user, (feature or "unknown")[:64]),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def feature_usage_counts(days: int = 30) -> list[tuple[str, int]]:
    """Return [(feature, count)] over the last N days, most-used first."""
    if get_neon_conn is None:
        return []
    try:
        conn = get_neon_conn()
        if conn is None:
            return []
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(feature, 'unknown') AS feature, COUNT(*) AS n
            FROM ai_usage
            WHERE called_at > NOW() - (%s || ' days')::interval
            GROUP BY 1
            ORDER BY 2 DESC
            """,
            (str(int(days)),),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        out: list[tuple[str, int]] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append((str(r.get("feature")), int(r.get("n", 0))))
            else:
                out.append((str(r[0]), int(r[1])))
        return out
    except Exception:
        return []
