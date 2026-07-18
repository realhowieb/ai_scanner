"""Persistence for alert outcome scoring ("did this alert's fires work out?").

Each fired alert event gets one outcome row once its forward window completes:
entry close on the fire date, whether it hit the target within the horizon, and
the horizon return. The alerts UI aggregates these into a per-alert scorecard.
Plain cursor + commit + close pattern, like the rest of db/.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn

HIT_TARGET_PCT = 5.0
HORIZON_DAYS = 3


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_outcomes (
            event_id BIGINT PRIMARY KEY,
            alert_id BIGINT,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            fired_at TIMESTAMPTZ NOT NULL,
            entry_price DOUBLE PRECISION,
            max_gain_pct DOUBLE PRECISION,
            horizon_return_pct DOUBLE PRECISION,
            hit BOOLEAN,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_alert_outcomes_alert "
        "ON alert_outcomes (alert_id, fired_at DESC)"
    )
    conn.commit()
    cur.close()


def list_unscored_events(min_age_days: int = 6, limit: int = 500) -> List[Dict[str, Any]]:
    """Fired events with a ticker, old enough to have a complete forward window,
    that don't have an outcome row yet."""
    conn = get_neon_conn()
    if conn is None:
        return []
    _ensure_schema(conn)
    cur = conn.cursor()
    # DISTINCT ON (alert_id, ticker, day): forced/manual runs bypass the alert
    # throttle and can record the same fire several times in one day; scoring
    # each duplicate would overweight it in the scorecard. Keep the earliest
    # event per (alert, ticker, day) and mark the rest scored via the same
    # outcome row rule below (they simply never get selected).
    cur.execute(
        """
        SELECT DISTINCT ON (e.alert_id, e.ticker, (e.fired_at AT TIME ZONE 'UTC')::date)
               e.id, e.alert_id, e.user_id, e.ticker, e.fired_at
        FROM alert_events e
        LEFT JOIN alert_outcomes o ON o.event_id = e.id
        WHERE o.event_id IS NULL
          AND e.ticker IS NOT NULL
          AND e.fired_at < NOW() - make_interval(days => %s)
          AND NOT EXISTS (
              SELECT 1 FROM alert_outcomes o2
              WHERE o2.alert_id IS NOT DISTINCT FROM e.alert_id
                AND o2.ticker = UPPER(e.ticker)
                AND (o2.fired_at AT TIME ZONE 'UTC')::date
                    = (e.fired_at AT TIME ZONE 'UTC')::date
          )
        ORDER BY e.alert_id, e.ticker, (e.fired_at AT TIME ZONE 'UTC')::date,
                 e.fired_at ASC
        LIMIT %s
        """,
        (int(min_age_days), int(limit)),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(dict(r))
        else:
            out.append(
                {"id": r[0], "alert_id": r[1], "user_id": r[2], "ticker": r[3], "fired_at": r[4]}
            )
    return out


def save_outcome(
    *,
    event_id: int,
    alert_id: Optional[int],
    user_id: str,
    ticker: str,
    fired_at,
    entry_price: Optional[float],
    max_gain_pct: Optional[float],
    horizon_return_pct: Optional[float],
    hit: Optional[bool],
) -> bool:
    conn = get_neon_conn()
    if conn is None:
        return False
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alert_outcomes
            (event_id, alert_id, user_id, ticker, fired_at, entry_price,
             max_gain_pct, horizon_return_pct, hit)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING
        """,
        (
            int(event_id),
            int(alert_id) if alert_id is not None else None,
            user_id,
            (ticker or "").upper(),
            fired_at,
            entry_price,
            max_gain_pct,
            horizon_return_pct,
            hit,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def outcome_sequences_for_user(user_id: str, last_n: int = 10) -> Dict[int, List[bool]]:
    """Per-alert hit/miss sequence, oldest-first (for the scorecard dot strip)."""
    conn = get_neon_conn()
    if conn is None:
        return {}
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT alert_id, hit FROM (
            SELECT alert_id, hit, fired_at,
                   ROW_NUMBER() OVER (PARTITION BY alert_id ORDER BY fired_at DESC) AS rn
            FROM alert_outcomes
            WHERE user_id = %s AND alert_id IS NOT NULL AND hit IS NOT NULL
        ) recent
        WHERE rn <= %s
        ORDER BY alert_id, fired_at ASC
        """,
        (user_id, int(last_n)),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out: Dict[int, List[bool]] = {}
    for r in rows:
        aid = int(r["alert_id"] if isinstance(r, dict) else r[0])
        hit = bool(r["hit"] if isinstance(r, dict) else r[1])
        out.setdefault(aid, []).append(hit)
    return out


def scorecards_by_type(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate scored outcomes by alert TYPE — the 'which kind of alert pays
    off' edge scorecard.

    Joins outcomes to user_alerts for the type. Returns
    {alert_type: {"fires": n, "hits": k, "hit_rate": k/n,
                  "avg_return_pct": x, "avg_max_gain_pct": y}} for each type with
    at least one scored (hit IS NOT NULL) outcome. Deleted alerts drop out of the
    join and simply aren't counted.
    """
    conn = get_neon_conn()
    if conn is None:
        return {}
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.alert_type AS alert_type,
               COUNT(*) AS fires,
               COUNT(*) FILTER (WHERE o.hit) AS hits,
               AVG(o.horizon_return_pct) AS avg_return_pct,
               AVG(o.max_gain_pct) AS avg_max_gain_pct
        FROM alert_outcomes o
        JOIN user_alerts a ON a.id = o.alert_id
        WHERE o.user_id = %s AND o.hit IS NOT NULL
        GROUP BY a.alert_type
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = r if isinstance(r, dict) else {
            "alert_type": r[0], "fires": r[1], "hits": r[2],
            "avg_return_pct": r[3], "avg_max_gain_pct": r[4],
        }
        fires = int(d["fires"] or 0)
        if fires <= 0:
            continue
        hits = int(d["hits"] or 0)
        out[str(d["alert_type"])] = {
            "fires": fires,
            "hits": hits,
            "hit_rate": hits / fires,
            "avg_return_pct": float(d["avg_return_pct"]) if d["avg_return_pct"] is not None else None,
            "avg_max_gain_pct": float(d["avg_max_gain_pct"]) if d["avg_max_gain_pct"] is not None else None,
        }
    return out


def scorecards_for_user(user_id: str, last_n: int = 10) -> Dict[int, Dict[str, Any]]:
    """Per-alert aggregate over each alert's most recent scored fires.

    Returns {alert_id: {"fires": n, "hits": k, "avg_return_pct": x}} for alerts
    with at least one scored outcome. Unscoreable rows (hit IS NULL) excluded.
    """
    conn = get_neon_conn()
    if conn is None:
        return {}
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT alert_id,
               COUNT(*) AS fires,
               COUNT(*) FILTER (WHERE hit) AS hits,
               AVG(horizon_return_pct) AS avg_return_pct
        FROM (
            SELECT alert_id, hit, horizon_return_pct,
                   ROW_NUMBER() OVER (PARTITION BY alert_id ORDER BY fired_at DESC) AS rn
            FROM alert_outcomes
            WHERE user_id = %s AND alert_id IS NOT NULL AND hit IS NOT NULL
        ) recent
        WHERE rn <= %s
        GROUP BY alert_id
        """,
        (user_id, int(last_n)),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict):
            out[int(r["alert_id"])] = {
                "fires": int(r["fires"]),
                "hits": int(r["hits"]),
                "avg_return_pct": float(r["avg_return_pct"]) if r["avg_return_pct"] is not None else None,
            }
        else:
            out[int(r[0])] = {
                "fires": int(r[1]),
                "hits": int(r[2]),
                "avg_return_pct": float(r[3]) if r[3] is not None else None,
            }
    return out
