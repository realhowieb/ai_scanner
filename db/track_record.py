"""Persistence for the signal track record (forward-return performance).

A single small summary row per horizon, recomputed daily by the cron and read
cheaply by the results UI and the morning digest. Follows the proven plain
cursor + commit + close pattern used elsewhere in db/.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_track_record (
            id BIGSERIAL PRIMARY KEY,
            horizon_days INTEGER NOT NULL,
            avg_return DOUBLE PRECISION,
            median_return DOUBLE PRECISION,
            win_rate DOUBLE PRECISION,
            sample_size INTEGER NOT NULL DEFAULT 0,
            runs_used INTEGER NOT NULL DEFAULT 0,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    # Benchmark-relative fields (added after initial ship; the numbers stored in
    # avg/median/win_rate are excess vs this benchmark, top_n candidates each).
    cur.execute("ALTER TABLE signal_track_record ADD COLUMN IF NOT EXISTS benchmark TEXT")
    cur.execute("ALTER TABLE signal_track_record ADD COLUMN IF NOT EXISTS top_n INTEGER")
    # Which ranking produced the top-N: 'breakout' (BreakoutScore) or
    # 'prebreakout' (PreBreakout model). Lets us A/B the two signals.
    cur.execute("ALTER TABLE signal_track_record ADD COLUMN IF NOT EXISTS ranking TEXT")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_track_record_horizon_computed "
        "ON signal_track_record (horizon_days, computed_at DESC)"
    )
    conn.commit()
    cur.close()


def save_daily_excess(ranking: str, horizon_days: int, daily) -> int:
    """Upsert per-scan-day mean excess rows (for the calendar heatmap)."""
    if not daily:
        return 0
    conn = get_neon_conn()
    if conn is None:
        return 0
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_track_daily (
            day DATE NOT NULL,
            ranking TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            avg_excess DOUBLE PRECISION,
            n_picks INTEGER,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (day, ranking, horizon_days)
        )
        """
    )
    saved = 0
    for day, avg_excess, n_picks in daily:
        cur.execute(
            """
            INSERT INTO signal_track_daily (day, ranking, horizon_days, avg_excess, n_picks)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (day, ranking, horizon_days)
            DO UPDATE SET avg_excess = EXCLUDED.avg_excess,
                          n_picks = EXCLUDED.n_picks,
                          computed_at = NOW()
            """,
            (day, ranking, int(horizon_days), float(avg_excess), int(n_picks)),
        )
        saved += 1
    conn.commit()
    cur.close()
    conn.close()
    return saved


def load_daily_excess(ranking: str = "breakout", horizon_days: int = 5, days: int = 120):
    """[(day, avg_excess)] oldest-first for the calendar heatmap."""
    conn = get_neon_conn()
    if conn is None:
        return []
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT day, avg_excess FROM signal_track_daily
            WHERE ranking = %s AND horizon_days = %s
              AND day >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY day ASC
            """,
            (ranking, int(horizon_days), int(days)),
        )
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    cur.close()
    conn.close()
    out = []
    for r in rows:
        day = r["day"] if isinstance(r, dict) else r[0]
        val = r["avg_excess"] if isinstance(r, dict) else r[1]
        out.append((day, float(val) if val is not None else None))
    return out


def save_track_record(
    *,
    horizon_days: int,
    avg_return: Optional[float],
    median_return: Optional[float],
    win_rate: Optional[float],
    sample_size: int,
    runs_used: int,
    benchmark: Optional[str] = None,
    top_n: Optional[int] = None,
    ranking: Optional[str] = None,
) -> bool:
    """Insert a fresh track-record summary row. Returns False if Neon is down.

    avg/median/win_rate are excess-vs-benchmark when `benchmark` is set (win_rate
    then = share of candidates that beat the benchmark). `ranking` records which
    signal chose the top-N ('breakout' or 'prebreakout').
    """
    conn = get_neon_conn()
    if conn is None:
        return False
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signal_track_record
            (horizon_days, avg_return, median_return, win_rate, sample_size,
             runs_used, benchmark, top_n, ranking)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            int(horizon_days),
            avg_return,
            median_return,
            win_rate,
            int(sample_size),
            int(runs_used),
            benchmark,
            int(top_n) if top_n is not None else None,
            ranking,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def load_latest_track_record(
    horizon_days: int = 5, ranking: str = "breakout"
) -> Optional[Dict[str, Any]]:
    """Return the most recent summary for a horizon + ranking, or None.

    Older rows may have NULL ranking (pre-A/B); treat those as 'breakout' so the
    default display keeps working during the transition.
    """
    conn = get_neon_conn()
    if conn is None:
        return None
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT horizon_days, avg_return, median_return, win_rate,
               sample_size, runs_used, computed_at, benchmark, top_n, ranking
        FROM signal_track_record
        WHERE horizon_days = %s
          AND (ranking = %s OR (ranking IS NULL AND %s = 'breakout'))
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        (int(horizon_days), ranking, ranking),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    if isinstance(row, dict):
        return dict(row)
    return {
        "horizon_days": row[0],
        "avg_return": row[1],
        "median_return": row[2],
        "win_rate": row[3],
        "sample_size": row[4],
        "runs_used": row[5],
        "computed_at": row[6],
        "benchmark": row[7],
        "top_n": row[8],
        "ranking": row[9],
    }
