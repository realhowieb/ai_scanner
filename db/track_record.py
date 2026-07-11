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
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_track_record_horizon_computed "
        "ON signal_track_record (horizon_days, computed_at DESC)"
    )
    conn.commit()
    cur.close()


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
) -> bool:
    """Insert a fresh track-record summary row. Returns False if Neon is down.

    avg/median/win_rate are excess-vs-benchmark when `benchmark` is set (win_rate
    then = share of candidates that beat the benchmark).
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
             runs_used, benchmark, top_n)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def load_latest_track_record(horizon_days: int = 5) -> Optional[Dict[str, Any]]:
    """Return the most recent summary for a horizon, or None."""
    conn = get_neon_conn()
    if conn is None:
        return None
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT horizon_days, avg_return, median_return, win_rate,
               sample_size, runs_used, computed_at, benchmark, top_n
        FROM signal_track_record
        WHERE horizon_days = %s
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        (int(horizon_days),),
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
    }
