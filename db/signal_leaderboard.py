"""Persistence for the signal leaderboard (Strategy Lab).

One row per (signal, horizon) recomputed daily by the cron and read cheaply by
the Strategy Lab UI. Same plain cursor + commit + close pattern as the rest of
db/. Numbers are benchmark-excess forward returns (see analytics.signal_leaderboard).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_leaderboard (
            signal TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            display TEXT,
            avg_excess DOUBLE PRECISION,
            median_excess DOUBLE PRECISION,
            win_rate DOUBLE PRECISION,
            sample_size INTEGER NOT NULL DEFAULT 0,
            runs_used INTEGER NOT NULL DEFAULT 0,
            benchmark TEXT,
            top_n INTEGER,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (signal, horizon_days)
        )
        """
    )
    conn.commit()
    cur.close()


def save_leaderboard(horizon_days: int, rows: List[Dict[str, Any]]) -> int:
    """Upsert the per-signal leaderboard for a horizon. Returns rows written."""
    if not rows:
        return 0
    conn = get_neon_conn()
    if conn is None:
        return 0
    _ensure_schema(conn)
    cur = conn.cursor()
    saved = 0
    for r in rows:
        cur.execute(
            """
            INSERT INTO signal_leaderboard
                (signal, horizon_days, display, avg_excess, median_excess,
                 win_rate, sample_size, runs_used, benchmark, top_n, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (signal, horizon_days) DO UPDATE SET
                display = EXCLUDED.display,
                avg_excess = EXCLUDED.avg_excess,
                median_excess = EXCLUDED.median_excess,
                win_rate = EXCLUDED.win_rate,
                sample_size = EXCLUDED.sample_size,
                runs_used = EXCLUDED.runs_used,
                benchmark = EXCLUDED.benchmark,
                top_n = EXCLUDED.top_n,
                computed_at = NOW()
            """,
            (
                str(r.get("signal")), int(horizon_days), r.get("display"),
                r.get("avg_excess"), r.get("median_excess"), r.get("win_rate"),
                int(r.get("sample_size") or 0), int(r.get("runs_used") or 0),
                r.get("benchmark"), int(r.get("top_n")) if r.get("top_n") else None,
            ),
        )
        saved += 1
    conn.commit()
    cur.close()
    conn.close()
    return saved


def load_leaderboard(horizon_days: int = 5) -> List[Dict[str, Any]]:
    """Per-signal leaderboard for a horizon, best average-excess first."""
    conn = get_neon_conn()
    if conn is None:
        return []
    _ensure_schema(conn)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT signal, horizon_days, display, avg_excess, median_excess,
                   win_rate, sample_size, runs_used, benchmark, top_n, computed_at
            FROM signal_leaderboard
            WHERE horizon_days = %s
            ORDER BY avg_excess DESC NULLS LAST
            """,
            (int(horizon_days),),
        )
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    cur.close()
    conn.close()
    out: List[Dict[str, Any]] = []
    cols = ("signal", "horizon_days", "display", "avg_excess", "median_excess",
            "win_rate", "sample_size", "runs_used", "benchmark", "top_n", "computed_at")
    for r in rows:
        out.append(dict(r) if isinstance(r, dict) else dict(zip(cols, r)))
    return out


def leaderboard_horizons() -> List[int]:
    """Distinct horizons with stored leaderboard data (ascending)."""
    conn = get_neon_conn()
    if conn is None:
        return []
    _ensure_schema(conn)
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT horizon_days FROM signal_leaderboard ORDER BY horizon_days")
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    cur.close()
    conn.close()
    return [int(r["horizon_days"] if isinstance(r, dict) else r[0]) for r in rows]
