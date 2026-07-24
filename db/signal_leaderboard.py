"""Persistence for the signal leaderboard (Strategy Lab).

One row per (signal, horizon, entry_mode) recomputed daily by the cron and read
cheaply by the Strategy Lab UI. entry_mode is 'close' (enter at the signal-day
close) or 'open' (enter at the signal-day open — the realistic fill for an early
signal). Same plain cursor + commit + close pattern as the rest of db/. Numbers
are benchmark-excess forward returns (see analytics.signal_leaderboard).
"""
from __future__ import annotations

from typing import Any, Dict, List

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_leaderboard (
            signal TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            entry_mode TEXT NOT NULL DEFAULT 'close',
            display TEXT,
            avg_excess DOUBLE PRECISION,
            median_excess DOUBLE PRECISION,
            win_rate DOUBLE PRECISION,
            sample_size INTEGER NOT NULL DEFAULT 0,
            runs_used INTEGER NOT NULL DEFAULT 0,
            benchmark TEXT,
            top_n INTEGER,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (signal, horizon_days, entry_mode)
        )
        """
    )
    conn.commit()
    # Migrate installs that predate entry_mode: add the column, then move the
    # primary key from (signal, horizon_days) to include entry_mode so both
    # entry modes can coexist. Each step is best-effort and independently
    # committed so a partial/older state can't poison the rest.
    for stmt in (
        "ALTER TABLE signal_leaderboard "
        "ADD COLUMN IF NOT EXISTS entry_mode TEXT NOT NULL DEFAULT 'close'",
    ):
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception:
            conn.rollback()
    # If the PK doesn't already include entry_mode, swap it in.
    try:
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.key_column_usage
            WHERE table_name = 'signal_leaderboard'
              AND constraint_name = 'signal_leaderboard_pkey'
              AND column_name = 'entry_mode'
            """
        )
        row = cur.fetchone()
        has_it = int((row[0] if not isinstance(row, dict) else list(row.values())[0]) or 0) > 0
        if not has_it:
            cur.execute("ALTER TABLE signal_leaderboard DROP CONSTRAINT IF EXISTS signal_leaderboard_pkey")
            cur.execute(
                "ALTER TABLE signal_leaderboard "
                "ADD PRIMARY KEY (signal, horizon_days, entry_mode)"
            )
            conn.commit()
    except Exception:
        conn.rollback()
    cur.close()


def save_leaderboard(
    horizon_days: int, rows: List[Dict[str, Any]], entry_mode: str = "close"
) -> int:
    """Upsert the per-signal leaderboard for a (horizon, entry_mode). Returns rows written."""
    if not rows:
        return 0
    conn = get_neon_conn()
    if conn is None:
        return 0
    _ensure_schema(conn)
    mode = "open" if str(entry_mode).lower() == "open" else "close"
    cur = conn.cursor()
    saved = 0
    for r in rows:
        cur.execute(
            """
            INSERT INTO signal_leaderboard
                (signal, horizon_days, entry_mode, display, avg_excess, median_excess,
                 win_rate, sample_size, runs_used, benchmark, top_n, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (signal, horizon_days, entry_mode) DO UPDATE SET
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
                str(r.get("signal")), int(horizon_days), mode, r.get("display"),
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


def load_leaderboard(horizon_days: int = 5, entry_mode: str = "close") -> List[Dict[str, Any]]:
    """Per-signal leaderboard for a (horizon, entry_mode), best average-excess first."""
    conn = get_neon_conn()
    if conn is None:
        return []
    _ensure_schema(conn)
    mode = "open" if str(entry_mode).lower() == "open" else "close"
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT signal, horizon_days, entry_mode, display, avg_excess, median_excess,
                   win_rate, sample_size, runs_used, benchmark, top_n, computed_at
            FROM signal_leaderboard
            WHERE horizon_days = %s AND entry_mode = %s
            ORDER BY avg_excess DESC NULLS LAST
            """,
            (int(horizon_days), mode),
        )
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    cur.close()
    conn.close()
    out: List[Dict[str, Any]] = []
    cols = ("signal", "horizon_days", "entry_mode", "display", "avg_excess", "median_excess",
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
