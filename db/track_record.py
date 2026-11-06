"""Persistence for the signal track record (forward-return performance).

A single small summary row per horizon, recomputed daily by the cron and read
cheaply by the results UI and the morning digest. Follows the proven plain
cursor + commit + close pattern used elsewhere in db/.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from analytics.track_method import METHOD
from db.engine import get_neon_conn, schema_once

# Separate stored results from the old method that re-scored historical scans.
# Public callers keep using "prebreakout"; legacy rows remain for audit only.
PREBREAKOUT_SNAPSHOT = "prebreakout_snapshot_v1"


def _stored_ranking(ranking, methodology=None):
    if methodology:
        return f"{methodology}:{ranking}"
    return PREBREAKOUT_SNAPSHOT if ranking == "prebreakout" else ranking


def _public_summary(row):
    out = dict(row)
    payload = out.pop("details", None)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or payload.get("methodology") != METHOD:
        return None
    stored = out.get("ranking") or ""
    if stored not in {f"{METHOD}:breakout", f"{METHOD}:prebreakout"}:
        return None
    out.update(payload)
    out['ranking'] = stored.split(':', 1)[1]
    return out


@schema_once
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
    cur.execute("ALTER TABLE signal_track_record ADD COLUMN IF NOT EXISTS details JSONB")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_track_record_horizon_computed "
        "ON signal_track_record (horizon_days, computed_at DESC)"
    )
    conn.commit()
    cur.close()


def save_daily_excess(ranking: str, horizon_days: int, daily, *, methodology=None) -> int:
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
            (day, _stored_ranking(ranking, methodology), int(horizon_days), float(avg_excess), int(n_picks)),
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
            (_stored_ranking(ranking, METHOD), int(horizon_days), int(days)),
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
        # Drop weekend rows written before the weekend-snapshot fix — they
        # duplicate the adjacent trading day and shouldn't paint their own cell.
        try:
            if day is not None and day.weekday() >= 5:
                continue
        except Exception:
            pass
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
    details: Optional[Dict[str, Any]] = None,
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
             runs_used, benchmark, top_n, ranking, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            _stored_ranking(ranking, (details or {}).get("methodology")),
            json.dumps({k: v for k, v in details.items() if k != "daily"}, default=str, allow_nan=False) if details else None,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


_SUMMARY_COLS = ("horizon_days", "avg_return", "median_return", "win_rate", "sample_size",
                 "runs_used", "computed_at", "benchmark", "top_n", "ranking", "details")


def load_latest_track_records() -> List[Dict[str, Any]]:
    """The most recent summary for every horizon + ranking, in one query.

    Only the current methodology namespace and a matching payload are public.
    Older rows remain stored for audit and are never relabeled at read time.
    """
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT ON (horizon_days, COALESCE(ranking, 'breakout'))
                   horizon_days, avg_return, median_return, win_rate,
                   sample_size, runs_used, computed_at, benchmark, top_n,
                   ranking, details
            FROM signal_track_record
            WHERE ranking IN (%s, %s)
            ORDER BY horizon_days, COALESCE(ranking, 'breakout'), computed_at DESC
            """, (_stored_ranking('breakout', METHOD), _stored_ranking('prebreakout', METHOD))
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        rows = []  # old schema/results are not new-method evidence
    finally:
        conn.close()
    return [v for r in rows if (v := _public_summary(r if isinstance(r, dict) else dict(zip(_SUMMARY_COLS, r))))]


def load_latest_track_record(
    horizon_days: int = 5, ranking: str = "breakout"
) -> Optional[Dict[str, Any]]:
    """Return the most recent summary for a horizon + ranking, or None.

    No compatibility fallback to old methodology rows.
    """
    return next((r for r in load_latest_track_records()
                 if r.get("horizon_days") == horizon_days and r.get("ranking") == ranking), None)
