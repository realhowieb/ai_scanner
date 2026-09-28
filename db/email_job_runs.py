"""P1-36 — one summary row per email-job run (counts only, never addresses).

Written by the morning digest, evening wrap and alert runner at the end of each
run; read by analytics.email_health. Recording never raises.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .engine import get_neon_conn

JOBS = ("digest", "evening", "alerts")
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hsf_email_job_runs (
    id BIGSERIAL PRIMARY KEY,
    at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job TEXT NOT NULL,
    stats TEXT NOT NULL
)
"""


def record_email_run(job: str, stats: Dict[str, Any]) -> bool:
    """stats: digest/evening {sent, skipped: {reason: n}}; alerts {fired, emailed, email_failed}."""
    if job not in JOBS:
        return False
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        cur.execute("INSERT INTO hsf_email_job_runs (job, stats) VALUES (%s, %s)",
                    (job, json.dumps(stats, default=str, sort_keys=True)))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def recent_email_runs(days: int = 5) -> List[Dict[str, Any]]:
    """[{at, job, stats}] newest first for the last `days` days; [] if unavailable."""
    try:
        conn = get_neon_conn()
        if conn is None:
            return []
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        conn.commit()
        cur.execute("SELECT at, job, stats FROM hsf_email_job_runs "
                    "WHERE at > NOW() - make_interval(days => %s) ORDER BY at DESC, id DESC",
                    (max(1, int(days)),))
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        return []
    out = []
    for r in rows:
        at, job, stats = list(r.values()) if isinstance(r, dict) else list(r)
        try:
            parsed = json.loads(stats) if stats else {}
        except (TypeError, ValueError):
            parsed = {}
        out.append({"at": at, "job": job, "stats": parsed})
    return out
