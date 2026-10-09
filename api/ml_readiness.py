"""ML v4 data readiness (admin only, read-only): GET /v1/ml/readiness.

Reads the whole opportunity dataset with slim projections (two bulk queries:
opportunity rows without their feature payload, and the scan-record index the
backward join needs), builds ``analytics.ml_readiness`` and returns its stable
API view. Aggregates only: no tickers, observation ids or per-row returns.

Cached 30 minutes, then served stale for up to 2 hours while one background
reload runs: readiness moves once per maturation cycle, and every read costs
Neon egress.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict

from analytics import ml_readiness as mr
from analytics import research_dataset as rd
from api.store import DatabaseUnavailable
from api.today import TTLCache

log = logging.getLogger("hsf_api.ml_readiness")

TTL_S = 1800
STALE_S = 7200
_cache = TTLCache(2)


def clear_cache() -> None:
    _cache.clear()


def _bounds(now: dt.datetime) -> tuple:
    s = mr.DATASET_START
    return dt.datetime(s.year, s.month, s.day, tzinfo=dt.timezone.utc), now + dt.timedelta(minutes=1)


def build(now: dt.datetime | None = None) -> Dict[str, Any]:
    """The full readiness report (uncached)."""
    from db.research_datasets import ResearchDataUnavailable, fetch_readiness_rows, fetch_scan_index

    now = now or dt.datetime.now(dt.timezone.utc)
    lo, hi = _bounds(now)
    t0 = time.perf_counter()
    try:
        rows = fetch_readiness_rows(lo, hi)
        scans = fetch_scan_index({r.get("ticker") for r in rows}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
    except ResearchDataUnavailable as e:
        raise DatabaseUnavailable("database unavailable") from e
    t1 = time.perf_counter()
    report = mr.build_report(rows, scans, now=now)
    log.info(json.dumps({"event": "ml_readiness_built", "rows": len(rows), "scan_index_rows": len(scans),
                         "read_ms": round((t1 - t0) * 1000, 1),
                         "build_ms": round((time.perf_counter() - t1) * 1000, 1), "status": report["status"]}))
    return report


def readiness() -> Dict[str, Any]:
    return _cache.get(("readiness",), lambda: mr.api_view(build()), TTL_S, STALE_S)
