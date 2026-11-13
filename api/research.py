"""Point-in-time research window loader (used by scripts/ml_v4_shadow.py).

Builds analytics.research_dataset records over persisted rows: two bulk reads
per date window, no provider calls.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict

from analytics import research_dataset as rd
from api.store import DatabaseUnavailable

log = logging.getLogger("hsf_api.research")


def _bounds(start: dt.date, end: dt.date) -> tuple:
    utc = dt.timezone.utc
    return (dt.datetime(start.year, start.month, start.day, tzinfo=utc),
            dt.datetime(end.year, end.month, end.day, tzinfo=utc) + dt.timedelta(days=1))


def load_window(start: dt.date, end: dt.date, *, ids=None) -> Dict[str, Any]:
    """Rows, scan records and built records for one date window (uncached)."""
    from db.research_datasets import ResearchDataUnavailable, fetch_opportunity_rows, fetch_scan_records

    t0 = time.perf_counter()
    lo, hi = _bounds(start, end)
    try:
        rows = fetch_opportunity_rows(lo, hi, ids=ids) if ids is not None else fetch_opportunity_rows(lo, hi)
        t1 = time.perf_counter()
        scans = fetch_scan_records({r.get("ticker") for r in rows}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
    except ResearchDataUnavailable as e:
        raise DatabaseUnavailable("database unavailable") from e
    t2 = time.perf_counter()
    records = rd.build_records(rows, scans)
    t3 = time.perf_counter()
    timings = {"opportunity_read_ms": round((t1 - t0) * 1000, 1), "scan_read_ms": round((t2 - t1) * 1000, 1),
               "build_ms": round((t3 - t2) * 1000, 1), "rows": len(rows), "scan_records": len(scans)}
    joins = sum(1 for r in records if r["features"].join.get("status") != rd.JOIN_MATCHED)
    log.info(json.dumps({"event": "research_window_built", "start": start.isoformat(), "end": end.isoformat(),
                         "observations": len(records), "temporal_join_misses": joins, **timings}))
    return {"rows": rows, "scans": scans, "records": records, "timings": timings, "start": start, "end": end}
