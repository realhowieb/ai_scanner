"""Internal research endpoints (admin only, read-only): the point-in-time HSF
research dataset for ML audits, walk-forward validation and label experiments.

Everything comes from analytics.research_dataset over persisted rows (two bulk
reads per date window, no provider calls). Features and outcomes are always
separate objects: /features/{id} never carries a label, and outcomes appear
only under an explicit `outcome` key when asked for.

Caching: the record set of a date window is cached (TTL + stale window, the
same TTLCache as /v1/today). Finalized dataset versions are immutable rows in
the registry, so caching their metadata can't change reproducibility.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Dict, List, Optional

from analytics import research_dataset as rd
from analytics import research_schema as rs
from api.store import DatabaseUnavailable
from api.today import TTLCache

log = logging.getLogger("hsf_api.research")

DEFAULT_DAYS = 90
MAX_DAYS = 400
MAX_PAGE = 500
RECORDS_TTL_S = 900        # 15 min fresh
RECORDS_STALE_S = 3600     # then served stale for up to an hour while refreshing
_cache = TTLCache(16)


class BadRequest(ValueError):
    """A filter the API refuses (422)."""


def clear_cache() -> None:
    _cache.clear()


def _today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def window(start_date: Optional[dt.date], end_date: Optional[dt.date]) -> tuple:
    """Explicit inclusive date window; default the last DEFAULT_DAYS days."""
    end = end_date or _today()
    start = start_date or (end - dt.timedelta(days=DEFAULT_DAYS - 1))
    if start > end:
        raise BadRequest("start_date must be on or before end_date")
    if (end - start).days + 1 > MAX_DAYS:
        raise BadRequest(f"date window is limited to {MAX_DAYS} days")
    return start, end


def _bounds(start: dt.date, end: dt.date) -> tuple:
    utc = dt.timezone.utc
    return (dt.datetime(start.year, start.month, start.day, tzinfo=utc),
            dt.datetime(end.year, end.month, end.day, tzinfo=utc) + dt.timedelta(days=1))


def load_window(start: dt.date, end: dt.date) -> Dict[str, Any]:
    """Rows, scan records and built records for one date window (uncached)."""
    from db.research_datasets import ResearchDataUnavailable, fetch_opportunity_rows, fetch_scan_records

    t0 = time.perf_counter()
    lo, hi = _bounds(start, end)
    try:
        rows = fetch_opportunity_rows(lo, hi)
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


def records_for(start: dt.date, end: dt.date) -> Dict[str, Any]:
    return _cache.get(("window", start, end), lambda: load_window(start, end), RECORDS_TTL_S, RECORDS_STALE_S)


def _filters(**kw: Any) -> Dict[str, Any]:
    try:
        return rd.normalize_filters(**kw)
    except ValueError as e:
        raise BadRequest(str(e)) from None


def _version_members(dataset_version: Optional[str]) -> Optional[set]:
    if not dataset_version:
        return None
    entry = dataset(dataset_version, verify=False, _raw=True)
    if entry is None:
        raise LookupError("dataset version")
    ids = entry.get("observation_ids") or []
    if isinstance(ids, str):
        ids = json.loads(ids)
    return {int(i) for i in ids}


def observation_out(rec: Dict[str, Any], *, include_outcome: bool = False) -> Dict[str, Any]:
    """Observation + features as separate keys; `outcome` only when asked."""
    snap = rec["features"].to_dict()
    out = {"observation": rec["observation"], "features": snap["features"],
           "feature_schema_version": snap["feature_schema_version"], "feature_join": snap["join"]}
    if include_outcome:
        out["outcome"] = rec["outcome"].to_dict()
    return out


def coverage(start_date: Optional[dt.date], end_date: Optional[dt.date]) -> Dict[str, Any]:
    start, end = window(start_date, end_date)
    w = records_for(start, end)
    records = w["records"]
    return {"window": {"start_date": start.isoformat(), "end_date": end.isoformat()},
            "coverage": rd.coverage(records), "data_quality": rd.quality_report(w["rows"], records),
            "timings": w["timings"]}


def observations(*, limit: int, offset: int, include_outcomes: bool, dataset_version: Optional[str] = None,
                 **filter_kw: Any) -> Dict[str, Any]:
    f = _filters(**filter_kw)
    start, end = window(f["start_date"], f["end_date"])
    f["start_date"], f["end_date"] = start, end
    members = _version_members(dataset_version)
    records = rd.apply_filters(records_for(start, end)["records"], f)
    if members is not None:
        records = [r for r in records if r["observation"]["observation_id"] in members]
    limit = max(1, min(int(limit), MAX_PAGE))
    page = records[offset: offset + limit]
    nxt = offset + limit if offset + limit < len(records) else None
    return {"filters": {**{k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in f.items()},
                        "dataset_version": dataset_version},
            "total": len(records), "limit": limit, "offset": offset, "next_offset": nxt,
            "items": [observation_out(r, include_outcome=include_outcomes) for r in page]}


def _single(observation_id: int) -> Optional[Dict[str, Any]]:
    from db.research_datasets import ResearchDataUnavailable, fetch_opportunity_row, fetch_scan_records

    try:
        found = fetch_opportunity_row(observation_id)
        if found is None:
            return None
        obs_at = rd.to_dt(found["row"].get("fired_at"))
        scans = fetch_scan_records([found["row"].get("ticker")], obs_at - rd.MAX_SCAN_LAG - dt.timedelta(hours=1),
                                   obs_at) if obs_at else []
    except ResearchDataUnavailable as e:
        raise DatabaseUnavailable("database unavailable") from e
    # Ranks need the whole snapshot; the record we want is the one with our id.
    records = rd.build_records(found["snapshot"] or [found["row"]], scans)
    return next((r for r in records if r["observation"]["observation_id"] == int(observation_id)), None)


def observation(observation_id: int, *, include_outcome: bool) -> Optional[Dict[str, Any]]:
    rec = _single(observation_id)
    return observation_out(rec, include_outcome=include_outcome) if rec else None


def features(observation_id: int) -> Optional[Dict[str, Any]]:
    """ONLY what was known at observed_at: the FeatureSnapshot, nothing else."""
    rec = _single(observation_id)
    return rec["features"].to_dict() if rec else None


def _entry_out(e: Dict[str, Any], *, with_ids: bool = False) -> Dict[str, Any]:
    meta = e.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    out = {"dataset_version": e["dataset_version"], "created_at": e.get("created_at"),
           "feature_schema_version": e.get("feature_schema_version"),
           "label_schema_version": e.get("label_schema_version"), "fingerprint": e.get("fingerprint"),
           "observation_count": e.get("observation_count"), "finalized": True, "metadata": meta}
    if with_ids:
        out["observation_ids"] = e.get("observation_ids")
    return out


def datasets() -> Dict[str, Any]:
    from db.research_datasets import ResearchDataUnavailable, list_dataset_versions

    def load() -> List[Dict[str, Any]]:
        try:
            return [_entry_out(e) for e in list_dataset_versions()]
        except ResearchDataUnavailable as e:
            raise DatabaseUnavailable("database unavailable") from e

    items = _cache.get(("datasets",), load, 300, 0)
    return {"items": items, "feature_schema": rs.schema_as_dict(), "label_schema": rs.label_schema_as_dict(),
            "feature_schema_version": rs.FEATURE_SCHEMA_VERSION, "label_schema_version": rs.LABEL_SCHEMA_VERSION}


def dataset(name: str, *, verify: bool, _raw: bool = False) -> Optional[Dict[str, Any]]:
    """A finalized version. `verify=true` rebuilds it from its stored filters and
    members with current data and code, and reports whether the fingerprint
    still matches (it never rewrites the stored version)."""
    from db.research_datasets import ResearchDataUnavailable, get_dataset_version

    try:
        entry = _cache.get(("dataset", name), lambda: get_dataset_version(name), 3600, 0)
    except ResearchDataUnavailable as e:
        raise DatabaseUnavailable("database unavailable") from e
    if entry is None:
        _cache.clear_key(("dataset", name))
        return None
    if _raw:
        return entry
    out = _entry_out(entry)
    if verify:
        out["verification"] = verify_entry(entry)
    return out


def verify_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    meta = entry.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    ids = entry.get("observation_ids") or []
    if isinstance(ids, str):
        ids = json.loads(ids)
    f = dict(meta.get("filters") or {})
    start = dt.date.fromisoformat(f["start_date"]) if f.get("start_date") else None
    end = dt.date.fromisoformat(f["end_date"]) if f.get("end_date") else None
    if start is None or end is None:
        return {"reproducible": None, "reason": "stored version has no explicit date window"}
    w = load_window(start, end)  # uncached: verification must see current data
    built = rd.build_dataset(w["rows"], w["scans"], filters=f, members=ids,
                             feature_schema_version=int(entry["feature_schema_version"]),
                             label_schema_version=int(entry["label_schema_version"]))
    fp = built["metadata"]["fingerprint"]
    return {"reproducible": fp == entry.get("fingerprint"), "rebuilt_fingerprint": fp,
            "stored_fingerprint": entry.get("fingerprint"),
            "rebuilt_count": built["metadata"]["observation_count"],
            "stored_count": entry.get("observation_count")}
