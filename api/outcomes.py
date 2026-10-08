"""Outcome Intelligence for API clients: /v1/outcomes/* (Pro, can_track_record).

Thin API layer over analytics.outcome_intelligence (all math lives there).

Caching (process-local, like the rest of hsf-api):
  * stamp    ("stamp")                      COUNT + latest outcome/benchmark write of
                                            the outcome rows; re-read every STAMP_TTL_S.
  * dataset  ("dataset", stamp)             canonical records built once per stamp;
                                            DATASET_TTL_S, served stale while reloading.
  * views    ("view", stamp, name, args)    one computed response per endpoint and
                                            exact parameter set; VIEW_TTL_S[name].
A new frozen or newly matured row changes the stamp, so every cached aggregate is
recomputed within STAMP_TTL_S of the cron writing it. If the database is down the
last good dataset keeps being served and the response says ``stale: true``; with
no dataset at all the endpoint answers 503.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from analytics import outcome_intelligence as oi
from api.store import DatabaseUnavailable
from api.today import TTLCache

log = logging.getLogger("hsf_api.outcomes")

STAMP_TTL_S = 300
DATASET_TTL_S = 6 * 3600
DATASET_STALE_S = 6 * 3600
VIEW_TTL_S = {"summary": 1800, "scores": 1800, "horizons": 1800, "setups": 1800, "timeseries": 1800,
              "symbol": 600, "query": 600}
MAX_ROWS = 100_000   # hard cap on rows loaded; reported as truncated when hit

_cache = TTLCache(256)
_state_lock = threading.Lock()
_state: Dict[str, Any] = {"stamp": None, "stale": False}


def clear_cache() -> None:
    _cache.clear()
    with _state_lock:
        _state.update(stamp=None, stale=False)


def _load_stamp() -> Tuple[str, ...]:
    from db.signal_outcomes import outcome_dataset_stamp

    stamp = outcome_dataset_stamp()
    if stamp is None:
        raise DatabaseUnavailable("database unavailable")
    return stamp


def _current_stamp() -> Tuple[str, ...]:
    """The live stamp, or the last good one (marked stale) when the probe fails."""
    try:
        stamp = _cache.get("stamp", _load_stamp, ttl_s=STAMP_TTL_S)
    except Exception as e:
        with _state_lock:
            last = _state["stamp"]
            _state["stale"] = last is not None
        if last is None:
            raise DatabaseUnavailable("database unavailable") from e
        log.warning(json.dumps({"event": "outcomes_stamp_failed", "error": type(e).__name__}))
        return last
    with _state_lock:
        old = _state["stamp"]
        _state.update(stamp=stamp, stale=False)
    if old is not None and old != stamp:
        _cache.clear_key(("dataset", old))  # views keyed by the old stamp age out by TTL
    return stamp


def _load_dataset() -> Dict[str, Any]:
    from db.signal_outcomes import fetch_outcome_rows

    started = time.monotonic()
    try:
        rows = fetch_outcome_rows(limit=MAX_ROWS)
    except Exception as e:
        log.error(json.dumps({"event": "outcomes_load_failed", "error": type(e).__name__}))
        raise DatabaseUnavailable("database unavailable") from e
    records = oi.build_records(rows)
    out = {"records": records, "rows_loaded": len(rows), "dropped": len(rows) - len(records),
           "truncated": len(rows) >= MAX_ROWS,
           "loaded_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    log.info(json.dumps({"event": "outcomes_dataset_loaded", "rows": len(rows), "records": len(records),
                         "ms": round((time.monotonic() - started) * 1000, 1)}))
    return out


def dataset() -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    stamp = _current_stamp()
    return stamp, _cache.get(("dataset", stamp), _load_dataset, ttl_s=DATASET_TTL_S, stale_s=DATASET_STALE_S)


def _observe(name: str, out: Dict[str, Any], hit: bool, started: float) -> None:
    """One structured line per request: latency, cache, records and coverage. No user data."""
    m = out.get("metrics") or {}
    cov = out.get("coverage") or {}
    log.info(json.dumps({
        "event": "outcomes_request", "endpoint": name, "cache": "hit" if hit else "miss",
        "ms": round((time.monotonic() - started) * 1000, 1),
        "records": out.get("raw_observations"), "unit": out.get("unit"),
        "matured": m.get("matured_count"), "pending": m.get("pending_count"),
        "missing_benchmark": cov.get("missing_benchmark"), "missing_mfe_mae": cov.get("missing_mfe_mae"),
        "stale": out.get("dataset", {}).get("stale"),
    }))


def view(name: str, args: Dict[str, Any], compute: Callable[[List[Dict[str, Any]]], Dict[str, Any]]) -> Dict[str, Any]:
    """Cached computation of one endpoint for one exact parameter set."""
    started = time.monotonic()
    stamp, ds = dataset()
    key = ("view", stamp, name, json.dumps(args, sort_keys=True, default=str))
    computed = {"hit": True}

    def make() -> Dict[str, Any]:
        computed["hit"] = False
        return compute(ds["records"])

    try:
        out = _cache.get(key, make, ttl_s=VIEW_TTL_S.get(name, 600))
    except ValueError:
        raise
    except Exception as e:
        log.error(json.dumps({"event": "outcomes_aggregation_failed", "endpoint": name, "error": type(e).__name__}))
        raise
    with _state_lock:
        stale = bool(_state["stale"])
    out = {**out, "dataset": {"rows_loaded": ds["rows_loaded"], "truncated": ds["truncated"],
                              "loaded_at": ds["loaded_at"], "stale": stale,
                              "source": "signal_outcomes (HSF opportunities)"},
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    _observe(name, out, computed["hit"], started)
    return out


# --------------------------------------------------------------------------- endpoints
def summary(filters: Dict[str, Any], horizon: int, unit: str) -> Dict[str, Any]:
    return view("summary", {"f": filters, "h": horizon, "u": unit},
                lambda recs: oi.summary(recs, filters, horizon=horizon, unit=unit))


def scores(filters: Dict[str, Any], horizon: int, unit: str, buckets: Optional[str]) -> Dict[str, Any]:
    bks = oi.parse_buckets(buckets)
    return view("scores", {"f": filters, "h": horizon, "u": unit, "b": bks},
                lambda recs: oi.by_score(recs, filters, horizon=horizon, unit=unit, buckets=bks))


def horizons(filters: Dict[str, Any], unit: str) -> Dict[str, Any]:
    return view("horizons", {"f": filters, "u": unit}, lambda recs: oi.by_horizon(recs, filters, unit=unit))


def setups(filters: Dict[str, Any], horizon: int, unit: str, group_by: str) -> Dict[str, Any]:
    return view("setups", {"f": filters, "h": horizon, "u": unit, "g": group_by},
                lambda recs: oi.by_group(recs, filters, horizon=horizon, unit=unit, group_by=group_by))


def timeseries(filters: Dict[str, Any], horizon: int, unit: str, period: str) -> Dict[str, Any]:
    return view("timeseries", {"f": filters, "h": horizon, "u": unit, "p": period},
                lambda recs: oi.timeseries(recs, filters, period=period, horizon=horizon, unit=unit))


def symbol(ticker: str, filters: Dict[str, Any], horizon: Optional[int], unit: str,
           page: int, page_size: int) -> Dict[str, Any]:
    return view("symbol", {"t": ticker, "f": filters, "h": horizon, "u": unit, "p": page, "s": page_size},
                lambda recs: oi.symbol(recs, ticker, filters, horizon=horizon, unit=unit,
                                       page=page, page_size=page_size))


def query(filters: Dict[str, Any], horizon: int, unit: str, page: int, page_size: int) -> Dict[str, Any]:
    return view("query", {"f": filters, "h": horizon, "u": unit, "p": page, "s": page_size},
                lambda recs: oi.query(recs, filters, horizon=horizon, unit=unit, page=page, page_size=page_size))


def ai_evidence(ticker: str) -> Optional[str]:
    """Evidence text for the AI layer: the ticker's FULL matured history at every
    horizon, no filters, default unit. None when unavailable or nothing matured;
    never raises (an AI note still works without it)."""
    try:
        v = symbol(ticker, oi.normalize_filters(), None, oi.DEFAULT_UNIT, 1, 1)
    except Exception:
        return None
    return oi.ai_evidence_text(v)
