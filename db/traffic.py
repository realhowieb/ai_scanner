"""Payload estimates, never wire/billing measurements. No SQL or values are logged."""
from __future__ import annotations

import contextvars
import functools
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger("hsf.db_traffic")
_current = contextvars.ContextVar("db_traffic", default=None)


def _limit(name, default):
    try:
        return max(0, float(os.getenv(name, default)))
    except ValueError:
        return float(default)


def size(value):
    """Approximate UTF-8 application payload size; no serialized values retained."""
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, dict):
        return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8"))
    if isinstance(value, (tuple, list)):
        return sum(size(v) for v in value)
    return len(str(value).encode("utf-8"))


def row_size(row):
    # Row dictionary keys are column metadata, but keys inside JSONB cells are payload.
    return sum(size(v) for v in row.values()) if isinstance(row, dict) else size(row)


@dataclass
class Metrics:
    calls: int = 0
    errors: int = 0
    rows_returned: int = 0
    rows_fetched: int = 0
    db_to_client_payload_bytes: int = 0
    client_to_db_payload_bytes: int = 0
    execute_ms: float = 0
    fetch_ms: float = 0
    connections_opened: int = 0
    connections_reused: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stale_hits: int = 0
    cache_activity: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **counts):
        with self._lock:
            for key, value in counts.items():
                setattr(self, key, getattr(self, key) + value)

    def snapshot(self):
        with self._lock:
            return {k: ({name: dict(counts) for name, counts in v.items()} if k == "cache_activity"
                        else round(v, 3) if isinstance(v, float) else v)
                    for k, v in vars(self).items() if k != "_lock"}


def count(**counts):
    metrics = _current.get()
    if metrics is not None:
        metrics.add(**counts)


def cache(kind, *, hits=0, misses=0, stale_hits=0):
    """Static cache families keep symbol counts separate from API lookup counts."""
    metrics = _current.get()
    if metrics is None:
        return
    metrics.add(cache_hits=hits, cache_misses=misses, cache_stale_hits=stale_hits)
    with metrics._lock:
        counts = metrics.cache_activity.setdefault(kind, {'hits': 0, 'misses': 0, 'stale_hits': 0})
        counts['hits'] += hits
        counts['misses'] += misses
        counts['stale_hits'] += stale_hits


@contextmanager
def scope(label):
    """Use only code-owned labels (route templates/module names), never user input."""
    metrics = Metrics()
    token = _current.set(metrics)
    started = time.perf_counter()
    try:
        yield metrics
    finally:
        _current.reset(token)
        elapsed_ms = (time.perf_counter()-started)*1000
        summary = dict(event="db_traffic", scope=label, elapsed_ms=round(elapsed_ms, 2),
                       **metrics.snapshot())
        exceeded = (metrics.db_to_client_payload_bytes > _limit("DB_TRAFFIC_WARN_BYTES", "100000000") or
                    metrics.client_to_db_payload_bytes > _limit("DB_TRAFFIC_WARN_UPLOAD_BYTES", "100000000") or
                    metrics.calls > _limit("DB_TRAFFIC_WARN_CALLS", "10000") or
                    elapsed_ms > _limit("DB_JOB_WARN_MS", "900000"))
        summary["budget_exceeded"] = exceeded
        log.log(logging.WARNING if exceeded else logging.INFO, json.dumps(summary))
        # Action annotation contains only aggregate numbers and fixed text.
        if exceeded and os.getenv("GITHUB_ACTIONS") == "true":
            print("::warning title=Database traffic budget::Estimated payload/call budget exceeded; inspect db_traffic summary.")


class TrafficCursorMixin:
    def execute(self, query, params=None, **kwargs):
        metrics = _current.get()
        if metrics is None:
            return super().execute(query, params, **kwargs)
        # SQL is counted in outgoing bytes but never emitted.
        metrics.add(calls=1, client_to_db_payload_bytes=size(query) + size(params))
        started = time.perf_counter()
        try:
            result = super().execute(query, params, **kwargs)
            if getattr(self, "description", None) is not None:
                metrics.add(rows_returned=max(0, getattr(self, "rowcount", 0)))
            return result
        except BaseException:
            metrics.add(errors=1)
            raise
        finally:
            elapsed = (time.perf_counter()-started)*1000
            metrics.add(execute_ms=elapsed)
            if elapsed > _limit("DB_QUERY_WARN_MS", "2000"):
                log.warning(json.dumps({"event": "db_query_slow", "execute_ms": round(elapsed, 2)}))

    def executemany(self, query, params_seq, **kwargs):
        metrics = _current.get()
        if metrics is None:
            return super().executemany(query, params_seq, **kwargs)
        def measured():
            for params in params_seq:
                metrics.add(calls=1, client_to_db_payload_bytes=size(query) + size(params))
                yield params
        started = time.perf_counter()
        try:
            return super().executemany(query, measured(), **kwargs)
        except BaseException:
            metrics.add(errors=1)
            raise
        finally:
            metrics.add(execute_ms=(time.perf_counter()-started)*1000)

    def _fetch(self, method, *args, **kwargs):
        if _current.get() is None:
            return method(*args, **kwargs)
        started = time.perf_counter()
        result = method(*args, **kwargs)
        rows = [] if result is None else [result] if method.__name__ == "fetchone" else result
        count(rows_fetched=len(rows), db_to_client_payload_bytes=sum(row_size(r) for r in rows),
              fetch_ms=(time.perf_counter()-started)*1000)
        return result

    def fetchone(self):
        return self._fetch(super().fetchone)

    def fetchmany(self, *args, **kwargs):
        return self._fetch(super().fetchmany, *args, **kwargs)

    def fetchall(self):
        return self._fetch(super().fetchall)

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

@functools.lru_cache(maxsize=2)
def cursor_factory(driver="psycopg"):
    if driver == "psycopg2":
        from psycopg2.extensions import cursor as Cursor
    else:
        from psycopg import Cursor

    class TrafficCursor(TrafficCursorMixin, Cursor):
        pass

    return TrafficCursor


def connect_options(driver="psycopg"):
    if os.getenv("DB_TRAFFIC_ENABLED", "1").strip() == "0":
        return {}
    try:
        return {"cursor_factory": cursor_factory(driver)}
    except ImportError:  # minimal driver shims used by tests
        return {}
