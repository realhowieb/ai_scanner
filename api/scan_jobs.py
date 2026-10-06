"""Custom scan jobs for the API (Web v2 client-readiness run, 2026-10-06).

POST /v1/scans queues a scan; GET /v1/scans/{scan_id} returns its status and,
when complete, the ranked results. Scans take seconds (S&P 500) to minutes
(US market: 224.6 s on 2026-10-06), longer than a client should hold a request
open, so they run on a small in-process worker pool (API_SCAN_WORKERS, default
1, which also bounds memory on the API's instance) and their state lives in
the API-owned `api_scan_jobs` table.

The scan itself is the web app's: universe selection
(scan.universe_selection.resolve_scan_universe with the same loaders and the
same liquidity pre-filter) and execution
(scan.execution.run_manual_scan_execution with scan.engine.run_breakout_scan).
Plan rules are checked here, server-side, before anything is queued
(api.scan_rules); the client's request is never trusted for entitlements.

A job left queued or running when the process restarts (deploy, crash) is
marked failed ("interrupted") the next time anyone looks at it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from api.store import DatabaseUnavailable, _conn
from db.engine import schema_once

log = logging.getLogger("hsf_api.scans")

ACTIVE = ("queued", "running")
STALE_AFTER_S = 20 * 60          # running with no heartbeat this long = the worker is gone
STALE_QUEUED_S = 60 * 60         # queued this long = the process that queued it is gone
MAX_ACTIVE_JOBS = 10             # across all users; beyond this the API answers 503
KEEP_DAYS = 7

_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()


class ScanBusy(RuntimeError):
    """Too many scans queued across the service (503, retry later)."""


class ScanInProgress(RuntimeError):
    """This account already has a scan queued or running (409)."""

    def __init__(self, scan_id: str):
        super().__init__(scan_id)
        self.scan_id = scan_id


def _workers() -> int:
    try:
        return max(1, min(4, int(os.getenv("API_SCAN_WORKERS", "1"))))
    except ValueError:
        return 1


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="hsf-scan")
        return _pool


# ---- storage --------------------------------------------------------------------------------------
@schema_once
def ensure_scan_jobs_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_scan_jobs (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            status TEXT NOT NULL,
            universe TEXT NOT NULL,
            params JSONB NOT NULL,
            progress JSONB,
            result JSONB,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS api_scan_jobs_user ON api_scan_jobs (username, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS api_scan_jobs_status ON api_scan_jobs (status)")
    conn.commit()
    cur.close()


def _rows(cur) -> List[Dict[str, Any]]:
    cols = [d.name for d in cur.description]
    out = []
    for r in cur.fetchall():
        row = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
        for k in ("params", "progress", "result"):
            if isinstance(row.get(k), str):
                row[k] = json.loads(row[k])
        out.append(row)
    return out


def _execute(sql: str, params: tuple, *, fetch: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        ensure_scan_jobs_schema(conn)
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = _rows(cur) if fetch else []
        conn.commit()
        cur.close()
        return rows
    finally:
        conn.close()


def _expire_stale() -> None:
    _execute("UPDATE api_scan_jobs SET status = 'failed', error = 'interrupted', finished_at = NOW(), "
             "updated_at = NOW() WHERE (status = 'running' AND updated_at < NOW() - make_interval(secs => %s)) "
             "OR (status = 'queued' AND updated_at < NOW() - make_interval(secs => %s))",
             (STALE_AFTER_S, STALE_QUEUED_S))


def create_job(username: str, universe: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a queued job: one active job per account, MAX_ACTIVE_JOBS overall."""
    _expire_stale()
    conn = _conn()
    try:
        ensure_scan_jobs_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"api_scan_jobs:{username}",))
        cur.execute("SELECT id FROM api_scan_jobs WHERE username = %s AND status IN ('queued', 'running') "
                    "ORDER BY created_at DESC LIMIT 1", (username,))
        mine = _rows(cur)
        if mine:
            conn.rollback()
            raise ScanInProgress(mine[0]["id"])
        cur.execute("SELECT COUNT(*) AS n FROM api_scan_jobs WHERE status IN ('queued', 'running')")
        if int(_rows(cur)[0]["n"]) >= MAX_ACTIVE_JOBS:
            conn.rollback()
            raise ScanBusy("too many scans queued")
        cur.execute("DELETE FROM api_scan_jobs WHERE created_at < NOW() - make_interval(days => %s)", (KEEP_DAYS,))
        job_id = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO api_scan_jobs (id, username, status, universe, params, progress) "
            "VALUES (%s, %s, 'queued', %s, %s, %s) "
            "RETURNING id, status, universe, params, progress, result, error, created_at, started_at, finished_at",
            (job_id, username, universe, json.dumps(params), json.dumps({"phase": "queued"})),
        )
        row = _rows(cur)[0]
        conn.commit()
        cur.close()
        return row
    finally:
        conn.close()


def get_job(username: str, job_id: str) -> Optional[Dict[str, Any]]:
    _expire_stale()
    rows = _execute("SELECT id, status, universe, params, progress, result, error, created_at, started_at, "
                    "finished_at FROM api_scan_jobs WHERE id = %s AND username = %s", (job_id, username), fetch=True)
    return rows[0] if rows else None


def list_jobs(username: str, limit: int = 10) -> List[Dict[str, Any]]:
    _expire_stale()
    return _execute("SELECT id, status, universe, params, progress, NULL::jsonb AS result, error, created_at, "
                    "started_at, finished_at FROM api_scan_jobs WHERE username = %s "
                    "ORDER BY created_at DESC LIMIT %s", (username, int(limit)), fetch=True)


def _mark_running(job_id: str) -> bool:
    """Start a queued job; False when it's no longer queued (expired meanwhile)."""
    return bool(_execute("UPDATE api_scan_jobs SET status = 'running', started_at = NOW(), progress = %s, "
                         "updated_at = NOW() WHERE id = %s AND status = 'queued' RETURNING id",
                         (json.dumps({"phase": "starting"}), job_id), fetch=True))


def _set_progress(job_id: str, progress: Dict[str, Any]) -> None:
    _execute("UPDATE api_scan_jobs SET progress = %s, updated_at = NOW() WHERE id = %s",
             (json.dumps(progress), job_id))


def _finish(job_id: str, status: str, progress: Dict[str, Any], *,
            result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    _execute("UPDATE api_scan_jobs SET status = %s, progress = %s, result = %s, error = %s, "
             "finished_at = NOW(), updated_at = NOW() WHERE id = %s AND status = 'running'",
             (status, json.dumps(progress), json.dumps(result) if result is not None else None, error, job_id))


# ---- running --------------------------------------------------------------------------------------
def submit(job_id: str, work: Callable[[Callable[[Dict[str, Any]], None]], Dict[str, Any]]) -> None:
    """Run `work(report)` on the scan pool; it returns the result payload."""
    def run() -> None:
        started = time.perf_counter()
        try:
            if not _mark_running(job_id):
                return

            def report(progress: Dict[str, Any]) -> None:
                _set_progress(job_id, {**progress, "elapsed_s": round(time.perf_counter() - started, 1)})

            result = work(report)
            _finish(job_id, "complete", {"phase": "complete", "elapsed_s": round(time.perf_counter() - started, 1)},
                    result=result)
        except Exception as e:  # never leave a job running; the message is safe to show
            log.warning("scan job %s failed: %s", job_id, type(e).__name__)
            msg = str(e) if isinstance(e, ScanFailed) else "The scan failed. Try again in a few minutes."
            try:
                _finish(job_id, "failed", {"phase": "failed", "elapsed_s": round(time.perf_counter() - started, 1)},
                        error=msg[:300])
            except DatabaseUnavailable:
                pass
        finally:
            try:
                from db.engine import release_thread_connection

                release_thread_connection()
            except Exception:
                pass

    _executor().submit(run)


class ScanFailed(RuntimeError):
    """A scan that could not run, with a message for the user."""
