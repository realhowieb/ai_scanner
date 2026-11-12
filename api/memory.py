"""Keep hsf-api's memory flat on a 512 MB instance.

The API answers on many threads (uvicorn's thread pool, background cache rebuilds,
the alert worker). glibc gives each busy thread its own malloc arena and rarely hands
freed memory back, so memory crept up with traffic even though the caches stay the
same size: a local load test grew 200 -> 314 MB and kept climbing; with two arenas it
levelled off at 271 MB. tune_allocator() caps the arenas (HSF_MALLOC_ARENA_MAX,
default 2; 0 leaves glibc's default) and must run before the worker threads start.
Elsewhere than glibc (macOS, musl) it does nothing.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional

M_ARENA_MAX = -8  # glibc mallopt parameter

_libc: Optional[ctypes.CDLL] = None
_page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _glibc() -> Optional[ctypes.CDLL]:
    global _libc
    if _libc is None:
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            lib.mallopt  # noqa: B018 - glibc only; AttributeError elsewhere
            lib.malloc_trim  # noqa: B018
            lib.malloc_trim.argtypes = [ctypes.c_size_t]
            lib.malloc_trim.restype = ctypes.c_int
            _libc = lib
        except (OSError, AttributeError):
            return None
    return _libc


def tune_allocator() -> Optional[int]:
    """Cap glibc malloc arenas. Returns the cap applied, or None when nothing was done."""
    try:
        arenas = int(os.environ.get("HSF_MALLOC_ARENA_MAX", "2").strip() or "2")
    except ValueError:
        arenas = 2
    lib = _glibc()
    if arenas <= 0 or lib is None:
        return None
    return arenas if lib.mallopt(M_ARENA_MAX, arenas) == 1 else None


_STARTED = time.monotonic()  # when this process imported the module (about when it started)
_peak_mb = 0.0  # highest rss_mb() seen since this process started
_container_peak_mb = 0.0  # highest container_mb() seen since this process started

# cgroup v2 first, then v1. Render's memory graph and its out-of-memory restart count
# the container's memory minus reclaimable file cache (the "working set").
_CGROUP_FILES = (
    ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "inactive_file"),
    ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file"),
)


def rss_mb() -> Optional[float]:
    """This process's resident memory in MB (Linux), for the request log. It counts shared
    library pages too, so it reads 30-50 MB above Render's graph. Each reading also
    updates the peaks that /readyz reports."""
    global _peak_mb
    try:
        with open("/proc/self/statm", "rb") as f:
            mb = round(int(f.read().split()[1]) * _page / 1048576, 1)
    except (OSError, ValueError, IndexError):
        return None
    if mb > _peak_mb:
        _peak_mb = mb
    container_mb()
    return mb


def container_mb() -> Optional[float]:
    """The container's memory in MB the way Render measures it (cgroup usage minus
    inactive file cache), or None outside a container. Updates the container peak."""
    global _container_peak_mb
    for usage_path, stat_path, inactive_key in _CGROUP_FILES:
        try:
            with open(usage_path, "rb") as f:
                usage = int(f.read().strip())
        except (OSError, ValueError):
            continue
        inactive = 0
        try:
            with open(stat_path, "r") as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == inactive_key:
                        inactive = int(v)
                        break
        except (OSError, ValueError):
            pass
        mb = round(max(0, usage - inactive) / 1048576, 1)
        if mb > _container_peak_mb:
            _container_peak_mb = mb
        return mb
    return None


def peak_rss_mb() -> Optional[float]:
    return _peak_mb or None


def peak_container_mb() -> Optional[float]:
    return _container_peak_mb or None


def _env_mb(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _price_cache_entries() -> Optional[int]:
    try:
        from data.prices import price_cache_size

        return price_cache_size()
    except Exception:  # informational only
        return None


def memory_status() -> dict:
    """Memory now, the peak since start, the instance limit (HSF_MEMORY_LIMIT_MB, 512 on
    Render Starter) and whether the peak crossed HSF_MEMORY_WARN_MB (default 400), so
    /readyz and the nightly health report warn before Render restarts the API for
    running out of memory. The warning uses the container figure (what Render restarts
    on) when it can be read, else the process figure."""
    now = rss_mb()
    peak = peak_rss_mb()
    c_now = container_mb()
    c_peak = peak_container_mb()
    judged = c_peak if c_peak is not None else peak
    warn = _env_mb("HSF_MEMORY_WARN_MB", 400.0)
    return {"rss_mb": now, "rss_peak_mb": peak, "container_mb": c_now, "container_peak_mb": c_peak,
            "memory_limit_mb": _env_mb("HSF_MEMORY_LIMIT_MB", 512.0), "memory_warn_mb": warn,
            "memory_warning": bool(judged is not None and judged >= warn),
            "price_cache_entries": _price_cache_entries(),
            "cache_counts": sys.modules["api.today"].TTLCache.statistics() if "api.today" in sys.modules else None}


# These observations contain aggregate counts only: never cache keys/usernames/tickers.
@contextmanager
def observe_memory(operation: str):
    """Log substantial retained RSS growth around a known operation, even on failure."""
    import json
    import logging

    before = rss_mb()
    try:
        yield
    finally:
        after = rss_mb()
        if before is not None and after is not None and after - before >= 16:
            logging.getLogger("hsf.memory").info(json.dumps({
                "event": "memory_growth", "operation": operation,
                "rss_before_mb": before, "rss_after_mb": after, "rss_delta_mb": round(after - before, 1)}))


# ---- heavy loads one at a time -----------------------------------------------------------------------
# Opening Admin starts several big research reads at once (signal research, evidence, ML
# readiness). Together they pushed the process past 510 MB and, by holding the interpreter
# on three threads, slowed every other request for about a minute. heavy_load() makes such
# loads take turns, so the peak is the largest single load instead of their sum, and hands
# the freed pages back to the OS before the next one starts.
_heavy_lock = threading.RLock()


def release_memory() -> bool:
    """Collect garbage and return free allocator pages to the OS. True when trimmed."""
    import gc

    gc.collect()
    lib = _glibc()
    return bool(lib.malloc_trim(0)) if lib is not None else False


@contextmanager
def heavy_load(operation: str):
    """Run one memory-heavy load at a time (process-wide; re-entrant on the same thread)."""
    if os.getenv("HSF_HEAVY_LOAD_SERIAL", "1").strip() == "0":
        yield
        return
    with _heavy_lock:
        try:
            yield
        finally:
            try:
                release_memory()
            except Exception:  # noqa: BLE001 - tidying must never fail the load
                pass


def maintain_memory() -> dict:
    """Five-minute ops maintenance: expire unused entries; trim only under current pressure.

    No valid cache entries are evicted. malloc_trim returns free allocator pages
    to the OS; it does not restart the process or release live Python objects.
    """
    import gc
    import json
    import logging

    if os.getenv("HSF_MEMORY_MAINTENANCE_ENABLED", "1").strip() == "0":
        return {"disabled": True}
    from api.today import TTLCache

    before = container_mb()
    if before is None:
        before = rss_mb()
    expired = TTLCache.prune_all()
    trimmed = False
    if before is not None and before >= _env_mb("HSF_MEMORY_WARN_MB", 400.0):
        gc.collect()
        lib = _glibc()
        if lib is not None:
            trimmed = bool(lib.malloc_trim(0))
    after = container_mb()
    if after is None:
        after = rss_mb()
    result = {"expired_cache_entries": expired, "before_mb": before, "after_mb": after,
              "allocator_trimmed": trimmed, "cache_counts": TTLCache.statistics()}
    if expired or (before is not None and before >= _env_mb("HSF_MEMORY_WARN_MB", 400.0)):
        logging.getLogger("hsf.memory").info(json.dumps({"event": "memory_maintenance", **result}))
    return result


# ---- nightly fresh start ----------------------------------------------------------------------------
# Memory climbs through each trading day and is never handed back (the Brief, Day Trader and
# scheduled scans leave the allocator holding their peak), so a day with no deploy starts
# the next one already high. Between 3:00 and 4:00 ET on weekdays, when nobody uses the
# app, a process still above HSF_MEMORY_RESTART_MB (default 420; 0 turns this off) that
# has run for at least an hour and holds no custom scan asks itself to stop gracefully;
# Render starts it again (about a minute) and the day begins from a clean slate.
RESTART_WINDOW_ET = (3, 4)  # [start hour, end hour)
RESTART_MIN_UPTIME_S = 3600


def restart_due(now, mb: Optional[float], uptime_s: float, scans_held: int) -> bool:
    from analytics import market_calendar as mc

    limit = _env_mb("HSF_MEMORY_RESTART_MB", 420.0)
    if limit <= 0 or mb is None or mb < limit or uptime_s < RESTART_MIN_UPTIME_S or scans_held:
        return False
    et = now.astimezone(mc.ET)
    return et.weekday() < 5 and RESTART_WINDOW_ET[0] <= et.hour < RESTART_WINDOW_ET[1]


def maybe_restart_for_memory(now=None) -> bool:
    """Ops-jobs hook (every pass). Sends this process SIGTERM when restart_due(); uvicorn
    finishes the requests in flight and exits, and Render starts a fresh instance."""
    import datetime as dt
    import json
    import logging
    import signal

    now = now or dt.datetime.now(dt.timezone.utc)
    mb = container_mb()
    if mb is None:
        mb = rss_mb()
    try:
        from api import scan_jobs

        with scan_jobs._held_lock:
            held = len(scan_jobs._held)
    except Exception:
        held = 0
    uptime = time.monotonic() - _STARTED
    if not restart_due(now, mb, uptime, held):
        return False
    logging.getLogger("hsf.memory").warning(json.dumps({
        "event": "memory_restart", "container_mb": mb, "uptime_h": round(uptime / 3600, 1),
        "limit_mb": _env_mb("HSF_MEMORY_RESTART_MB", 420.0)}))
    os.kill(os.getpid(), signal.SIGTERM)
    return True
