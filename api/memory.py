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


_peak_mb = 0.0  # highest rss_mb() seen since this process started


def rss_mb() -> Optional[float]:
    """This process's resident memory in MB (Linux), for the request log. Each reading
    also updates the peak that /readyz reports."""
    global _peak_mb
    try:
        with open("/proc/self/statm", "rb") as f:
            mb = round(int(f.read().split()[1]) * _page / 1048576, 1)
    except (OSError, ValueError, IndexError):
        return None
    if mb > _peak_mb:
        _peak_mb = mb
    return mb


def peak_rss_mb() -> Optional[float]:
    return _peak_mb or None


def _env_mb(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def memory_status() -> dict:
    """Memory now, the peak since start, the instance limit (HSF_MEMORY_LIMIT_MB, 512 on
    Render Starter) and whether the peak crossed HSF_MEMORY_WARN_MB (default 400), so
    /readyz and the nightly health report warn before Render restarts the API for
    running out of memory."""
    now = rss_mb()
    peak = peak_rss_mb()
    warn = _env_mb("HSF_MEMORY_WARN_MB", 400.0)
    return {"rss_mb": now, "rss_peak_mb": peak, "memory_limit_mb": _env_mb("HSF_MEMORY_LIMIT_MB", 512.0),
            "memory_warn_mb": warn, "memory_warning": bool(peak is not None and peak >= warn)}
