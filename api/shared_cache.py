"""Shared state in Render Key Value (Redis-compatible), with an in-process fallback.

Set REDIS_URL on the API (Render Key Value "Internal URL", e.g. redis://red-xxxx:6379)
and rate limits plus third-party lookups (news, SEC filings) are shared by every API
instance and survive restarts and deploys. Without REDIS_URL, or while Key Value can't be
reached, everything falls back to this process's memory: requests never fail because of
the cache. HSF_REDIS_DISABLED=1 forces the fallback.

Only JSON values are stored. Keys carry the hsf: prefix; rate-limit keys hash the caller
(IP or account) so no email address is written to Key Value.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger("hsf_api")

PREFIX = "hsf:"
RETRY_AFTER_FAILURE_S = 30  # after a connection error, use memory for this long before trying again
_MEM_MAX = 2000

_client: Any = None
_client_url: Optional[str] = None
_down_until = 0.0
_lock = threading.Lock()
_mem: Dict[str, Tuple[float, Any]] = {}
_mem_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "errors": 0}


def url() -> Optional[str]:
    if os.environ.get("HSF_REDIS_DISABLED", "").strip() == "1":
        return None
    value = (os.environ.get("REDIS_URL") or os.environ.get("HSF_REDIS_URL") or "").strip()
    return value if value.startswith(("redis://", "rediss://")) else None


def configured() -> bool:
    return url() is not None


def client() -> Any:
    """The Redis client, or None (not configured, package missing, or recently down)."""
    global _client, _client_url
    target = url()
    if target is None or time.monotonic() < _down_until:
        return None
    with _lock:
        if _client is None or _client_url != target:
            try:
                import redis
            except ImportError:
                return None
            _client = redis.Redis.from_url(target, socket_timeout=0.5, socket_connect_timeout=0.5,
                                           health_check_interval=30, decode_responses=True)
            _client_url = target
        return _client


def mark_down(error: BaseException) -> None:
    """A Key Value call failed: count it and use memory for a short while."""
    global _down_until
    _stats["errors"] += 1
    _down_until = time.monotonic() + RETRY_AFTER_FAILURE_S
    log.warning(json.dumps({"event": "shared_cache_unavailable", "error": type(error).__name__}))


def hashed(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


# ---- JSON values ------------------------------------------------------------------------
def _mem_get(key: str) -> Any:
    with _mem_lock:
        hit = _mem.get(key)
        if hit and time.monotonic() < hit[0]:
            return hit[1]
        _mem.pop(key, None)
        return None


def _mem_set(key: str, value: Any, ttl_s: float) -> None:
    now = time.monotonic()
    with _mem_lock:
        if len(_mem) >= _MEM_MAX:
            for k in [k for k, (exp, _) in _mem.items() if exp <= now]:
                del _mem[k]
            while len(_mem) >= _MEM_MAX:
                del _mem[min(_mem, key=lambda k: _mem[k][0])]
        _mem[key] = (now + ttl_s, value)


def get(key: str) -> Any:
    """The cached JSON value, or None."""
    full = PREFIX + key
    r = client()
    if r is not None:
        try:
            raw = r.get(full)
            return json.loads(raw) if raw is not None else None
        except Exception as e:
            mark_down(e)
    return _mem_get(full)


def put(key: str, value: Any, ttl_s: float) -> None:
    full = PREFIX + key
    r = client()
    if r is not None:
        try:
            r.set(full, json.dumps(value, separators=(",", ":"), default=str), ex=max(1, int(ttl_s)))
            return
        except Exception as e:
            mark_down(e)
    _mem_set(full, value, ttl_s)


def cached(key: str, loader: Callable[[], Any], ttl_s: float, *, empty_ttl_s: Optional[float] = None,
           is_empty: Callable[[Any], bool] = lambda v: not v) -> Any:
    """The value under `key`, or loader()'s answer stored for ttl_s (empty answers for
    empty_ttl_s; not stored when that is None). A loader error propagates and stores nothing."""
    hit = get(key)
    if hit is not None:
        _stats["hits"] += 1
        return hit
    _stats["misses"] += 1
    value = loader()
    if is_empty(value):
        if empty_ttl_s:
            put(key, value, empty_ttl_s)
    else:
        put(key, value, ttl_s)
    return value


def status() -> Dict[str, Any]:
    """For /readyz: whether Key Value is configured and answers. Never raises."""
    out: Dict[str, Any] = {"configured": configured(), "ok": None, "ping_ms": None, **_stats}
    r = client()
    if r is None:
        out["ok"] = False if configured() else None
        return out
    try:
        t0 = time.perf_counter()
        r.ping()
        out.update(ok=True, ping_ms=round((time.perf_counter() - t0) * 1000, 1))
    except Exception as e:
        mark_down(e)
        out["ok"] = False
    return out


def reset() -> None:
    """For tests."""
    global _client, _client_url, _down_until
    with _lock:
        _client, _client_url, _down_until = None, None, 0.0
    with _mem_lock:
        _mem.clear()
    for k in _stats:
        _stats[k] = 0
