"""Per-IP request limits for the public auth endpoints (P1-59 account step).

In-process sliding windows: one Render instance today. Behind more than one
instance each keeps its own counts, so the effective limit grows with the
instance count; the per-account limits in the database still apply.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request

# bucket -> (max requests, window seconds)
LIMITS: Dict[str, Tuple[int, int]] = {
    "login": (30, 600),
    "signup": (5, 3600),
    "password_reset": (5, 3600),
    "verify": (30, 600),
    "verify_resend": (3, 3600),
    "scan": (30, 3600),          # per account (custom scans run on a small shared pool)
    "ai": (60, 3600),            # per account, on top of the daily AI_DAILY_LIMIT
}
_MAX_KEYS = 50_000  # bound memory if someone sprays many addresses

_hits: Dict[Tuple[str, str], Deque[float]] = {}
_lock = threading.Lock()


def client_ip(request: Request) -> str:
    """The caller's address. Render's proxy appends the real client address as the
    LAST X-Forwarded-For entry; earlier entries come from the client and can be
    forged, so they are ignored."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff.strip():
        return xff.split(",")[-1].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def check(bucket: str, key: str, *, now: Optional[float] = None) -> None:
    """Count one request; raise 429 with Retry-After when over the limit."""
    limit, window = LIMITS[bucket]
    now = time.monotonic() if now is None else now
    with _lock:
        if len(_hits) > _MAX_KEYS:
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > window]:
                del _hits[k]
        q = _hits.setdefault((bucket, key), deque())
        while q and now - q[0] >= window:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(window - (now - q[0])))
            raise HTTPException(429, "Too many requests. Try again later.",
                                headers={"Retry-After": str(retry)})
        q.append(now)


def limit(bucket: str):
    """FastAPI dependency: per-IP limit for `bucket`."""
    def dep(request: Request) -> None:
        check(bucket, client_ip(request))
    return dep


def reset() -> None:
    with _lock:
        _hits.clear()
