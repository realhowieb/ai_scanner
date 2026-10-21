"""Per-IP request limits for the public auth endpoints (P1-59 account step).

In-process sliding windows: one Render instance today. Behind more than one
instance each keeps its own counts, so the effective limit grows with the
instance count; the per-account limits in the database still apply.
"""
from __future__ import annotations

import hmac
import ipaddress
import math
import os
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
    "paper_connect": (5, 3600),  # per account: key validation attempts
    "paper_order": (30, 3600),   # per account
    "paper_status": (60, 60),    # per account: read-only confirmation polling
    "research_preferences": (60, 600),
    "symbol_search": (120, 60),
    "notification_retry": (10, 600),
    "operations": (60, 60),
    "billing_sync": (20, 600),  # per account: bounded return/focus checks against Stripe
    "billing_diagnostics": (10, 600),  # admin-only Stripe configuration probes
    "delete_account": (5, 3600), # per account: password attempts on DELETE /v1/me
    "events": (120, 3600),       # per IP: anonymous funnel events from the web landing page
    "unsubscribe": (30, 600),    # per IP: token lookups on the emailed unsubscribe link
    "client_errors": (30, 600),  # per IP: crash reports from the web app
    "export": (10, 3600),        # per account: "Download my data"
}
_MAX_KEYS = 50_000  # bound memory if someone sprays many addresses

_hits: Dict[Tuple[str, str], Deque[float]] = {}
_lock = threading.Lock()


def _trusted_proxy_ip(request: Request) -> Optional[str]:
    """The visitor address the web server forwards (web/src/server/clientAddress.ts).
    Trusted only when the call carries HSF_PROXY_SECRET, which only the web server
    knows; otherwise every visitor of the website would share the web server's limit."""
    secret = os.environ.get("HSF_PROXY_SECRET", "").strip()
    if not secret:
        return None
    given = request.headers.get("x-hsf-proxy-key", "")
    if not given or not hmac.compare_digest(given.encode(), secret.encode()):
        return None
    try:
        return str(ipaddress.ip_address(request.headers.get("x-hsf-client-ip", "").strip()))
    except ValueError:
        return None


def client_ip(request: Request) -> str:
    """The caller's address. Calls relayed by the web server use the visitor address
    it forwards (see _trusted_proxy_ip). Otherwise Render's proxy appends the real
    client address as the LAST X-Forwarded-For entry; earlier entries come from the
    client and can be forged, so they are ignored."""
    relayed = _trusted_proxy_ip(request)
    if relayed:
        return relayed
    xff = request.headers.get("x-forwarded-for", "")
    if xff.strip():
        return xff.split(",")[-1].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def check(bucket: str, key: str, *, now: Optional[float] = None) -> None:
    """Count one request; raise 429 with Retry-After when over the limit."""
    limit, window = LIMITS[bucket]
    now = time.monotonic() if now is None else now
    with _lock:
        identity = (bucket, key)
        if identity not in _hits and len(_hits) >= _MAX_KEYS:
            # Each bucket has its own window. Never evict an active quota to make
            # room: rotating addresses must not reset another caller's limit.
            for k in [k for k, q in _hits.items()
                      if not q or now - q[-1] >= LIMITS[k[0]][1]]:
                del _hits[k]
            if len(_hits) >= _MAX_KEYS:
                retry = max(1, math.ceil(min(
                    q[-1] + LIMITS[k[0]][1] - now for k, q in _hits.items()
                )))
                raise HTTPException(429, "Too many requests. Try again later.",
                                    headers={"Retry-After": str(retry)})
        q = _hits.setdefault(identity, deque())
        while q and now - q[0] >= window:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, math.ceil(window - (now - q[0])))
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
