"""Cloudflare Turnstile: a free bot check on sign-up and password reset.

Off until TURNSTILE_SECRET_KEY is set on the API. Set it together with
NEXT_PUBLIC_TURNSTILE_SITE_KEY on the web (which shows the widget); with only the
secret set, sign-ups from the web would be refused. Once on, those two endpoints need
the widget's one-time token (turnstile_token), checked with Cloudflare's siteverify.
If Cloudflare can't be reached the request is allowed (rate limits still apply):
an outage there must not stop people signing up.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import HTTPException

log = logging.getLogger("hsf_api")

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
FAILED = "The security check didn't pass. Reload the page and try again."


def enabled() -> bool:
    return bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip())


def verify(token: Optional[str], ip: Optional[str], *, action: str, client=None) -> None:
    """Raise 400 when Turnstile is on and the token is missing or rejected."""
    secret = os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    if not secret:
        return
    token = (token or "").strip()
    if not token or len(token) > 2048:
        raise HTTPException(400, FAILED)
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=5.0)
    try:
        data = {"secret": secret, "response": token}
        if ip and ip != "unknown":
            data["remoteip"] = ip
        r = client.post(VERIFY_URL, data=data)
        body = r.json() if r.status_code == 200 else None
    except Exception as e:
        log.warning(json.dumps({"event": "turnstile_unreachable", "error": type(e).__name__}))
        return
    finally:
        if own:
            client.close()
    if not isinstance(body, dict):
        log.warning(json.dumps({"event": "turnstile_unreachable", "status": r.status_code}))
        return
    if not body.get("success"):
        log.info(json.dumps({"event": "turnstile_failed", "action": action,
                             "codes": [str(c)[:40] for c in (body.get("error-codes") or [])][:5]}))
        raise HTTPException(400, FAILED)
