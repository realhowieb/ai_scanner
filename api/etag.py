"""ETag revalidation for read-only API answers.

A successful GET under /v1/ that answers JSON gets a weak ETag (a hash of the body)
and `Cache-Control: private, no-cache`, so a caller may keep the answer but must ask
again each time. A repeat request carrying the same ETag in If-None-Match gets an
empty 304 instead of the full body, which keeps big unchanged answers (latest scan,
track record, stock pages) off the network. The answer is still computed; only the
transfer is saved. An endpoint that sets its own Cache-Control (no-store) is left alone.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List

Message = Dict[str, Any]
MAX_BODY_BYTES = 8 * 1024 * 1024  # larger answers pass through untouched


# Today and the latest scan stamp every answer with the request time, so their hash never
# repeated and they never got a 304. For the hash only: request-time "checked_at" values
# are dropped, and Today's top-level "as_of" counts only to the hour (its greeting follows
# it). The body sent is unchanged; a 304 keeps the caller's earlier copy.
_CHECKED_AT = re.compile(rb'"checked_at":"[^"]*"')
_TOP_AS_OF = re.compile(rb'^\{"as_of":"([0-9-]{10}T[0-9]{2})[^"]*"')
CLOCK_STAMPED = {"/v1/today", "/v1/scans/latest"}


def hash_view(path: str, body: bytes) -> bytes:
    if path not in CLOCK_STAMPED:
        return body
    body = _CHECKED_AT.sub(b'"checked_at":null', body)
    return _TOP_AS_OF.sub(rb'{"as_of":"\1"', body, count=1)


def etag_for(body: bytes) -> str:
    return 'W/"' + hashlib.sha256(body).hexdigest()[:32] + '"'


def matches(if_none_match: str, etag: str) -> bool:
    """RFC 9110 weak comparison: W/ prefixes are ignored; "*" matches anything."""
    want = etag[2:] if etag.startswith("W/") else etag
    for tag in if_none_match.split(","):
        tag = tag.strip()
        if tag == "*":
            return True
        if (tag[2:] if tag.startswith("W/") else tag) == want:
            return True
    return False


class ETagMiddleware:
    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["method"] != "GET" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        if_none_match = ""
        for k, v in scope.get("headers") or []:
            if k == b"if-none-match":
                if_none_match = v.decode("latin-1")
                break

        start: Message = {}
        chunks: List[bytes] = []
        size = 0
        passthrough = False

        async def wrapped(message: Message) -> None:
            nonlocal start, size, passthrough
            if passthrough:
                await send(message)
                return
            if message["type"] == "http.response.start":
                headers = {k.lower(): v for k, v in message.get("headers") or []}
                if (message["status"] != 200 or b"cache-control" in headers or b"etag" in headers
                        or not headers.get(b"content-type", b"").startswith(b"application/json")):
                    passthrough = True
                    await send(message)
                    return
                start = message
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > MAX_BODY_BYTES:  # too big to buffer: send what we have, stream the rest
                passthrough = True
                await send(start)
                await send({"type": "http.response.body", "body": b"".join(chunks),
                            "more_body": message.get("more_body", False)})
                return
            if message.get("more_body", False):
                return
            body = b"".join(chunks)
            tag = etag_for(hash_view(scope["path"], body))
            extra = [(b"etag", tag.encode("latin-1")), (b"cache-control", b"private, no-cache")]
            if if_none_match and matches(if_none_match, tag):
                keep = [(k, v) for k, v in start.get("headers") or []
                        if k.lower() not in (b"content-length", b"content-type")]
                await send({**start, "status": 304, "headers": keep + extra})
                await send({"type": "http.response.body", "body": b""})
                return
            await send({**start, "headers": list(start.get("headers") or []) + extra})
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, wrapped)
