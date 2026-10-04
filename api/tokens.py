"""Access tokens (short-lived signed JWTs) and refresh tokens (opaque, stored hashed).

The access token carries only the account id and expiry; plan and admin status
are read from the database on each request, so a downgrade applies at once.
"""
from __future__ import annotations

import hashlib
import secrets
import time

import jwt

ALGORITHM = "HS256"


def create_access_token(username: str, settings, *, now: float | None = None) -> str:
    now = int(now if now is not None else time.time())
    claims = {"sub": username, "iat": now, "exp": now + settings.access_ttl_s,
              "iss": settings.issuer, "typ": "access"}
    return jwt.encode(claims, settings.jwt_secret, algorithm=ALGORITHM)


def verify_access_token(token: str, settings) -> str | None:
    """The account id in a valid, unexpired access token; None otherwise."""
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM],
                            issuer=settings.issuer, options={"require": ["exp", "iat", "sub", "iss"]})
    except jwt.PyJWTError:
        return None
    if claims.get("typ") != "access":
        return None
    sub = str(claims.get("sub") or "").strip().lower()
    return sub or None


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Only this hash is stored, so a database leak doesn't hand out sessions."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
