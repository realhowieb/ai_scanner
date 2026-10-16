"""Access tokens (short-lived signed JWTs) and refresh tokens (opaque, stored hashed).

Access tokens are bound to the current password hash with a keyed digest.
Neither the password nor its bcrypt hash is exposed in the token.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time

import jwt

ALGORITHM = "HS256"


def credential_version(username: str, password_hash: str, settings) -> str:
    material = f"access-credential-v1\0{username.strip().lower()}\0{password_hash}"
    return hmac.new(settings.jwt_secret.encode(), material.encode(), hashlib.sha256).hexdigest()


def create_access_token(username: str, settings, *, password_hash: str, now: float | None = None) -> str:
    now = int(now if now is not None else time.time())
    claims = {"sub": username, "iat": now, "exp": now + settings.access_ttl_s,
              "iss": settings.issuer, "typ": "access",
              "cv": credential_version(username, password_hash, settings)}
    return jwt.encode(claims, settings.jwt_secret, algorithm=ALGORITHM)


def access_claims(token: str, settings) -> dict | None:
    """Verified claims. Legacy tokens must refresh to acquire credential binding."""
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM],
                            issuer=settings.issuer, options={"require": ["exp", "iat", "sub", "iss", "cv"]})
    except jwt.PyJWTError:
        return None
    version = claims.get("cv")
    if (claims.get("typ") != "access" or not isinstance(version, str)
            or len(version) != 64 or any(c not in "0123456789abcdef" for c in version)):
        return None
    return claims


def verify_access_token(token: str, settings) -> str | None:
    claims = access_claims(token, settings)
    return (str(claims.get("sub") or "").strip().lower() or None) if claims else None


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Only this hash is stored, so a database leak doesn't hand out sessions."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
