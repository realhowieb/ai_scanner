"""API settings, read from the environment once per process."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger("hsf_api")

MIN_SECRET_LEN = 32


@dataclass(frozen=True)
class Settings:
    jwt_secret: str
    access_ttl_s: int
    refresh_ttl_s: int
    cors_origins: tuple[str, ...]
    issuer: str = "hsf-api"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# scheme://host[:port], nothing else. https anywhere; http only for local development.
_ORIGIN = re.compile(r"^(https://[a-z0-9.-]+|http://(localhost|127\.0\.0\.1))(:[0-9]{1,5})?$")


def parse_cors_origins(raw: str) -> tuple[str, ...]:
    """Browser origins from API_CORS_ORIGINS (comma-separated). Entries that aren't
    an exact origin (a wildcard, a path, plain http to a real host, a typo) are
    dropped with a warning instead of stopping the service; the rest still work."""
    good, bad = [], 0
    for entry in (raw or "").split(","):
        origin = entry.strip().rstrip("/").lower()
        if not origin:
            continue
        if _ORIGIN.match(origin):
            if origin not in good:
                good.append(origin)
        else:
            bad += 1
    if bad:
        log.warning("API_CORS_ORIGINS: ignored %d entr%s that %s not an exact https origin "
                    "(scheme://host[:port], no wildcard or path)", bad, "y" if bad == 1 else "ies",
                    "is" if bad == 1 else "are")
    return tuple(good)


def load_settings() -> Settings:
    """API_JWT_SECRET signs access tokens (32+ chars, e.g. `openssl rand -base64 48`).
    API_CORS_ORIGINS: comma-separated browser origins allowed to call the API."""
    secret = os.getenv("API_JWT_SECRET", "").strip()
    if len(secret) < MIN_SECRET_LEN:
        raise RuntimeError(f"API_JWT_SECRET must be set to at least {MIN_SECRET_LEN} characters")
    origins = parse_cors_origins(os.getenv("API_CORS_ORIGINS", ""))
    return Settings(
        jwt_secret=secret,
        access_ttl_s=_int("API_ACCESS_TTL_S", 15 * 60),
        refresh_ttl_s=_int("API_REFRESH_TTL_S", 30 * 24 * 3600),
        cors_origins=origins,
    )
