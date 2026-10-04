"""API settings, read from the environment once per process."""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def load_settings() -> Settings:
    """API_JWT_SECRET signs access tokens (32+ chars, e.g. `openssl rand -base64 48`).
    API_CORS_ORIGINS: comma-separated browser origins allowed to call the API."""
    secret = os.getenv("API_JWT_SECRET", "").strip()
    if len(secret) < MIN_SECRET_LEN:
        raise RuntimeError(f"API_JWT_SECRET must be set to at least {MIN_SECRET_LEN} characters")
    origins = tuple(o.strip().rstrip("/") for o in os.getenv("API_CORS_ORIGINS", "").split(",") if o.strip())
    return Settings(
        jwt_secret=secret,
        access_ttl_s=_int("API_ACCESS_TTL_S", 15 * 60),
        refresh_ttl_s=_int("API_REFRESH_TTL_S", 30 * 24 * 3600),
        cors_origins=origins,
    )
