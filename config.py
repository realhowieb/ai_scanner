# ai_scanner/config.py
from __future__ import annotations
import os
from dataclasses import dataclass


def _get(name: str, default: str|None=None):
    # 1) Streamlit secrets if present, 2) env, 3) default
    try:
        import streamlit as st
        if "secrets" in dir(st) and st.secrets is not None:
            val = st.secrets.get(name)
            if val is not None:
                return val
    except Exception:
        pass
    return os.getenv(name, default)


def _get_any(names: tuple[str, ...], default: str | None = None):
    for name in names:
        val = _get(name)
        if val:
            return val
    return default


def _get_db_url() -> str:
    try:
        import streamlit as st
        nested = st.secrets["neon"]["database_url"]  # type: ignore[index]
        if nested:
            return str(nested)
    except Exception:
        pass

    return str(
        _get_any(
            ("NEON_DATABASE_URL", "DATABASE_URL", "database_url", "DB_URL"),
            "sqlite:///scanner.sqlite",
        )
    )


@dataclass(frozen=True)
class Settings:
    profile: str = _get("PROFILE", "dev")
    db_url: str = _get_db_url()
    tz: str = _get("TZ", "America/New_York")
    max_workers: int = int(_get("MAX_WORKERS", "4"))
    chunk_size: int = int(_get("CHUNK_SIZE", "70"))
    yfinance_timeout_ms: int = int(_get("YF_TIMEOUT_MS", "15000"))
    show_diagnostics_default: bool = _get("SHOW_DIAGNOSTICS", "0") == "1"

SETTINGS = Settings()

# --- Scan engine constants ---
# Minimum universe size to activate DB price caching (admin-only runs).
DB_CACHE_MIN_TICKERS: int = 800
# Default chunk size for parallel price fetching.
PRICE_FETCH_CHUNK_SIZE: int = 150
# Bounds for the chunk size parameter.
PRICE_FETCH_CHUNK_MIN: int = 25
PRICE_FETCH_CHUNK_MAX: int = 400
# Maximum age for DB-cached price data before it is considered stale.
DB_CACHE_MAX_AGE_MINUTES: int = 30
# Throttle interval for scan progress UI updates (seconds).
PROGRESS_UI_THROTTLE_SEC: float = 0.25

# --- Login rate limiting ---
LOGIN_RATE_LIMIT_WINDOW_SEC: int = 600   # 10-minute window
LOGIN_RATE_LIMIT_MAX_ATTEMPTS: int = 10  # max failures before lockout

# ---------- Stripe Billing Configuration ----------
STRIPE_MONTHLY_LINKS = {
    "basic":  "https://buy.stripe.com/fZu7sNgEi4JW4j98zo1Jm00",
    "pro":    "https://buy.stripe.com/7sY28tbjYfoA7vl3f41Jm01",
    "premium": "https://buy.stripe.com/28E8wR0Fk90cbLBaHw1Jm02",
}

STRIPE_YEARLY_LINKS = {
    "basic":  "https://buy.stripe.com/7sY8wRds62BOdTJ4j81Jm05",
    "pro":    "https://buy.stripe.com/aFa9AV9bQ1xKeXN7vk1Jm04",
    "premium": "https://buy.stripe.com/7sY28t1Jo6S416X7vk1Jm03",
}

# ---------- Tier Metadata ----------
TIERS_CONFIG = {
    "basic": {
        "name": "Basic",
        "price_monthly": 19,
        "price_yearly": 190,
        "features": ["SP500 Scan"],
        "max_results": 25,
    },
    "pro": {
        "name": "Pro",
        "price_monthly": 25,
        "price_yearly": 250,
        "features": [
            "SP500 Scan",
            "NASDAQ",
            "ExportCSV",
            "Premarket",
            "AfterHours",
            "UnusualVolume",
        ],
        "max_results": 100,
    },
    "premium": {
        "name": "Premium",
        "price_monthly": 49,
        "price_yearly": 490,
        "features": [
            "SP500 Scan",
            "NASDAQ",
            "ExportCSV",
            "AI Notes",
            "Premarket",
            "AfterHours",
            "UnusualVolume",
            "Snapshots",
        ],
        "max_results": 200,
    },
    "admin": {
        "name": "Admin",
        "price_monthly": 0,
        "price_yearly": 0,
        "features": [
            "SP500 Scan",
            "NASDAQ",
            "ExportCSV",
            "AI Notes",
            "Premarket",
            "AfterHours",
            "UnusualVolume",
            "Snapshots",
        ],
        "max_results": 9999,
    },
}
