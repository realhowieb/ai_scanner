# ai_scanner/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

# Snapshots before this date predate the BreakoutScore clipping fix; their
# scores are on a different (un-clipped) scale, so including them poisons any
# forward-return / track-record aggregate. Single source of truth — imported by
# analytics.track_record and ui.alert_preview. Bump this only when a scoring
# change invalidates earlier snapshots.
SCORE_EPOCH = date(2026, 7, 1)


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


def _get_int(name: str, default: int) -> int:
    """Parse an int env/secret, tolerating unset or empty values.

    A workflow that passes `${{ secrets.FOO }}` for an unset secret injects an
    empty string, so a bare int(_get(...)) would crash config import (and the
    whole cron). Fall back to the default on empty/invalid input.
    """
    raw = _get(name, None)
    try:
        text = str(raw).strip()
        return int(text) if text else int(default)
    except (TypeError, ValueError):
        return int(default)


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

# --- Session lifetime ---
SESSION_TTL_DAYS: int = int(_get("SESSION_TTL_DAYS", "14"))

# --- Billing service ---
BILLING_API_BASE: str = _get("BILLING_API_BASE", "https://ai-scanner-h2c8.onrender.com")

# --- AI scan summary (Anthropic) ---
ANTHROPIC_API_KEY: str | None = _get("ANTHROPIC_API_KEY")
# Default to the most capable Opus model; override via env if desired.
ANTHROPIC_MODEL: str = _get("ANTHROPIC_MODEL", "claude-opus-4-8")
# Global kill switch — set AI_ENABLED=0 to disable all AI features instantly.
AI_ENABLED: bool = _get("AI_ENABLED", "1") != "0"
# Per-user AI calls allowed per rolling 24h (0 = unlimited).
AI_DAILY_LIMIT: int = int(_get("AI_DAILY_LIMIT", "25"))
# Hard timeout (seconds) on each Claude request so the UI never hangs.
AI_REQUEST_TIMEOUT_SECONDS: float = float(_get("AI_REQUEST_TIMEOUT_SECONDS", "30"))
# Scheduled per-user watchlist email digests (opt-in; off by default).
WATCHLIST_ALERTS_ENABLED: bool = _get("WATCHLIST_ALERTS_ENABLED", "0") == "1"
# Max users emailed per run, to bound cost/volume.
WATCHLIST_ALERTS_MAX_USERS: int = int(_get("WATCHLIST_ALERTS_MAX_USERS", "100"))

# --- Per-user alerts (breakout / watchlist / price) ---
# Master switch for the alert engine (UI panel + scheduled evaluation).
ALERTS_ENABLED: bool = _get("ALERTS_ENABLED", "1") == "1"
# Don't re-fire the same standing alert more often than this (hours).
ALERT_THROTTLE_HOURS: float = float(_get("ALERT_THROTTLE_HOURS", "12"))
# Max alerts a single user may create (guards table growth / email volume).
ALERT_MAX_PER_USER: int = int(_get("ALERT_MAX_PER_USER", "25"))

# --- Pre-open morning digest email (Pro+, structured, no AI dependency) ---
# Off by default: enable per-environment once deliverability is verified so a
# broken run can't email the whole user base. The cron sends at most once/day.
MORNING_DIGEST_ENABLED: bool = str(_get("MORNING_DIGEST_ENABLED", "0")).strip() == "1"
# Cap recipients per run to bound email volume/cost.
MORNING_DIGEST_MAX_USERS: int = _get_int("MORNING_DIGEST_MAX_USERS", 500)

# --- Day Trader live panel (intraday snapshots: gappers / VWAP / RVOL) ---
# Master switch for the live day-trader monitor. On by default; set to 0 to hide.
DAY_TRADER_ENABLED: bool = _get("DAY_TRADER_ENABLED", "1") == "1"

# --- Alerting (ops) ---
SLACK_WEBHOOK_URL: str | None = _get("SLACK_WEBHOOK_URL")
ALERT_EMAIL: str | None = _get("ALERT_EMAIL")
SCAN_ERROR_ALERT_THRESHOLD: int = int(_get("SCAN_ERROR_ALERT_THRESHOLD", "5"))
SCAN_ERROR_ALERT_WINDOW_MINUTES: int = int(_get("SCAN_ERROR_ALERT_WINDOW_MINUTES", "15"))

# --- Password reset ---
RESET_TOKEN_TTL_MINUTES: int = int(_get("RESET_TOKEN_TTL_MINUTES", "30"))
SMTP_HOST: str | None = _get("SMTP_HOST")
SMTP_PORT: int = int(_get("SMTP_PORT", "587"))
SMTP_USER: str | None = _get("SMTP_USER")
SMTP_PASS: str | None = _get("SMTP_PASS")
SMTP_FROM: str = _get("SMTP_FROM", "noreply@ai-scanner.app")
APP_BASE_URL: str = _get("APP_BASE_URL", "https://hsf-beta.streamlit.app")

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
