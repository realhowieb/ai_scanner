"""Single source of truth for Alpaca credentials and endpoint URLs.

Before this module existed there were four independent readers with three
different precedence rules (env-first, secrets-first, and secrets-only) — the
secrets-only and secrets-first variants broke twice in headless contexts (the
cron has no Streamlit secrets file). Every reader now resolves identically:

  1. Environment variables (works everywhere, including the cron)
  2. Streamlit secrets, guarded (app context only; absent file never raises)
  3. Default (URLs only — credentials have no default)

Expected keys: ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, and optionally
ALPACA_DATA_URL / ALPACA_BASE_URL.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

DEFAULT_DATA_URL = "https://data.alpaca.markets"
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def alpaca_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve one config value: env first, then guarded Streamlit secrets."""
    env_val = os.getenv(key)
    if env_val:
        return env_val
    try:  # pragma: no cover - depends on app vs headless context
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            val = secrets.get(key) if hasattr(secrets, "get") else secrets[key]
            if val:
                return str(val)
    except Exception:
        pass
    return default


def get_alpaca_config() -> Optional[Dict[str, str]]:
    """Return Alpaca config, or None when credentials are not configured.

    Shape: {"api_key", "api_secret", "data_url", "base_url"} — URLs are
    normalized without a trailing slash.
    """
    api_key = alpaca_secret("ALPACA_API_KEY_ID")
    api_secret = alpaca_secret("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        return None
    data_url = alpaca_secret("ALPACA_DATA_URL", DEFAULT_DATA_URL) or DEFAULT_DATA_URL
    base_url = alpaca_secret("ALPACA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "data_url": data_url.rstrip("/"),
        "base_url": base_url.rstrip("/"),
    }


def get_alpaca_headers() -> Optional[Dict[str, str]]:
    """Return Alpaca auth headers, or None when credentials are not configured."""
    cfg = get_alpaca_config()
    if cfg is None:
        return None
    return {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }
