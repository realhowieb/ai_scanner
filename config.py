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

@dataclass(frozen=True)
class Settings:
    profile: str = _get("PROFILE", "dev")
    db_url: str = _get("DB_URL", "sqlite:///scanner.sqlite")
    tz: str = _get("TZ", "America/New_York")
    max_workers: int = int(_get("MAX_WORKERS", "4"))
    chunk_size: int = int(_get("CHUNK_SIZE", "70"))
    yfinance_timeout_ms: int = int(_get("YF_TIMEOUT_MS", "10000"))
    show_diagnostics_default: bool = _get("SHOW_DIAGNOSTICS", "0") == "1"

SETTINGS = Settings()

# ---------- Stripe Billing Configuration ----------
STRIPE_MONTHLY_LINKS = {
    "basic":  "https://buy.stripe.com/basic-monthly",
    "pro":    "https://buy.stripe.com/pro-monthly",
    "premium": "https://buy.stripe.com/premium-monthly",
}

STRIPE_YEARLY_LINKS = {
    "basic":  "https://buy.stripe.com/basic-yearly",
    "pro":    "https://buy.stripe.com/pro-yearly",
    "premium": "https://buy.stripe.com/premium-yearly",
}

# ---------- Tier Metadata ----------
TIERS_CONFIG = {
    "basic": {
        "name": "Basic",
        "price_monthly": 10,
        "price_yearly": 100,
        "features": ["SP500 Scan"],
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
    },
}