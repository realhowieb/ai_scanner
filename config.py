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