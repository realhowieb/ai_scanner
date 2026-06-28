"""Optional Sentry error monitoring.

No-op unless SENTRY_DSN is configured (env var, or Streamlit secret). Safe to
call from both the Streamlit app and the headless cron; never raises.
"""
from __future__ import annotations

import os

_initialized = False


def _get_dsn() -> str | None:
    dsn = os.getenv("SENTRY_DSN")
    if dsn:
        return dsn
    # Streamlit secrets (app context only); guarded — no secrets file in cron.
    try:
        import streamlit as st  # type: ignore

        val = st.secrets.get("SENTRY_DSN")  # may raise if no secrets
        return str(val) if val else None
    except Exception:
        return None


def init_sentry(component: str = "app") -> bool:
    """Initialize Sentry once. Returns True if active, False if not configured."""
    global _initialized
    if _initialized:
        return True
    dsn = _get_dsn()
    if not dsn:
        print(f"[monitoring] Sentry not configured (no SENTRY_DSN) — component={component}")
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            # Errors only — no performance tracing (avoids overhead/cost).
            traces_sample_rate=0.0,
            environment=os.getenv("SENTRY_ENV", "production"),
            release=os.getenv("SENTRY_RELEASE") or None,
        )
        sentry_sdk.set_tag("component", component)
        _initialized = True
        print(f"[monitoring] Sentry active — component={component}")
        return True
    except Exception as e:
        print(f"[monitoring] Sentry init failed: {type(e).__name__}: {e}")
        return False


def capture(exc: BaseException) -> None:
    """Send an exception to Sentry if initialized; otherwise a no-op."""
    if not _initialized:
        return
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
