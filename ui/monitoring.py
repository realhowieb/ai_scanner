"""Optional Sentry error monitoring.

No-op unless SENTRY_DSN is configured (env var, or Streamlit secret). Safe to
call from both the Streamlit app and the headless cron; never raises.
"""
from __future__ import annotations

import os
from typing import Any

_initialized = False

_STREAMLIT_COMPONENT_LOGGER = "streamlit.web.server.component_request_handler"
_COOKIE_COMPONENT_BUILD_PATH = "streamlit_cookies_manager/build"


def _is_handled_cookie_component_directory_read(event: dict[str, Any]) -> bool:
    """Return whether *event* is Streamlit's harmless bare-component probe.

    Streamlit 1.54 attempts to open a component's build directory when a client
    requests the component route without an asset filename. The handler catches
    the resulting ``IsADirectoryError`` and returns 404, but logs it with an
    exception, which Sentry otherwise promotes to a production issue. Keep the
    filter deliberately narrow so real component read errors still report.
    """
    if event.get("logger") != _STREAMLIT_COMPONENT_LOGGER:
        return False

    exceptions = event.get("exception", {}).get("values", [])
    for exception in exceptions:
        if not isinstance(exception, dict) or exception.get("type") != "IsADirectoryError":
            continue
        message = str(exception.get("value") or "").replace("\\", "/")
        if _COOKIE_COMPONENT_BUILD_PATH in message:
            return True
    return False


def _before_send(event: dict[str, Any], _hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop one known handled Streamlit component probe; retain all other events."""
    if _is_handled_cookie_component_directory_read(event):
        return None
    return event


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
            before_send=_before_send,
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
