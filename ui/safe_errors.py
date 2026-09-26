"""Run 62 — friendly error presentation for user-facing sections.

Normal users see a plain sentence ("HSF couldn't load the market brief right
now. Try again shortly."). The technical detail is never swallowed: it is logged
through the standard `logging` module and sent to Sentry when monitoring is
initialized (ui.monitoring.capture). Admins additionally see the exception type
and message in a collapsed "Technical details" expander.

Never raises.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_LOG = logging.getLogger("hsf.ui")


def friendly_message(section: str) -> str:
    """The user-facing sentence for a failed section."""
    what = (section or "this section").strip()
    return f"HSF couldn't load {what} right now. Try again shortly."


def _is_admin() -> bool:
    try:
        return bool(st is not None and st.session_state.get("is_admin"))
    except Exception:
        return False


def technical_detail(exc: Optional[BaseException]) -> str:
    """`Type: message` for admins (empty when there is no exception)."""
    if exc is None:
        return ""
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def report_error(section: str, exc: Optional[BaseException] = None) -> None:
    """Log and forward an error without showing anything."""
    try:
        if exc is not None:
            _LOG.error("UI section failed: %s", section, exc_info=(type(exc), exc, exc.__traceback__))
        else:
            _LOG.error("UI section failed: %s", section)
    except Exception:
        pass
    if exc is not None:
        try:
            from ui.monitoring import capture

            capture(exc)
        except Exception:
            pass


def show_error(section: str, exc: Optional[BaseException] = None, *, level: str = "warning",
               message: Optional[str] = None) -> None:
    """Show a friendly message for a failed section; details only to admins.

    level: "warning" (default), "error" or "info". `message` overrides the
    default sentence when a section needs more specific guidance.
    """
    report_error(section, exc)
    if st is None:
        return
    text = message or friendly_message(section)
    try:
        {"error": st.error, "info": st.info}.get(level, st.warning)(text)
        if _is_admin() and exc is not None:
            with st.expander("Technical details (admin)", expanded=False):
                st.code(technical_detail(exc))
    except Exception:
        pass


STARTUP_MESSAGE = "HSF is having trouble starting right now. Try again in a few minutes."


def show_startup_problem(detail: object) -> None:
    """Startup failure (import or auth). `detail` is an exception or a text
    description; either way it is logged, and shown only to admins."""
    exc = detail if isinstance(detail, BaseException) else None
    if exc is None and detail:
        try:
            _LOG.error("Startup problem: %s", detail)
        except Exception:
            pass
    show_error("HSF", exc, level="error", message=STARTUP_MESSAGE)
    if exc is None and detail and _is_admin() and st is not None:
        try:
            with st.expander("Technical details (admin)", expanded=False):
                st.code(str(detail))
        except Exception:
            pass
