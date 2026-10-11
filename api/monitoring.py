"""Sentry for the API and the web app's crash reports.

No-op unless SENTRY_DSN is set on hsf-api (the Streamlit app reads the same
variable through ui.monitoring). The API's own unhandled errors reach Sentry
through sentry_sdk's FastAPI integration; crash reports from the web app arrive
on POST /v1/client-errors and are forwarded here as messages tagged web.
Sentry's default scrubbing drops Authorization and Cookie headers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def init_api_monitoring() -> bool:
    try:
        from ui.monitoring import init_sentry

        return init_sentry("api")
    except Exception:  # monitoring never stops the API from starting
        return False


def capture_client_error(report: Dict[str, Any], stack: Optional[str] = None) -> None:
    """Forward one web crash report to Sentry when it's active; otherwise a no-op."""
    try:
        from ui import monitoring

        if not monitoring._initialized:
            return
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("component", "web")
            scope.set_tag("web_error_kind", report.get("kind"))
            if report.get("path"):
                scope.set_tag("web_path", report["path"])
            if report.get("request_id"):
                scope.set_tag("request_id", report["request_id"])
            scope.set_context("web_error", {**report, "stack": (stack or "")[:2000] or None})
            scope.fingerprint = ["web", str(report.get("kind")), str(report.get("message"))[:120]]
            sentry_sdk.capture_message(f"Web: {report.get('message')}", level="error")
    except Exception:
        pass
