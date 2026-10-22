"""Sentry for the API and the web app's crash reports.

No-op unless SENTRY_DSN is set on hsf-api (the Streamlit app reads the same
variable through ui.monitoring). The API's own unhandled errors reach Sentry
through sentry_sdk's FastAPI integration; crash reports from the web app arrive
on POST /v1/client-errors and are forwarded here as messages tagged web.
Sentry's default scrubbing drops Authorization and Cookie headers.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# One JavaScript stack line: Chrome "at fn (url:line:col)" / "at url:line:col",
# Firefox and Safari "fn@url:line:col".
_FRAME = re.compile(r"^\s*(?:at\s+)?(?:(?P<fn>[^\s(@]+)\s+\(|(?P<fn2>[^\s@]*)@)?"
                    r"(?P<url>https?://[^\s()]+?):(?P<line>\d+):(?P<col>\d+)\)?\s*$")


def parse_js_stack(stack: Optional[str]) -> List[Dict[str, Any]]:
    """Sentry frames (oldest call first) from a browser stack string. Query strings are
    dropped so frames match the uploaded files."""
    frames = []
    for line in (stack or "").splitlines():
        m = _FRAME.match(line)
        if not m:
            continue
        frames.append({"function": (m.group("fn") or m.group("fn2") or "?")[:200],
                       "abs_path": m.group("url").split("?")[0].split("#")[0][:500],
                       "lineno": int(m.group("line")), "colno": int(m.group("col")), "in_app": True})
    return list(reversed(frames))[-50:]


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
            frames = parse_js_stack(stack)
            if frames and report.get("release"):
                # A real JavaScript exception: Sentry applies the source maps uploaded for
                # web@<commit> (web/scripts/sentry-sourcemaps.mjs) to these frames.
                value = str(report.get("message") or "Error")
                head = re.match(r"^([A-Za-z]*Error): ", (stack or "").lstrip())  # Chrome: "TypeError: ..."
                kind = head.group(1) if head else "Error"
                sentry_sdk.capture_event({
                    "platform": "javascript", "level": "error", "release": f"web@{report['release']}",
                    "exception": {"values": [{"type": kind, "value": value[:500],
                                              "stacktrace": {"frames": frames}}]},
                }, scope=scope)
            else:
                sentry_sdk.capture_message(f"Web: {report.get('message')}", level="error")
    except Exception:
        pass
