"""P1-60: hsf-api (always-on plan) runs the real-time price-alert worker.

It starts only when REALTIME_ALERTS_ENABLED=1, and hsf-api installs the
worker's own libraries (psycopg2, httpx).
"""
import importlib.util
from pathlib import Path

import pytest

from billing_service import realtime_alerts

ROOT = Path(__file__).resolve().parents[1]
needs_api = pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "jwt")),
                               reason="needs fastapi and PyJWT (hsf-api deps)")


def test_api_requirements_include_alert_worker_libraries():
    lines = [line.split("#", 1)[0].strip() for line in (ROOT / "api" / "requirements.txt").read_text().splitlines()]
    names = {line.split("=")[0] for line in lines if line and not line.startswith("-")}
    assert {"psycopg2-binary", "httpx"} <= names


@needs_api
def test_worker_starts_only_when_enabled(monkeypatch):
    import api.main as api_main

    started = []
    monkeypatch.setattr(realtime_alerts.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw["name"])})())
    monkeypatch.delenv("REALTIME_ALERTS_ENABLED", raising=False)
    api_main._start_realtime_alerts()
    assert started == []
    monkeypatch.setenv("REALTIME_ALERTS_ENABLED", "1")
    api_main._start_realtime_alerts()
    assert started == ["realtime-alerts"]


@needs_api
def test_start_failure_never_breaks_the_api(monkeypatch):
    import api.main as api_main

    def boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(realtime_alerts, "start_background_worker", boom)
    api_main._start_realtime_alerts()  # logs a warning, does not raise
