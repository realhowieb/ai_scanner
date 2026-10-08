"""P1-60: hsf-api (always-on plan) runs the real-time price-alert worker.

It starts only when REALTIME_ALERTS_ENABLED=1, and hsf-api installs the
worker's own libraries (psycopg2, httpx).
"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

from billing_service import realtime_alerts

ROOT = Path(__file__).resolve().parents[1]
API_DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "jwt"))


class ApiRequirementsTest(unittest.TestCase):
    def test_api_requirements_include_alert_worker_libraries(self):
        lines = [line.split("#", 1)[0].strip()
                 for line in (ROOT / "api" / "requirements.txt").read_text().splitlines()]
        names = {line.split("=")[0] for line in lines if line and not line.startswith("-")}
        self.assertLessEqual({"psycopg2-binary", "httpx"}, names)


@unittest.skipUnless(API_DEPS, "needs fastapi and PyJWT (hsf-api deps)")
class ApiStartsWorkerTest(unittest.TestCase):
    def test_worker_starts_only_when_enabled(self):
        import api.main as api_main

        started = []
        fake_thread = lambda **kw: type("T", (), {"start": lambda self: started.append(kw["name"])})()  # noqa: E731
        with mock.patch.object(realtime_alerts.threading, "Thread", fake_thread):
            with mock.patch.dict(os.environ, {"REALTIME_ALERTS_ENABLED": "0"}):
                api_main._start_realtime_alerts()
            self.assertEqual(started, [])
            with mock.patch.dict(os.environ, {"REALTIME_ALERTS_ENABLED": "1"}):
                api_main._start_realtime_alerts()
        self.assertEqual(started, ["realtime-alerts"])

    def test_start_failure_never_breaks_the_api(self):
        import api.main as api_main

        with mock.patch.object(realtime_alerts, "start_background_worker", side_effect=RuntimeError("no db")):
            api_main._start_realtime_alerts()  # logs a warning, does not raise
