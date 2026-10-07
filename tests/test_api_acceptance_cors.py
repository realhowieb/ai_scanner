import os
import sys
import unittest

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import api_acceptance  # noqa: E402


def _run_with(allow_origin):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"database": "ok"})
        if request.url.path == "/docs":
            return httpx.Response(200, text="swagger")
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"paths": {}})
        headers = {"access-control-allow-origin": allow_origin} if allow_origin else {}
        return httpx.Response(400 if not allow_origin else 200, headers=headers)

    run = api_acceptance.Run("https://api.test")
    run.http = httpx.Client(transport=httpx.MockTransport(handler))
    return run


def _cors(run):
    return {r["check"]: r["status"] for r in run.results if "CORS" in r["check"]}


class CorsChecksWithoutOrigin(unittest.TestCase):
    def test_no_origin_still_checks_unlisted_origin_is_refused(self):
        run = _run_with(None)
        api_acceptance.deployment_checks(run, None)
        self.assertEqual(_cors(run), {"CORS refuses an unlisted origin": "PASS"})

    def test_no_origin_fails_when_any_origin_is_allowed(self):
        run = _run_with("https://evil.example")
        api_acceptance.deployment_checks(run, None)
        self.assertEqual(_cors(run), {"CORS refuses an unlisted origin": "FAIL"})


if __name__ == "__main__":
    unittest.main()
