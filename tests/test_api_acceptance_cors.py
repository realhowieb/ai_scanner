import importlib.util
import os
import sys
import unittest

HAS_HTTPX = importlib.util.find_spec("httpx") is not None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _run_with(allow_origin):
    import api_acceptance
    import httpx

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


def _deployment_checks(run):
    import api_acceptance
    api_acceptance.deployment_checks(run, None)


def _cors(run):
    return {r["check"]: r["status"] for r in run.results if "CORS" in r["check"]}


@unittest.skipUnless(HAS_HTTPX, "needs httpx")
class CorsChecksWithoutOrigin(unittest.TestCase):
    def test_no_origin_still_checks_unlisted_origin_is_refused(self):
        run = _run_with(None)
        _deployment_checks(run)
        self.assertEqual(_cors(run), {"CORS refuses an unlisted origin": "PASS"})

    def test_no_origin_fails_when_any_origin_is_allowed(self):
        run = _run_with("https://evil.example")
        _deployment_checks(run)
        self.assertEqual(_cors(run), {"CORS refuses an unlisted origin": "FAIL"})


if __name__ == "__main__":
    unittest.main()
