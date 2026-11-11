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


@unittest.skipUnless(HAS_HTTPX, "needs httpx")
class PlanLimitInputs(unittest.TestCase):
    def test_valid_distinct_thresholds_reach_limit_and_cleanup(self):
        import api_acceptance
        import httpx

        for limit in (1, 5, 25):
            with self.subTest(limit=limit):
                alerts = {}
                submitted = []

                def handler(request):
                    import json
                    if request.method == "GET":
                        return httpx.Response(200, json={"limit": limit, "used": len(alerts)})
                    if request.method == "DELETE":
                        alerts.pop(int(request.url.path.rsplit("/", 1)[-1]))
                        return httpx.Response(204)
                    body = json.loads(request.content)
                    threshold = body["threshold"]
                    submitted.append(threshold)
                    if not 0 < threshold <= 1_000_000:
                        return httpx.Response(422)
                    if len(alerts) >= limit:
                        return httpx.Response(403)
                    if threshold in alerts.values():
                        return httpx.Response(409)
                    aid = len(alerts) + 1
                    alerts[aid] = threshold
                    return httpx.Response(201, json={"id": aid})

                run = api_acceptance.Run("https://api.test")
                run.http = httpx.Client(transport=httpx.MockTransport(handler))
                api_acceptance.plan_limit(run, {"h": {}, "role": "TEST"})
                self.assertEqual(run.results[-1]["status"], "PASS")
                self.assertEqual(len(set(submitted)), limit + 1)
                self.assertEqual(alerts, {})
                run.http.close()


if __name__ == "__main__":
    unittest.main()
