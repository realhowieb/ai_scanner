"""P1-59: HSF API v1 (api/): sign-in, token refresh/rotation, /me and /today.

The database is mocked at api.store's boundary and the Today builder at
api.today, so these run without Postgres or network.
"""
import datetime as dt
import importlib.util
import json
import time
import unittest
from unittest import mock

DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "jwt", "bcrypt"))

SECRET = "test-secret-" + "x" * 40


def _hash(pw: str) -> str:
    import bcrypt

    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(4)).decode()


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class ApiTestCase(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from api import main, ratelimit
        from api.settings import Settings

        ratelimit.reset()  # per-IP limits are process-wide; each test starts clean
        self.addCleanup(ratelimit.reset)

        self.settings = Settings(jwt_secret=SECRET, access_ttl_s=900, refresh_ttl_s=3600,
                                 cors_origins=("https://app.example.com",))
        self.accounts = {
            "pro@example.com": {"username": "pro@example.com", "full_name": "Pro User", "password": _hash("right pw"),
                                "tier": "pro", "is_admin": False, "is_active": True},
            "off@example.com": {"username": "off@example.com", "full_name": "Off", "password": _hash("right pw"),
                                "tier": "pro", "is_admin": False, "is_active": False},
            "boss@example.com": {"username": "boss@example.com", "full_name": "Boss", "password": _hash("right pw"),
                                 "tier": "basic", "is_admin": True, "is_active": True},
        }
        self.refresh = {}  # hash -> {"username", "revoked"}
        self.attempts = []
        patches = [
            mock.patch("api.store.get_account", side_effect=lambda u: self.accounts.get((u or "").strip().lower())),
            mock.patch("api.store.save_refresh_token", side_effect=self._save),
            mock.patch("api.store.use_refresh_token", side_effect=self._use),
            mock.patch("api.store.revoke_refresh_token", side_effect=self._revoke),
            mock.patch("db.users.is_login_rate_limited", side_effect=lambda u: u == "locked@example.com"),
            mock.patch("db.users.record_login_attempt",
                       side_effect=lambda u, success, **k: self.attempts.append((u, success))),
            # P1-64 push devices: in-memory, same contract as api.devices
            mock.patch("api.devices.register", side_effect=self._dev_register),
            mock.patch("api.devices.list_devices", side_effect=self._dev_list),
            mock.patch("api.devices.remove",
                       side_effect=lambda u, i: self._dev_drop(lambda d: d["username"] == u and d["id"] == i) > 0),
            mock.patch("api.devices.remove_token",
                       side_effect=lambda u, t: self._dev_drop(lambda d: d["username"] == u and d["token"] == t)),
            mock.patch("api.devices.remove_all", side_effect=lambda u: self._dev_drop(lambda d: d["username"] == u)),
        ]
        self.devices = []
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.client = TestClient(main.create_app(self.settings))

    def _save(self, token_hash, username, ttl_s, client):
        self.refresh[token_hash] = {"username": username, "revoked": False}

    def _use(self, token_hash):
        row = self.refresh.get(token_hash)
        if row is None:
            return "invalid", None
        if row["revoked"]:
            for r in self.refresh.values():
                if r["username"] == row["username"]:
                    r["revoked"] = True
            return "reused", row["username"]
        row["revoked"] = True
        return "ok", row["username"]

    def _revoke(self, token_hash):
        if token_hash in self.refresh:
            self.refresh[token_hash]["revoked"] = True
            return self.refresh[token_hash]["username"]
        return None

    _PUBLIC = ("id", "provider", "platform", "device_name", "app_version", "created_at", "last_seen_at")

    def _dev_register(self, username, token, provider, platform, device_name, app_version):
        old = next((d for d in self.devices if d["token"] == token), None)
        if old:
            self.devices.remove(old)
        d = {"id": old["id"] if old else len(self.devices) + 100, "username": username, "token": token,
             "provider": provider, "platform": platform, "device_name": device_name, "app_version": app_version,
             "created_at": "2026-10-06T00:00:00+00:00", "last_seen_at": "2026-10-06T00:00:00+00:00"}
        self.devices.append(d)
        return {k: d[k] for k in self._PUBLIC}

    def _dev_list(self, username):
        return [{k: d[k] for k in self._PUBLIC} for d in self.devices if d["username"] == username]

    def _dev_drop(self, match):
        gone = [d for d in self.devices if match(d)]
        self.devices = [d for d in self.devices if not match(d)]
        return len(gone)

    def login(self, email="pro@example.com", password="right pw"):
        return self.client.post("/v1/auth/login", json={"email": email, "password": password})

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}


class HealthAndSettingsTests(ApiTestCase):
    def test_healthz_needs_nothing(self):
        self.assertEqual(self.client.get("/healthz").json(), {"ok": True})

    def test_root_redirects_to_docs(self):
        r = self.client.get("/", follow_redirects=False)
        self.assertIn(r.status_code, (302, 307))
        self.assertEqual(r.headers["location"], "/docs")

    def test_short_secret_refuses_to_start(self):
        from api.settings import load_settings

        with mock.patch.dict("os.environ", {"API_JWT_SECRET": "short"}):
            with self.assertRaises(RuntimeError):
                load_settings()

    def test_cors_only_for_listed_origins(self):
        ok = self.client.options("/v1/me", headers={"Origin": "https://app.example.com",
                                                     "Access-Control-Request-Method": "GET"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "https://app.example.com")
        bad = self.client.options("/v1/me", headers={"Origin": "https://evil.example",
                                                      "Access-Control-Request-Method": "GET"})
        self.assertNotIn("access-control-allow-origin", bad.headers)

    def test_cors_preflight_is_cached_and_exposes_backoff_headers(self):
        ok = self.client.options("/v1/me/devices", headers={"Origin": "https://app.example.com",
                                                             "Access-Control-Request-Method": "DELETE"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers.get("access-control-max-age"), "600")
        r = self.client.get("/healthz", headers={"Origin": "https://app.example.com"})
        exposed = r.headers.get("access-control-expose-headers", "").lower()
        self.assertIn("retry-after", exposed)
        self.assertIn("x-request-id", exposed)

    def test_cors_origins_parsing_drops_unsafe_entries(self):
        from api.settings import parse_cors_origins

        with self.assertLogs("hsf_api", level="WARNING") as logs:
            got = parse_cors_origins(" https://App.Example.com/ ,*, https://x.example/path, http://evil.example,"
                                     "http://localhost:8081, https://app.example.com, https://*.example.com,,"
                                     "https://staging.example.com:8443")
        self.assertEqual(got, ("https://app.example.com", "http://localhost:8081", "https://staging.example.com:8443"))
        self.assertIn("ignored 4 entries", logs.output[0])
        self.assertEqual(parse_cors_origins(""), ())


class CompressionTests(ApiTestCase):
    def test_large_responses_are_gzipped_when_asked_and_small_ones_are_not(self):
        big = self.client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(big.headers.get("content-encoding"), "gzip")
        self.assertIn("paths", big.json())  # the client unpacks it transparently
        small = self.client.get("/healthz", headers={"Accept-Encoding": "gzip"})
        self.assertIsNone(small.headers.get("content-encoding"))
        self.assertTrue(small.headers.get("x-request-id"))


class ClientErrorTests(ApiTestCase):
    def test_web_crash_reports_are_logged_without_query_strings_and_rate_limited(self):
        from api import ratelimit

        ratelimit.reset()
        with self.assertLogs("hsf_api", level="WARNING") as logs, \
                mock.patch("api.monitoring.capture_client_error") as capture:
            r = self.client.post("/v1/client-errors", json={"message": "boom", "kind": "boundary",
                                                            "path": "/stocks/AAPL?x=secret", "digest": "d1"})
        self.assertEqual(r.status_code, 202)
        line = next(json.loads(m.split(":", 2)[2]) for m in logs.output if "web_client_error" in m)
        self.assertEqual((line["message"], line["path"], line["digest"]), ("boom", "/stocks/AAPL", "d1"))
        capture.assert_called_once()
        self.assertEqual(self.client.post("/v1/client-errors", json={"message": ""}).status_code, 422)
        for _ in range(40):
            last = self.client.post("/v1/client-errors", json={"message": "again"})
        self.assertEqual(last.status_code, 429)
        ratelimit.reset()

    def test_forwarding_is_a_no_op_without_sentry(self):
        from api.monitoring import capture_client_error

        capture_client_error({"kind": "window", "message": "x"}, "stack")  # must not raise


class DocsTests(ApiTestCase):
    def test_openapi_declares_bearer_auth_so_docs_show_authorize(self):
        spec = self.client.get("/openapi.json").json()
        schemes = spec["components"]["securitySchemes"]
        self.assertEqual(list(schemes.values())[0]["scheme"], "bearer")
        self.assertTrue(spec["paths"]["/v1/me"]["get"].get("security"))
        self.assertTrue(spec["paths"]["/v1/today"]["get"].get("security"))
        self.assertFalse(spec["paths"]["/v1/auth/login"]["post"].get("security"))

    def test_wrong_scheme_is_rejected(self):
        token = self.login().json()["access_token"]
        self.assertEqual(self.client.get("/v1/me", headers={"Authorization": f"Basic {token}"}).status_code, 401)
        self.assertEqual(self.client.get("/v1/me", headers={"Authorization": token}).status_code, 401)


class ReviewFixTests(ApiTestCase):
    """P1-59 review: refresh grace window, typed schema, 503 on outage, clear startup failure."""

    def test_grace_retry_gets_a_new_pair_without_revoking_anything(self):
        first = self.login().json()
        with mock.patch("api.store.use_refresh_token", return_value=("grace", "pro@example.com")):
            r = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["refresh_token"])

    def test_openapi_describes_every_payload(self):
        spec = self.client.get("/openapi.json").json()
        schemas = spec["components"]["schemas"]
        for name in ("TokenPair", "Me", "Today", "Recap", "SessionCard", "Mover", "Setup", "TopSetups"):
            self.assertIn(name, schemas)
        ref = lambda path, method="get": spec["paths"][path][method]["responses"]["200"]["content"][
            "application/json"]["schema"]["$ref"]
        self.assertTrue(ref("/v1/me").endswith("/Me"))
        self.assertTrue(ref("/v1/today").endswith("/Today"))
        self.assertTrue(ref("/v1/auth/login", "post").endswith("/TokenPair"))
        self.assertEqual(set(schemas["Recap"]["properties"]),
                         {"day", "title", "scans", "premarket_scans", "postmarket_scans", "entered", "left", "standouts"})

    def test_database_outage_is_503_not_500(self):
        from api import store

        with mock.patch("api.store.get_account", side_effect=store.DatabaseUnavailable("down")):
            r = self.client.post("/v1/auth/login", json={"email": "pro@example.com", "password": "right pw"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get("retry-after"), "30")
        self.assertNotIn("down", r.text)

    @unittest.skipUnless(importlib.util.find_spec("psycopg"), "needs psycopg")
    def test_connection_drop_mid_query_is_503(self):
        import psycopg

        token = self.login().json()["access_token"]
        with mock.patch("api.store.get_account", side_effect=psycopg.OperationalError("server closed")):
            r = self.client.get("/v1/me", headers=self.auth(token))
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("server closed", r.text)

    def test_missing_secret_fails_startup_with_the_reason(self):
        import asyncio

        from api import main

        with mock.patch.dict("os.environ", {"API_JWT_SECRET": ""}):
            app = main._module_app()
        sent = []

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(msg):
            sent.append(msg)

        asyncio.run(app({"type": "lifespan"}, receive, send))
        self.assertEqual(sent[0]["type"], "lifespan.startup.failed")
        self.assertIn("API_JWT_SECRET", sent[0]["message"])


class LoginTests(ApiTestCase):
    def test_login_returns_tokens_and_records_success(self):
        r = self.login(" Pro@Example.com ")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["token_type"], "bearer")
        self.assertEqual(body["expires_in"], 900)
        self.assertTrue(body["refresh_token"])
        self.assertIn(("pro@example.com", True), self.attempts)
        # only the hash is stored, never the token itself
        self.assertNotIn(body["refresh_token"], self.refresh)

    def test_password_with_stray_spaces_is_accepted_like_the_web_app(self):
        self.assertEqual(self.login(password="right pw  ").status_code, 200)

    def test_wrong_password_unknown_and_inactive_get_the_same_answer(self):
        for email, pw in (("pro@example.com", "nope"), ("ghost@example.com", "right pw"), ("off@example.com", "right pw")):
            r = self.login(email, pw)
            self.assertEqual(r.status_code, 401, email)
            self.assertEqual(r.json()["detail"], "Email or password is incorrect.")
        self.assertIn(("pro@example.com", False), self.attempts)

    def test_rate_limited(self):
        self.assertEqual(self.login("locked@example.com").status_code, 429)

    def test_plain_text_stored_password_is_never_accepted(self):
        self.accounts["pro@example.com"]["password"] = "right pw"
        self.assertEqual(self.login().status_code, 401)


class TokenTests(ApiTestCase):
    def test_me_with_access_token(self):
        token = self.login().json()["access_token"]
        r = self.client.get("/v1/me", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        me = r.json()
        self.assertEqual(me["email"], "pro@example.com")
        self.assertEqual(me["plan"], "pro")
        self.assertEqual(me["plan_label"], "Pro")
        self.assertEqual(me["alert_limit"], 5)
        self.assertTrue(me["entitlements"]["can_day_trader"])
        self.assertFalse(me["entitlements"]["can_ai_notes"])
        self.assertFalse(me["entitlements"]["can_admin_panel"])
        self.assertNotIn("password", r.text)

    def test_admin_gets_everything(self):
        token = self.login("boss@example.com").json()["access_token"]
        me = self.client.get("/v1/me", headers=self.auth(token)).json()
        self.assertEqual(me["plan"], "admin")
        self.assertTrue(all(me["entitlements"].values()))

    def test_missing_bad_expired_and_wrong_secret_tokens(self):
        import jwt

        from api import tokens

        self.assertEqual(self.client.get("/v1/me").status_code, 401)
        self.assertEqual(self.client.get("/v1/me", headers=self.auth("garbage")).status_code, 401)
        old = tokens.create_access_token("pro@example.com", self.settings, now=time.time() - 3600)
        self.assertEqual(self.client.get("/v1/me", headers=self.auth(old)).status_code, 401)
        forged = jwt.encode({"sub": "pro@example.com", "iat": int(time.time()), "exp": int(time.time()) + 600,
                             "iss": "hsf-api", "typ": "access"}, "another-secret-" + "y" * 40, algorithm="HS256")
        self.assertEqual(self.client.get("/v1/me", headers=self.auth(forged)).status_code, 401)
        none_alg = jwt.encode({"sub": "pro@example.com", "exp": int(time.time()) + 600}, None, algorithm="none")
        self.assertEqual(self.client.get("/v1/me", headers=self.auth(none_alg)).status_code, 401)

    def test_deactivated_after_sign_in_loses_access(self):
        token = self.login().json()["access_token"]
        self.accounts["pro@example.com"]["is_active"] = False
        self.assertEqual(self.client.get("/v1/me", headers=self.auth(token)).status_code, 401)

    def test_plan_change_applies_to_existing_token(self):
        token = self.login().json()["access_token"]
        self.accounts["pro@example.com"]["tier"] = "basic"
        me = self.client.get("/v1/me", headers=self.auth(token)).json()
        self.assertEqual(me["plan"], "basic")
        self.assertFalse(me["entitlements"]["can_day_trader"])


class RefreshTests(ApiTestCase):
    def test_refresh_rotates_and_old_token_stops_working(self):
        first = self.login().json()
        r = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(r.status_code, 200)
        second = r.json()
        self.assertNotEqual(second["refresh_token"], first["refresh_token"])
        self.assertEqual(self.client.get("/v1/me", headers=self.auth(second["access_token"])).status_code, 200)

    def test_reusing_a_rotated_token_revokes_every_session(self):
        first = self.login().json()
        second = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}).json()
        replay = self.client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(replay.status_code, 401)
        # the legitimate newer token is revoked too
        again = self.client.post("/v1/auth/refresh", json={"refresh_token": second["refresh_token"]})
        self.assertEqual(again.status_code, 401)

    def test_logout_revokes(self):
        tok = self.login().json()["refresh_token"]
        self.assertEqual(self.client.post("/v1/auth/logout", json={"refresh_token": tok}).status_code, 204)
        self.assertEqual(self.client.post("/v1/auth/refresh", json={"refresh_token": tok}).status_code, 401)

    def test_unknown_refresh_token(self):
        r = self.client.post("/v1/auth/refresh", json={"refresh_token": "z" * 64})
        self.assertEqual(r.status_code, 401)


class TodayEndpointTests(ApiTestCase):
    def test_today_passes_the_accounts_entitlements(self):
        token = self.login().json()["access_token"]
        payload = {"as_of": "2026-10-05T12:40:00+00:00", "market": {"phase": "premarket"}, "errors": []}
        with mock.patch("api.today.build_today", return_value=payload) as build:
            r = self.client.get("/v1/today", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["market"]["phase"], "premarket")
        self.assertIsNone(r.json()["recap"])
        now, ent = build.call_args[0]
        self.assertTrue(ent["can_day_trader"])
        self.assertEqual(now.tzinfo, dt.timezone.utc)

    def test_today_requires_sign_in(self):
        self.assertEqual(self.client.get("/v1/today").status_code, 401)


    def test_market_tape(self):
        token = self.login().json()["access_token"]
        quotes = [{"symbol": "SPY", "last": 777.3, "chg_pct": -0.23}, {"symbol": "DIA", "last": 511.0, "chg_pct": None}]
        with mock.patch("api.today.tape_quotes", return_value=quotes):
            r = self.client.get("/v1/market/tape", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"quotes": quotes})
        self.assertEqual(self.client.get("/v1/market/tape").status_code, 401)

    def test_today_personal_passes_the_user_and_marker(self):
        token = self.login().json()["access_token"]
        payload = {"new_since": {"marker": "11:10", "tickers": [{"ticker": "TER", "score": 75}], "total": 1}, "errors": []}
        with mock.patch("api.today.build_personal", return_value=payload) as build:
            r = self.client.get("/v1/today/me?seen=11&baseline=10", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["new_since"]["tickers"], [{"ticker": "TER", "score": 75}])
        self.assertEqual(build.call_args[0][1:], (11, 10))
        self.assertEqual(self.client.get("/v1/today/me").status_code, 401)

if __name__ == "__main__":
    unittest.main()
