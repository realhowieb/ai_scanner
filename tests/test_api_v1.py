"""P1-59: HSF API v1 (api/): sign-in, token refresh/rotation, /me and /today.

The database is mocked at api.store's boundary and the Today builder at
api.today, so these run without Postgres or network.
"""
import datetime as dt
import importlib.util
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

        from api import main
        from api.settings import Settings

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
        ]
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

    def login(self, email="pro@example.com", password="right pw"):
        return self.client.post("/v1/auth/login", json={"email": email, "password": password})

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}


class HealthAndSettingsTests(ApiTestCase):
    def test_healthz_needs_nothing(self):
        self.assertEqual(self.client.get("/healthz").json(), {"ok": True})

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
        with mock.patch("api.today.build_today", return_value={"ok": 1}) as build:
            r = self.client.get("/v1/today", headers=self.auth(token))
        self.assertEqual(r.json(), {"ok": 1})
        now, ent = build.call_args[0]
        self.assertTrue(ent["can_day_trader"])
        self.assertEqual(now.tzinfo, dt.timezone.utc)

    def test_today_requires_sign_in(self):
        self.assertEqual(self.client.get("/v1/today").status_code, 401)


if __name__ == "__main__":
    unittest.main()
