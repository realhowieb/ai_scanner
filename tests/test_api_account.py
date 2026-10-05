"""P1-59 account step: sign-up, email verification, password reset and change,
email preferences, billing links, per-IP limits.

The web app's account functions are mocked at their module boundary (db.users,
db.password_reset, db.email_verification, db.email_prefs, ui.email_utils,
ui.checkout, ui.auth_sessions), so these tests check the API's wiring and rules
against the same functions the Streamlit pages call.
"""
import json
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

GOOD = "Blue-Harbor-Lantern-42"
NEW = "Quiet-Maple-Orbit-77"


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class AccountApiTestCase(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.created, self.sent, self.verified = [], [], set()
        self.reset_tokens = {}       # token -> username
        self.password_updates, self.web_signouts, self.app_signouts = [], [], []
        self.prefs = {}
        p = mock.patch
        for target, fn in (
            ("db.users.get_user_by_username", lambda u: self.accounts.get((u or "").lower())),
            ("db.users.find_username_by_display_name",
             lambda n: next((k for k, a in self.accounts.items() if (a.get("full_name") or "").lower() == n.lower()), None)),
            ("db.users.create_user_account", self._create),
            ("db.users.update_neon_user_password", self._update_pw),
            ("db.email_verification.create_verification_token", lambda u: f"verify-{u}-token"),
            ("db.email_verification.consume_verification_token", self._consume_verify),
            ("db.email_verification.is_email_verified", lambda u: u in self.verified),
            ("ui.email_utils.send_verification_email", lambda to_address, verify_url: self.sent.append(("verify", to_address, verify_url)) or True),
            ("ui.email_utils.send_password_reset_email", lambda to, url: self.sent.append(("reset", to, url)) or True),
            ("db.password_reset.create_reset_token", self._new_reset),
            ("db.password_reset.peek_reset_token", lambda t: self.reset_tokens.get(t)),
            ("db.password_reset.consume_reset_token", lambda t: self.reset_tokens.pop(t, None)),
            ("ui.auth_sessions.revoke_user_sessions", lambda u: self.web_signouts.append(u) or 1),
            ("api.store.revoke_all_refresh_tokens", self._revoke_all),
            ("db.email_prefs.get_prefs", lambda u: {"digest": True, "evening": True, "alerts": True, **self.prefs}),
            ("db.email_prefs.set_prefs", lambda u, **ch: self.prefs.update(ch) or True),
        ):
            p(target, side_effect=fn).start()

    # fakes ---------------------------------------------------------------------------------------
    def _create(self, email, password_hash, tier="basic", full_name=None):
        self.created.append({"email": email, "hash": password_hash, "tier": tier, "name": full_name})
        self.accounts[email] = {"username": email, "full_name": full_name, "password": password_hash,
                                "tier": tier, "is_admin": False, "is_active": True}
        return {"username": email}

    def _update_pw(self, username, hashed):
        self.password_updates.append((username, hashed))
        self.accounts[username]["password"] = hashed
        return True

    def _consume_verify(self, token):
        user = token[len("verify-"):-len("-token")] if token.startswith("verify-") else None
        if user:
            self.verified.add(user)
        return user

    def _new_reset(self, username, ttl_minutes=30):
        token = f"reset{len(self.reset_tokens)}abcdefgh"
        self.reset_tokens[token] = username
        return token

    def _revoke_all(self, username, reason):
        self.app_signouts.append((username, reason))
        for r in self.refresh.values():
            if r["username"] == username and not r["revoked"]:
                r["revoked"], r["reason"] = True, reason
        return 1

    def _use(self, token_hash):
        """Mirrors api.store.use_refresh_token: password sign-outs answer invalid, no sweep."""
        row = self.refresh.get(token_hash)
        if row and row["revoked"] and row.get("reason") in ("password_change", "password_reset"):
            return "invalid", None
        return super()._use(token_hash)

    def signup(self, **over):
        body = {"email": "New@Example.com", "password": GOOD, "username": "newbie", "accept_terms": True, **over}
        return self.client.post("/v1/auth/signup", json=body, headers=over.pop("headers", None) or {})


class SignupTests(AccountApiTestCase):
    def test_signup_creates_a_free_account_and_signs_in(self):
        r = self.signup()
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual((body["email"], body["verification_sent"]), ("new@example.com", True))
        self.assertTrue(body["access_token"] and body["refresh_token"])
        made = self.created[0]
        self.assertEqual((made["email"], made["tier"], made["name"]), ("new@example.com", "basic", "newbie"))
        self.assertTrue(made["hash"].startswith("$2"))
        self.assertNotIn(GOOD, json.dumps(self.created))
        self.assertTrue(self.sent[0][2].endswith("/verify_email?token=verify-new@example.com-token"))
        me = self.client.get("/v1/me", headers=self.auth(body["access_token"])).json()
        self.assertEqual((me["plan"], me["email_verified"]), ("basic", False))

    def test_signup_rules_match_the_web_form(self):
        cases = [({"email": "pro@example.com"}, 409), ({"username": "Pro User"}, 400),
                 ({"username": "PRO"}, 201), ({"password": "password123"}, 400), ({"password": "short"}, 400),
                 ({"accept_terms": False}, 400), ({"email": "not-an-email"}, 400), ({"username": "a@b"}, 400)]
        for i, (over, code) in enumerate(cases):
            from api import ratelimit

            ratelimit.reset()
            body = {"email": f"case{i}@example.com", **over}
            r = self.signup(**body)
            self.assertEqual(r.status_code, code, (over, r.text))
        r = self.signup(email="x9@example.com", username="Pro User".replace(" ", ""))
        # a display name already used by another account is taken
        self.accounts["pro@example.com"]["full_name"] = "taken"
        from api import ratelimit

        ratelimit.reset()
        self.assertEqual(self.signup(email="y@example.com", username="TAKEN").status_code, 409)

    def test_email_not_configured_still_creates_the_account(self):
        with mock.patch("ui.email_utils.send_verification_email", return_value=False):
            r = self.signup()
        self.assertEqual(r.status_code, 201)
        self.assertFalse(r.json()["verification_sent"])

    def test_signup_is_limited_per_address(self):
        codes = [self.signup(email=f"n{i}@example.com", username=f"user{i}").status_code for i in range(6)]
        self.assertEqual(codes[:5], [201] * 5)
        self.assertEqual(codes[5], 429)

    def test_forwarded_for_cannot_be_spoofed_to_dodge_the_limit(self):
        for i in range(6):
            r = self.client.post("/v1/auth/signup", headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"},
                                 json={"email": f"s{i}@example.com", "password": GOOD, "username": f"s{i}",
                                       "accept_terms": True})
        self.assertEqual(r.status_code, 429)  # the last (proxy-added) address is the one counted


class VerificationTests(AccountApiTestCase):
    def test_verify_with_token(self):
        self.assertEqual(self.client.post("/v1/auth/verify-email", json={"token": "verify-pro@example.com-token"}).status_code, 200)
        self.assertIn("pro@example.com", self.verified)
        self.assertEqual(self.client.post("/v1/auth/verify-email", json={"token": "bogus-token-123"}).status_code, 400)

    def test_resend_only_for_the_signed_in_account_and_limited(self):
        h = self.auth(self.login().json()["access_token"])
        codes = [self.client.post("/v1/me/verify-email", headers=h).status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        self.assertTrue(all(s[1] == "pro@example.com" for s in self.sent))
        self.verified.add("pro@example.com")
        self.assertIn("already verified", self.client.post("/v1/me/verify-email", headers=h).json()["message"])
        self.assertEqual(self.client.post("/v1/me/verify-email").status_code, 401)


class PasswordResetTests(AccountApiTestCase):
    def test_same_answer_whether_or_not_the_account_exists(self):
        a = self.client.post("/v1/auth/password-reset", json={"email": "pro@example.com"})
        b = self.client.post("/v1/auth/password-reset", json={"email": "nobody@example.com"})
        c = self.client.post("/v1/auth/password-reset", json={"email": "off@example.com"})  # deactivated
        self.assertEqual((a.status_code, b.status_code, c.status_code), (202, 202, 202))
        self.assertEqual(a.json(), b.json())
        self.assertEqual([s[1] for s in self.sent], ["pro@example.com"])
        self.assertIn("/reset_password?token=", self.sent[0][2])

    def test_confirm_sets_the_password_and_signs_out_everywhere(self):
        pair = self.login().json()
        self.client.post("/v1/auth/password-reset", json={"email": "pro@example.com"})
        token = next(iter(self.reset_tokens))
        weak = self.client.post("/v1/auth/password-reset/confirm", json={"token": token, "new_password": "password123"})
        self.assertEqual(weak.status_code, 400)
        self.assertIn(token, self.reset_tokens)  # a rejected password doesn't use up the link
        ok = self.client.post("/v1/auth/password-reset/confirm", json={"token": token, "new_password": NEW})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(self.web_signouts, ["pro@example.com"])
        self.assertEqual(self.app_signouts, [("pro@example.com", "password_reset")])
        self.assertEqual(self.client.post("/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}).status_code, 401)
        self.assertEqual(self.login(password=NEW).status_code, 200)
        again = self.client.post("/v1/auth/password-reset/confirm", json={"token": token, "new_password": NEW})
        self.assertEqual(again.status_code, 400)

    def test_reset_request_limited_per_address(self):
        codes = [self.client.post("/v1/auth/password-reset", json={"email": "x@example.com"}).status_code for _ in range(6)]
        self.assertEqual(codes[-1], 429)


class PasswordChangeTests(AccountApiTestCase):
    def test_change_password(self):
        pair = self.login().json()
        h = self.auth(pair["access_token"])
        bad = self.client.post("/v1/me/password", headers=h, json={"current_password": "nope", "new_password": NEW})
        self.assertEqual(bad.status_code, 400)
        weak = self.client.post("/v1/me/password", headers=h, json={"current_password": "right pw", "new_password": "short"})
        self.assertEqual(weak.status_code, 400)
        ok = self.client.post("/v1/me/password", headers=h, json={"current_password": "right pw", "new_password": NEW})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(self.app_signouts, [("pro@example.com", "password_change")])
        self.assertEqual(self.client.post("/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}).status_code, 401)
        self.assertEqual(self.client.post("/v1/auth/refresh", json={"refresh_token": ok.json()["refresh_token"]}).status_code, 200)
        self.assertEqual(self.login(password=NEW).status_code, 200)


class PrefsAndBillingTests(AccountApiTestCase):
    def setUp(self):
        super().setUp()
        self.h = self.auth(self.login().json()["access_token"])

    def test_email_preferences(self):
        self.assertEqual(self.client.get("/v1/me/email-preferences", headers=self.h).json(),
                         {"digest": True, "evening": True, "alerts": True})
        r = self.client.patch("/v1/me/email-preferences", headers=self.h, json={"evening": False})
        self.assertEqual(r.json(), {"digest": True, "evening": False, "alerts": True})

    def test_checkout_needs_a_verified_email(self):
        r = self.client.post("/v1/billing/checkout", headers=self.h, json={"plan": "pro"})
        self.assertEqual(r.status_code, 403)
        self.verified.add("pro@example.com")
        with mock.patch("ui.checkout.create_checkout_url",
                        return_value=("https://checkout.stripe.com/c/pay/cs_test_1", None)) as m:
            r = self.client.post("/v1/billing/checkout", headers=self.h, json={"plan": "premium", "interval": "year"})
        self.assertEqual(r.json(), {"url": "https://checkout.stripe.com/c/pay/cs_test_1", "mode": "checkout"})
        m.assert_called_once_with("pro@example.com", "premium", "year")
        self.assertEqual(self.client.post("/v1/billing/checkout", headers=self.h, json={"plan": "gold"}).status_code, 422)

    def test_billing_failures_never_leak_details(self):
        self.verified.add("pro@example.com")
        with mock.patch("ui.checkout.create_checkout_url", return_value=(None, "sk_live_secret stack trace")):
            r = self.client.post("/v1/billing/checkout", headers=self.h, json={"plan": "pro"})
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("sk_live", r.text)

    def test_portal(self):
        calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append((url, json, headers))
            return mock.Mock(status_code=200, json=lambda: {"portal_url": "https://billing.stripe.com/p/session/x"})

        with mock.patch("ui.checkout.billing_auth_headers", return_value={"X-HSF-Auth": "billing-token"}), \
                mock.patch("requests.post", side_effect=fake_post):
            r = self.client.post("/v1/billing/portal", headers=self.h, json={"flow": "cancel"})
        self.assertEqual(r.json(), {"url": "https://billing.stripe.com/p/session/x", "mode": "portal"})
        url, body, headers = calls[0]
        self.assertTrue(url.endswith("/create-portal-session"))
        self.assertEqual((body["email"], body.get("flow"), headers), ("pro@example.com", "cancel", {"X-HSF-Auth": "billing-token"}))

        with mock.patch("ui.checkout.billing_auth_headers", return_value={"X-HSF-Auth": "t"}), \
                mock.patch("requests.post", return_value=mock.Mock(status_code=400)):
            r = self.client.post("/v1/billing/portal", headers=self.h, json={})
        self.assertEqual(r.status_code, 404)
        with mock.patch("ui.checkout.billing_auth_headers", return_value={"X-HSF-Auth": "t"}), \
                mock.patch("requests.post", return_value=mock.Mock(status_code=500)):
            r = self.client.post("/v1/billing/portal", headers=self.h, json={})
        self.assertEqual(r.status_code, 502)
        with mock.patch("config.BILLING_API_BASE", "file:///etc/passwd"):
            r = self.client.post("/v1/billing/portal", headers=self.h, json={})
        self.assertEqual(r.status_code, 502)  # only http(s) billing URLs are called
        self.assertEqual(self.client.post("/v1/billing/portal", headers=self.h, json={"flow": "delete"}).status_code, 422)

    def test_all_account_routes_need_sign_in(self):
        self.assertEqual(self.client.get("/v1/me/email-preferences").status_code, 401)
        for path in ("/v1/me/password", "/v1/billing/checkout", "/v1/billing/portal"):
            self.assertEqual(self.client.post(path, json={}).status_code, 401, path)


if __name__ == "__main__":
    unittest.main()
