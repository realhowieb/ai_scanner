"""iPhone app push (api/apns.py): provider token, request shape, plan check and cleanup
of device tokens Apple no longer accepts."""
import importlib.util
import json
import os
import unittest
from unittest import mock

CRYPTO = all(importlib.util.find_spec(m) for m in ("cryptography", "jwt", "httpx"))
TOKEN = "a" * 64


def _p8() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


class _Client:
    def __init__(self, answers):
        self.answers, self.posts = answers, []

    def post(self, url, content=None, headers=None):
        self.posts.append((url, json.loads(content), headers))
        return self.answers.get(url.rsplit("/", 1)[-1], _Resp(200))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@unittest.skipUnless(CRYPTO, "needs cryptography, PyJWT and httpx")
class ApnsTests(unittest.TestCase):
    def setUp(self):
        from api import apns

        apns._jwt_cache.clear()
        self.env = mock.patch.dict(os.environ, {"APNS_KEY_P8": _p8().replace("\n", "\\n"), "APNS_KEY_ID": "KEY123",
                                                "APNS_TEAM_ID": "TEAM456"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_off_until_the_key_is_set_or_when_switched_off(self):
        from api import apns, webpush

        self.assertTrue(apns.enabled())
        self.assertTrue(webpush.config()["apns_enabled"])
        with mock.patch.dict(os.environ, {"HSF_APNS_ENABLED": "0"}):
            self.assertFalse(apns.enabled())
        with mock.patch.dict(os.environ, {"APNS_KEY_ID": ""}):
            self.assertFalse(apns.enabled())
            self.assertEqual(apns.notify_user("a@example.com", "t", "b"), 0)

    def test_provider_token_is_es256_with_key_id_and_is_reused(self):
        import jwt

        from api import apns

        tok = apns.provider_token(now=1_000_000)
        self.assertEqual(jwt.get_unverified_header(tok), {"alg": "ES256", "kid": "KEY123", "typ": "JWT"})
        self.assertEqual(jwt.decode(tok, options={"verify_signature": False}), {"iss": "TEAM456", "iat": 1_000_000})
        self.assertEqual(apns.provider_token(now=1_000_000 + 600), tok)
        self.assertNotEqual(apns.provider_token(now=1_000_000 + 3000), tok)

    def test_sends_alerts_and_drops_tokens_apple_refuses(self):
        from api import apns

        gone, stale, ok = "b" * 64, "c" * 64, TOKEN
        targets = [{"id": 1, "provider": "apns", "token": ok}, {"id": 2, "provider": "apns", "token": gone},
                   {"id": 3, "provider": "apns", "token": stale}, {"id": 4, "provider": "webpush", "token": "{}"}]
        client = _Client({gone: _Resp(410, {"reason": "Unregistered"}),
                          stale: _Resp(400, {"reason": "BadDeviceToken"})})
        with mock.patch("api.devices.devices_for_user", return_value=targets), \
                mock.patch("api.store.get_account", return_value={"tier": "pro"}), \
                mock.patch("api.devices.remove") as remove, \
                mock.patch("httpx.Client", return_value=client):
            self.assertEqual(apns.notify_user("A@example.com", "HSF alert", "NVDA crossed 200"), 1)
        self.assertEqual(len(client.posts), 3)
        url, body, headers = client.posts[0]
        self.assertEqual(url, f"https://api.push.apple.com/3/device/{ok}")
        self.assertEqual(body["aps"]["alert"], {"title": "HSF alert", "body": "NVDA crossed 200"})
        self.assertEqual(body["url"], "/alerts")
        self.assertEqual(headers["apns-topic"], "com.hsfinest.app")
        self.assertEqual(headers["apns-push-type"], "alert")
        self.assertTrue(headers["authorization"].startswith("bearer "))
        self.assertEqual(sorted(c.args for c in remove.call_args_list), [("a@example.com", 2), ("a@example.com", 3)])

    def test_sandbox_and_plan_check(self):
        from api import apns

        with mock.patch.dict(os.environ, {"APNS_SANDBOX": "1"}):
            self.assertEqual(apns.host(), "api.sandbox.push.apple.com")
        targets = [{"id": 1, "provider": "apns", "token": TOKEN}]
        with mock.patch("api.devices.devices_for_user", return_value=targets), \
                mock.patch("api.store.get_account", return_value={"tier": "basic"}), \
                mock.patch.object(apns, "send") as send:
            self.assertEqual(apns.notify_user("a@example.com", "t", "b"), 0)
        send.assert_not_called()

    def test_alert_fired_reaches_the_iphone_app_even_if_browsers_fail(self):
        from api import webpush

        with mock.patch.object(webpush, "notify_user", return_value=0), \
                mock.patch("api.apns.notify_user", return_value=1) as native:
            webpush.alert_fired("a@example.com", "AAPL crossed 200")
        native.assert_called_once_with("a@example.com", "HSF alert", "AAPL crossed 200", "/alerts")


if __name__ == "__main__":
    unittest.main()
