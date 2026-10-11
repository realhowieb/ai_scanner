"""Browser push (api/webpush.py): RFC 8291 encryption, VAPID header, subscription
checks, sending and cleanup of subscriptions the browser dropped."""
import base64
import importlib.util
import json
import os
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

CRYPTO = all(importlib.util.find_spec(m) for m in ("cryptography", "jwt", "httpx"))


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _keys():
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    return _b64(key.private_numbers().private_value.to_bytes(32, "big"))


# The receiving browser's public point and 16-byte auth value from RFC 8291 Appendix A
# (published test data, not credentials).
UA_PUBLIC, UA_AUTH = ("BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4",
                      "BTBZMqHH6r4Tts7J_aSIgg")


@unittest.skipUnless(CRYPTO, "needs cryptography, PyJWT and httpx")
class EncryptionTests(unittest.TestCase):
    def test_matches_the_rfc_8291_appendix_a_vector(self):
        from cryptography.hazmat.primitives.asymmetric import ec

        from api.webpush import _b64d, encrypt

        server = ec.derive_private_key(int.from_bytes(_b64d("yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw"), "big"),
                                       ec.SECP256R1())
        out = encrypt(b"When I grow up, I want to be a watermelon", UA_PUBLIC, UA_AUTH,
                      salt=_b64d("DGv6ra1nlYgDCS1FRnbzlw"), server_key=server)
        self.assertEqual(_b64(out), "DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYW"
                                    "AmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPTpK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQ"
                                    "exSgSxsj_Qulcy4a-fN")


class SubscriptionTests(unittest.TestCase):
    def test_accepts_a_real_subscription_and_refuses_anything_else(self):
        from api.webpush import InvalidSubscription, endpoint_of, subscription_token

        tok = subscription_token("https://fcm.googleapis.com/fcm/send/abc", UA_PUBLIC, UA_AUTH)
        self.assertEqual(endpoint_of(tok), "https://fcm.googleapis.com/fcm/send/abc")
        self.assertEqual(tok, subscription_token(" https://fcm.googleapis.com/fcm/send/abc ", UA_PUBLIC, UA_AUTH))
        for bad in (("http://push.example/x", UA_PUBLIC, UA_AUTH), ("https://push.example/x", UA_AUTH, UA_AUTH),
                    ("https://push.example/x", UA_PUBLIC, UA_PUBLIC), ("javascript:alert(1)", UA_PUBLIC, UA_AUTH)):
            with self.assertRaises(InvalidSubscription):
                subscription_token(*bad)


@unittest.skipUnless(CRYPTO, "needs cryptography, PyJWT and httpx")
class SendTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"VAPID_PRIVATE_KEY": _keys(), "VAPID_SUBJECT": "mailto:ops@example.com"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_off_without_keys_or_subject(self):
        from api import webpush

        self.assertTrue(webpush.enabled())
        with mock.patch.dict(os.environ, {"VAPID_SUBJECT": ""}):
            self.assertFalse(webpush.enabled())
        with mock.patch.dict(os.environ, {"HSF_WEB_PUSH_ENABLED": "0"}):
            self.assertFalse(webpush.config()["enabled"])
        with mock.patch.dict(os.environ, {"VAPID_PRIVATE_KEY": ""}):
            self.assertEqual(webpush.config(), {"enabled": False, "allowed": True, "public_key": None})
            self.assertEqual(webpush.notify_user("a@example.com", "t", "b"), 0)

    def test_posts_encrypted_payload_with_vapid_and_drops_gone_subscriptions(self):
        import jwt

        from api import webpush

        live = webpush.subscription_token("https://push.example/live", UA_PUBLIC, UA_AUTH)
        gone = webpush.subscription_token("https://push.example/gone", UA_PUBLIC, UA_AUTH)
        posted = []

        class _Resp:
            def __init__(self, code):
                self.status_code = code

        class _Client:
            def __init__(self, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, content, headers):
                posted.append((url, content, headers))
                return _Resp(410 if url.endswith("gone") else 201)

        targets = [{"id": 1, "provider": "webpush", "token": live}, {"id": 2, "provider": "webpush", "token": gone},
                   {"id": 3, "provider": "expo", "token": "ExpoPushToken[abcdefghijkl]"}]
        pro = {"username": "a@example.com", "tier": "pro", "is_active": True}
        with mock.patch("api.devices.devices_for_user", return_value=targets), \
                mock.patch("api.store.get_account", return_value=pro), \
                mock.patch("api.devices.remove") as remove, mock.patch("httpx.Client", _Client):
            self.assertEqual(webpush.notify_user("A@Example.com", "HSF alert", "AAPL crossed 200"), 1)
        self.assertEqual([p[0] for p in posted], ["https://push.example/live", "https://push.example/gone"])
        remove.assert_called_once_with("a@example.com", 2)
        headers = posted[0][2]
        self.assertEqual(headers["Content-Encoding"], "aes128gcm")
        token, k = headers["Authorization"][len("vapid t="):].split(", k=")
        self.assertEqual(k, webpush.public_key())
        claims = jwt.decode(token, options={"verify_signature": False})
        self.assertEqual((claims["aud"], claims["sub"]), ("https://push.example", "mailto:ops@example.com"))
        self.assertGreater(len(posted[0][1]), 86)

    def test_a_dead_push_service_never_raises(self):
        from api import webpush

        tok = webpush.subscription_token("https://push.example/x", UA_PUBLIC, UA_AUTH)

        class _Boom:
            def __init__(self, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, *a, **k):
                raise OSError("down")

        with mock.patch("api.devices.devices_for_user", return_value=[{"id": 1, "provider": "webpush", "token": tok}]), \
                mock.patch("api.store.get_account", return_value={"tier": "pro"}), mock.patch("httpx.Client", _Boom):
            self.assertEqual(webpush.notify_user("a@example.com", "t", "b"), 0)

    def test_only_pro_and_above_get_pushed(self):
        from api import webpush

        tok = webpush.subscription_token("https://push.example/x", UA_PUBLIC, UA_AUTH)
        targets = [{"id": 1, "provider": "webpush", "token": tok}]
        for account in ({"tier": "basic"}, {"tier": "premium", "is_active": False}, None):
            with mock.patch("api.devices.devices_for_user", return_value=targets), \
                    mock.patch("api.store.get_account", return_value=account), \
                    mock.patch.object(webpush, "send") as send:
                self.assertEqual(webpush.notify_user("a@example.com", "t", "b"), 0)
            send.assert_not_called()
        self.assertTrue(webpush.plan_allows({"tier": "basic", "is_admin": True}))
        self.assertTrue(webpush.plan_allows({"tier": "premium"}))
        self.assertFalse(webpush.plan_allows({"tier": "basic"}))


@unittest.skipUnless(DEPS and CRYPTO, "needs fastapi, httpx, PyJWT, bcrypt and cryptography")
class RouteTests(ApiTestCase):
    def test_config_subscribe_and_unsubscribe(self):
        h = self.auth(self.login().json()["access_token"])
        self.assertEqual(self.client.get("/v1/web-push/config", headers=h).json(),
                         {"enabled": False, "allowed": True, "public_key": None})
        body = {"endpoint": "https://push.example/x", "keys": {"p256dh": UA_PUBLIC, "auth": UA_AUTH}, "device_name": "Chrome"}
        self.assertEqual(self.client.post("/v1/me/web-push", json=body, headers=h).status_code, 503)
        device = {"id": 7, "provider": "webpush", "platform": "web", "device_name": "Chrome", "app_version": None,
                  "created_at": None, "last_seen_at": None}
        with mock.patch.dict(os.environ, {"VAPID_PRIVATE_KEY": _keys(), "VAPID_SUBJECT": "mailto:ops@example.com"}), \
                mock.patch("api.devices.register", return_value=device) as register, \
                mock.patch("api.devices.remove_web_endpoint") as unsub:
            cfg = self.client.get("/v1/web-push/config", headers=h).json()
            r = self.client.post("/v1/me/web-push", json=body, headers=h)
            bad = self.client.post("/v1/me/web-push", json={**body, "endpoint": "http://push.example/x"}, headers=h)
            gone = self.client.request("DELETE", "/v1/me/web-push", json={"endpoint": "https://push.example/x"}, headers=h)
        self.assertTrue(cfg["enabled"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["provider"], "webpush")
        self.assertEqual(json.loads(register.call_args.args[1])["endpoint"], "https://push.example/x")
        self.assertEqual(register.call_args.args[2:4], ("webpush", "web"))
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(gone.status_code, 204)
        unsub.assert_called_once()

    def test_free_plan_gets_the_upgrade_note_and_cant_subscribe(self):
        self.accounts["free@example.com"] = {**self.accounts["pro@example.com"], "username": "free@example.com",
                                             "tier": "basic"}
        h = self.auth(self.login("free@example.com").json()["access_token"])
        body = {"endpoint": "https://push.example/x", "keys": {"p256dh": UA_PUBLIC, "auth": UA_AUTH}}
        with mock.patch.dict(os.environ, {"VAPID_PRIVATE_KEY": _keys(), "VAPID_SUBJECT": "mailto:ops@example.com"}), \
                mock.patch("api.devices.register") as register:
            cfg = self.client.get("/v1/web-push/config", headers=h).json()
            r = self.client.post("/v1/me/web-push", json=body, headers=h)
        self.assertEqual(cfg, {"enabled": True, "allowed": False, "public_key": None})
        self.assertEqual(r.status_code, 403)
        self.assertIn("part of Pro", r.json()["detail"])
        register.assert_not_called()


class HookTests(unittest.TestCase):
    def test_price_alert_fire_hooks_run_and_never_raise(self):
        from billing_service import realtime_alerts as rt

        seen = []
        ok = lambda u, m: seen.append((u, m))  # noqa: E731

        def boom(u, m):
            raise RuntimeError("x")

        with mock.patch.object(rt, "_fire_hooks", []):
            rt.register_fire_hook(boom)
            rt.register_fire_hook(ok)
            rt.register_fire_hook(ok)
            rt.run_fire_hooks("a@example.com", "AAPL crossed 200")
        self.assertEqual(seen, [("a@example.com", "AAPL crossed 200")])


if __name__ == "__main__":
    unittest.main()
