"""P1-64: push device registration for the mobile app.

API wiring runs against the in-memory device fake in tests.test_api_v1; the
SQL in api.devices runs against a throwaway Postgres when HSF_TEST_PG_URL is set.
"""
import os
import unittest

from tests.test_api_account import NEW, AccountApiTestCase
from tests.test_api_v1 import DEPS, ApiTestCase

APNS = "a1" * 32                                  # 64 hex chars
FCM = "fGcM_token:APA91b" + "x" * 120
EXPO = "ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]"
PG_URL = os.environ.get("HSF_TEST_PG_URL")


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class DeviceApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        pair = self.login().json()
        self.h, self.refresh_token = self.auth(pair["access_token"]), pair["refresh_token"]

    def register(self, token=APNS, platform="ios", headers=None, **extra):
        return self.client.post("/v1/me/devices", headers=headers or self.h,
                                json={"push_token": token, "platform": platform, **extra})

    def test_register_list_remove(self):
        r = self.register(device_name="Howie's iPhone", app_version="1.0.0")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["provider"], d["platform"], d["device_name"]), ("apns", "ios", "Howie's iPhone"))
        self.assertNotIn("push_token", d)
        self.assertNotIn("token", d)
        listed = self.client.get("/v1/me/devices", headers=self.h).json()
        self.assertEqual([x["id"] for x in listed], [d["id"]])
        self.assertNotIn("token", listed[0])
        self.assertEqual(self.client.delete(f"/v1/me/devices/{d['id']}", headers=self.h).status_code, 204)
        self.assertEqual(self.client.get("/v1/me/devices", headers=self.h).json(), [])
        self.assertEqual(self.client.delete(f"/v1/me/devices/{d['id']}", headers=self.h).status_code, 404)

    def test_provider_defaults_and_validation(self):
        self.assertEqual(self.register(FCM, "android").json()["provider"], "fcm")
        self.assertEqual(self.register(EXPO, "ios").json()["provider"], "expo")
        self.assertEqual(self.register("not a token!!", "ios").status_code, 400)
        self.assertEqual(self.register(FCM, "ios", provider="apns").status_code, 400)   # not hex
        self.assertEqual(self.register(APNS, "android", provider="apns").status_code, 400)
        self.assertEqual(self.register(APNS, "web").status_code, 422)
        self.assertEqual(self.register(APNS, "ios", provider="sms").status_code, 422)

    def test_reregister_is_idempotent(self):
        a = self.register().json()
        b = self.register(app_version="1.0.1").json()
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(len(self.devices), 1)
        self.assertEqual(b["app_version"], "1.0.1")

    def test_needs_sign_in_and_is_per_account(self):
        self.assertEqual(self.client.post("/v1/me/devices", json={"push_token": APNS, "platform": "ios"}).status_code, 401)
        self.assertEqual(self.client.get("/v1/me/devices").status_code, 401)
        mine = self.register().json()
        boss = self.auth(self.login("boss@example.com").json()["access_token"])
        self.assertEqual(self.client.get("/v1/me/devices", headers=boss).json(), [])
        self.assertEqual(self.client.delete(f"/v1/me/devices/{mine['id']}", headers=boss).status_code, 404)
        self.assertEqual(len(self.devices), 1)
        # the same phone signing in to another account moves the token
        self.register(headers=boss)
        self.assertEqual(self.client.get("/v1/me/devices", headers=self.h).json(), [])
        self.assertEqual(len(self.client.get("/v1/me/devices", headers=boss).json()), 1)

    def test_logout_with_push_token_removes_only_that_device(self):
        self.register()
        self.register(FCM, "android")
        r = self.client.post("/v1/auth/logout", json={"refresh_token": self.refresh_token, "push_token": APNS})
        self.assertEqual(r.status_code, 204)
        self.assertEqual([d["token"] for d in self.devices], [FCM])

    def test_logout_cannot_remove_another_accounts_device(self):
        boss = self.auth(self.login("boss@example.com").json()["access_token"])
        self.register(headers=boss)
        self.client.post("/v1/auth/logout", json={"refresh_token": self.refresh_token, "push_token": APNS})
        self.assertEqual(len(self.devices), 1)
        r = self.client.post("/v1/auth/logout", json={"refresh_token": "x" * 40, "push_token": APNS})
        self.assertEqual(r.status_code, 204)
        self.assertEqual(len(self.devices), 1)

    def test_logout_without_push_token_keeps_devices(self):
        self.register()
        self.client.post("/v1/auth/logout", json={"refresh_token": self.refresh_token})
        self.assertEqual(len(self.devices), 1)

    def test_refresh_token_theft_removes_all_devices(self):
        self.register()
        new = self.client.post("/v1/auth/refresh", json={"refresh_token": self.refresh_token})
        self.assertEqual(new.status_code, 200)
        replay = self.client.post("/v1/auth/refresh", json={"refresh_token": self.refresh_token})
        self.assertEqual(replay.status_code, 401)
        self.assertEqual(self.devices, [])


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class DeviceSignOutTests(AccountApiTestCase):
    def test_password_change_removes_every_device(self):
        h = self.auth(self.login().json()["access_token"])
        self.client.post("/v1/me/devices", headers=h, json={"push_token": APNS, "platform": "ios"})
        self.client.post("/v1/me/devices", headers=h, json={"push_token": FCM, "platform": "android"})
        boss = self.auth(self.login("boss@example.com").json()["access_token"])
        self.client.post("/v1/me/devices", headers=boss, json={"push_token": EXPO, "platform": "ios"})
        ok = self.client.post("/v1/me/password", headers=h, json={"current_password": "right pw", "new_password": NEW})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual([d["username"] for d in self.devices], ["boss@example.com"])


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class DeviceStorePostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            conn.execute("DROP TABLE IF EXISTS api_push_devices")
        os.environ["DATABASE_URL"] = PG_URL
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)
        from api import devices

        self.d = devices
        with psycopg.connect(PG_URL) as conn:  # schema_once may already have run in this process
            devices.ensure_devices_schema.__wrapped__(conn)

    def test_register_moves_caps_and_removes(self):
        d = self.d
        a = d.register("pro@example.com", APNS, "apns", "ios", "iPhone", "1.0")
        again = d.register("pro@example.com", APNS, "apns", "ios", "iPhone", "1.1")
        self.assertEqual(a["id"], again["id"])
        self.assertEqual(again["app_version"], "1.1")
        self.assertNotIn("token", again)
        moved = d.register("boss@example.com", APNS, "apns", "ios", None, None)
        self.assertEqual(moved["id"], a["id"])
        self.assertEqual(d.list_devices("pro@example.com"), [])
        self.assertEqual([x["token"] for x in d.devices_for_user("boss@example.com")], [APNS])
        for i in range(d.MAX_DEVICES + 3):
            d.register("pro@example.com", f"{i:02d}" * 32, "apns", "ios", None, None)
        kept = d.list_devices("pro@example.com")
        self.assertEqual(len(kept), d.MAX_DEVICES)
        self.assertFalse(d.remove("pro@example.com", moved["id"]))  # not theirs
        self.assertTrue(d.remove("pro@example.com", kept[0]["id"]))
        self.assertEqual(d.remove_token("pro@example.com", APNS), 0)
        self.assertEqual(d.remove_token("boss@example.com", APNS), 1)
        self.assertEqual(d.remove_all("pro@example.com"), d.MAX_DEVICES - 1)


if __name__ == "__main__":
    unittest.main()
