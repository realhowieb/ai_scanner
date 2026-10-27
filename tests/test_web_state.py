"""Getting started checklist, unread alert badge, in-app feedback and the shared stock page
(api/web_state.py, api/share.py, api/routes/onboarding.py), through the API.

db.web_state runs its real SQL on an in-memory SQLite database; watchlists and ticker
alerts use the in-memory fakes from test_api_v1_data. No Postgres, no network.
"""
import datetime as dt
import sqlite3
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS
from tests.test_api_v1_data import DataApiBase

UTC = dt.timezone.utc


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class WebStateApiTests(DataApiBase):
    def setUp(self):
        super().setUp()
        from db import web_state

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        web_state.set_connection_factory(lambda: self.conn)
        self.addCleanup(lambda: web_state.set_connection_factory(None))
        mock.patch("api.alert_rules.list_rules", return_value=[]).start()
        mock.patch("api.chat_delivery.summary", return_value={"telegram": {"connected": False},
                                                              "discord": {"connected": False}}).start()
        mock.patch("api.account.is_verified", return_value=True).start()
        self.notified = []
        mock.patch("services.ops_notify.notify_ops",
                   side_effect=lambda subject, text, **kw: self.notified.append((subject, text))).start()

    def steps(self, h=None):
        r = self.client.get("/v1/me/getting-started", headers=h or self.h)
        self.assertEqual(r.status_code, 200)
        return r.json(), {s["id"]: s["done"] for s in r.json()["steps"]}

    def test_requires_sign_in(self):
        for method, path in (("get", "/v1/me/getting-started"), ("post", "/v1/me/getting-started/dismiss"),
                             ("get", "/v1/me/alerts/unread"), ("post", "/v1/me/alerts/seen"),
                             ("post", "/v1/me/feedback"), ("get", "/v1/admin/feedback")):
            self.assertEqual(getattr(self.client, method)(path).status_code, 401, path)

    def test_steps_tick_from_what_the_account_did(self):
        body, done = self.steps()
        self.assertTrue(body["show"])
        self.assertEqual(done, {"watchlist": False, "alert": False, "delivery": False, "guide": False})
        delivery = next(s for s in body["steps"] if s["id"] == "delivery")
        self.assertEqual(delivery["title"], "Get alerts on your phone")  # Pro: Telegram/Discord/browser

        wid = self.client.post("/v1/watchlists", json={"name": "Mine"}, headers=self.h).json()["id"]
        _, done = self.steps()
        self.assertFalse(done["watchlist"])  # an empty list doesn't count
        self.client.post(f"/v1/watchlists/{wid}/tickers", json={"tickers": ["NVDA"]}, headers=self.h)
        self.client.post("/v1/alerts", json={"type": "price", "ticker": "NVDA", "threshold": 200, "direction": "above"},
                         headers=self.h)
        self.devices.append({"id": 1, "username": "pro@example.com", "token": "t" * 20, "provider": "webpush",
                             "platform": "web", "device_name": None, "app_version": None,
                             "created_at": "2026-10-06T00:00:00+00:00", "last_seen_at": "2026-10-06T00:00:00+00:00"})
        body = self.client.post("/v1/me/getting-started/guide", headers=self.h).json()
        done = {s["id"]: s["done"] for s in body["steps"]}
        self.assertTrue(done["watchlist"])
        self.assertTrue(done["delivery"])
        self.assertTrue(done["guide"])
        self.assertTrue(done["alert"])
        self.assertFalse(body["show"])  # every step done: the card goes away by itself
        self.assertEqual(body["done"], 4)

    def test_free_plan_confirms_email_instead(self):
        h = self.auth(self.login("free@example.com").json()["access_token"])
        with mock.patch("api.account.is_verified", return_value=False):
            body, done = self.steps(h)
        delivery = next(s for s in body["steps"] if s["id"] == "delivery")
        self.assertEqual(delivery["title"], "Confirm your email")
        self.assertFalse(done["delivery"])

    def test_dismiss_hides_the_card_for_good(self):
        body = self.client.post("/v1/me/getting-started/dismiss", headers=self.h).json()
        self.assertFalse(body["show"])
        self.assertTrue(body["dismissed"])
        self.assertFalse(self.steps()[0]["show"])

    def test_kill_switch(self):
        with mock.patch.dict("os.environ", {"HSF_GETTING_STARTED": "0"}):
            self.assertFalse(self.steps()[0]["show"])

    def test_unread_counts_since_the_last_alerts_visit(self):
        now = dt.datetime.now(UTC)
        fired = [now - dt.timedelta(hours=h) for h in (1, 30, 24 * 10)]
        calls = []

        def events(user, *, limit, triggered_after=None, **kw):
            calls.append(triggered_after)
            return [{"id": i} for i, t in enumerate(fired) if triggered_after is None or t > triggered_after][:limit]

        with mock.patch("api.alert_rules.list_events", side_effect=events):
            first = self.client.get("/v1/me/alerts/unread", headers=self.h).json()
            self.assertEqual(first["count"], 2)  # no visit yet: the last 7 days
            self.assertIsNone(first["seen_at"])
            seen = self.client.post("/v1/me/alerts/seen", headers=self.h).json()
            self.assertEqual(seen["count"], 0)
            self.assertTrue(seen["seen_at"])
            self.assertEqual(self.client.get("/v1/me/alerts/unread", headers=self.h).json()["count"], 0)

    def test_unread_survives_a_feed_failure(self):
        with mock.patch("api.alert_rules.list_events", side_effect=RuntimeError("down")):
            r = self.client.get("/v1/me/alerts/unread", headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 0)

    def test_feedback_is_stored_and_sent_to_admins(self):
        r = self.client.post("/v1/me/feedback", headers=self.h, json={
            "kind": "problem", "message": "The chart is blank on my phone", "path": "/stocks/NVDA?x=1",
            "build": "abc123"})
        self.assertEqual(r.status_code, 201)
        rows = self.conn.execute("SELECT user_id, kind, message, path, build FROM hsf_feedback").fetchall()
        self.assertEqual([tuple(x) for x in rows],
                         [("pro@example.com", "problem", "The chart is blank on my phone", "/stocks/NVDA", "abc123")])
        self.assertEqual(self.notified[0][0], "HSF feedback: Problem")
        self.assertIn("pro@example.com", self.notified[0][1])
        self.assertIn("/stocks/NVDA", self.notified[0][1])

    def test_feedback_validation_and_limit(self):
        self.assertEqual(self.client.post("/v1/me/feedback", headers=self.h, json={"message": "x"}).status_code, 422)
        from api import ratelimit

        with mock.patch.dict(ratelimit.LIMITS, {"feedback": (2, 3600)}):
            ratelimit.reset()
            codes = [self.client.post("/v1/me/feedback", headers=self.h, json={"message": "hello there"}).status_code
                     for _ in range(3)]
        self.assertEqual(codes, [201, 201, 429])

    def test_admin_feedback_is_admins_only(self):
        self.assertEqual(self.client.get("/v1/admin/feedback", headers=self.h).status_code, 403)


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class PublicStockTests(DataApiBase):
    DETAIL = {"ticker": "MXL", "price": 12.5, "change_pct": 3.2, "scan_at": "2026-10-23T14:35:00+00:00",
              "in_latest_scan": True, "has_setup": True, "hsf_score": 77.4, "primary_setup": "Breakout",
              "reasons": ["Broke above the 20-day high", "Volume 3x normal"], "signals": ["breakout", "gapper"],
              "risks": ["Earnings in 4 days"]}

    def setUp(self):
        super().setUp()
        mock.patch("api.symbol_search._rows", return_value=[{"ticker": "MXL", "name": "MaxLinear, Inc."},
                                                            {"ticker": "NVDA", "name": "NVIDIA"}]).start()

    def test_ranked_stock_shows_score_and_one_reason_without_sign_in(self):
        with mock.patch("api.scans.stock_detail", return_value=self.DETAIL):
            r = self.client.get("/v1/public/stocks/mxl")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ticker": "MXL", "name": "MaxLinear, Inc.", "price": 12.5, "change_pct": 3.2,
                                    "scan_at": "2026-10-23T14:35:00+00:00", "ranked": True, "hsf_score": 77,
                                    "setup": "Breakout", "reason": "Broke above the 20-day high",
                                    "hidden_insights": 4})

    def test_unranked_stock_shows_price_only(self):
        detail = {**self.DETAIL, "ticker": "NVDA", "hsf_score": 5, "has_setup": False}
        with mock.patch("api.scans.stock_detail", return_value=detail):
            body = self.client.get("/v1/public/stocks/NVDA").json()
        self.assertFalse(body["ranked"])
        self.assertIsNone(body["hsf_score"])
        self.assertIsNone(body["reason"])
        self.assertEqual(body["hidden_insights"], 0)

    def test_unknown_symbol_is_refused_before_any_lookup(self):
        with mock.patch("api.scans.stock_detail") as detail:
            self.assertEqual(self.client.get("/v1/public/stocks/ZZZZQ").status_code, 404)
        detail.assert_not_called()
