"""Signed-out endpoints for the web app: plans and pricing, funnel events and the
emailed unsubscribe link."""
import unittest
from unittest import mock

from tests.test_api_account import AccountApiTestCase
from tests.test_api_v1 import DEPS


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class PublicApiTests(AccountApiTestCase):
    def setUp(self):
        super().setUp()
        self.events = []
        mock.patch("ui.acquisition.track_event",
                   side_effect=lambda name, **kw: self.events.append((name, kw)) or True).start()

    def test_plans_come_from_the_billing_page_source(self):
        from ui import pricing

        r = self.client.get("/v1/plans")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([t["id"] for t in body["tiers"]], ["basic", "pro", "premium"])
        self.assertEqual([t["price"] for t in body["tiers"]], [pricing.PRICES[t] for t in pricing.TIERS])
        self.assertEqual(len(body["rows"]), len(pricing.ROWS))
        ai = next(r for r in body["rows"] if r["label"].startswith("AI scan summaries"))
        self.assertEqual((ai["basic"], ai["pro"], ai["premium"]), (False, False, True))
        alerts = next(r for r in body["rows"] if r["label"].startswith("Alerts"))
        self.assertEqual(alerts["premium"], pricing.ALERT_LIMIT_BY_TIER["premium"])
        self.assertTrue(body["tiers"][1]["highlights"])

    def test_funnel_events_keep_only_allowed_names_and_clean_tags(self):
        r = self.client.post("/v1/events", json={"event": "landing_visit", "surface": "landing",
                                                 "attribution": {"utm_source": "reddit", "utm_campaign": "launch<script>"}})
        self.assertEqual(r.status_code, 202, r.text)
        name, kw = self.events[0]
        self.assertEqual(name, "landing_visit")
        self.assertEqual(kw["attribution"]["source"], "reddit")
        self.assertEqual(kw["attribution"]["utm_campaign"], "launchscript")
        self.assertEqual(kw["metadata"], {"surface": "landing", "app": "web"})
        self.assertEqual(self.client.post("/v1/events", json={"event": "successful_paid_conversion"}).status_code, 422)

    def test_web_signup_records_signup_completed_with_its_first_visit_tags(self):
        r = self.signup(attribution={"utm_source": "reddit", "referrer": "https://www.reddit.com/r/stocks"})
        self.assertEqual(r.status_code, 201, r.text)
        name, kw = self.events[0]
        self.assertEqual((name, kw["username"], kw["plan"]), ("signup_completed", "new@example.com", "basic"))
        self.assertEqual(kw["attribution"]["referrer_domain"], "www.reddit.com")
        self.events.clear()
        self.signup(email="other@example.com", username="other")   # app clients send no attribution
        self.assertEqual(self.events, [])

    def test_unsubscribe_link_shows_then_changes_only_on_post(self):
        with mock.patch("db.email_prefs.user_for_token", side_effect=lambda t: "pro@example.com" if t == "linktoken1234" else None):
            r = self.client.get("/v1/email-preferences/unsubscribe", params={"t": "linktoken1234"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["prefs"], {"digest": True, "evening": True, "alerts": True})
            self.assertNotIn("pro@example.com", r.text)
            self.assertEqual(self.prefs, {})
            r = self.client.post("/v1/email-preferences/unsubscribe", json={"token": "linktoken1234", "kind": "alerts"})
            self.assertEqual(r.json()["prefs"], {"digest": True, "evening": True, "alerts": False})
            r = self.client.post("/v1/email-preferences/unsubscribe", json={"token": "linktoken1234", "kind": "all"})
            self.assertEqual(r.json()["prefs"], {"digest": False, "evening": False, "alerts": False})
            bad = self.client.get("/v1/email-preferences/unsubscribe", params={"t": "wrongtoken99"})
            self.assertEqual(bad.status_code, 400)
            self.assertIn("isn't valid", bad.json()["detail"])

    def test_unsubscribe_save_failure_is_a_503(self):
        with mock.patch("db.email_prefs.user_for_token", return_value="pro@example.com"), \
                mock.patch("db.email_prefs.set_prefs", return_value=False):
            r = self.client.post("/v1/email-preferences/unsubscribe", json={"token": "linktoken1234", "kind": "digest"})
        self.assertEqual(r.status_code, 503)


del AccountApiTestCase  # don't run the parent's tests twice
if __name__ == "__main__":
    unittest.main()
