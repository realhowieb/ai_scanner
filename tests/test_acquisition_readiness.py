from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class AcquisitionAttributionTests(unittest.TestCase):
    def test_reddit_utm_classification(self):
        from ui.acquisition import attribution_from_params

        attr = attribution_from_params({
            "utm_source": "reddit",
            "utm_medium": "community",
            "utm_campaign": "beta25",
        })
        self.assertEqual(attr["source"], "reddit")
        self.assertEqual(attr["utm_medium"], "community")
        self.assertEqual(attr["utm_campaign"], "beta25")

    def test_direct_and_other_unknown_classification(self):
        from ui.acquisition import classify_source

        self.assertEqual(classify_source({}), "direct")
        self.assertEqual(classify_source({"utm_source": "newsletter"}), "other_unknown")
        self.assertEqual(classify_source({}, "https://www.reddit.com/r/stocks/"), "reddit")

    def test_user_hash_does_not_expose_email(self):
        from ui.acquisition import user_hash

        hashed = user_hash("Trader@Example.com")
        self.assertIsNotNone(hashed)
        self.assertNotIn("Trader", hashed)
        self.assertNotIn("@", hashed)
        self.assertEqual(len(hashed), 24)

    def test_track_event_uses_hash_and_attribution_without_pii(self):
        from ui import acquisition

        captured = {}

        class Cursor:
            def execute(self, sql, params=None):
                if "INSERT INTO acquisition_events" in sql:
                    captured["params"] = params

            def close(self):
                pass

        class Conn:
            def cursor(self):
                return Cursor()

            def commit(self):
                pass

            def close(self):
                pass

        with mock.patch("db.engine.get_neon_conn", return_value=Conn()), \
             mock.patch.object(acquisition, "current_attribution", return_value={
                 "source": "reddit",
                 "utm_source": "reddit",
                 "utm_medium": "community",
                 "utm_campaign": "beta25",
             }):
            self.assertTrue(acquisition.track_event("signup_completed", username="new@example.com", plan="basic"))

        params = captured["params"]
        self.assertEqual(params[0], "signup_completed")
        self.assertEqual(params[1], "reddit")
        self.assertEqual(params[2], "reddit")
        self.assertNotIn("new@example.com", repr(params))
        self.assertEqual(len(params[8]), 24)

    def test_authenticated_session_event_is_not_emitted_on_every_rerun(self):
        from ui import acquisition

        events = []
        fake_st = mock.Mock()
        fake_st.session_state = {"hsf_restored_session": True}
        with mock.patch.object(acquisition, "st", fake_st), \
             mock.patch.object(acquisition, "track_event", side_effect=lambda name, **kw: events.append(name)):
            acquisition.track_authenticated_session_once("u@example.com", plan="basic")
            acquisition.track_authenticated_session_once("u@example.com", plan="basic")
        self.assertEqual(events, ["return_session"])


class PublicAcquisitionCopyTests(unittest.TestCase):
    def test_landing_has_primary_free_cta_and_no_credit_card_copy(self):
        from ui.landing import details_html, hero_html

        hero = hero_html("")
        details = details_html()
        self.assertIn("Start scanning free", hero)
        self.assertIn("No credit card required", hero)
        self.assertIn("What HSF AI surfaces and why", details)
        self.assertIn("Thousands of symbols", details)
        self.assertIn("Free", details)
        self.assertIn("Pro", details)
        self.assertIn("Premium", details)

    def test_canonical_positioning_is_preserved(self):
        from ui.product_copy import POSITIONING_SHORT, TAGLINE

        self.assertEqual(TAGLINE, "Turn the whole market into a short list.")
        self.assertIn("Signal intelligence, not a prediction machine.", POSITIONING_SHORT)

    def test_signup_and_post_auth_routing_contract(self):
        auth = (ROOT / "ui" / "auth.py").read_text()
        app = (ROOT / "app.py").read_text()
        self.assertIn('st.form_submit_button("Start scanning free")', auth)
        self.assertIn('track_event("signup_started"', auth)
        self.assertIn('track_event("signup_completed"', auth)
        self.assertIn('st.session_state["hsf_start_scanner_after_auth"] = True', auth)
        self.assertIn('st.session_state.pop("hsf_start_scanner_after_auth", False)', app)
        self.assertIn('track_scanner_view_once(username, plan=tier_key)', app)

    def test_pricing_still_comes_from_entitlement_source(self):
        landing = (ROOT / "ui" / "landing.py").read_text()
        pricing = (ROOT / "ui" / "pricing.py").read_text()
        self.assertIn("plans_html()", landing)
        self.assertIn("FEATURE_MIN_TIER", pricing)
        self.assertIn("ALERT_LIMIT_BY_TIER", pricing)


class AcquisitionEventWiringTests(unittest.TestCase):
    def test_scanner_billing_and_webhook_events_are_wired(self):
        scans = (ROOT / "ui" / "scans.py").read_text()
        billing = (ROOT / "pages" / "billing.py").read_text()
        service = (ROOT / "billing_service" / "main.py").read_text()
        self.assertIn('track_event(\n                        "first_scanner_use"', scans)
        self.assertIn('"upgrade_started"', billing)
        self.assertIn("def _record_paid_conversion", service)
        self.assertIn('"successful_paid_conversion"', service)


if __name__ == "__main__":
    unittest.main()
