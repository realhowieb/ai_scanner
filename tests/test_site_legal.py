"""Terms of Service for every account: the /terms text, the HSF_LEGAL_PAGES_ENABLED switch,
and recording acceptance at sign-up and at monthly/yearly checkout. Plus the landing page's
live top-setups preview."""
import importlib.util as _ilu
import os
import unittest
from unittest import mock

from domain import legal
from tests.test_founding_legal import DbBase

ON = {legal.GENERAL_PAGES_FLAG: "1"}
# api.legal reaches api.founding, which imports fastapi (absent on the slim CI jobs).
_NEEDS_API = unittest.skipUnless(_ilu.find_spec("fastapi"), "needs fastapi")


class TermsTextTests(unittest.TestCase):
    def test_terms_of_service_text(self):
        cur = legal.current(legal.load(env={"HSF_SUPPORT_EMAIL": "help@hsf.test", "HSF_GOVERNING_LAW": "Ohio"}))
        terms = cur[legal.SITE_TERMS]
        self.assertEqual((terms["version"], terms["status"], terms["path"]), (1, "proposed", "/terms"))
        text = terms["text"].lower()
        for phrase in ("not give financial, investment", "renew automatically", "cancel anytime from account",
                       "at least 18", "help@hsf.test", "laws of ohio", "hsfinest llc", "/privacy",
                       "acceptable use", "founding membership terms"):
            self.assertIn(phrase, text)

    def test_general_pages_switch(self):
        self.assertFalse(legal.general_pages_open({}))
        self.assertTrue(legal.general_pages_open(ON))
        self.assertEqual(legal.GENERAL_POLICIES, ("terms", "privacy"))
        self.assertNotIn(legal.SITE_TERMS, legal.CHECKOUT_POLICIES, "founder checkout is unchanged")


@_NEEDS_API
class SwitchTests(unittest.TestCase):
    def test_status_and_page_open(self):
        from api import legal as svc

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(legal.GENERAL_PAGES_FLAG, None)
            os.environ.pop("FOUNDING_LEGAL_PAGES_ENABLED", None)
            self.assertEqual(svc.status(), {"open": False, "policies": [], "signup_statement": None,
                                            "subscription_statement": None})
            self.assertFalse(any(svc.page_open(t) for t in legal.POLICY_TYPES))
        with mock.patch.dict(os.environ, ON):
            os.environ.pop("FOUNDING_LEGAL_PAGES_ENABLED", None)
            st = svc.status()
            self.assertTrue(st["open"])
            self.assertEqual([p["policy_type"] for p in st["policies"]], ["terms", "privacy"])
            self.assertIn("Terms of Service", st["signup_statement"])
            self.assertIn("renews automatically", st["subscription_statement"])
            self.assertEqual({t: svc.page_open(t) for t in legal.POLICY_TYPES},
                             {"terms": True, "privacy": True, "founding_terms": False, "founding_refunds": False})
        with mock.patch.dict(os.environ, {"FOUNDING_LEGAL_PAGES_ENABLED": "1"}):
            os.environ.pop(legal.GENERAL_PAGES_FLAG, None)
            self.assertEqual({t: svc.page_open(t) for t in legal.POLICY_TYPES},
                             {"terms": False, "privacy": True, "founding_terms": True, "founding_refunds": True})


@_NEEDS_API
class RecordingTests(DbBase):
    def setUp(self):
        super().setUp()
        from api import founding
        from api import legal as svc

        self.svc = svc
        founding.clear_cache()
        self.addCleanup(founding.clear_cache)

    def test_nothing_recorded_while_off(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(legal.GENERAL_PAGES_FLAG, None)
            self.assertEqual(self.svc.record_signup("A@x.com"), 0)

    def test_signup_and_checkout_record_terms_and_privacy(self):
        with mock.patch.dict(os.environ, ON):
            self.assertEqual(self.svc.record_signup("A@x.com"), 2)
            self.assertEqual(self.svc.record_signup("a@x.com"), 0, "a retry adds nothing")
            self.assertEqual(self.svc.record_checkout("a@x.com", "https://checkout.stripe.com/c/pay/cs_live_a1B2#x", "pro"), 2)
            self.assertEqual(self.svc.record_checkout("a@x.com", "https://billing.stripe.com/p/session/x", "pro"), 0)
        rows = self.ldb.acceptances(user_id="a@x.com")
        self.assertEqual({(r["kind"], r["policy_type"], r["checkout_session_id"]) for r in rows},
                         {("signup", "terms", "signup:a@x.com"), ("signup", "privacy", "signup:a@x.com"),
                          ("subscription", "terms", "cs_live_a1B2"), ("subscription", "privacy", "cs_live_a1B2")})

    def test_database_errors_never_block(self):
        with mock.patch.dict(os.environ, ON), mock.patch("api.founding.recorded", side_effect=RuntimeError("db down")):
            self.assertEqual(self.svc.record_signup("a@x.com"), 0)


class ApiTests(DbBase):
    def setUp(self):
        super().setUp()
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs fastapi, httpx, PyJWT and bcrypt")
        from api import founding
        from tests.test_api_account import AccountApiTestCase

        founding.clear_cache()
        self.addCleanup(founding.clear_cache)
        case = type("Case", (AccountApiTestCase,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case, self.client = case, case.client
        p = mock.patch.dict(os.environ, {})
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(legal.GENERAL_PAGES_FLAG, None)
        os.environ.pop("FOUNDING_LEGAL_PAGES_ENABLED", None)

    def test_terms_page_and_status(self):
        self.assertEqual(self.client.get("/v1/founding/policies/terms").status_code, 404)
        self.assertFalse(self.client.get("/v1/legal/status").json()["open"])
        os.environ[legal.GENERAL_PAGES_FLAG] = "1"
        page = self.client.get("/v1/founding/policies/terms").json()
        self.assertEqual((page["policy_type"], page["path"], page["version"]), ("terms", "/terms", 1))
        self.assertEqual(self.client.get("/v1/founding/policies/privacy").status_code, 200)
        self.assertEqual(self.client.get("/v1/founding/policies/founding_terms").status_code, 404)
        self.assertTrue(self.client.get("/v1/legal/status").json()["open"])

    def test_signup_and_checkout_record_acceptance(self):
        os.environ[legal.GENERAL_PAGES_FLAG] = "1"
        self.assertEqual(self.case.signup().status_code, 201)
        self.assertEqual({r["kind"] for r in self.ldb.acceptances(user_id="new@example.com")}, {"signup"})
        self.case.verified.add("pro@example.com")
        h = self.case.auth(self.case.login().json()["access_token"])
        with mock.patch("services.billing.create_checkout_url",
                        return_value=("https://checkout.stripe.com/c/pay/cs_test_9#f", None)):
            r = self.client.post("/v1/billing/checkout", headers=h, json={"plan": "pro"})
        self.assertEqual(r.status_code, 200, r.text)
        acc = self.ldb.acceptances(checkout_session_id="cs_test_9")
        self.assertEqual({(a["policy_type"], a["tier"], a["kind"]) for a in acc},
                         {("terms", "pro", "subscription"), ("privacy", "pro", "subscription")})

    def test_public_top_setups(self):
        rows = [{"ticker": t, "score": s, "primary_setup": "breakout", "chg_pct": 1.5} for t, s in
                (("AAA", 91.4), ("BBB", 80), ("CCC", 70))]
        fake = {"scan_at": "2026-10-29T13:35:00+00:00", "stale": False, "total": 40, "setups": rows}
        with mock.patch("api.scans.latest_scan", return_value=fake) as m:
            r = self.client.get("/v1/public/top-setups")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([s["ticker"] for s in body["setups"]], ["AAA", "BBB", "CCC"])
        self.assertEqual(body["setups"][0], {"ticker": "AAA", "hsf_score": 91, "setup": "breakout", "change_pct": 1.5})
        self.assertEqual((body["ranked_total"], body["free_rows"]), (40, 25))
        self.assertEqual(m.call_args.kwargs["min_score"], 40)
        with mock.patch("api.scans.latest_scan", side_effect=RuntimeError("no db")):
            self.assertEqual(self.client.get("/v1/public/top-setups").status_code, 503)
