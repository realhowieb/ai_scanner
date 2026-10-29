"""Founding 100 legal pages, policy versions, the checkout acknowledgment and the founder
checkout (Stripe mocked). Real-Postgres races for seats are in test_founding_postgres.py."""
import datetime as dt
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from domain import founding as rules
from domain import legal

UTC = dt.timezone.utc
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE
ALL_ON = {name: "1" for name in rules.FLAGS}


class PolicyTextTests(unittest.TestCase):
    def setUp(self):
        self.p = legal.load(env={})

    def test_all_three_policies_load_with_a_version_and_hash(self):
        for ptype in legal.POLICY_TYPES:
            cur = legal.current(self.p)[ptype]
            self.assertEqual(cur["version"], 1)
            self.assertEqual(len(cur["content_hash"]), 64)
            self.assertEqual(cur["status"], "proposed")
            self.assertTrue(cur["sections"])
            self.assertNotRegex(cur["text"], r"\{[a-z_]+\}", "every placeholder filled")

    def test_terms_cover_every_required_section(self):
        headings = [s["heading"] for s in legal.current(self.p)[legal.TERMS]["sections"] if s["heading"]]
        self.assertEqual(len(headings), 17)
        for i, word in enumerate(["eligibility", "One-time payment", "Five-year", "activation", "features and exclusions",
                                  "API access", "Refund eligibility", "cancellation", "upgrades", "badge",
                                  "security", "Financial information", "availability", "discontinuation",
                                  "Changes", "Governing law", "Support contact"], start=1):
            self.assertTrue(headings[i - 1].startswith(f"{i}. "), headings[i - 1])
            self.assertIn(word.lower(), headings[i - 1].lower())

    def test_terms_state_the_required_conditions(self):
        text = legal.current(self.p)[legal.TERMS]["text"].lower()
        for phrase in ("$499 usd", "$999 usd", "75 founder pro and 25 founder elite", "five calendar years",
                       "never renews automatically", "no recurring membership charges",
                       "only after the payment provider confirms your payment", "cannot be transferred",
                       "not ownership of", "premium third-party market data is not included",
                       "marked \"planned\"", "consumer rights that cannot be waived", "hsfinest llc"):
            self.assertIn(phrase, text)

    def test_refund_policy_terms(self):
        text = legal.current(self.p)[legal.REFUNDS]["text"]
        for phrase in ("14 calendar days", "don't need to give a reason", "does not, by itself, stop you",
                       "within seven business days", "can't promise a date", "$999 × 36 ÷ 60 = $599.40",
                       "its own 14-day refund window", "never refused automatically",
                       "from your upgrade to your original end date"):
            self.assertIn(phrase, text)

    def test_missing_business_details_are_marked_not_invented(self):
        text = legal.current(self.p)[legal.TERMS]["text"]
        self.assertIn("[support email to be added]", text)
        self.assertIn("[governing-law state to be added]", text)
        self.assertEqual(legal.missing_details({}), ["HSF_SUPPORT_EMAIL", "HSF_GOVERNING_LAW"])

    def test_details_change_the_hash(self):
        filled = legal.load(env={"HSF_SUPPORT_EMAIL": "help@example.com", "HSF_GOVERNING_LAW": "the State of Texas"})
        a, b = legal.current(self.p)[legal.TERMS], legal.current(filled)[legal.TERMS]
        self.assertIn("help@example.com", b["text"])
        self.assertNotEqual(a["content_hash"], b["content_hash"])
        self.assertEqual(legal.missing_details({"HSF_SUPPORT_EMAIL": "x@y.z", "HSF_GOVERNING_LAW": "Texas"}), [])

    def test_bad_files_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "privacy.v2.txt").write_text("title: T\nversion: 3\neffective_date: x\nstatus: proposed\n---\nHi\n")
            with self.assertRaises(legal.PolicyError):
                legal.load(Path(d), env={})
            Path(d, "privacy.v2.txt").write_text("title: T\nversion: 2\neffective_date: x\nstatus: proposed\n---\n{nope}\n")
            with self.assertRaises(legal.PolicyError):
                legal.load(Path(d), env={})

    def test_disclosures(self):
        self.assertEqual(rules.disclosure(PRO), {
            "name": "HSF AI Founder Pro", "payment": "One-time payment: $499", "duration": "Membership duration: Five years",
            "recurring": "Recurring payments: None", "refund": "Refund guarantee: 14 days"})
        self.assertEqual(rules.disclosure(ELITE)["payment"], "One-time payment: $999")
        self.assertEqual(rules.disclosure(ELITE, "upgrade")["payment"], "One-time payment: $500")

    def test_hold_outlasts_the_stripe_session(self):
        self.assertGreaterEqual(rules.CHECKOUT_MINUTES, 30)  # Stripe's minimum
        self.assertGreater(rules.RESERVATION_MINUTES, rules.CHECKOUT_MINUTES)


class DbBase(unittest.TestCase):
    def setUp(self):
        from db import founding
        from db import legal as legal_db

        self.fdb, self.ldb = founding, legal_db
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))
        founding.set_connection_factory(lambda: self.conn)
        self.addCleanup(lambda: founding.set_connection_factory(None))

    def files(self, d, terms="Hello {legal_entity}.", version=1):
        Path(d, f"founding_terms.v{version}.txt").write_text(
            f"title: Terms\nversion: {version}\neffective_date: soon\nstatus: proposed\n---\n## One\n{terms}\n")
        return legal.load(Path(d), env={})


class PolicyVersionTests(DbBase):
    def test_sync_records_versions_once(self):
        first = self.ldb.sync(legal.load(env={}))
        again = self.ldb.sync(legal.load(env={}))
        self.assertEqual(first, again)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM legal_policy_versions").fetchone()[0],
                         len(legal.POLICY_TYPES))
        self.assertTrue(all(r["available"] for r in first.values()))

    def test_draft_text_can_change_until_someone_accepts_it(self):
        with tempfile.TemporaryDirectory() as d:
            self.ldb.sync(self.files(d))
            changed = self.ldb.sync(self.files(d, "Hello again."))
            self.assertTrue(changed["founding_terms"]["available"])
            self.assertIn("again", self.ldb.version("founding_terms", 1)["content"])
            self.ldb.record_acceptance("a@x.com", [changed["founding_terms"]["id"]], checkout_session_id="cs_1",
                                       tier=PRO, kind="purchase")
            frozen = self.ldb.sync(self.files(d, "Sneaky edit."))
            self.assertFalse(frozen["founding_terms"]["available"])
            self.assertEqual(frozen["founding_terms"]["problem"], "accepted_version_changed")
            row = self.ldb.version("founding_terms", 1)
            self.assertIn("again", row["content"], "accepted text is never replaced")
            self.assertEqual(row["content_hash"], changed["founding_terms"]["content_hash"])

    def test_new_versions_keep_history(self):
        with tempfile.TemporaryDirectory() as d:
            self.ldb.sync(self.files(d))
            self.files(d, "Version two.", version=2)
            out = self.ldb.sync(legal.load(Path(d), env={}))
            self.assertEqual(out["founding_terms"]["version"], 2)
            self.assertEqual([r["version"] for r in self.ldb.history("founding_terms")], [1, 2])

    def test_acceptances_are_insert_only_and_idempotent(self):
        ids = [r["id"] for r in self.ldb.sync(legal.load(env={})).values()]
        self.assertEqual(self.ldb.record_acceptance("A@x.com", ids, checkout_session_id="cs_1", tier=PRO,
                                                    kind="purchase", reservation_id=7), len(legal.POLICY_TYPES))
        self.assertEqual(self.ldb.record_acceptance("a@x.com", ids, checkout_session_id="cs_1", tier=PRO,
                                                    kind="purchase", reservation_id=7), 0)
        rows = self.ldb.acceptances(checkout_session_id="cs_1")
        self.assertEqual({r["policy_type"] for r in rows}, set(legal.POLICY_TYPES))
        self.assertTrue(all(r["user_id"] == "a@x.com" and len(r["content_hash"]) == 64 for r in rows))
        self.assertEqual(len(self.ldb.acceptances(user_id="a@x.com")), len(legal.POLICY_TYPES))
        with self.assertRaises(ValueError):
            self.ldb.record_acceptance("a@x.com", ids, checkout_session_id="", tier=PRO, kind="purchase")
        with self.assertRaises(sqlite3.IntegrityError):  # not a real policy version
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.ldb.record_acceptance("a@x.com", [999], checkout_session_id="cs_2", tier=PRO, kind="purchase")


def _fake_billing(user=None, subs=()):
    stripe = mock.MagicMock()
    stripe.Subscription.list.return_value = {"data": list(subs), "has_more": False}
    counter = {"n": 0}

    def create(**kw):
        counter["n"] += 1
        return types.SimpleNamespace(id=f"cs_test_{kw['metadata']['founding_reservation_id']}",
                                     url=f"https://checkout.stripe.test/{counter['n']}")

    stripe.checkout.Session.create.side_effect = create
    stripe.checkout.Session.retrieve.return_value = {"status": "open", "url": "https://checkout.stripe.test/open"}
    return types.SimpleNamespace(stripe=stripe, _get_user_by_email=lambda email: dict(user or {"email": email}))


class ApiTests(DbBase):
    def setUp(self):
        super().setUp()
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs fastapi, httpx, PyJWT and bcrypt")
        from api import founding as svc
        from tests.test_api_v1 import ApiTestCase

        self.svc = svc
        svc.clear_cache()
        self.addCleanup(svc.clear_cache)
        case = type("Case", (ApiTestCase,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case, self.client = case, case.client
        self.case.accounts["pro@example.com"]["tier"] = "basic"
        self.h = case.auth(case.login().json()["access_token"])
        self.admin = case.auth(case.login("boss@example.com").json()["access_token"])
        self.b = _fake_billing()
        for p in (mock.patch.dict(sys.modules, {"billing_service.main": self.b}),
                  mock.patch("api.billing.embedded_enabled", return_value=True)):
            p.start()
            self.addCleanup(p.stop)

    def flags(self, **over):
        env = {**ALL_ON, **{k: v for k, v in over.items()}}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)

    def body(self, tier=PRO, **over):
        offer = self.client.get("/v1/founding/offer", headers=self.h).json()
        policies = [{k: p[k] for k in ("policy_type", "version", "content_hash")} for p in offer["policies"]]
        return {"tier": tier, "acknowledged": True, "policies": policies, **over}

    def checkout(self, body, headers=None):
        return self.client.post("/v1/founding/checkout", json=body, headers=headers or self.h)

    # pages
    def test_policy_pages_hidden_until_switched_on_but_admins_can_preview(self):
        self.assertEqual(self.client.get("/v1/founding/policies/founding_terms").status_code, 404)
        self.assertEqual(self.client.get("/v1/admin/founding/policies/founding_terms", headers=self.h).status_code, 403)
        pre = self.client.get("/v1/admin/founding/policies/founding_terms", headers=self.admin)
        self.assertEqual(pre.status_code, 200)
        self.assertIn("HSF_SUPPORT_EMAIL", pre.json()["missing_details"])
        self.assertEqual(pre.json()["versions"], [1])
        self.flags()
        page = self.client.get("/v1/founding/policies/founding_refunds").json()
        self.assertEqual((page["version"], page["status"]), (1, "proposed"))
        self.assertEqual(self.client.get("/v1/founding/policies/founding_refunds?version=9").status_code, 404)
        self.assertEqual(self.client.get("/v1/founding/policies/cookies").status_code, 422)

    def test_offer_hidden_from_members_while_closed(self):
        self.assertEqual(self.client.get("/v1/founding/offer", headers=self.h).status_code, 404)
        preview = self.client.get("/v1/founding/offer", headers=self.admin).json()
        self.assertTrue(preview["preview"])
        self.assertFalse(preview["checkout_open"])
        self.assertFalse(any(t["on_sale"] for t in preview["tiers"]))

    def test_offer_shows_disclosures_seats_and_policies(self):
        self.flags()
        offer = self.client.get("/v1/founding/offer", headers=self.h).json()
        self.assertTrue(offer["checkout_open"])
        self.assertEqual([(t["tier"], t["remaining"], t["price_cents"]) for t in offer["tiers"]],
                         [(PRO, 75, 49_900), (ELITE, 25, 99_900)])
        self.assertEqual(offer["tiers"][0]["disclosure"]["recurring"], "Recurring payments: None")
        self.assertEqual([p["policy_type"] for p in offer["policies"]], list(legal.CHECKOUT_POLICIES))
        self.assertEqual(offer["acknowledgment"], legal.ACKNOWLEDGMENT)
        self.assertFalse(offer["upgrade"]["eligible"])

    # checkout refusals
    def test_checkout_refused_while_any_switch_is_off(self):
        self.flags()
        body = self.body()
        for off in ("FOUNDING_100_ENABLED", "FOUNDING_CHECKOUT_ENABLED", "FOUNDING_LEGAL_PAGES_ENABLED"):
            with self.subTest(off=off), mock.patch.dict(os.environ, {off: "0"}):
                r = self.checkout(body)
                self.assertEqual((r.status_code, r.json()["code"]), (403, "checkout_closed"))
        with mock.patch.dict(os.environ, {"FOUNDING_ELITE_ENABLED": "0"}):
            self.assertEqual(self.checkout({**body, "tier": ELITE}).status_code, 403)
        self.b.stripe.checkout.Session.create.assert_not_called()

    def test_missing_acknowledgment_is_refused(self):
        self.flags()
        r = self.checkout(self.body(acknowledged=False))
        self.assertEqual((r.status_code, r.json()["code"]), (400, "acknowledgment_required"))
        self.assertEqual(self.checkout({k: v for k, v in self.body().items() if k != "acknowledged"}).status_code, 422)
        self.b.stripe.checkout.Session.create.assert_not_called()
        self.assertEqual(self.fdb.availability()["tiers"][PRO]["held"], 0)

    def test_wrong_or_missing_policy_version_is_refused(self):
        self.flags()
        body = self.body()
        stale = [dict(p, version=p["version"] + 1) if p["policy_type"] == "founding_terms" else p for p in body["policies"]]
        r = self.checkout({**body, "policies": stale})
        self.assertEqual((r.status_code, r.json()["code"], r.json()["changed"]), (409, "policy_changed", ["founding_terms"]))
        tampered = [dict(p, content_hash="0" * 64) if p["policy_type"] == "privacy" else p for p in body["policies"]]
        self.assertEqual(self.checkout({**body, "policies": tampered}).json()["changed"], ["privacy"])
        self.assertEqual(self.checkout({**body, "policies": body["policies"][:2]}).json()["changed"], ["privacy"])
        self.b.stripe.checkout.Session.create.assert_not_called()

    def test_live_subscription_must_be_cancelled_first(self):
        self.flags()
        self.b._get_user_by_email = lambda e: {"stripe_customer_id": "cus_1"}
        self.b.stripe.Subscription.list.return_value = {"data": [{"status": "active"}], "has_more": False}
        r = self.checkout(self.body())
        self.assertEqual((r.status_code, r.json()["code"]), (409, "cancel_subscription_first"))
        self.b.stripe.Subscription.list.return_value = {"data": [{"status": "canceled"}], "has_more": False}
        self.assertEqual(self.checkout(self.body()).status_code, 200)

    # checkout
    def test_checkout_holds_a_seat_charges_the_server_price_and_records_acceptance(self):
        self.flags()
        r = self.checkout(self.body(ELITE))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["checkout_url"].startswith("https://checkout.stripe.test/"))
        kw = self.b.stripe.checkout.Session.create.call_args.kwargs
        self.assertEqual(kw["mode"], "payment")
        self.assertEqual(kw["payment_method_types"], ["card"])
        self.assertEqual(kw["line_items"][0]["price_data"]["unit_amount"], 99_900)
        self.assertEqual(kw["metadata"]["user_email"], "pro@example.com")
        self.assertEqual(kw["payment_intent_data"]["metadata"], kw["metadata"])
        self.assertTrue(kw["idempotency_key"].startswith("hsf-founding-"))
        self.assertIn("/account/founder?checkout=success", kw["success_url"])
        res = self.fdb.open_reservation("pro@example.com")
        self.assertEqual(res["checkout_session_id"], f"cs_test_{res['id']}")
        acc = self.ldb.acceptances(checkout_session_id=res["checkout_session_id"])
        self.assertEqual({(a["policy_type"], a["tier"], a["user_id"]) for a in acc},
                         {(t, ELITE, "pro@example.com") for t in legal.CHECKOUT_POLICIES})
        # Clicking again reuses the same open session and seat.
        again = self.checkout(self.body(ELITE))
        self.assertEqual(again.json()["checkout_url"], "https://checkout.stripe.test/open")
        self.assertEqual(self.b.stripe.checkout.Session.create.call_count, 1)
        self.assertEqual(self.fdb.availability()["tiers"][ELITE]["held"], 1)
        self.assertEqual(len(self.ldb.acceptances(user_id="pro@example.com")), 3)

    def test_failed_session_create_keeps_no_acceptance(self):
        self.flags()
        self.b.stripe.checkout.Session.create.side_effect = RuntimeError("stripe down")
        r = self.checkout(self.body())
        self.assertEqual((r.status_code, r.json()["code"]), (503, "billing_unavailable"))
        self.assertEqual(self.ldb.acceptances(user_id="pro@example.com"), [])

    def test_member_and_sold_out_are_refused(self):
        self.flags()
        res = self.fdb.reserve("pro@example.com", PRO)
        self.fdb.fulfill(res["id"], payment_reference="pi_1", amount_cents=49_900)
        r = self.checkout(self.body())
        self.assertEqual((r.status_code, r.json()["code"]), (409, "already_member"))
        offer = self.client.get("/v1/founding/offer", headers=self.h).json()
        self.assertEqual(offer["membership"]["badge"], "HSF AI Founder #001")
        self.assertTrue(offer["upgrade"]["eligible"] and offer["upgrade"]["on_sale"])
        self.conn.execute("UPDATE founding_inventory SET sold = capacity WHERE tier = ?", (ELITE,))
        r = self.checkout(self.body(ELITE, upgrade=True))
        self.assertEqual((r.status_code, r.json()["code"]), (409, "sold_out"))

    def test_upgrade_checkout_charges_500(self):
        self.flags()
        res = self.fdb.reserve("pro@example.com", PRO)
        self.fdb.fulfill(res["id"], payment_reference="pi_1", amount_cents=49_900)
        self.assertEqual(self.checkout(self.body(PRO, upgrade=True)).status_code, 403)  # upgrades go to Elite
        with mock.patch.dict(os.environ, {"FOUNDING_UPGRADES_ENABLED": "0"}):
            self.assertEqual(self.checkout(self.body(ELITE, upgrade=True)).status_code, 403)
        r = self.checkout(self.body(ELITE, upgrade=True))
        self.assertEqual(r.status_code, 200, r.text)
        kw = self.b.stripe.checkout.Session.create.call_args.kwargs
        self.assertEqual((kw["line_items"][0]["price_data"]["unit_amount"], kw["metadata"]["founding_kind"]),
                         (50_000, "upgrade"))

    def test_cancel_releases_the_hold_but_keeps_a_paid_one(self):
        self.flags()
        self.checkout(self.body())
        self.b.stripe.checkout.Session.retrieve.return_value = {"status": "complete"}
        self.assertEqual(self.client.post("/v1/founding/checkout/cancel", headers=self.h).json(),
                         {"released": False, "paid": True})
        self.b.stripe.checkout.Session.retrieve.return_value = {"status": "open"}
        self.assertEqual(self.client.post("/v1/founding/checkout/cancel", headers=self.h).json()["released"], True)
        self.b.stripe.checkout.Session.expire.assert_called_once()
        self.assertEqual(self.fdb.availability()["tiers"][PRO]["held"], 0)

    def test_acceptance_needs_sign_in(self):
        self.flags()
        self.assertEqual(self.client.post("/v1/founding/checkout", json={}).status_code, 401)
        self.assertEqual(self.client.get("/v1/founding/offer").status_code, 401)


class WebhookTests(DbBase):
    """Founding Checkout events go to db/founding, never to the subscription sync."""

    def setUp(self):
        super().setUp()
        import importlib.util

        if importlib.util.find_spec("fastapi") is None:
            self.skipTest("needs fastapi")
        from tests.test_billing_service import _configured_module

        self.b = _configured_module()
        self.b._get_user_by_email = mock.MagicMock(side_effect=lambda email: {"tier": "basic", "is_admin": False})
        self.b._set_user_plan_by_email = mock.MagicMock()
        self.b._record_paid_conversion = mock.MagicMock()
        self.engine = self.b._billing_engine
        self.res = self.fdb.reserve("buyer@example.com", ELITE)
        self.fdb.attach_checkout(self.res["id"], "cs_1")

    def send(self, kind="checkout.session.completed", event_id="evt_1", **over):
        data = {"id": "cs_1", "object": "checkout.session", "mode": "payment", "payment_status": "paid",
                "payment_intent": "pi_1", "amount_total": 99_900, "currency": "usd", "customer": "cus_1",
                "metadata": {"founding_reservation_id": str(self.res["id"]), "user_email": "buyer@example.com",
                             "founding_tier": ELITE, "founding_kind": "purchase"}}
        data.update(over)
        self.b.stripe.Webhook.construct_event.return_value = {"id": event_id, "type": kind, "livemode": False,
                                                              "data": {"object": data}}
        return self.engine.webhook(b"raw", "sig")

    def body(self, r):
        import json

        return json.loads(r.body)

    def test_paid_session_activates_without_touching_the_plan(self):
        out = self.body(self.send())
        self.assertEqual(out["founding"], "activated")
        m = self.fdb.membership("buyer@example.com")
        self.assertEqual((m["tier"], m["membership_number"], m["status"]), (ELITE, 1, "active"))
        self.b._set_user_plan_by_email.assert_not_called()
        self.b.stripe.Subscription.list.assert_not_called()
        self.b._record_paid_conversion.assert_not_called()

    def test_duplicate_event_and_duplicate_payment_activate_once(self):
        self.send()
        self.assertEqual(self.body(self.send())["note"], "already_processed")
        self.assertEqual(self.body(self.send(event_id="evt_2"))["founding"], "already_done")
        self.assertEqual(self.fdb.availability()["tiers"][ELITE]["sold"], 1)

    def test_tampered_amount_or_account_is_not_granted(self):
        self.assertEqual(self.body(self.send(amount_total=100))["founding"], "needs_review")
        self.assertIsNone(self.fdb.membership("buyer@example.com"))
        res2 = self.fdb.reserve("other@example.com", PRO)
        r = self.send(event_id="evt_3", id="cs_9", metadata={"founding_reservation_id": str(res2["id"]),
                                                              "user_email": "buyer@example.com"})
        self.assertEqual(self.body(r)["founding"], "ignored")
        self.assertIsNone(self.fdb.membership("other@example.com"))

    def test_unpaid_and_failed_payments(self):
        self.assertEqual(self.body(self.send(payment_status="unpaid"))["founding"], "waiting")
        self.assertEqual(self.body(self.send("checkout.session.async_payment_failed", "evt_2",
                                             payment_status="unpaid"))["founding"], "released")
        self.assertEqual(self.fdb.availability()["tiers"][ELITE]["held"], 0)

    def test_subscription_checkouts_still_take_the_old_path(self):
        self.b._get_username_by_customer = mock.MagicMock(return_value="buyer@example.com")
        self.b.stripe.Subscription.list.return_value = {"data": [], "has_more": False}
        self.send(mode="subscription", metadata={})
        self.b.stripe.Subscription.list.assert_called()
        self.assertIsNone(self.fdb.membership("buyer@example.com"))

    def test_processing_failure_is_retryable(self):
        with mock.patch("db.founding.fulfill", side_effect=RuntimeError("db down")):
            from fastapi import HTTPException

            with self.assertRaises(HTTPException) as ctx:
                self.send()
            self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(self.body(self.send())["founding"], "activated")


if __name__ == "__main__":
    unittest.main()
