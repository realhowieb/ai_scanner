"""Founding 100 refunds: the 14-day guarantee, later requests, admin approval, prorated
discontinuation refunds, Stripe refund events and what they do to access (Stripe mocked).
Real-Postgres checks of the refund tables are in test_founding_postgres.py."""
import datetime as dt
import json
import os
import sys
import types
import unittest
from decimal import Decimal
from unittest import mock

from domain import founding as rules
from domain import founding_refunds as rr
from tests.test_founding_legal import ALL_ON, DbBase, _fake_billing

UTC = dt.timezone.utc
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE
T0 = dt.datetime(2026, 11, 2, 15, 0, tzinfo=UTC)  # a Monday


class RuleTests(unittest.TestCase):
    def test_policy_example_is_exact(self):
        start = dt.datetime(2026, 11, 2, tzinfo=UTC)
        end = rr.add_months(start, 60)
        self.assertEqual(rr.prorated_cents(99_900, start, end, rr.add_months(start, 24)), 59_940)

    def test_rounding_is_half_up_and_capped(self):
        start = dt.datetime(2026, 1, 1, tzinfo=UTC)
        end = rr.add_months(start, 60)
        # $499 x 59/60 = 490.6833.. -> 49068
        self.assertEqual(rr.prorated_cents(49_900, start, end, rr.add_months(start, 1)), 49_068)
        self.assertEqual(rr.prorated_cents(49_900, start, end, start), 49_900)
        self.assertEqual(rr.prorated_cents(49_900, start, end, start, already_refunded_cents=40_000), 9_900)
        self.assertEqual(rr.prorated_cents(49_900, start, end, end + dt.timedelta(days=9)), 0)

    def test_part_months_count_their_share(self):
        a = dt.datetime(2026, 2, 1, tzinfo=UTC)
        self.assertEqual(rr.months_between(a, dt.datetime(2026, 2, 15, tzinfo=UTC)), Decimal(14) / Decimal(28))
        self.assertEqual(rr.months_between(dt.datetime(2026, 1, 31, tzinfo=UTC), dt.datetime(2026, 2, 28, tzinfo=UTC)), 1)

    def test_upgrade_spreads_over_upgrade_to_original_end(self):
        start = dt.datetime(2026, 1, 1, tzinfo=UTC)
        end = rr.add_months(start, 60)
        upgraded = rr.add_months(start, 12)  # 48 months left
        # stopped 24 months after the upgrade: 24 of 48 unused -> half of $500
        self.assertEqual(rr.prorated_cents(50_000, upgraded, end, rr.add_months(upgraded, 24)), 25_000)

    def test_guarantee_window_and_business_days(self):
        self.assertTrue(rr.within_guarantee(T0, T0 + dt.timedelta(days=14)))
        self.assertFalse(rr.within_guarantee(T0, T0 + dt.timedelta(days=14, seconds=1)))
        friday = dt.datetime(2026, 11, 6, 12, tzinfo=UTC)
        self.assertEqual(rr.start_due(friday).date(), dt.date(2026, 11, 17))  # 7 weekdays later

    def test_default_amounts(self):
        self.assertEqual(rr.default_amount("guarantee", 99_900, 0), 99_900)
        self.assertEqual(rr.default_amount("discontinuation", 99_900, 0, prorated=59_940), 59_940)
        with self.assertRaises(ValueError):
            rr.default_amount("discontinuation", 99_900, 0)
        with self.assertRaises(ValueError):
            rr.default_amount("because", 99_900, 0)


class RefundDbBase(DbBase):
    def setUp(self):
        super().setUp()
        from db import founding_refunds

        self.r = founding_refunds

    def buy(self, user="buyer@example.com", tier=ELITE, ref="pi_1", paid_at=T0):
        res = self.fdb.reserve(user, tier, now=paid_at)
        out = self.fdb.fulfill(res["id"], payment_reference=ref, amount_cents=rules.PRICE_CENTS[tier],
                               paid_at=paid_at, now=paid_at)
        self.assertEqual(out["outcome"], "activated")
        return out["membership"]

    def upgrade(self, user="buyer@example.com", ref="pi_up", paid_at=T0):
        res = self.fdb.reserve(user, ELITE, kind="upgrade", now=paid_at)
        out = self.fdb.fulfill(res["id"], payment_reference=ref, amount_cents=rules.UPGRADE_PRICE_CENTS,
                               paid_at=paid_at, now=paid_at)
        self.assertEqual(out["outcome"], "upgraded")

    def audit_actions(self):
        return [a["action"] for a in reversed(self.r.history())]


class RefundDbTests(RefundDbBase):
    def test_payments_are_recorded_with_their_deadline(self):
        self.buy()
        (p,) = self.r.payments("buyer@example.com", now=T0 + dt.timedelta(days=3))
        self.assertEqual((p["kind"], p["amount_cents"], p["refundable_cents"]), ("purchase", 99_900, 99_900))
        self.assertTrue(p["within_guarantee"])
        self.assertEqual(p["deadline"], (T0 + dt.timedelta(days=14)).isoformat(timespec="seconds"))

    def test_request_within_14_days(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", reason="changed my mind", now=T0 + dt.timedelta(days=13))
        self.assertEqual((r["status"], int(r["within_guarantee"])), ("requested", 1))
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active", "a request never changes access")

    def test_request_after_14_days_is_kept_for_review(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0 + dt.timedelta(days=20))
        self.assertEqual((r["status"], int(r["within_guarantee"])), ("requested", 0))
        with self.assertRaises(self.r.RefundError):
            self.r.approve(r["id"], "boss@example.com", basis="guarantee", now=T0 + dt.timedelta(days=21))
        ok = self.r.approve(r["id"], "boss@example.com", basis="billing_error", reason="charged twice")
        self.assertEqual((ok["status"], ok["amount_cents"]), ("approved", 99_900))

    def test_duplicate_request_is_refused(self):
        self.buy()
        self.r.request_refund("buyer@example.com", "purchase", now=T0)
        with self.assertRaises(self.r.AlreadyRequested):
            self.r.request_refund("buyer@example.com", "purchase", now=T0)

    def test_no_payment_or_wrong_kind(self):
        with self.assertRaises(self.r.NothingToRefund):
            self.r.request_refund("nobody@example.com", "purchase")
        self.buy()
        with self.assertRaises(self.r.NothingToRefund):
            self.r.request_refund("buyer@example.com", "upgrade")
        with self.assertRaises(self.r.RefundError):
            self.r.request_refund("buyer@example.com", "everything")

    def test_amount_can_never_exceed_what_is_left(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        with self.assertRaises(self.r.RefundError):
            self.r.approve(r["id"], "boss@example.com", basis="guarantee", amount_cents=99_901)
        with self.assertRaises(self.r.RefundError):
            self.r.approve(r["id"], "boss@example.com", basis="guarantee", amount_cents=0)

    def test_full_refund_ends_access_but_keeps_the_account_and_records(self):
        m = self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        self.r.mark_initiated(r["id"], "re_1", "boss@example.com")
        out = self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_1")
        self.assertEqual((out["outcome"], out["full"], out["access"]), ("completed", True, "refunded"))
        after = self.fdb.membership("buyer@example.com")
        self.assertEqual((after["status"], after["membership_number"], after["former_number"]),
                         ("refunded", None, m["membership_number"]))
        self.assertEqual(self.fdb.availability()["tiers"][ELITE]["sold"], 0, "the seat goes back")
        self.assertEqual(self.r.get_request(r["id"])["status"], "completed")
        self.assertEqual(self.r.payments("buyer@example.com")[0]["refunded_cents"], 99_900, "payment record kept")
        self.assertEqual(self.audit_actions(), ["refund_requested", "refund_approved", "refund_initiated",
                                                "refund_completed"])

    def test_duplicate_refund_event_changes_nothing(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_1")
        again = self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_1")
        self.assertEqual(again["outcome"], "already_done")
        self.assertEqual(self.audit_actions().count("refund_completed"), 1)
        with self.assertRaises(self.r.NothingToRefund):
            self.r.request_refund("buyer@example.com", "purchase")

    def test_partial_refund_keeps_access(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0 + dt.timedelta(days=40))
        self.r.approve(r["id"], "boss@example.com", basis="service_failure", amount_cents=20_000, reason="outage")
        out = self.r.apply_provider_refund("pi_1", refunded_total_cents=20_000, provider_refund_id="re_1")
        self.assertEqual((out["outcome"], out["full"]), ("completed", False))
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")
        (p,) = self.r.payments("buyer@example.com")
        self.assertEqual((p["refunded_cents"], p["refundable_cents"]), (20_000, 79_900))
        # a second request can only take what is left
        r2 = self.r.request_refund("buyer@example.com", "purchase")
        with self.assertRaises(self.r.RefundError):
            self.r.approve(r2["id"], "boss@example.com", basis="exception", amount_cents=80_000)

    def test_upgrade_only_refund_returns_to_founder_pro(self):
        m = self.buy(tier=PRO)
        self.upgrade(paid_at=T0 + dt.timedelta(days=30))
        r = self.r.request_refund("buyer@example.com", "upgrade", now=T0 + dt.timedelta(days=35))
        self.assertEqual(int(r["within_guarantee"]), 1, "the upgrade has its own 14 days")
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        out = self.r.apply_provider_refund("pi_up", refunded_total_cents=50_000, provider_refund_id="re_9")
        self.assertEqual(out["access"], "upgrade_reversed")
        after = self.fdb.membership("buyer@example.com")
        self.assertEqual((after["tier"], after["status"], after["membership_number"], after["expires_at"]),
                         (PRO, "active", m["membership_number"], m["expires_at"]))
        # the original purchase is still refundable on its own terms
        self.assertEqual(self.r.payments("buyer@example.com")[0]["refundable_cents"], 49_900)

    def test_discontinuation_is_prorated_per_payment(self):
        self.buy(tier=PRO)
        m = self.fdb.membership("buyer@example.com")
        up_at = rr.add_months(T0, 12)
        self.upgrade(paid_at=up_at)
        stop = rr.add_months(T0, 36)
        r1 = self.r.request_refund("buyer@example.com", "purchase", now=stop)
        q1 = self.r.prorated_quote(r1["id"], stop)
        start = self.fdb.parse_ts(m["activated_at"])
        end = self.fdb.parse_ts(m["expires_at"])
        self.assertEqual(q1["amount_cents"], rr.prorated_cents(49_900, start, end, stop))
        r2 = self.r.request_refund("buyer@example.com", "upgrade", now=stop)
        q2 = self.r.prorated_quote(r2["id"], stop)
        self.assertEqual((q2["start"], q2["amount_cents"]),
                         (up_at.isoformat(timespec="seconds"), rr.prorated_cents(50_000, up_at, end, stop)))
        with self.assertRaises(self.r.RefundError):
            self.r.approve(r2["id"], "boss@example.com", basis="discontinuation")  # needs the stop date
        ok = self.r.approve(r2["id"], "boss@example.com", basis="discontinuation", stopped_at=stop)
        self.assertEqual(ok["amount_cents"], q2["amount_cents"])

    def test_deny_needs_a_reason_and_is_final(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0 + dt.timedelta(days=60))
        with self.assertRaises(self.r.RefundError):
            self.r.deny(r["id"], "boss@example.com", reason=" ")
        self.r.deny(r["id"], "boss@example.com", reason="Outside 14 days, no exception applies.")
        with self.assertRaises(self.r.BadState):
            self.r.approve(r["id"], "boss@example.com", basis="exception")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")

    def test_provider_failure_keeps_access_and_allows_a_new_request(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        self.r.mark_initiated(r["id"], "re_1", "boss@example.com")
        self.r.mark_failed(r["id"], "expired_or_canceled_card", "payment_provider")
        self.assertEqual(self.r.get_request(r["id"])["status"], "failed")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")
        self.assertEqual(self.r.request_refund("buyer@example.com", "purchase")["status"], "requested")

    def test_refund_made_directly_in_stripe_is_still_applied(self):
        self.buy()
        out = self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_x")
        self.assertEqual((out["outcome"], out["request_id"], out["access"]), ("completed", None, "refunded"))
        self.assertEqual(self.r.apply_provider_refund("pi_nope", refunded_total_cents=1, provider_refund_id=None),
                         {"outcome": "unknown_payment"})

    def test_access_change_is_retried_after_a_crash(self):
        self.buy()
        with mock.patch.object(self.fdb, "refund", side_effect=RuntimeError("db down")):
            with self.assertRaises(RuntimeError):
                self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_1")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")
        again = self.r.apply_provider_refund("pi_1", refunded_total_cents=99_900, provider_refund_id="re_1")
        self.assertEqual((again["outcome"], again["access"]), ("already_done", "refunded"))
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "refunded")


def _refund_billing():
    b = _fake_billing()
    n = {"i": 0}

    def create(**kw):
        n["i"] += 1
        return types.SimpleNamespace(id=f"re_{n['i']}", status="pending")

    b.stripe.Refund.create.side_effect = create
    return b


class RefundApiTests(RefundDbBase):
    def setUp(self):
        super().setUp()
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs fastapi, httpx, PyJWT and bcrypt")
        from api import founding as svc
        from tests.test_api_v1 import ApiTestCase

        svc.clear_cache()
        self.addCleanup(svc.clear_cache)
        case = type("Case", (ApiTestCase,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.client = case.client
        self.h = case.auth(case.login().json()["access_token"])
        self.admin = case.auth(case.login("boss@example.com").json()["access_token"])
        self.b = _refund_billing()
        from api import plan_access
        from services import founding_emails

        plan_access.clear_cache()  # founder access is cached per account
        self.addCleanup(plan_access.clear_cache)
        self.mails = types.SimpleNamespace()
        for name in ("refund_received", "refund_approved", "refund_denied", "refund_initiated"):
            p = mock.patch.object(founding_emails, name)
            setattr(self.mails, name, p.start())
            self.addCleanup(p.stop)
        for p in (mock.patch.dict(sys.modules, {"billing_service.main": self.b}),
                  mock.patch("api.billing.embedded_enabled", return_value=True),
                  mock.patch.dict(os.environ, {**ALL_ON, "FOUNDING_REFUNDS_ENABLED": "0"})):
            p.start()
            self.addCleanup(p.stop)
        self.buy("pro@example.com", tier=PRO, ref="pi_mine", paid_at=dt.datetime.now(UTC) - dt.timedelta(days=2))
        self.buy("other@example.com", tier=ELITE, ref="pi_theirs", paid_at=dt.datetime.now(UTC) - dt.timedelta(days=2))

    def ask(self, kind="purchase", headers=None, **body):
        return self.client.post("/v1/founding/refunds/request", json={"kind": kind, **body}, headers=headers or self.h)

    def test_member_sees_only_their_own_payments(self):
        out = self.client.get("/v1/founding/refunds/me", headers=self.h).json()
        self.assertEqual([(p["kind"], p["amount_cents"]) for p in out["payments"]], [("purchase", 49_900)])
        self.assertTrue(out["payments"][0]["within_guarantee"])
        self.assertNotIn("payment_reference", out["payments"][0])
        self.assertEqual(out["guarantee_days"], 14)
        self.assertEqual(self.client.get("/v1/founding/refunds/me").status_code, 401)

    def test_request_is_tied_to_the_signed_in_account(self):
        r = self.ask(reason="not for me", payment_reference="pi_theirs", user_id="other@example.com")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "requested")
        (req,) = self.r.all_requests()
        self.assertEqual((req["user_id"], req["payment_reference"]), ("pro@example.com", "pi_mine"))
        self.mails.refund_received.assert_called_once()
        self.assertEqual(self.ask().json()["code"], "refund_already_requested")
        self.assertEqual(self.ask("upgrade").status_code, 409)

    def test_admin_endpoints_need_an_admin(self):
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        for method, path, body in (("get", "/v1/admin/founding/refunds", None),
                                   ("get", f"/v1/admin/founding/refunds/{rid}", None),
                                   ("get", f"/v1/admin/founding/refunds/{rid}/prorated?stopped_at=2027-01-01T00:00:00Z", None),
                                   ("post", f"/v1/admin/founding/refunds/{rid}/approve", {"basis": "guarantee"}),
                                   ("post", f"/v1/admin/founding/refunds/{rid}/deny", {"reason": "no"})):
            kw = {"json": body} if body else {}
            self.assertEqual(getattr(self.client, method)(path, headers=self.h, **kw).status_code, 403, path)
            self.assertEqual(getattr(self.client, method)(path, **kw).status_code, 401, path)
        self.assertEqual(self.r.get_request(rid)["status"], "requested")

    def test_admin_detail_shows_policies_and_audit(self):
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        out = self.client.get(f"/v1/admin/founding/refunds/{rid}", headers=self.admin).json()
        self.assertEqual(out["request"]["user_id"], "pro@example.com")
        self.assertEqual(out["request"]["number"], "001")
        self.assertEqual([a["action"] for a in out["audit"]], ["refund_requested"])
        self.assertFalse(out["refunds_live"])
        listed = self.client.get("/v1/admin/founding/refunds?status=requested", headers=self.admin).json()
        self.assertEqual(len(listed["requests"]), 1)
        self.assertEqual(self.client.get("/v1/admin/founding/refunds/999", headers=self.admin).status_code, 404)

    def test_approval_while_refunds_are_off_sends_nothing(self):
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json={"basis": "guarantee"}, headers=self.admin)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["sent"], r.json()["request"]["status"]), (False, "approved"))
        self.b.stripe.Refund.create.assert_not_called()
        self.mails.refund_approved.assert_called_once()
        self.assertEqual(self.fdb.membership("pro@example.com")["status"], "active")

    def test_approval_sends_one_stripe_refund_with_an_idempotency_key(self):
        os.environ["FOUNDING_REFUNDS_ENABLED"] = "1"
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json={"basis": "guarantee"}, headers=self.admin)
        self.assertEqual((r.json()["sent"], r.json()["request"]["status"]), (True, "initiated"))
        kw = self.b.stripe.Refund.create.call_args.kwargs
        self.assertEqual((kw["payment_intent"], kw["amount"], kw["idempotency_key"]),
                         ("pi_mine", 49_900, f"hsf-founding-refund-{rid}"))
        self.mails.refund_initiated.assert_called_once()
        self.assertEqual(self.fdb.membership("pro@example.com")["status"], "active", "until Stripe confirms")
        again = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json={"basis": "guarantee"},
                                 headers=self.admin)
        self.assertEqual(again.json()["code"], "refund_wrong_state")
        self.assertEqual(self.b.stripe.Refund.create.call_count, 1)

    def test_stripe_error_keeps_it_approved_and_retryable(self):
        os.environ["FOUNDING_REFUNDS_ENABLED"] = "1"
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        self.b.stripe.Refund.create.side_effect = RuntimeError("card_declined")
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json={"basis": "guarantee"}, headers=self.admin)
        self.assertEqual((r.json()["sent"], r.json()["request"]["status"]), (False, "approved"))
        self.assertIn("RuntimeError", r.json()["request"]["failure_reason"])
        self.b.stripe.Refund.create.side_effect = lambda **kw: types.SimpleNamespace(id="re_ok", status="pending")
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json={"basis": "guarantee"}, headers=self.admin)
        self.assertEqual((r.json()["sent"], r.json()["request"]["provider_refund_id"]), (True, "re_ok"))
        self.assertEqual(self.b.stripe.Refund.create.call_args.kwargs["idempotency_key"], f"hsf-founding-refund-{rid}")

    def test_prorated_refunds_need_their_switch(self):
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        os.environ["FOUNDING_PRORATED_REFUNDS_ENABLED"] = "0"
        body = {"basis": "discontinuation", "stopped_at": "2028-01-01T00:00:00Z"}
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json=body, headers=self.admin)
        self.assertEqual((r.status_code, r.json()["code"]), (403, "prorated_refunds_off"))
        os.environ["FOUNDING_PRORATED_REFUNDS_ENABLED"] = "1"
        q = self.client.get(f"/v1/admin/founding/refunds/{rid}/prorated?stopped_at=2028-01-01T00:00:00Z",
                            headers=self.admin).json()
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/approve", json=body, headers=self.admin)
        self.assertEqual(r.json()["request"]["amount_cents"], q["amount_cents"])
        self.assertLess(q["amount_cents"], 49_900)

    def test_deny_tells_the_member_why(self):
        self.ask()
        rid = self.r.all_requests()[0]["id"]
        self.assertEqual(self.client.post(f"/v1/admin/founding/refunds/{rid}/deny", json={"reason": ""},
                                          headers=self.admin).status_code, 422)
        r = self.client.post(f"/v1/admin/founding/refunds/{rid}/deny", json={"reason": "Already refunded by bank."},
                             headers=self.admin)
        self.assertEqual(r.json()["request"]["status"], "denied")
        mine = self.client.get("/v1/founding/refunds/me", headers=self.h).json()["requests"][0]
        self.assertEqual((mine["status"], mine["decision_reason"]), ("denied", "Already refunded by bank."))
        self.mails.refund_denied.assert_called_once()


class RefundWebhookTests(RefundDbBase):
    def setUp(self):
        super().setUp()
        import importlib.util

        if importlib.util.find_spec("fastapi") is None:
            self.skipTest("needs fastapi")
        from tests.test_billing_service import _configured_module

        self.b = _configured_module()
        self.b._get_user_by_email = mock.MagicMock(side_effect=lambda email: {"tier": "basic", "is_admin": False})
        self.b._set_user_plan_by_email = mock.MagicMock()
        self.engine = self.b._billing_engine
        self.buy()
        p = mock.patch("services.founding_emails.refund_completed")
        self.notify = p.start()
        self.addCleanup(p.stop)

    def send(self, kind, data, event_id="evt_r1"):
        self.b.stripe.Webhook.construct_event.return_value = {"id": event_id, "type": kind, "livemode": False,
                                                              "data": {"object": data}}
        return json.loads(self.engine.webhook(b"raw", "sig").body)

    def charge(self, refunded=99_900, intent="pi_1"):
        return {"id": "ch_1", "object": "charge", "payment_intent": intent, "amount": 99_900,
                "amount_refunded": refunded, "customer": "cus_1", "refunds": {"data": [{"id": "re_1"}]}}

    def test_refund_event_ends_access_once(self):
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        self.assertEqual(self.send("charge.refunded", self.charge())["founding_refund"], "completed")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "refunded")
        self.assertEqual(self.send("charge.refunded", self.charge())["note"], "already_processed")
        self.assertEqual(self.send("charge.refunded", self.charge(), event_id="evt_r2")["founding_refund"], "already_done")
        self.notify.assert_called_once_with("buyer@example.com", kind="purchase",
                                            amount_cents=99_900, full=True)
        self.b._set_user_plan_by_email.assert_not_called()

    def test_other_refunds_are_ignored(self):
        self.assertEqual(self.send("charge.refunded", self.charge(intent="pi_subscription"))["note"], "ignored")
        self.assertEqual(self.send("refund.failed", {"id": "re_2", "metadata": {}})["note"], "ignored")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")

    def test_refund_failed_event(self):
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0)
        self.r.approve(r["id"], "boss@example.com", basis="guarantee")
        self.r.mark_initiated(r["id"], "re_1", "boss@example.com")
        out = self.send("refund.failed", {"id": "re_1", "failure_reason": "lost_or_stolen_card",
                                          "metadata": {"founding_refund_request_id": str(r["id"])}})
        self.assertEqual(out["founding_refund"], "failed")
        self.assertEqual(self.r.get_request(r["id"])["failure_reason"], "lost_or_stolen_card")
        self.assertEqual(self.fdb.membership("buyer@example.com")["status"], "active")


class EmailTests(RefundDbBase):
    def setUp(self):
        super().setUp()
        p = mock.patch("ui.email_utils._send_smtp", return_value=True)
        self.smtp = p.start()
        self.addCleanup(p.stop)
        from services import founding_emails

        self.e = founding_emails

    def sent(self):
        to, subject, text, html = self.smtp.call_args.args
        return to, subject, text, html

    def test_purchase_confirmation_has_the_facts_and_no_card_details(self):
        self.buy(paid_at=dt.datetime.now(UTC))
        self.assertTrue(self.e.purchase_confirmation("buyer@example.com"))
        to, subject, text, html = self.sent()
        self.assertEqual(to, "buyer@example.com")
        for phrase in ("$999.00", "does not renew", "14-day refund guarantee"):
            self.assertIn(phrase, text)
        self.assertNotIn("pi_1", text + html)

    def test_refund_emails(self):
        self.buy()
        r = self.r.request_refund("buyer@example.com", "purchase", now=T0 + dt.timedelta(days=20))
        self.assertTrue(self.e.refund_received("buyer@example.com", {**r}))
        self.assertIn("after the 14-day guarantee", self.sent()[2])
        self.assertTrue(self.e.refund_completed("buyer@example.com", kind="upgrade", amount_cents=50_000, full=True))
        self.assertIn("back on Founder Pro", self.sent()[2])
        self.assertTrue(self.e.refund_initiated("buyer@example.com", {"amount_cents": 99_900}))
        self.assertIn("up to your bank", self.sent()[2])

    def test_a_send_failure_never_raises(self):
        self.smtp.side_effect = RuntimeError("smtp down")
        self.assertFalse(self.e.refund_denied("buyer@example.com", {"decision_reason": "x"}))


if __name__ == "__main__":
    unittest.main()
