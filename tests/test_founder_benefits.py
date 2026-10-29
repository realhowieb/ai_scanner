"""Founder benefits: founder rate after five years, early access, founder invites, Elite
limits, the founder badge on shared links and in the digest, founder feedback and the
founder poll."""
import datetime as dt
import os
import sqlite3
import sys
import types
import unittest
from unittest import mock

from domain import founding as rules
from domain import legal
from tests.test_founding_legal import DbBase

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE


class RuleTests(unittest.TestCase):
    def test_poll_options(self):
        self.assertIn("api", rules.poll_options({}))
        self.assertEqual(rules.poll_options({"HSF_FOUNDER_POLL": "a:Alpha| b : Beta |bad|:x|c d:No"}),
                         {"a": "Alpha", "b": "Beta"})
        self.assertEqual(rules.poll_options({"HSF_FOUNDER_POLL": ""}), {})

    def test_early_access_features(self):
        self.assertEqual(rules.early_access_features({}), ())
        self.assertEqual(rules.early_access_features({"HSF_FOUNDER_EARLY_ACCESS": "can_early_breakout, can_ai_notes"}),
                         ("can_early_breakout", "can_ai_notes"))

    def test_terms_describe_the_benefits(self):
        cur = legal.current(legal.load(env={}))[legal.TERMS]
        text = cur["text"]
        for phrase in ("50% off its price at that time", "up to 50 active alerts", "twice Premium's daily AI allowance",
                       "Early access", "60 days of Pro", "never your name or email", "Founder polls are product feedback"):
            self.assertIn(phrase, text)
        self.assertEqual(len([s for s in cur["sections"] if s["heading"]]), 17)


class Base(DbBase):
    def setUp(self):
        super().setUp()
        from api import plan_access
        from db import referrals

        self.pa, self.refs = plan_access, referrals
        plan_access.clear_cache()
        self.addCleanup(plan_access.clear_cache)
        self.rconn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.rconn)
        self.addCleanup(lambda: referrals.set_connection_factory(None))

    def buy(self, user, tier=PRO, ref="pi_1", at=NOW):
        res = self.fdb.reserve(user, tier, now=at)
        out = self.fdb.fulfill(res["id"], payment_reference=ref, amount_cents=rules.PRICE_CENTS[tier], paid_at=at, now=at)
        self.assertEqual(out["outcome"], "activated")
        self.pa.clear_cache()
        return out["membership"]


class InviteTests(Base):
    def invite(self, inviter, friend):
        from api import referrals as api_refs

        code = self.refs.get_or_create_code(inviter)
        self.assertTrue(self.refs.record_signup(code, friend))
        with mock.patch.object(api_refs, "reward_days", return_value=30):
            self.assertTrue(api_refs.reward_on_verify(friend))
        return self.refs.grant_until(friend)

    def test_first_three_friends_of_a_founder_get_60_days(self):
        self.buy("f@example.com")
        for i in range(3):
            until = self.invite("f@example.com", f"friend{i}@example.com")
            self.assertGreater(until, NOW + dt.timedelta(days=59))
        until = self.invite("f@example.com", "friend3@example.com")
        self.assertLess(until, NOW + dt.timedelta(days=31))
        from api import referrals as api_refs

        self.assertEqual(api_refs.founder_invites_left("f@example.com"), 0)

    def test_other_inviters_give_the_usual_reward(self):
        until = self.invite("n@example.com", "friend@example.com")
        self.assertLess(until, NOW + dt.timedelta(days=31))
        from api import referrals as api_refs

        self.assertIsNone(api_refs.founder_invites_left("n@example.com"))

    def test_no_bonus_when_invite_rewards_are_off(self):
        self.buy("f@example.com")
        code = self.refs.get_or_create_code("f@example.com")
        self.refs.record_signup(code, "friend@example.com")
        self.refs.reward_verified("friend@example.com", days=0, cap_days=180,
                                  invited_days=lambda *a: 60)
        self.assertIsNone(self.refs.grant_until("friend@example.com"))


class LimitTests(Base):
    def test_elite_alert_and_ai_limits(self):
        self.buy("e@example.com", ELITE)
        self.buy("p@example.com", PRO, "pi_2")
        self.assertEqual(self.pa.founder_alert_limit("e@example.com", 25), rules.ELITE_ALERT_LIMIT)
        self.assertEqual(self.pa.founder_alert_limit("p@example.com", 5), 5)
        self.assertEqual(self.pa.ai_daily_limit("e@example.com", 25), 50)
        self.assertEqual(self.pa.ai_daily_limit("p@example.com", 25), 25)
        self.assertEqual(self.pa.ai_daily_limit("e@example.com", 0), 0, "0 still means unlimited")

    def test_entitlements_and_early_access(self):
        from api.deps import entitlements_for

        ent = entitlements_for({"tier": "premium", "founder_tier": ELITE})
        self.assertEqual(ent["alert_limit"], rules.ELITE_ALERT_LIMIT)
        self.assertEqual(entitlements_for({"tier": "premium"})["alert_limit"], 25)
        self.assertFalse(entitlements_for({"tier": "pro", "founder_tier": PRO})["entitlements"]["can_early_breakout"])
        with mock.patch.dict(os.environ, {"HSF_FOUNDER_EARLY_ACCESS": "can_early_breakout,not_a_feature"}):
            self.assertTrue(entitlements_for({"tier": "pro", "founder_tier": PRO})["entitlements"]["can_early_breakout"])
            self.assertFalse(entitlements_for({"tier": "pro"})["entitlements"]["can_early_breakout"])
            self.assertNotIn("not_a_feature", entitlements_for({"tier": "pro", "founder_tier": PRO})["entitlements"])

    def test_realtime_and_scheduled_alerts_use_the_elite_limit(self):
        from billing_service import realtime_alerts as rt
        from scheduler import alert_runner

        self.buy("e@example.com", ELITE)
        alerts = [{"id": i, "user_id": "e@example.com", "tier": "basic"} for i in range(60)]
        rt._apply_founder_plans(alerts)
        self.assertEqual(len(rt._apply_plan_limits(alerts)), rules.ELITE_ALERT_LIMIT)
        with mock.patch("auth.tier_sync.resolve_user_tier", return_value={"tier_key": "basic"}):
            self.assertEqual(alert_runner._alert_limit_for_user("e@example.com"), rules.ELITE_ALERT_LIMIT)


class AlumniRateTests(Base):
    def test_coupon_only_for_members_whose_five_years_ended(self):
        from billing_service.founding_checkout import alumni_coupon

        m = self.buy("a@example.com")
        with mock.patch.dict(os.environ, {"HSF_FOUNDER_ALUMNI_COUPON": "founder50"}):
            self.assertIsNone(alumni_coupon("a@example.com"), "still active")
            self.conn.execute("UPDATE founding_memberships SET status = 'expired' WHERE id = ?", (m["id"],))
            self.assertEqual(alumni_coupon("a@example.com"), "founder50")
            self.assertIsNone(alumni_coupon("never@example.com"))
            self.conn.execute("UPDATE founding_memberships SET status = 'refunded' WHERE id = ?", (m["id"],))
            self.assertIsNone(alumni_coupon("a@example.com"))
        self.conn.execute("UPDATE founding_memberships SET status = 'expired' WHERE id = ?", (m["id"],))
        self.assertIsNone(alumni_coupon("a@example.com"), "no coupon configured")

    def test_subscription_checkout_applies_the_coupon(self):
        import importlib.util

        if importlib.util.find_spec("fastapi") is None:
            self.skipTest("needs fastapi")
        from tests.test_billing_service import _configured_module

        b = _configured_module()
        b.stripe.checkout.Session.create.return_value = types.SimpleNamespace(id="cs_1", url="https://stripe.test/1")
        with mock.patch.object(b, "_founder_alumni_coupon", return_value="founder50"):
            kw = self._create(b)
        self.assertEqual(kw["discounts"], [{"coupon": "founder50"}])
        self.assertNotIn("allow_promotion_codes", kw)
        with mock.patch.object(b, "_founder_alumni_coupon", return_value=None):
            kw = self._create(b)
        self.assertTrue(kw["allow_promotion_codes"])
        self.assertNotIn("discounts", kw)

    def _create(self, b):
        b.stripe.checkout.Session.create.reset_mock()
        b.STRIPE_PRICE_PRO = "price_pro"
        with mock.patch.object(b, "_require_env"):
            b._create_checkout_for_account("a@example.com", {"plan": "pro"}, user={"stripe_customer_id": None})
        return b.stripe.checkout.Session.create.call_args.kwargs


class ApiTests(Base):
    def setUp(self):
        super().setUp()
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs fastapi, httpx, PyJWT and bcrypt")
        from tests.test_api_v1 import ApiTestCase

        case = type("Case", (ApiTestCase,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.client = case.client
        self.h = case.auth(case.login().json()["access_token"])
        self.admin = case.auth(case.login("boss@example.com").json()["access_token"])

    def test_non_members_get_404(self):
        r = self.client.get("/v1/founding/benefits", headers=self.h)
        self.assertEqual((r.status_code, r.json()["code"]), (404, "founding_not_member"))
        self.assertEqual(self.client.post("/v1/founding/poll", json={"option": "api"}, headers=self.h).status_code, 404)
        self.assertEqual(self.client.get("/v1/founding/benefits").status_code, 401)

    def test_member_benefits_badge_and_poll(self):
        self.buy("pro@example.com", ELITE)
        out = self.client.get("/v1/founding/benefits", headers=self.h).json()
        self.assertEqual((out["badge"], out["alumni_discount_percent"], out["show_badge"]), ("HSF AI Elite Founder #001", 50, True))
        self.assertEqual(out["elite_limits"], {"alert_limit": 50, "ai_multiplier": 2})
        self.assertEqual(out["invites"], {"days": 60, "total": 3, "left": 3})
        self.assertTrue(out["poll"]["open"])
        code = self.refs.get_or_create_code("pro@example.com")
        self.assertEqual(self.client.get(f"/v1/public/founder-badge/{code}").json(), {"badge": "HSF AI Elite Founder #001"})
        out = self.client.post("/v1/founding/benefits/badge", json={"show": False}, headers=self.h).json()
        self.assertFalse(out["show_badge"])
        self.assertEqual(self.client.get(f"/v1/public/founder-badge/{code}").status_code, 404)
        self.assertEqual(self.client.get("/v1/public/founder-badge/nosuchcode").status_code, 404)
        self.assertEqual(self.client.post("/v1/founding/poll", json={"option": "nope"}, headers=self.h).json()["code"],
                         "poll_unknown_option")
        self.assertEqual(self.client.post("/v1/founding/poll", json={"option": "api"}, headers=self.h).json()["poll"]["choice"], "api")
        self.client.post("/v1/founding/poll", json={"option": "crypto"}, headers=self.h)
        self.assertEqual(self.client.get("/v1/admin/founding/poll", headers=self.h).status_code, 403)
        res = self.client.get("/v1/admin/founding/poll", headers=self.admin).json()
        self.assertEqual(res["total"], 1)
        self.assertEqual({o["key"]: o["votes"] for o in res["options"]}["crypto"], 1)

    def test_refunded_member_loses_benefits_and_badge(self):
        self.buy("pro@example.com")
        code = self.refs.get_or_create_code("pro@example.com")
        self.fdb.refund("pi_1", now=NOW)
        self.pa.clear_cache()
        self.assertEqual(self.client.get("/v1/founding/benefits", headers=self.h).status_code, 404)
        self.assertEqual(self.client.get(f"/v1/public/founder-badge/{code}").status_code, 404)

    def test_expired_member_keeps_badge_but_poll_closes(self):
        m = self.buy("pro@example.com")
        self.conn.execute("UPDATE founding_memberships SET status = 'expired' WHERE id = ?", (m["id"],))
        self.pa.clear_cache()
        out = self.client.get("/v1/founding/benefits", headers=self.h).json()
        self.assertEqual((out["status"], out["poll"]["open"], out["invites"], out["early_access"]), ("expired", False, None, []))
        self.assertEqual(self.client.post("/v1/founding/poll", json={"option": "api"}, headers=self.h).json()["code"], "poll_closed")


class FeedbackAndDigestTests(Base):
    def test_founder_feedback_is_marked(self):
        from api import web_state

        with mock.patch("db.web_state.add_feedback") as add, \
                mock.patch.dict(os.environ, {"HSF_FEEDBACK_NOTIFY": "1"}), \
                mock.patch("services.ops_notify.notify_ops") as notify:
            web_state.send_feedback({"username": "f@example.com", "tier": "pro", "founder_tier": PRO},
                                    "idea", "More scans", "/today", None)
        self.assertEqual(add.call_args.args[4], "pro, Founder Pro")
        self.assertIn("(founder)", notify.call_args.args[0])

    def test_digest_header_shows_the_badge(self):
        from scheduler.morning_digest import _founder_badge

        self.assertEqual(_founder_badge("f@example.com"), "")
        self.buy("f@example.com")
        self.assertEqual(_founder_badge("f@example.com"), "HSF AI Founder #001")


if __name__ == "__main__":
    unittest.main()
