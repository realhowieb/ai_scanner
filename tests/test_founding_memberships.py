"""Founding 100 memberships (domain/founding.py, db/founding.py, api/plan_access.py).

db.founding runs its real SQL on an in-memory SQLite database here; the real-Postgres
concurrency tests are tests/test_founding_postgres.py. No Stripe, no network.
"""
import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from domain import founding as rules

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE


class RulesTests(unittest.TestCase):
    def test_capacity_and_prices(self):
        self.assertEqual(rules.CAPACITY, {PRO: 75, ELITE: 25})
        self.assertEqual(sum(rules.CAPACITY.values()), 100)
        self.assertEqual(rules.expected_amount("purchase", PRO), 49_900)
        self.assertEqual(rules.expected_amount("purchase", ELITE), 99_900)
        self.assertEqual(rules.expected_amount("upgrade", ELITE), 50_000)
        with self.assertRaises(ValueError):
            rules.expected_amount("upgrade", PRO)
        with self.assertRaises(ValueError):
            rules.expected_amount("purchase", "founder_gold")

    def test_five_calendar_years(self):
        self.assertEqual(rules.term_end(dt.datetime(2026, 10, 28, 3, 4, 5, tzinfo=UTC)),
                         dt.datetime(2031, 10, 28, 3, 4, 5, tzinfo=UTC))
        # Leap day: 2028-02-29 + 5 years -> 2033-02-28 (no 29 Feb in 2033).
        self.assertEqual(rules.term_end(dt.datetime(2028, 2, 29, 12, tzinfo=UTC)),
                         dt.datetime(2033, 2, 28, 12, tzinfo=UTC))
        # Five calendar years is not 1,825 days when a leap day falls in between.
        start = dt.datetime(2027, 1, 1, tzinfo=UTC)
        self.assertEqual((rules.term_end(start) - start).days, 1826)
        # A naive time is read as UTC; another zone is converted first.
        self.assertEqual(rules.term_end(dt.datetime(2026, 1, 1)), dt.datetime(2031, 1, 1, tzinfo=UTC))
        est = dt.timezone(dt.timedelta(hours=-5))
        self.assertEqual(rules.term_end(dt.datetime(2026, 12, 31, 22, tzinfo=est)),
                         dt.datetime(2032, 1, 1, 3, tzinfo=UTC))

    def test_flags_default_off(self):
        for name in rules.FLAGS:
            self.assertFalse(rules.flag(name, {}))
        self.assertFalse(rules.tier_available(PRO, {}))
        on = {"FOUNDING_100_ENABLED": "1", "FOUNDING_CHECKOUT_ENABLED": "1"}
        self.assertTrue(rules.tier_available(PRO, on))
        self.assertFalse(rules.tier_available(ELITE, on))
        self.assertTrue(rules.tier_available(ELITE, {**on, "FOUNDING_ELITE_ENABLED": "1"}))
        self.assertFalse(rules.tier_available(PRO, {"FOUNDING_100_ENABLED": "1"}))  # checkout off
        with self.assertRaises(KeyError):
            rules.flag("FOUNDING_MADE_UP", {})

    def test_badges(self):
        self.assertEqual(rules.badge_title(PRO, 17), "HSF AI Founder #017")
        self.assertEqual(rules.badge_title(ELITE, 4), "HSF AI Elite Founder #004")
        self.assertIsNone(rules.badge_title(PRO, None))

    def test_entitlements_map_onto_real_flags(self):
        from domain.plans import FEATURE_MIN_TIER, has_min_tier

        for name, rule in rules.ENTITLEMENTS.items():
            for f in rule["flags"]:
                self.assertIn(f, FEATURE_MIN_TIER, (name, f))
                # A flag is only listed under an entitlement whose plan already unlocks it.
                if "min_plan" in rule:
                    self.assertTrue(has_min_tier(rule["min_plan"], FEATURE_MIN_TIER[f]), (name, f))
        pro = rules.entitlements_for("pro", PRO)
        self.assertTrue(pro["scanner.advanced"] and pro["founder.priority_feedback"])
        self.assertFalse(pro["ml.premium_insights"])
        elite = rules.entitlements_for("premium", ELITE)
        self.assertTrue(elite["ml.premium_insights"] and elite["scanner.custom_configurations"])
        # Planned entitlements grant nothing, even to Elite.
        self.assertFalse(elite["api.personal_access"])
        self.assertFalse(elite["founder.early_access"])
        self.assertFalse(rules.entitlements_for("premium")["founder.priority_feedback"])


class StoreBase(unittest.TestCase):
    def setUp(self):
        from db import founding

        self.db = founding
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        founding.set_connection_factory(lambda: self.conn)
        self.addCleanup(lambda: founding.set_connection_factory(None))

    def buy(self, user, tier=PRO, *, at=T0, ref=None):
        res = self.db.reserve(user, tier, now=at)
        out = self.db.fulfill(res["id"], payment_reference=ref or f"pi_{user}", amount_cents=rules.PRICE_CENTS[tier],
                              paid_at=at, now=at)
        self.assertEqual(out["outcome"], "activated", out)
        return out["membership"]

    def inv(self, tier):
        return dict(self.conn.execute("SELECT capacity, held, sold FROM founding_inventory WHERE tier = ?", (tier,)).fetchone())


class StoreTests(StoreBase):
    def test_schema_seeds_75_and_25_and_refuses_overselling(self):
        a = self.db.availability(T0)
        self.assertEqual((a["capacity"], a["remaining"]), (100, 100))
        self.assertEqual(a["tiers"][PRO]["capacity"], 75)
        self.assertEqual(a["tiers"][ELITE]["capacity"], 25)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("UPDATE founding_inventory SET sold = 26 WHERE tier = ?", (ELITE,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO founding_memberships (user_id, membership_number, tier, status, "
                              "purchase_price_cents, created_at, updated_at) VALUES ('x', 101, ?, 'expired', 0, 'n', 'n')",
                              (PRO,))
        with self.assertRaises(sqlite3.IntegrityError):  # active needs a number and dates
            self.conn.execute("INSERT INTO founding_memberships (user_id, tier, status, purchase_price_cents, "
                              "created_at, updated_at) VALUES ('x', ?, 'active', 0, 'n', 'n')", (PRO,))
        self.conn.rollback()

    def test_purchase_activates_after_verified_payment_only(self):
        res = self.db.reserve("A@x.com", PRO, now=T0)
        self.assertEqual((res["status"], res["user_id"]), ("held", "a@x.com"))
        self.assertEqual(self.inv(PRO), {"capacity": 75, "held": 1, "sold": 0})
        self.assertEqual(self.db.active_tier("a@x.com", T0), (None, None))  # a hold grants nothing
        out = self.db.fulfill(res["id"], payment_reference="pi_1", amount_cents=49_900, paid_at=T0, now=T0)
        m = out["membership"]
        self.assertEqual(out["outcome"], "activated")
        self.assertEqual((m["membership_number"], m["tier"], m["status"], m["purchase_price_cents"]), (1, PRO, "active", 49_900))
        self.assertEqual(m["expires_at"], "2031-11-02T15:00:00+00:00")
        self.assertEqual(self.inv(PRO), {"capacity": 75, "held": 0, "sold": 1})
        self.assertEqual(self.db.active_tier("a@x.com", T0)[0], PRO)
        self.assertEqual(self.db.reservation(res["id"])["status"], "converted")

    def test_duplicate_webhook_is_a_no_op(self):
        res = self.db.reserve("a@x.com", PRO, now=T0)
        first = self.db.fulfill(res["id"], payment_reference="pi_1", amount_cents=49_900, now=T0)
        again = self.db.fulfill(res["id"], payment_reference="pi_1", amount_cents=49_900, now=T0)
        self.assertEqual(again["outcome"], "already_done")
        self.assertEqual(again["membership"]["id"], first["membership"]["id"])
        self.assertEqual(self.inv(PRO)["sold"], 1)

    def test_price_tampering_and_wrong_currency_get_no_seat(self):
        for i, (amount, currency) in enumerate(((100, "usd"), (99_900, "usd"), (49_900, "eur"))):
            res = self.db.reserve(f"u{i}@x.com", PRO, now=T0)
            out = self.db.fulfill(res["id"], payment_reference=f"pi_bad{i}", amount_cents=amount, currency=currency, now=T0)
            self.assertEqual((out["outcome"], out["reason"]), ("needs_review", "amount_mismatch"))
            self.assertEqual(self.db.active_tier(f"u{i}@x.com", T0), (None, None))
        self.assertEqual(self.inv(PRO), {"capacity": 75, "held": 0, "sold": 0})  # holds released
        self.assertEqual(len(self.db.needs_review()), 3)

    def test_one_membership_per_account(self):
        self.buy("a@x.com")
        with self.assertRaises(self.db.AlreadyMember):
            self.db.reserve("a@x.com", ELITE, now=T0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO founding_memberships (user_id, membership_number, tier, status, "
                              "purchase_price_cents, activated_at, expires_at, created_at, updated_at) "
                              "VALUES ('a@x.com', 50, ?, 'active', 0, 'n', 'n', 'n', 'n')", (ELITE,))
        self.conn.rollback()

    def test_asking_again_returns_the_same_hold_and_switching_tier_replaces_it(self):
        a = self.db.reserve("a@x.com", PRO, now=T0)
        b = self.db.reserve("a@x.com", PRO, now=T0 + dt.timedelta(minutes=1))
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(self.inv(PRO)["held"], 1)
        c = self.db.reserve("a@x.com", ELITE, now=T0 + dt.timedelta(minutes=2))
        self.assertNotEqual(c["id"], a["id"])
        self.assertEqual((self.inv(PRO)["held"], self.inv(ELITE)["held"]), (0, 1))
        self.assertEqual(self.db.reservation(a["id"])["status"], "released")

    def test_holds_count_against_seats_and_expire_after_the_hold_time(self):
        for i in range(25):
            self.db.reserve(f"e{i}@x.com", ELITE, now=T0)
        self.assertEqual(self.db.availability(T0)["tiers"][ELITE]["remaining"], 0)
        with self.assertRaises(self.db.SoldOut):
            self.db.reserve("late@x.com", ELITE, now=T0 + dt.timedelta(minutes=14))
        later = T0 + dt.timedelta(minutes=rules.RESERVATION_MINUTES)
        self.assertEqual(self.db.availability(later)["tiers"][ELITE]["remaining"], 25)  # display ignores stale holds
        res = self.db.reserve("late@x.com", ELITE, now=later)  # the sweep frees them
        self.assertEqual(res["status"], "held")
        self.assertEqual(self.inv(ELITE), {"capacity": 25, "held": 1, "sold": 0})

    def test_last_seat(self):
        for i in range(24):
            self.buy(f"e{i}@x.com", ELITE)
        last = self.db.reserve("last@x.com", ELITE, now=T0)
        with self.assertRaises(self.db.SoldOut):
            self.db.reserve("other@x.com", ELITE, now=T0)
        self.db.fulfill(last["id"], payment_reference="pi_last", amount_cents=99_900, now=T0)
        self.assertEqual(self.inv(ELITE), {"capacity": 25, "held": 0, "sold": 25})
        self.assertEqual(self.db.availability(T0)["tiers"][ELITE]["remaining"], 0)
        self.assertEqual(self.db.availability(T0)["tiers"][PRO]["remaining"], 75)

    def test_numbers_follow_payment_order_across_tiers(self):
        a = self.db.reserve("a@x.com", PRO, now=T0)
        b = self.db.reserve("b@x.com", ELITE, now=T0)
        nb = self.db.fulfill(b["id"], payment_reference="pi_b", amount_cents=99_900, now=T0)["membership"]
        na = self.db.fulfill(a["id"], payment_reference="pi_a", amount_cents=49_900, now=T0)["membership"]
        self.assertEqual((nb["membership_number"], na["membership_number"]), (1, 2))

    def test_payment_after_the_hold_expired(self):
        res = self.db.reserve("a@x.com", ELITE, now=T0)
        late = T0 + dt.timedelta(minutes=rules.RESERVATION_MINUTES + 5)
        out = self.db.fulfill(res["id"], payment_reference="pi_a", amount_cents=99_900, paid_at=late, now=late)
        self.assertEqual(out["outcome"], "activated")  # a seat was still free
        self.assertEqual(self.inv(ELITE), {"capacity": 25, "held": 0, "sold": 1})

    def test_payment_after_expiry_when_sold_out_is_flagged_not_granted(self):
        mine = self.db.reserve("slow@x.com", ELITE, now=T0)
        later = T0 + dt.timedelta(minutes=rules.RESERVATION_MINUTES + 1)
        for i in range(25):
            self.buy(f"e{i}@x.com", ELITE, at=later)
        out = self.db.fulfill(mine["id"], payment_reference="pi_slow", amount_cents=99_900, now=later)
        self.assertEqual((out["outcome"], out["reason"]), ("needs_review", "sold_out_after_hold_expired"))
        self.assertEqual(self.db.active_tier("slow@x.com", later), (None, None))
        self.assertEqual(self.inv(ELITE), {"capacity": 25, "held": 0, "sold": 25})
        self.assertEqual([r["payment_reference"] for r in self.db.needs_review()], ["pi_slow"])
        # The same webhook again stays flagged, never granted.
        self.assertEqual(self.db.fulfill(mine["id"], payment_reference="pi_slow", amount_cents=99_900, now=later)["outcome"],
                         "needs_review")

    def test_released_hold_then_paid_still_needs_a_free_seat(self):
        res = self.db.reserve("a@x.com", PRO, now=T0)
        self.assertTrue(self.db.release(res["id"], reason="checkout.session.expired", now=T0))
        self.assertFalse(self.db.release(res["id"], now=T0))  # already closed
        self.assertEqual(self.inv(PRO)["held"], 0)
        out = self.db.fulfill(res["id"], payment_reference="pi_a", amount_cents=49_900, now=T0)
        self.assertEqual(out["outcome"], "activated")

    def test_upgrade_keeps_number_and_dates_and_moves_one_seat(self):
        self.buy("x@x.com")
        m = self.buy("a@x.com")
        with self.assertRaises(self.db.NotEligible):
            self.db.reserve("nobody@x.com", ELITE, kind="upgrade", now=T0)
        with self.assertRaises(self.db.NotEligible):
            self.db.reserve("a@x.com", PRO, kind="upgrade", now=T0)
        res = self.db.reserve("a@x.com", ELITE, kind="upgrade", now=T0 + dt.timedelta(days=30))
        bad = self.db.fulfill(res["id"], payment_reference="pi_up_wrong", amount_cents=99_900, now=T0)
        self.assertEqual(bad["outcome"], "needs_review")  # the upgrade costs $500, not $999
        res = self.db.reserve("a@x.com", ELITE, kind="upgrade", now=T0 + dt.timedelta(days=30))
        up = self.db.fulfill(res["id"], payment_reference="pi_up", amount_cents=50_000, now=T0 + dt.timedelta(days=30))
        u = up["membership"]
        self.assertEqual(up["outcome"], "upgraded")
        self.assertEqual((u["id"], u["membership_number"], u["tier"]), (m["id"], m["membership_number"], ELITE))
        self.assertEqual((u["activated_at"], u["expires_at"]), (m["activated_at"], m["expires_at"]))
        self.assertEqual((self.inv(PRO)["sold"], self.inv(ELITE)["sold"]), (1, 1))
        # Charging the same upgrade twice does nothing.
        self.assertEqual(self.db.fulfill(res["id"], payment_reference="pi_up", amount_cents=50_000, now=T0)["outcome"],
                         "already_done")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM founding_memberships").fetchone()[0], 2)

    def test_upgrade_when_elite_is_full(self):
        self.buy("a@x.com")
        for i in range(25):
            self.buy(f"e{i}@x.com", ELITE)
        with self.assertRaises(self.db.SoldOut):
            self.db.reserve("a@x.com", ELITE, kind="upgrade", now=T0)

    def test_refund_gives_seat_and_number_back(self):
        a = self.buy("a@x.com")
        self.buy("b@x.com")
        out = self.db.refund("pi_a@x.com", now=T0)
        r = out["membership"]
        self.assertEqual((out["outcome"], r["status"], r["membership_number"], r["former_number"]),
                         ("refunded", "refunded", None, a["membership_number"]))
        self.assertIsNotNone(r["refunded_at"])
        self.assertEqual(self.inv(PRO)["sold"], 1)
        self.assertEqual(self.db.active_tier("a@x.com", T0), (None, None))
        self.assertEqual(self.db.refund("pi_a@x.com", now=T0)["outcome"], "already_done")
        c = self.buy("c@x.com")
        self.assertEqual(c["membership_number"], 1)  # the refunded number is free again
        # The refunded account may buy again (its old row isn't live).
        self.assertEqual(self.buy("a@x.com", ref="pi_a2")["membership_number"], 3)

    def test_dispute_revokes(self):
        self.buy("a@x.com", ELITE)
        out = self.db.revoke("pi_a@x.com", reason="dispute", now=T0)
        self.assertEqual((out["outcome"], out["membership"]["status"]), ("revoked", "revoked"))
        self.assertEqual(self.inv(ELITE)["sold"], 0)
        with self.assertRaises(self.db.NotFound):
            self.db.refund("pi_unknown", now=T0)

    def test_refunding_an_upgrade_moves_back_to_founder_pro(self):
        m = self.buy("a@x.com")
        res = self.db.reserve("a@x.com", ELITE, kind="upgrade", now=T0)
        self.db.fulfill(res["id"], payment_reference="pi_up", amount_cents=50_000, now=T0)
        out = self.db.refund("pi_up", now=T0)
        b = out["membership"]
        self.assertEqual((out["outcome"], b["tier"], b["status"], b["membership_number"]),
                         ("upgrade_reversed", PRO, "active", m["membership_number"]))
        self.assertEqual((self.inv(PRO)["sold"], self.inv(ELITE)["sold"]), (1, 0))

    def test_expiry_ends_access_but_keeps_the_badge(self):
        m = self.buy("a@x.com")
        end = dt.datetime(2031, 11, 2, 15, 0, tzinfo=UTC)
        self.assertEqual(self.db.active_tier("a@x.com", end - dt.timedelta(seconds=1))[0], PRO)
        self.assertEqual(self.db.active_tier("a@x.com", end), (None, None))
        self.assertEqual(self.db.expire_due(end), 1)
        kept = self.db.membership("a@x.com")
        self.assertEqual((kept["status"], kept["membership_number"]), ("expired", m["membership_number"]))
        self.assertEqual(self.inv(PRO)["sold"], 1)  # an expired founder still holds the seat and number

    def test_sweep_holds(self):
        self.db.reserve("a@x.com", PRO, now=T0)
        self.db.reserve("b@x.com", ELITE, now=T0)
        self.assertEqual(self.db.sweep_holds(T0 + dt.timedelta(minutes=10)), 0)
        self.assertEqual(self.db.sweep_holds(T0 + dt.timedelta(minutes=rules.RESERVATION_MINUTES)), 2)
        self.assertEqual((self.inv(PRO)["held"], self.inv(ELITE)["held"]), (0, 0))

    def test_a_failed_step_rolls_everything_back(self):
        res = self.db.reserve("a@x.com", PRO, now=T0)
        with mock.patch.object(self.db, "_next_number", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.db.fulfill(res["id"], payment_reference="pi_a", amount_cents=49_900, now=T0)
        self.assertEqual(self.inv(PRO), {"capacity": 75, "held": 1, "sold": 0})
        self.assertEqual(self.db.reservation(res["id"])["status"], "held")
        self.assertEqual(self.db.fulfill(res["id"], payment_reference="pi_a", amount_cents=49_900, now=T0)["outcome"],
                         "activated")


class PlanAccessTests(StoreBase):
    def setUp(self):
        super().setUp()
        from api import plan_access

        self.pa = plan_access
        plan_access.clear_cache()
        self.addCleanup(plan_access.clear_cache)
        # No invite trials here (db.referrals would need its own tables).
        p = mock.patch("api.referrals.grant_until", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def tier(self, user, plan, admin=False):
        self.pa.clear_cache()
        return self.pa.effective_tier(user, plan, admin)

    def test_no_members_means_no_lookup(self):
        with mock.patch.object(self.db, "active_tier") as looked:
            self.assertEqual(self.tier("a@x.com", "basic")["tier"], "basic")
            looked.assert_not_called()

    def test_founder_tiers_unlock_their_plan_and_never_lower_one(self):
        now = dt.datetime.now(UTC)
        self.buy("pro@x.com", at=now)
        self.buy("elite@x.com", ELITE, at=now)
        self.buy("sub@x.com", at=now)
        self.assertEqual(self.tier("pro@x.com", "basic")["tier"], "pro")
        self.assertEqual(self.tier("elite@x.com", "basic")["tier"], "premium")
        self.assertEqual(self.tier("elite@x.com", "pro")["tier"], "premium")
        self.assertEqual(self.tier("sub@x.com", "premium")["tier"], "premium")  # a Premium subscriber keeps Premium
        self.assertEqual(self.tier("nobody@x.com", "basic")["tier"], "basic")
        self.assertEqual(self.tier("pro@x.com", "admin", admin=True)["tier"], "admin")
        acct = self.pa.apply_to_account({"username": "elite@x.com", "tier": "basic", "is_admin": False})
        self.assertEqual((acct["tier"], acct["founder_tier"]), ("premium", ELITE))
        self.assertTrue(acct["founder_until"].startswith(str(now.year + 5)))

    def test_expired_refunded_and_storage_errors_fall_back_to_own_plan(self):
        long_ago = dt.datetime.now(UTC) - dt.timedelta(days=5 * 366)
        self.buy("old@x.com", at=long_ago)
        self.buy("refunded@x.com", at=dt.datetime.now(UTC))
        self.db.refund("pi_refunded@x.com")
        self.assertEqual(self.tier("old@x.com", "basic")["tier"], "basic")
        self.assertEqual(self.tier("refunded@x.com", "basic")["tier"], "basic")
        self.buy("ok@x.com", at=dt.datetime.now(UTC))
        self.pa.clear_cache()
        with mock.patch.object(self.db, "active_tier", side_effect=RuntimeError("db down")):
            self.assertEqual(self.pa.effective_tier("ok@x.com", "basic")["tier"], "basic")

    def test_api_sees_founder_plan(self):
        """A signed-in Founder Elite on Starter gets Premium entitlements from the API."""
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs fastapi, httpx, PyJWT and bcrypt")
        from tests.test_api_v1 import ApiTestCase

        case = type("Case", (ApiTestCase,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.accounts["pro@example.com"]["tier"] = "basic"
        self.buy("pro@example.com", ELITE, at=dt.datetime.now(UTC))
        self.pa.clear_cache()
        h = case.auth(case.login().json()["access_token"])
        me = case.client.get("/v1/me", headers=h).json()
        self.assertEqual(me["plan"], "premium")
        self.assertTrue(me["entitlements"]["can_early_breakout"])


if __name__ == "__main__":
    unittest.main()
