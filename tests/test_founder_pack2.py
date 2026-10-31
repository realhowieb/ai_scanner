"""Founder pack 2: the 5-year math, the shareable founder card, the founder value meter
(founder page, monthly digest line), the Founders Wall, the monthly Founder Briefing and
founder priority support (feedback sorting, answered state, daily nudge, reply target)."""
import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from domain import founding as rules
from tests.test_founder_benefits import _NEEDS_API, Base

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE


class RuleTests(unittest.TestCase):
    def test_months_between(self):
        a = dt.datetime(2026, 10, 31, 12, tzinfo=UTC)
        self.assertEqual(rules.months_between(a, a), 0)
        self.assertEqual(rules.months_between(a, dt.datetime(2026, 11, 30, 12, tzinfo=UTC)), 0)
        self.assertEqual(rules.months_between(a, dt.datetime(2026, 12, 31, 12, tzinfo=UTC)), 2)
        self.assertEqual(rules.months_between(a, a - dt.timedelta(days=3)), 0)

    def test_five_year_math(self):
        self.assertEqual(rules.five_year_math(PRO, 2500),
                         {"monthly_cents": 2500, "five_years_monthly_cents": 150_000, "price_cents": 49_900,
                          "saved_cents": 100_100})
        self.assertEqual(rules.five_year_math(ELITE, 4000)["five_years_monthly_cents"], 240_000)

    def test_value_meter(self):
        start = dt.datetime(2026, 1, 15, tzinfo=UTC)
        v = rules.value_meter(PRO, start, 2500, now=dt.datetime(2026, 4, 20, tzinfo=UTC))
        self.assertEqual((v["months_used"], v["months_left"], v["monthly_cost_so_far_cents"], v["ahead_cents"],
                          v["break_even_month"]), (3, 57, 7500, 0, 20))
        late = rules.value_meter(PRO, start, 2500, now=dt.datetime(2028, 1, 15, tzinfo=UTC))
        self.assertEqual((late["months_used"], late["ahead_cents"]), (24, 60_000 - 49_900))
        done = rules.value_meter(ELITE, start, 4000, now=dt.datetime(2040, 1, 1, tzinfo=UTC))
        self.assertEqual((done["months_used"], done["months_left"]), (60, 0))

    def test_wall_names(self):
        for ok in ("Howard B.", "O'Neil", "trader_42", "Ann-Marie Lee"):
            self.assertEqual(rules.clean_wall_name(ok), ok)
        self.assertEqual(rules.clean_wall_name("  Two   spaces "), "Two spaces")
        for bad in ("", "x", "a@b.com", "mysite.com", "www.example", "http stuff", "<b>hi</b>", "x" * 31, "-dash"):
            self.assertIsNone(rules.clean_wall_name(bad), bad)

    def test_monthly_prices_follow_the_pricing_page(self):
        from services import founder_value

        self.assertEqual((founder_value.monthly_cents("pro"), founder_value.monthly_cents("premium")), (2500, 4000))
        self.assertEqual(founder_value.monthly_cents("basic"), 0)

    def test_first_weekday_of_month(self):
        from scheduler.morning_digest import first_weekday_of_month

        self.assertTrue(first_weekday_of_month(dt.date(2026, 12, 1)))   # Tuesday the 1st
        self.assertFalse(first_weekday_of_month(dt.date(2026, 12, 2)))
        self.assertFalse(first_weekday_of_month(dt.date(2026, 11, 1)))  # Sunday
        self.assertTrue(first_weekday_of_month(dt.date(2026, 11, 2)))   # Monday after
        self.assertFalse(first_weekday_of_month(dt.date(2026, 11, 3)))


class WallAndValueTests(Base):
    def test_wall_lists_every_seat_names_only_when_opted_in(self):
        from db import founding_benefits as extras

        self.buy("a@example.com", PRO, ref="pi_a")
        self.buy("b@example.com", ELITE, ref="pi_b")
        extras.set_wall_name("b@example.com", "Bea")
        seats = extras.wall()
        self.assertEqual(len(seats), 100)
        self.assertEqual(seats[0], {"number": "001", "taken": True, "tier": PRO, "label": "Founder Pro", "name": None,
                                    "since": f"{NOW:%Y-%m}"})
        self.assertEqual((seats[1]["name"], seats[1]["label"]), ("Bea", "Founder Elite"))
        self.assertEqual(seats[2], {"number": "003", "taken": False, "tier": None, "label": None, "name": None,
                                    "since": None})
        self.assertNotIn("example.com", repr(seats))
        extras.set_wall_name("b@example.com", None)
        self.assertIsNone(extras.wall()[1]["name"])

    def test_refunded_seat_leaves_the_wall(self):
        from db import founding_benefits as extras

        self.buy("a@example.com")
        extras.set_wall_name("a@example.com", "Al")
        self.fdb.refund("pi_1", now=NOW)
        self.assertFalse(extras.wall()[0]["taken"])

    def test_value_for_user_and_digest_card(self):
        from scheduler import morning_digest as md
        from services import founder_value

        self.assertIsNone(founder_value.for_user("a@example.com"))
        self.buy("a@example.com", at=NOW - dt.timedelta(days=100))
        v = founder_value.for_user("a@example.com")
        self.assertEqual((v["months_used"], v["paid_cents"], v["monthly_cents"]), (3, 49_900, 2500))
        first = dt.datetime(2026, 12, 1, 12, tzinfo=UTC)
        html, text = md._founder_value_card("a@example.com", now=first)
        self.assertIn("Your founder value", html)
        self.assertIn("Month 3 of 60", text)
        self.assertEqual(md._founder_value_card("a@example.com", now=first + dt.timedelta(days=1)), ("", ""))
        self.assertEqual(md._founder_value_card("nobody@example.com", now=first), ("", ""))

    def test_briefing_message_and_send_once(self):
        from db import founding_benefits as extras
        from services import founding_emails as mail

        self.buy("a@example.com", ref="pi_a")
        self.buy("b@example.com", ELITE, ref="pi_b")
        extras.vote("a@example.com", "crypto")
        parts = mail.briefing_parts("Faster scans.\n\nNew earnings view.")
        self.assertEqual(parts["poll"][0], ("Crypto scanner", 1))
        msg = mail.briefing_message(extras.active_members()[0], parts, "2026-11",
                                    {"months_used": 1, "months_left": 59, "term_months": 60, "monthly_cents": 2500,
                                     "paid_cents": 49_900, "monthly_cost_so_far_cents": 2500, "ahead_cents": 0,
                                     "break_even_month": 20, "alerts_used": None}, "crypto")
        self.assertEqual(msg["subject"], "Your Founder Briefing: November 2026")
        for s in ("Faster scans.", "New earnings view.", "Your vote: Crypto scanner.", "Month 1 of 60",
                  "pays for itself in month 20", "HSF AI Founder #001"):
            self.assertIn(s, msg["text"])
        self.assertIn("What shipped", msg["html"])
        with mock.patch("ui.email_utils._send_smtp", return_value=True) as smtp:
            out = mail.send_briefing("Faster scans.", month="2026-11")
        self.assertEqual(out, {"sent": 2, "failed": 0})
        self.assertEqual(sorted(c.args[0] for c in smtp.call_args_list), ["a@example.com", "b@example.com"])
        self.assertTrue(extras.claim_briefing("2026-11", "s", "boss@example.com"))
        self.assertFalse(extras.claim_briefing("2026-11", "s", "boss@example.com"))


class FeedbackPriorityTests(unittest.TestCase):
    def setUp(self):
        from db import web_state

        self.store = web_state
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        web_state.set_connection_factory(lambda: self.conn)
        self.addCleanup(lambda: web_state.set_connection_factory(None))

    def test_founders_first_until_answered(self):
        s = self.store
        s.add_feedback("a@example.com", "idea", "one", None, "pro, Founder Pro", None)
        s.add_feedback("b@example.com", "idea", "two", None, "basic", None)
        s.add_feedback("c@example.com", "idea", "three", None, "premium, Founder Elite", None)
        rows = s.recent_feedback()
        self.assertEqual([r["message"] for r in rows], ["three", "one", "two"])
        self.assertEqual([r["founder"] for r in rows], [True, True, False])
        self.assertEqual(s.unanswered_founder_feedback()["count"], 2)
        self.assertTrue(s.mark_answered(rows[0]["id"], "boss@example.com"))
        self.assertFalse(s.mark_answered(999, "boss@example.com"))
        rows = s.recent_feedback()
        self.assertEqual([r["message"] for r in rows], ["one", "three", "two"])
        self.assertEqual(rows[1]["answered_by"], "boss@example.com")
        self.assertEqual(s.unanswered_founder_feedback()["count"], 1)
        s.mark_answered(rows[1]["id"], "boss@example.com", answered=False)
        self.assertEqual(s.unanswered_founder_feedback()["count"], 2)

    def test_daily_nudge(self):
        from api import ops_jobs

        with mock.patch("services.ops_notify.notify_ops") as notify:
            self.assertFalse(ops_jobs.founder_feedback_nudge()["sent"])
            self.store.add_feedback("a@example.com", "idea", "one", None, "pro, Founder Pro", None)
            out = ops_jobs.founder_feedback_nudge()
        self.assertEqual((out["sent"], out["count"]), (True, 1))
        self.assertIn("1 founder message waiting", notify.call_args.args[1])
        self.assertEqual(ops_jobs.founder_nudge_due(dt.datetime(2026, 11, 2, 15, tzinfo=UTC)), "2026-11-02")
        self.assertIsNone(ops_jobs.founder_nudge_due(dt.datetime(2026, 11, 1, 15, tzinfo=UTC)))  # Sunday
        self.assertIsNone(ops_jobs.founder_nudge_due(dt.datetime(2026, 11, 2, 13, tzinfo=UTC)))


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
        from api import founding as svc

        self.svc = svc
        svc.clear_cache()
        self.addCleanup(svc.clear_cache)
        p = mock.patch.dict(os.environ, {"FOUNDING_100_ENABLED": "1"})
        p.start()
        self.addCleanup(p.stop)

    def test_public_status_carries_the_five_year_math(self):
        out = self.client.get("/v1/public/founding").json()
        self.assertEqual(out["tiers"][0]["five_year"]["five_years_monthly_cents"], 150_000)
        self.assertEqual(out["tiers"][1]["five_year"]["five_years_monthly_cents"], 240_000)
        offer = self.client.get("/v1/founding/offer", headers=self.h).json()
        self.assertEqual(offer["tiers"][0]["five_year"]["monthly_cents"], 2500)

    def test_benefits_wall_and_card(self):
        self.buy("pro@example.com", ELITE)
        with mock.patch("db.alert_rules.count_active", return_value=3):
            out = self.client.get("/v1/founding/benefits", headers=self.h).json()
        self.assertEqual((out["value"]["months_used"], out["value"]["alerts_used"], out["wall_name"]), (0, 3, None))
        code = self.refs.get_or_create_code("pro@example.com")
        self.assertEqual(out["card_path"], f"/founders/card/{code}")
        self.assertIsNone(out["reply_target"])
        bad = self.client.post("/v1/founding/wall", json={"show": True, "name": "me@x.com"}, headers=self.h)
        self.assertEqual((bad.status_code, bad.json()["code"]), (400, "wall_name_invalid"))
        out = self.client.post("/v1/founding/wall", json={"show": True, "name": "Pro Trader"}, headers=self.h).json()
        self.assertEqual(out["wall_name"], "Pro Trader")
        wall = self.client.get("/v1/public/founding/wall").json()
        self.assertEqual((wall["taken"], wall["seats"][0]["name"], len(wall["seats"])), (1, "Pro Trader", 100))
        self.client.post("/v1/founding/wall", json={"show": False}, headers=self.h)
        self.assertIsNone(self.client.get("/v1/public/founding/wall").json()["seats"][0]["name"])
        card = self.client.get(f"/v1/public/founder-card/{code}").json()
        self.assertEqual((card["badge"], card["number"], card["label"]), ("HSF AI Elite Founder #001", "001", "Founder Elite"))
        self.assertNotIn("example.com", repr(card))
        self.client.post("/v1/founding/benefits/badge", json={"show": False}, headers=self.h)
        self.assertEqual(self.client.get(f"/v1/public/founder-card/{code}").status_code, 404)

    def test_wall_is_closed_with_the_program(self):
        with mock.patch.dict(os.environ, {"FOUNDING_100_ENABLED": "0"}):
            self.svc.clear_cache()
            self.assertEqual(self.client.get("/v1/public/founding/wall").json(), {"open": False, "seats": [], "taken": 0})

    def test_non_members_cannot_join_the_wall(self):
        r = self.client.post("/v1/founding/wall", json={"show": True, "name": "Someone"}, headers=self.h)
        self.assertEqual(r.status_code, 404)

    def test_briefing_preview_and_send(self):
        self.buy("pro@example.com")
        self.assertEqual(self.client.post("/v1/admin/founding/briefing/preview", json={"note": "x"},
                                          headers=self.h).status_code, 403)
        prev = self.client.post("/v1/admin/founding/briefing/preview", json={"note": "Shipped the wall."},
                                headers=self.admin).json()
        self.assertEqual((prev["recipients"], prev["already_sent"]), (1, False))
        self.assertIn("Shipped the wall.", prev["text"])
        r = self.client.post("/v1/admin/founding/briefing/send", json={"note": "Shipped the wall.", "confirm": True},
                             headers=self.admin)
        self.assertEqual((r.status_code, r.json()["code"]), (400, "briefing_note_short"))
        note = "Shipped the Founders Wall and the value meter this month."
        self.assertEqual(self.client.post("/v1/admin/founding/briefing/send", json={"note": note, "confirm": False},
                                          headers=self.admin).status_code, 400)
        with mock.patch("ui.email_utils._send_smtp", return_value=True):
            r = self.client.post("/v1/admin/founding/briefing/send", json={"note": note, "confirm": True},
                                 headers=self.admin)
            self.assertEqual((r.status_code, r.json()["sent"]), (200, 1))
            again = self.client.post("/v1/admin/founding/briefing/send", json={"note": note, "confirm": True},
                                     headers=self.admin)
        self.assertEqual((again.status_code, again.json()["code"]), (409, "briefing_already_sent"))
        prev = self.client.post("/v1/admin/founding/briefing/preview", json={"note": ""}, headers=self.admin).json()
        self.assertTrue(prev["already_sent"])
        self.assertEqual(prev["history"][0]["sent"], 1)


@_NEEDS_API
class FeedbackApiTests(Base):
    def test_founder_reply_message(self):
        from tests.test_api_v1 import DEPS

        if not DEPS:
            self.skipTest("needs the API test stack")
        from tests.test_web_state import WebStateApiTests

        case = type("Case", (WebStateApiTests,), {"runTest": lambda s: None})()
        case.setUp()
        self.addCleanup(case.doCleanups)
        with mock.patch.dict(os.environ, {"HSF_FOUNDER_REPLY_TARGET": "one business day"}), \
                mock.patch("api.plan_access.founder_access", return_value=(PRO, NOW + dt.timedelta(days=9))):
            r = case.client.post("/v1/me/feedback", json={"kind": "idea", "message": "More scans"}, headers=case.h)
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["message"], "Thanks, founder. Your message goes to the top of our list. "
                                              "We aim to reply within one business day.")
        admin = case.auth(case.login("boss@example.com").json()["access_token"])
        items = case.client.get("/v1/admin/feedback", headers=admin).json()
        self.assertEqual(len(items), 1)
        r = case.client.post(f"/v1/admin/feedback/{items[0]['id']}/answered", json={"answered": True}, headers=admin)
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(case.client.get("/v1/admin/feedback", headers=admin).json()[0]["answered_at"])
        self.assertEqual(case.client.post("/v1/admin/feedback/999/answered", json={}, headers=admin).status_code, 404)
        st = case.client.get("/v1/admin/feedback/founders", headers=admin).json()
        self.assertEqual((st["count"], items[0]["founder"]), (0, True))


if __name__ == "__main__":
    unittest.main()
