"""2026-11-10 conversion pack: shorter sign-up (display name optional), the new-account
free Pro trial at email confirmation, the follow-up for unfinished checkouts, locked
feature previews, and company facts (market cap, industry, size filter)."""
import datetime as dt
import os
import sqlite3
import unittest
from unittest import mock

from tests.test_api_account import GOOD, AccountApiTestCase
from tests.test_api_v1 import DEPS

UTC = dt.timezone.utc


class _Store(unittest.TestCase):
    def setUp(self):
        from api import referrals as svc
        from db import referrals

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(lambda: (referrals.set_connection_factory(None), svc.clear_cache(), self.conn.close()))
        self.addCleanup(mock.patch.stopall)


class NewTrialTests(_Store):
    def _start(self, account, founder=None, **env):
        from api import referrals as svc

        with mock.patch.dict(os.environ, env), mock.patch("api.store.get_account", return_value=account), \
                mock.patch("api.plan_access.founder_access", return_value=(founder, None)):
            return svc.start_new_trial("new@example.com")

    def test_a_new_starter_account_gets_seven_days_once(self):
        until = self._start({"username": "new@example.com", "tier": "basic"})
        self.assertIsNotNone(until)
        self.assertAlmostEqual((until - dt.datetime.now(UTC)).total_seconds() / 86400, 7, delta=0.01)
        self.assertIsNone(self._start({"username": "new@example.com", "tier": "basic"}))  # never twice

    def test_paid_admin_founder_and_switched_off_get_none(self):
        self.assertIsNone(self._start({"tier": "pro"}))
        self.assertIsNone(self._start({"tier": "basic", "is_admin": True}))
        self.assertIsNone(self._start({"tier": "basic"}, founder="founder_pro"))
        self.assertIsNone(self._start({"tier": "basic"}, HSF_NEW_TRIAL_DAYS="0"))
        self.assertIsNone(self._start(None))

    def test_days_follow_the_env_with_a_cap(self):
        from api import referrals as svc

        for raw, want in (("3", 3), ("90", 30), ("x", 7)):
            with mock.patch.dict(os.environ, {"HSF_NEW_TRIAL_DAYS": raw}):
                self.assertEqual(svc.new_trial_days(), want)


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class SignupAndVerifyTests(AccountApiTestCase):
    def setUp(self):
        super().setUp()
        from api import referrals as svc
        from db import referrals

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(lambda: (referrals.set_connection_factory(None), svc.clear_cache(), self.conn.close()))

    def test_signup_without_a_display_name_takes_one_from_the_email(self):
        r = self.client.post("/v1/auth/signup", json={"email": "Jane.Doe+hsf@Example.com", "password": GOOD, "accept_terms": True})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(self.created[-1]["name"], "jane.doe")  # the API lowercases the email first
        from api import ratelimit

        ratelimit.reset()
        r = self.client.post("/v1/auth/signup", json={"email": "jane.doe@other.com", "password": GOOD, "accept_terms": True})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertRegex(self.created[-1]["name"], r"^jane\.doe-\d{4}$")

    def test_verify_starts_the_free_trial_and_says_so(self):
        self.client.post("/v1/auth/signup", json={"email": "trial@example.com", "password": GOOD, "accept_terms": True})
        with mock.patch("services.lifecycle_emails.send_welcome"):
            r = self.client.post("/v1/auth/verify-email", json={"token": "verify-trial@example.com-token"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("7 days of free Pro have started", r.json()["message"])
        token = self.login("trial@example.com", GOOD).json()["access_token"]
        me = self.client.get("/v1/me", headers=self.auth(token)).json()
        self.assertEqual(me["plan"], "pro")
        self.assertTrue(me["pro_trial_until"])

    def test_plans_say_how_long_the_trial_is(self):
        self.assertEqual(self.client.get("/v1/plans").json()["trial_days"], 7)
        with mock.patch.dict(os.environ, {"HSF_NEW_TRIAL_DAYS": "0"}):
            self.assertEqual(self.client.get("/v1/plans").json()["trial_days"], 0)

    def test_a_new_checkout_is_noted_for_the_follow_up(self):
        self.verified.add("pro@example.com")
        with mock.patch("api.account.checkout_url", return_value={"url": "https://checkout.stripe.com/x", "mode": "checkout"}), \
                mock.patch("services.checkout_followup.record_start") as rec:
            r = self.client.post("/v1/billing/checkout", json={"plan": "premium", "interval": "year"},
                                 headers=self.auth(self.login().json()["access_token"]))
        self.assertEqual(r.status_code, 200, r.text)
        rec.assert_called_once_with("pro@example.com", "premium", "year", "checkout")


class CheckoutFollowupTests(_Store):
    NOW = dt.datetime(2026, 11, 12, 16, 0, tzinfo=UTC)

    def log(self, user, plan, hours_ago, interval="month"):
        from services import checkout_followup as cf

        cf.record_start(user, plan, interval, "checkout", now=self.NOW - dt.timedelta(hours=hours_ago))

    def test_picks_each_accounts_latest_start_20_to_72_hours_old(self):
        from db import referrals
        from services import checkout_followup as cf

        self.log("a@x.com", "pro", 30)
        self.log("a@x.com", "premium", 25, "year")   # latest wins
        self.log("b@x.com", "pro", 5)                # too fresh
        self.log("c@x.com", "pro", 100)              # too old
        cf.record_start("d@x.com", "pro", "month", "portal", now=self.NOW)  # plan change, not a checkout
        rows = referrals.log_since(cf.STARTED, self.NOW - dt.timedelta(days=5))
        self.assertEqual(cf.pick(rows, now=self.NOW), [{"user": "a@x.com", "plan": "premium", "interval": "year"}])

    def test_skips_people_who_already_bought_and_admins(self):
        from services import checkout_followup as cf

        self.assertTrue(cf.still_wanted("pro", {"tier": "basic"}, "basic"))
        self.assertTrue(cf.still_wanted("premium", {"tier": "pro"}, "pro"))
        self.assertFalse(cf.still_wanted("pro", {"tier": "pro"}, "pro"))
        self.assertFalse(cf.still_wanted("pro", {"is_admin": True}, "basic"))
        self.assertFalse(cf.still_wanted("pro", None, "basic"))

    def test_sends_one_note_ever_with_the_link_and_founding_seats(self):
        from services import checkout_followup as cf

        self.log("a@x.com", "pro", 30)
        send = mock.patch("ui.email_utils.send_digest_email", return_value=True).start()
        mock.patch("db.email_prefs.wants_email", return_value=True).start()
        mock.patch("db.email_prefs.unsubscribe_url", return_value="https://u").start()
        mock.patch("api.store.get_account", return_value={"username": "a@x.com", "tier": "basic"}).start()
        mock.patch("api.plan_access.with_founder", side_effect=lambda u, t: t or "basic").start()
        mock.patch("api.founding.public_status", return_value={"checkout_open": True, "remaining": 100}).start()
        self.assertEqual(cf.run(now=self.NOW)["sent"], 1)
        subject, inner, text = send.call_args.args[1], send.call_args.args[2], send.call_args.args[3]
        self.assertEqual(subject, "You were one step from Pro")
        self.assertIn("/account?upgrade=pro&interval=month#plan", text)
        self.assertIn("100 of 100 seats are left", text)
        self.assertIn("Nothing was charged", inner)
        self.log("a@x.com", "pro", 21)
        self.assertEqual(cf.run(now=self.NOW)["sent"], 0)  # one follow-up per account, ever

    def test_off_switch(self):
        from services import checkout_followup as cf

        with mock.patch.dict(os.environ, {"HSF_CHECKOUT_FOLLOWUP_ENABLED": "0"}):
            self.assertTrue(cf.run(now=self.NOW).get("disabled"))


class PreviewTests(unittest.TestCase):
    def setUp(self):
        from api import previews

        previews._cache.clear()

    def test_day_trader_shows_the_top_unflagged_mover_and_a_count(self):
        from api import previews

        rows = [{"ticker": "bad", "quote_flags": ["stale"]}, {"ticker": "nvda", "chg_pct": 6.21, "rvol": 3.44},
                {"ticker": "amd", "chg_pct": 2.0}]
        with mock.patch("api.market.day_trader", return_value={"state": "open", "rows": rows}):
            p = previews.preview("day_trader")
        self.assertEqual(p["count"], 2)
        self.assertEqual(p["example"], {"ticker": "NVDA", "detail": "+6.2% · 3.4x volume"})

    def test_gappers_and_earnings(self):
        from api import previews

        with mock.patch("api.market.premarket_gappers", return_value={"active": True, "rows": [
                {"ticker": "AAA", "pct": 3.0}, {"ticker": "BBB", "pct": -9.5}]}):
            self.assertEqual(previews.preview("gappers")["example"], {"ticker": "BBB", "detail": "gapping -9.5%"})
        base = {"days": [{"date": "2026-11-12", "rows": [{"ticker": "SMALL", "dollar_volume": 1e6, "date": "2026-11-12"},
                                                        {"ticker": "BIG", "dollar_volume": 9e9, "date": "2026-11-12", "time": "amc"}]}]}
        with mock.patch("api.earnings_week.cached_base", return_value=base):
            p = previews.preview("earnings")
        self.assertEqual((p["count"], p["example"]), (2, {"ticker": "BIG", "detail": "reports Thursday after the close"}))

    def test_errors_give_an_empty_preview(self):
        from api import previews

        with mock.patch("api.market.day_trader", side_effect=RuntimeError("down")):
            p = previews.preview("day_trader")
        self.assertEqual((p["count"], p["example"]), (0, None))


class CompanyFactsTests(unittest.TestCase):
    def setUp(self):
        from api import sectors

        sectors.facts_map.cache_clear()
        sectors.sector_map.cache_clear()
        self.addCleanup(lambda: (sectors.facts_map.cache_clear(), sectors.sector_map.cache_clear()))

    def facts(self, text):
        import tempfile
        from pathlib import Path

        from api import sectors

        d = tempfile.mkdtemp()
        p = Path(d) / "sectors.csv"
        p.write_text(text)
        return mock.patch.object(sectors, "SECTORS_PATH", p)

    def test_reads_market_cap_and_industry_and_buckets_size(self):
        from api import sectors

        with self.facts("ticker,sector,industry,market_cap\nNVDA,Technology,Semiconductors,3200000000000\nTINY,Health Care,,120000000\n"):
            self.assertEqual(sectors.company_facts("NVDA"), {"sector": "Technology", "industry": "Semiconductors",
                                                            "market_cap": 3.2e12, "size": "Mega cap"})
            self.assertEqual(sectors.company_facts("tiny")["size"], "Micro cap")
            self.assertIsNone(sectors.company_facts("ZZZZ"))
            self.assertEqual(sectors.sector_for("NVDA"), "Technology")

    def test_an_older_two_column_file_still_works(self):
        from api import sectors

        with self.facts("ticker,sector\nAAPL,Technology\n"):
            self.assertEqual(sectors.company_facts("AAPL"), {"sector": "Technology", "industry": None,
                                                            "market_cap": None, "size": None})

    def test_buckets(self):
        from api.sectors import cap_bucket

        self.assertEqual([cap_bucket(v) for v in (None, 1e8, 3e8, 5e9, 5e10, 3e11)],
                         [None, "micro", "small", "mid", "large", "mega"])

    def test_scanner_size_filter(self):
        from tests.test_scanner_rank_sector import _scan, col

        caps = {"AAA": 3e12, "BBB": 5e9, "CCC": 5e9}
        with mock.patch("api.sectors.market_cap_for", side_effect=lambda t: caps.get(str(t).upper())):
            r = _scan(size="mid")
            self.assertEqual([s["ticker"] for s in r["setups"]], ["BBB", "CCC"])
            self.assertEqual(col(_scan(), "market_cap")["AAA"], 3e12)


if __name__ == "__main__":
    unittest.main()
