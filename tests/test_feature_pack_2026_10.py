"""Feature pack (2026-10-17): earnings heads-up alert rule, score movers, watchlist
performance, journal insights, and invite-a-friend referrals with a Pro trial.

No network, no Postgres: providers and observations are patched, referrals run on
SQLite through db.referrals.set_connection_factory.
"""
import datetime as dt
import importlib.util
import sqlite3
import unittest
from unittest import mock

from tests.test_alert_rules_engine import NOW, Store, observation, opp
from tests.test_api_v1 import DEPS, ApiTestCase, _hash

UTC = dt.timezone.utc
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


# ---- earnings heads-up --------------------------------------------------------------------------
class EarningsRuleCheckTests(unittest.TestCase):
    def setUp(self):
        from api.alert_rules import check

        self.check = check
        self.rule = {"rule_type": "EARNINGS_SOON", "threshold": 1.0}

    def test_fires_once_per_earnings_date_and_needs_no_scan_row(self):
        state = None
        fired = []
        for days, date in ((3, "2026-10-20"), (1, "2026-10-20"), (0, "2026-10-20"), (None, None),
                           (1, "2026-10-21")):
            f, value, _prev, new = self.check(self.rule, {"present": False, "earnings_days": days,
                                                          "earnings_date": date}, state)
            fired.append(f)
            state = {**(state or {}), **new}
        self.assertEqual(fired, [False, True, False, False, True])  # a moved date fires again

    def test_message_names_the_day_and_timing(self):
        from api.alert_rules import message_for

        msg = message_for(self.rule, "AAPL", 1.0, None, {"earnings_days": 1, "earnings_date": "2026-10-20",
                                                        "earnings_time": "amc"})
        self.assertEqual(msg, "AAPL reports earnings tomorrow after the close")
        msg = message_for(self.rule, "AAPL", 4.0, None, {"earnings_days": 4, "earnings_date": "2026-10-23"})
        self.assertEqual(msg, "AAPL reports earnings in 4 days (2026-10-23)")

    def test_validation_is_pro_with_pro_wording(self):
        from api.alert_rules import RuleForbidden, validate

        with self.assertRaises(RuleForbidden) as e:
            validate({"can_earnings": False}, rule_type="EARNINGS_SOON", threshold=1, value=None, ticker="AAPL",
                     watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        self.assertIn("part of Pro", str(e.exception))
        out = validate({"can_earnings": True}, rule_type="EARNINGS_SOON", threshold=2, value=None, ticker="AAPL",
                       watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        self.assertEqual(out["threshold"], 2.0)


@unittest.skipUnless(HAS_FASTAPI, "needs fastapi")
class EarningsEvaluatorTests(unittest.TestCase):
    def setUp(self):
        from api import alert_rules
        from db import alert_rules as store

        self.ar, self.store = alert_rules, store
        self.db = Store()
        self.addCleanup(self.db.close)
        self.db.user("a@x.com", "pro")
        self.db.user("b@x.com", "basic")
        mock.patch("api.alert_rules._send_email", return_value=(True, None)).start()
        mock.patch("api.watchlist_intel.observation_stale", return_value=False).start()
        self.addCleanup(mock.patch.stopall)

    def rule(self, user, **kw):
        return self.store.create_rule(user, rule_type="EARNINGS_SOON", operator="within_days", threshold=1.0,
                                      value=None, delivery_channels=["in_app"], cooldown_seconds=3600, **kw)

    def test_watchlist_names_outside_the_scan_get_a_heads_up(self):
        self.db.watchlist(1, "a@x.com", ["AAPL", "MXL"])
        self.rule("a@x.com", watchlist_id=1, ticker=None)
        self.rule("b@x.com", ticker="AAPL", watchlist_id=None)  # Starter: not on the plan, skipped
        cal = {"AAPL": {"earnings_days": 1, "earnings_date": "2026-10-09", "earnings_time": "bmo"}}
        with mock.patch("api.alert_rules.earnings_lookup", return_value=cal) as look:
            r = self.ar.evaluate_once(NOW, observation=observation(1, {"MXL": opp(70, 1)}))
            self.assertEqual(r["triggered"], 1)
            self.assertEqual(sorted(look.call_args[0][0]), ["AAPL", "MXL"])
            again = self.ar.evaluate_once(NOW + dt.timedelta(hours=2), observation=observation(2, {"MXL": opp(71, 1)}))
            self.assertEqual(again["triggered"], 0)  # same report: once
        evs = self.db.events()
        self.assertEqual([(e["user_id"], e["ticker"]) for e in evs], [("a@x.com", "AAPL")])
        self.assertEqual(evs[0]["message"], "AAPL reports earnings tomorrow before the open")

    def test_earnings_lookup_counts_days_in_new_york(self):
        rows = {"AAPL": (dt.date(2026, 10, 9), "amc"), "OLD": (dt.date(2026, 10, 1), None)}
        with mock.patch("db.earnings.load_earnings_details_map", return_value=rows):
            out = self.ar.earnings_lookup(["AAPL", "OLD", "NONE"], dt.datetime(2026, 10, 9, 2, 0, tzinfo=UTC))
        # 02:00 UTC is still Oct 8 in New York
        self.assertEqual(out, {"AAPL": {"earnings_days": 1, "earnings_date": "2026-10-09", "earnings_time": "amc"}})
        with mock.patch("db.earnings.load_earnings_details_map", side_effect=RuntimeError("db down")):
            self.assertEqual(self.ar.earnings_lookup(["AAPL"], NOW), {})


# ---- score movers -------------------------------------------------------------------------------
@unittest.skipUnless(HAS_FASTAPI, "needs fastapi")
class ScoreMoverTests(unittest.TestCase):
    def obs(self):
        o = observation(2, {"UP": opp(90, 1), "DOWN": opp(50, 30), "FLAT": opp(70, 2), "NEW": opp(88, 3),
                            "DEEP": opp(95, 150)},
                        prev={"UP": {"score": 70, "rank": 9}, "DOWN": {"score": 80, "rank": 4},
                              "FLAT": {"score": 70, "rank": 2}, "DEEP": {"score": 60, "rank": 140},
                              "FELL": {"score": 90, "rank": 20}})
        o["previous_run_id"] = 1
        o["previous_scan_at"] = NOW - dt.timedelta(hours=3)
        return o

    def test_rising_and_falling_within_the_plan_cap(self):
        from api.scans import score_movers

        with mock.patch("api.watchlist_intel.market_observation", return_value=self.obs()):
            basic = score_movers("basic", limit=5)
            premium = score_movers("premium", limit=5)
        self.assertEqual([r["ticker"] for r in basic["rising"]], ["UP"])  # DEEP is below Starter's cap
        self.assertEqual(basic["rising"][0]["change"], 20)
        self.assertEqual([r["ticker"] for r in basic["falling"]], ["DOWN"])
        self.assertEqual([r["ticker"] for r in premium["rising"]], ["DEEP", "UP"])
        self.assertIsNotNone(basic["previous_scan_at"])

    def test_no_previous_scan_or_an_error_reads_empty(self):
        from api.scans import score_movers

        first = observation(1, {"UP": opp(90, 1)})
        with mock.patch("api.watchlist_intel.market_observation", return_value=first):
            self.assertEqual(score_movers("pro")["rising"], [])
        with mock.patch("api.watchlist_intel.market_observation", side_effect=RuntimeError):
            self.assertEqual(score_movers("pro")["falling"], [])


# ---- watchlist performance ----------------------------------------------------------------------
class WatchlistPerformanceTests(unittest.TestCase):
    SERIES = {
        "AAA": [("2026-10-01", 10.0), ("2026-10-02", 11.0), ("2026-10-05", 12.0), ("2026-10-06", 15.0)],
        "BBB": [("2026-10-01", 20.0), ("2026-10-02", 19.0), ("2026-10-05", 18.0), ("2026-10-06", 16.0)],
        "SPY": [("2026-10-01", 500.0), ("2026-10-02", 505.0), ("2026-10-05", 510.0), ("2026-10-06", 520.0)],
    }

    def run_perf(self, items):
        from api import watchlist_perf

        with mock.patch("api.watchlist_perf._closes", return_value=self.SERIES):
            return watchlist_perf.performance({"id": 7, "name": "Tech", "items": items})

    def test_base_is_the_last_close_when_added_and_spy_is_the_benchmark(self):
        out = self.run_perf([
            # 10:00 ET Oct 5 (before the close): base = Oct 2 close 11
            {"ticker": "AAA", "added_at": dt.datetime(2026, 10, 5, 14, 0, tzinfo=UTC), "price_when_added": None},
            # 21:00 UTC Oct 2 = 5 PM ET, after the close: base = Oct 2 close 19
            {"ticker": "BBB", "added_at": "2026-10-02T21:00:00+00:00", "price_when_added": None},
            {"ticker": "ZZZ", "added_at": dt.datetime(2026, 10, 5, tzinfo=UTC), "price_when_added": None},
        ])
        rows = {r["ticker"]: r for r in out["items"]}
        self.assertEqual((rows["AAA"]["base_price"], rows["AAA"]["base_date"]), (11.0, "2026-10-02"))
        self.assertEqual(rows["AAA"]["change_pct"], 36.36)
        self.assertEqual(rows["AAA"]["benchmark_change_pct"], 2.97)
        self.assertEqual(rows["AAA"]["vs_benchmark"], 33.39)
        self.assertEqual((rows["BBB"]["base_price"], rows["BBB"]["change_pct"]), (19.0, -15.79))
        self.assertIsNone(rows["ZZZ"]["change_pct"])  # no bars: no number, not an estimate
        self.assertEqual([r["ticker"] for r in out["items"]], ["AAA", "BBB", "ZZZ"])
        s = out["summary"]
        self.assertEqual((s["count"], s["up"], s["down"], s["beat_benchmark"]), (2, 1, 1, 1))
        self.assertEqual((s["best"]["ticker"], s["worst"]["ticker"]), ("AAA", "BBB"))
        self.assertEqual(out["as_of"], "2026-10-06")

    def test_saved_price_wins_and_old_adds_use_the_first_bar(self):
        out = self.run_perf([
            {"ticker": "AAA", "added_at": "2026-10-05T14:00:00Z", "price_when_added": 12.5},
            {"ticker": "BBB", "added_at": "2025-01-01T14:00:00Z", "price_when_added": None},
        ])
        rows = {r["ticker"]: r for r in out["items"]}
        self.assertEqual((rows["AAA"]["base_source"], rows["AAA"]["change_pct"]), ("saved", 20.0))
        self.assertTrue(rows["BBB"]["since_first_bar"])
        self.assertEqual(rows["BBB"]["base_price"], 20.0)

    def test_empty_list(self):
        self.assertEqual(self.run_perf([])["items"], [])


# ---- journal insights ---------------------------------------------------------------------------
class JournalInsightTests(unittest.TestCase):
    def trades(self):
        t0 = dt.datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
        mk = lambda tk, e, x, sh, days, score, src="api": {  # noqa: E731
            "ticker": tk, "entry_price": e, "exit_price": x, "shares": sh, "source": src, "hsf_score": score,
            "entered_at": t0, "closed_at": t0 + dt.timedelta(days=days)}
        return [mk("A", 10, 12, 10, 0.2, 85), mk("B", 20, 18, 5, 3, 65), mk("C", 50, 55, 2, 10, None, "paper"),
                mk("D", 10, 9, 10, 30, 82)]

    def test_summary_curve_and_groups(self):
        from api import trading

        with mock.patch("db.trades.closed_trades", return_value=self.trades()):
            out = trading.insights("pro@example.com")
        s = out["summary"]
        self.assertEqual((out["closed"], s["wins"], s["losses"], s["win_rate"]), (4, 2, 2, 50.0))
        self.assertEqual(s["total_pnl"], 10.0)  # +20 -10 +10 -10
        self.assertEqual(s["profit_factor"], 1.5)
        self.assertEqual((s["best"]["ticker"], s["worst"]["ticker"]), ("A", "B"))
        self.assertEqual([p["cumulative_pnl"] for p in out["curve"]], [20.0, 10.0, 20.0, 10.0])
        score = {g["label"]: g for g in out["by_score"]}
        self.assertEqual((score["80+"]["count"], score["80+"]["wins"]), (2, 1))
        self.assertEqual(score["Not recorded"]["count"], 1)
        self.assertEqual([g["label"] for g in out["by_hold"]], ["Same day", "1-5 days", "6-20 days", "Over 20 days"])
        self.assertEqual({g["label"] for g in out["by_source"]}, {"Logged by hand", "Paper trading"})

    def test_no_closed_trades(self):
        from api import trading

        with mock.patch("db.trades.closed_trades", return_value=[]):
            out = trading.insights("x")
        self.assertEqual((out["closed"], out["summary"], out["curve"]), (0, None, []))


# ---- referrals ----------------------------------------------------------------------------------
class ReferralStoreTests(unittest.TestCase):
    def setUp(self):
        from api import referrals as svc
        from db import referrals

        self.r = referrals
        self.svc = svc
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(self._close)

    def _close(self):
        self.r.set_connection_factory(None)
        self.svc.clear_cache()
        self.conn.close()

    def test_code_signup_credit_and_trial_for_both_once(self):
        code = self.r.get_or_create_code("Inviter@x.com")
        self.assertEqual(self.r.get_or_create_code("inviter@x.com"), code)
        self.assertEqual(len(code), self.r.CODE_LEN)
        self.assertFalse(self.r.record_signup("nope", "new@x.com"))
        self.assertFalse(self.r.record_signup(code, "inviter@x.com"))  # self-invite
        self.assertTrue(self.r.record_signup(code.upper(), "new@x.com"))
        self.assertFalse(self.r.record_signup(code, "new@x.com"))  # once
        self.assertEqual(self.r.stats("inviter@x.com"), {"signed_up": 1, "verified": 0})
        stored = self.conn.execute("SELECT referred FROM hsf_referrals").fetchone()[0]
        self.assertNotIn("new@x.com", stored)  # only a hash of the invited email
        self.assertIsNone(self.r.grant_until("new@x.com"))
        self.assertEqual(self.r.reward_verified("new@x.com", days=30, cap_days=180), "inviter@x.com")
        self.assertIsNone(self.r.reward_verified("new@x.com", days=30, cap_days=180))  # never twice
        for who in ("new@x.com", "inviter@x.com"):
            until = self.r.grant_until(who)
            self.assertGreater(until, dt.datetime.now(UTC) + dt.timedelta(days=29))
        self.assertEqual(self.r.stats("inviter@x.com"), {"signed_up": 1, "verified": 1})

    def test_trials_stack_up_to_the_cap(self):
        code = self.r.get_or_create_code("inviter@x.com")
        for i in range(8):
            self.r.record_signup(code, f"f{i}@x.com")
            self.r.reward_verified(f"f{i}@x.com", days=30, cap_days=100)
        until = self.r.grant_until("inviter@x.com")
        self.assertLess(until, dt.datetime.now(UTC) + dt.timedelta(days=100, minutes=1))
        self.assertGreater(until, dt.datetime.now(UTC) + dt.timedelta(days=99))

    def test_effective_tier_only_lifts_starter(self):
        code = self.r.get_or_create_code("inviter@x.com")
        self.r.record_signup(code, "new@x.com")
        self.r.reward_verified("new@x.com", days=30, cap_days=180)
        self.assertEqual(self.svc.effective_tier("new@x.com", "basic")[0], "pro")
        self.assertEqual(self.svc.effective_tier("inviter@x.com", "premium"), ("premium", None))
        self.assertEqual(self.svc.effective_tier("other@x.com", "basic"), ("basic", None))
        with mock.patch.dict("os.environ", {"HSF_REFERRALS_ENABLED": "0"}):
            self.svc.clear_cache()
            self.assertEqual(self.svc.effective_tier("new@x.com", "basic"), ("basic", None))

    def test_storage_errors_never_lift_or_break(self):
        self.r.set_connection_factory(lambda: None)
        self.svc.clear_cache()
        self.assertEqual(self.svc.effective_tier("x@x.com", "basic"), ("basic", None))
        self.assertFalse(self.svc.credit_signup({"ref": "abc"}, "x@x.com"))
        self.assertFalse(self.svc.reward_on_verify("x@x.com"))


# ---- routes -------------------------------------------------------------------------------------
@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class FeaturePackRouteTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.accounts["free@example.com"] = {"username": "free@example.com", "full_name": "Free",
                                             "password": _hash("right pw"), "tier": "basic", "is_admin": False,
                                             "is_active": True}
        from api import referrals as svc
        from db import referrals

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        referrals.set_connection_factory(lambda: self.conn)
        referrals._ensure_schema.__wrapped__(self.conn, True)
        svc.clear_cache()
        self.addCleanup(lambda: (referrals.set_connection_factory(None), svc.clear_cache(), self.conn.close()))

    def token(self, email):
        return self.auth(self.login(email).json()["access_token"])

    def test_movers_route(self):
        with mock.patch("api.scans.score_movers", return_value={"scan_at": None, "previous_scan_at": None,
                                                                "max_results": 25, "rising": [], "falling": []}) as m:
            r = self.client.get("/v1/scans/latest/movers?limit=3", headers=self.token("pro@example.com"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(m.call_args.kwargs["limit"], 3)

    def test_journal_insights_is_pro(self):
        self.assertEqual(self.client.get("/v1/journal/insights", headers=self.token("free@example.com")).status_code, 403)
        with mock.patch("db.trades.closed_trades", return_value=[]):
            r = self.client.get("/v1/journal/insights", headers=self.token("pro@example.com"))
        self.assertEqual((r.status_code, r.json()["closed"]), (200, 0))

    def test_invite_link_then_trial_shows_as_pro(self):
        r = self.client.get("/v1/me/referrals", headers=self.token("pro@example.com"))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["path"], f"/signup?ref={body['code']}")
        self.assertEqual((body["signed_up"], body["reward_days"]), (0, 30))
        from db import referrals

        referrals.record_signup(body["code"], "free@example.com")
        referrals.reward_verified("free@example.com", days=30, cap_days=180)
        from api import referrals as svc

        svc.clear_cache()
        me = self.client.get("/v1/me", headers=self.token("free@example.com")).json()
        self.assertEqual(me["plan"], "pro")
        self.assertTrue(me["entitlements"]["can_earnings"])
        self.assertIsNotNone(me["pro_trial_until"])
        pro = self.client.get("/v1/me", headers=self.token("pro@example.com")).json()
        self.assertIsNone(pro["pro_trial_until"])  # a paying account is never "on trial"

    def test_verify_email_rewards_the_invite(self):
        from db import referrals

        code = referrals.get_or_create_code("pro@example.com")
        referrals.record_signup(code, "free@example.com")
        with mock.patch("api.account.verify_email", return_value="free@example.com"):
            r = self.client.post("/v1/auth/verify-email", json={"token": "t" * 20})
        self.assertEqual(r.status_code, 200)
        self.assertIn("30 days of Pro", r.json()["message"])
        self.assertIsNotNone(referrals.grant_until("pro@example.com"))


if __name__ == "__main__":
    unittest.main()
