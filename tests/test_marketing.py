"""Marketing: the weekly-setups email list (double opt-in, Resend sync), the public weekly
recap and the admin Marketing report."""
import datetime as dt
import sqlite3
import unittest
from unittest import mock

from tests.test_api_account import AccountApiTestCase
from tests.test_api_v1 import DEPS

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 11, 2, 15, 0, tzinfo=UTC)


class _Cur:
    def __init__(self, raw):
        self.raw = raw

    @property
    def description(self):
        return self.raw.description

    def execute(self, q, params=()):
        self.raw.execute(q.replace("%s", "?"), tuple(params))
        return self

    def fetchone(self):
        return self.raw.fetchone()

    def fetchall(self):
        return self.raw.fetchall()

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.raw = sqlite3.connect(":memory:", check_same_thread=False)

    def cursor(self):
        return _Cur(self.raw.cursor())

    def commit(self):
        self.raw.commit()

    def close(self):
        pass


class ListStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        p = mock.patch("db.marketing_list.get_neon_conn", return_value=self.db)
        p.start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict("os.environ", {"AI_SCANNER_SCHEMA_ONCE": "0"}).start()
        from db import marketing_list as ml

        self.ml = ml

    def test_normalize_email(self):
        self.assertEqual(self.ml.normalize_email("  Ann@Example.COM "), "ann@example.com")
        for bad in ("", "ann", "ann@x", "a b@x.com", "ann@@x.com", "a..b@x.com", "<a>@x.com"):
            self.assertIsNone(self.ml.normalize_email(bad), bad)

    def test_double_opt_in_lifecycle(self):
        ml = self.ml
        first = ml.subscribe("ann@example.com", {"source": "reddit", "utm_campaign": "w44"}, "landing", now=T0)
        self.assertEqual((first["status"], first["send_confirm"]), ("pending", True))
        again = ml.subscribe("ann@example.com", now=T0 + dt.timedelta(minutes=2))
        self.assertFalse(again["send_confirm"])            # no second email inside 10 minutes
        self.assertEqual(again["token"], first["token"])
        self.assertTrue(ml.subscribe("ann@example.com", now=T0 + dt.timedelta(minutes=11))["send_confirm"])

        self.assertEqual(ml.lookup(first["token"])["status"], "pending")
        done = ml.confirm(first["token"], now=T0)
        self.assertEqual((done["status"], done["changed"]), ("confirmed", True))
        self.assertFalse(ml.confirm(first["token"])["changed"])
        self.assertFalse(ml.subscribe("ann@example.com", now=T0 + dt.timedelta(days=1))["send_confirm"])

        self.assertEqual(ml.unsubscribe(first["token"])["status"], "unsubscribed")
        self.assertEqual(ml.confirm(first["token"])["status"], "unsubscribed")   # an old confirm link can't re-add
        self.assertTrue(ml.subscribe("ann@example.com", now=T0 + dt.timedelta(days=2))["send_confirm"])

        rows = ml.all_rows()
        self.assertEqual((rows[0]["source"], rows[0]["utm_campaign"], rows[0]["surface"]), ("reddit", "w44", "landing"))
        self.assertIsNone(ml.lookup("not-a-real-token"))
        self.assertIsNone(ml.subscribe("nope"))

    def test_no_database(self):
        with mock.patch("db.marketing_list.get_neon_conn", return_value=None):
            self.assertIsNone(self.ml.subscribe("ann@example.com"))
            self.assertIsNone(self.ml.confirm("tok" * 5))
            self.assertEqual(self.ml.all_rows(), [])


class _Res:
    def __init__(self, code):
        self.status_code = code


class ResendSyncTests(unittest.TestCase):
    def session(self, codes):
        calls = []

        class S:
            def request(self, method, url, json=None, timeout=None, headers=None):
                calls.append((method, url.replace("https://api.resend.com", ""), json, headers["Authorization"]))
                return _Res(codes.pop(0))

        return S(), calls

    def test_key_falls_back_to_the_resend_smtp_password(self):
        from services import marketing_list as s

        with mock.patch.dict("os.environ", {"SMTP_HOST": "smtp.resend.com", "SMTP_PASS": "re_abc"}, clear=True):
            self.assertEqual(s.resend_key(), "re_abc")
            self.assertFalse(s.sync_configured())   # no segment yet
        with mock.patch.dict("os.environ", {"SMTP_HOST": "smtp.other.com", "SMTP_PASS": "re_abc"}, clear=True):
            self.assertIsNone(s.resend_key())

    def test_sync_adds_contact_then_segment(self):
        from services import marketing_list as s

        env = {"RESEND_API_KEY": "re_full", "HSF_RESEND_SEGMENT_ID": "seg-1"}
        with mock.patch.dict("os.environ", env, clear=True):
            sess, calls = self.session([201, 200])
            self.assertTrue(s.sync_confirmed("ann@example.com", session=sess))
            self.assertEqual([(c[0], c[1]) for c in calls],
                             [("POST", "/contacts"), ("POST", "/contacts/ann%40example.com/segments/seg-1")])
            self.assertEqual(calls[0][2], {"email": "ann@example.com", "unsubscribed": False})
            self.assertEqual(calls[0][3], "Bearer re_full")
            sess, calls = self.session([409, 200, 200])   # already a contact: resubscribe it
            self.assertTrue(s.sync_confirmed("ann@example.com", session=sess))
            self.assertEqual(calls[1][:3], ("PATCH", "/contacts/ann%40example.com", {"unsubscribed": False}))
            sess, calls = self.session([401])
            self.assertFalse(s.sync_confirmed("ann@example.com", session=sess))
            sess, calls = self.session([200])
            self.assertTrue(s.sync_unsubscribed("ann@example.com", session=sess))
            self.assertEqual(calls[0][2], {"unsubscribed": True})
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(s.sync_confirmed("ann@example.com"))

    def test_confirm_email_has_the_link_and_no_performance_claims(self):
        from services import marketing_list as s

        with mock.patch("services.lifecycle_emails._base_url", return_value="https://app.hsfinest.com"):
            subject, text, html = s.compose_confirm("tok_123456789")
        self.assertIn("https://app.hsfinest.com/subscribe/confirm?t=tok_123456789", text)
        self.assertIn("Confirm my email", html)
        for claim in ("win rate", "profit", "beat the market", "guarantee"):
            self.assertNotIn(claim, (text + html).lower())


class RecapTests(unittest.TestCase):
    def test_weeks_publish_from_saturday(self):
        from api import public_recap as pr

        self.assertEqual(pr.latest_week(dt.datetime(2026, 11, 6, 21, 0, tzinfo=UTC)), "2026-W44")  # Fri 4pm ET
        self.assertEqual(pr.latest_week(dt.datetime(2026, 11, 7, 6, 0, tzinfo=UTC)), "2026-W45")   # Sat 1am ET
        now = dt.datetime(2026, 11, 4, tzinfo=UTC)
        self.assertTrue(pr.is_published("2026-W44", now))
        self.assertFalse(pr.is_published("2026-W45", now))
        self.assertEqual(pr.week_monday("2026-W45"), dt.date(2026, 11, 2))
        for bad in ("2026-45", "2026-W60", "x"):
            with self.assertRaises(pr.BadWeek):
                pr.week_monday(bad)

    def test_summarize_ranked_uses_each_days_first_run(self):
        from api import public_recap as pr

        mon = dt.datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
        later = mon + dt.timedelta(hours=3)
        tue = mon + dt.timedelta(days=1)
        rows = [("aaa", mon, 80, "Breakout"), ("BBB", mon, 70, "Gap"), ("CCC", later, 99, "Ignored"),
                ("AAA", tue, 85, "Momentum"), ("DDD", tue, 60, None), ("EEE", tue, None, None)]
        out = pr.summarize_ranked(rows, UTC, top=2)
        self.assertEqual(out["scan_days"], 2)
        self.assertEqual([(r["ticker"], r["days"], r["best_score"], r["setup"]) for r in out["ranked"]],
                         [("AAA", 2, 85, "Momentum"), ("BBB", 1, 70, "Gap"), ("DDD", 1, 60, None)])
        self.assertEqual(out["ranked"][0]["first_day"], "2026-11-02")

    def test_recap_is_built_once_then_stored(self):
        from api import public_recap as pr

        db = FakeDB()
        pr._cache.clear()
        payload = {"week": "2026-W44", "start": "2026-10-26", "end": "2026-10-30", "built_at": "x", "v": pr.VERSION,
                   "scan_days": 5, "ranked_names": 1, "ranked": [{"ticker": "AAA"}], "earnings": None}
        now = dt.datetime(2026, 11, 2, tzinfo=UTC)
        with mock.patch("db.engine.get_neon_conn", return_value=db), \
                mock.patch.object(pr, "build", return_value=payload) as build:
            self.assertEqual(pr.recap("2026-W44", now=now)["ranked"], [{"ticker": "AAA"}])
            pr._cache.clear()
            self.assertEqual(pr.recap("2026-W44", now=now)["week"], "2026-W44")
            self.assertEqual(build.call_count, 1)
            self.assertIsNone(pr.recap("2026-W45", now=now))        # not over yet
            self.assertEqual(pr.public_index(now)["weeks"][0]["week"], "2026-W44")

    def test_scores_below_the_ranked_floor_are_left_out(self):
        from api import public_recap as pr

        mon = dt.datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
        out = pr.summarize_ranked([("AAA", mon, 58, "Momentum"), ("ETF", mon, 29, "Gapper")], UTC)
        self.assertEqual([r["ticker"] for r in out["ranked"]], ["AAA"])
        self.assertEqual(pr.summarize_ranked([("ETF", mon, 29, None)], UTC)["scan_days"], 0)

    def test_a_week_stored_under_older_rules_is_rebuilt(self):
        from api import public_recap as pr

        db = FakeDB()
        pr._cache.clear()
        old = {"week": "2026-W44", "start": "2026-10-26", "end": "2026-10-30", "built_at": "a", "ranked": [{"ticker": "OLD"}]}
        new = {**old, "built_at": "b", "v": pr.VERSION, "ranked": [{"ticker": "NEW"}]}
        now = dt.datetime(2026, 11, 2, tzinfo=UTC)
        with mock.patch("db.engine.get_neon_conn", return_value=db):
            pr._store(old)
            with mock.patch.object(pr, "build", return_value=new):
                self.assertEqual(pr.recap("2026-W44", now=now)["ranked"], [{"ticker": "NEW"}])
            self.assertEqual(pr._stored("2026-W44")["ranked"], [{"ticker": "NEW"}])

    def test_empty_week_is_not_stored(self):
        from api import public_recap as pr

        pr._cache.clear()
        with mock.patch("db.engine.get_neon_conn", return_value=FakeDB()), \
                mock.patch.object(pr, "build", return_value={"week": "2026-W40", "ranked": []}):
            self.assertIsNone(pr.recap("2026-W40", now=T0))
            self.assertEqual(pr.archive(), [])

    def test_earnings_teaser_hides_scores(self):
        from api import public_recap as pr

        rows = [{"ticker": t, "time": "bmo", "last": 50.0, "rank": 1, "scan_score": 90} for t in ("A", "B", "C", "D")]
        base = {"start": dt.date(2026, 11, 2), "end": dt.date(2026, 11, 6), "logos": {},
                "days": [{"date": dt.date(2026, 11, 2), "rows": rows}]}
        out = pr.earnings_teaser(base)
        self.assertEqual(out["days"][0]["total"], 4)
        self.assertEqual(out["days"][0]["names"], [{"ticker": t, "time": "bmo"} for t in ("A", "B", "C")])


class MarketingReportTests(unittest.TestCase):
    def test_counts_by_source_and_campaign(self):
        from api.marketing_report import report
        from ui.acquisition import user_hash

        ev = lambda name, src, camp=None, uh=None: {"event": name, "source": src, "utm_source": src,
                                                     "utm_campaign": camp, "user_hash": uh}
        data = {"events": [ev("landing_visit", "x", "w44"), ev("landing_visit", "x", "w44"), ev("landing_visit", "reddit"),
                           ev("primary_cta_click", "x", "w44"), ev("signup_completed", "x", "w44", "h1"),
                           ev("successful_paid_conversion", "direct", None, "h1"), ev("signup_started", "x")],
                "sources": {"h1": "x", user_hash("f@x.com"): "reddit"},
                "founders": [{"user": "f@x.com", "status": "active", "cents": 49900},
                             {"user": "g@x.com", "status": "refunded", "cents": 99900}],
                "errors": {"founding": "boom"}}
        subs = [{"status": "confirmed", "source": "x", "utm_source": "x", "utm_campaign": "w44", "created_at": "2026-11-01"},
                {"status": "pending", "source": "google", "created_at": "2026-11-01"},
                {"status": "unsubscribed", "source": "x", "created_at": "2026-01-01"}]
        out = report(data, subs, since="2026-10-15", days=30, sync_configured=True)
        x = next(r for r in out["by_source"] if r["source"] == "X")
        self.assertEqual((x["visits"], x["clicks"], x["accounts"], x["paid"], x["list_signups"], x["list_confirmed"]),
                         (2, 1, 1, 1, 1, 1))
        self.assertEqual(out["by_source"][0]["source"], "X")
        self.assertEqual(next(r for r in out["by_source"] if r["source"] == "Reddit")["founders"], 1)
        self.assertEqual(out["founder_revenue_cents"], 49900)
        self.assertEqual(out["campaigns"], [{"utm_source": "x", "campaign": "w44", "visits": 2, "clicks": 1,
                                             "list_signups": 1, "accounts": 1}])
        self.assertEqual(out["list"], {"confirmed": 1, "pending": 1, "unsubscribed": 1, "new_in_period": 2,
                                       "sync_configured": True})
        self.assertEqual(out["totals"]["visits"], 3)
        self.assertEqual(out["unavailable"], ["founding"])

    def test_csv_keeps_confirmed_and_defuses_formulas(self):
        from api.marketing_report import subscribers_csv

        rows = [{"email": "a@x.com", "status": "confirmed", "utm_campaign": "=HYPERLINK(1)"},
                {"email": "b@x.com", "status": "pending"}]
        text = subscribers_csv(rows)
        self.assertTrue(text.startswith("email,status"))
        self.assertIn("a@x.com", text)
        self.assertNotIn("b@x.com", text)
        self.assertIn("'=HYPERLINK(1)", text)
        self.assertIn("b@x.com", subscribers_csv(rows, status="all"))


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class MarketingApiTests(AccountApiTestCase):
    def setUp(self):
        super().setUp()
        self.sent = []
        mock.patch("services.marketing_list.send_confirm", side_effect=lambda e, t: self.sent.append((e, t))).start()

    def test_subscribe_sends_one_confirm_and_never_reveals_membership(self):
        with mock.patch("db.marketing_list.subscribe", return_value={"token": "t" * 20, "status": "pending",
                                                                     "send_confirm": True}) as add:
            r = self.client.post("/v1/public/subscribe", json={"email": "Ann@Example.com", "surface": "landing",
                                                               "attribution": {"utm_source": "reddit"}})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(self.sent, [("ann@example.com", "t" * 20)])
            self.assertEqual(add.call_args[0][1]["source"], "reddit")
        with mock.patch("db.marketing_list.subscribe", return_value={"token": "t" * 20, "status": "confirmed",
                                                                     "send_confirm": False}):
            r2 = self.client.post("/v1/public/subscribe", json={"email": "ann@example.com"})
            self.assertEqual(r2.json(), r.json())
            self.assertEqual(len(self.sent), 1)

    def test_bad_email_bot_trap_and_switch(self):
        self.assertEqual(self.client.post("/v1/public/subscribe", json={"email": "nope@"}).status_code, 422)
        with mock.patch("db.marketing_list.subscribe") as add:
            r = self.client.post("/v1/public/subscribe", json={"email": "a@x.com", "website": "spam.example"})
            self.assertEqual(r.status_code, 200)
            add.assert_not_called()
        with mock.patch.dict("os.environ", {"HSF_MARKETING_LIST_ENABLED": "0"}):
            self.assertEqual(self.client.post("/v1/public/subscribe", json={"email": "a@x.com"}).status_code, 503)
            self.assertEqual(self.client.get("/v1/public/list").json(), {"enabled": False})

    def test_confirm_link_masks_the_address_and_syncs(self):
        tok = "k" * 20
        with mock.patch("db.marketing_list.lookup", return_value={"email": "ann@example.com", "status": "pending"}):
            r = self.client.get("/v1/public/subscribe/status", params={"t": tok})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertNotIn("ann@example.com", r.text)
        with mock.patch("db.marketing_list.confirm",
                        return_value={"email": "ann@example.com", "status": "confirmed", "changed": True}), \
                mock.patch("services.marketing_list.sync_confirmed") as sync:
            r = self.client.post("/v1/public/subscribe/action", json={"token": tok, "action": "confirm"})
            self.assertEqual(r.json()["status"], "confirmed")
            sync.assert_called_once_with("ann@example.com")
        with mock.patch("db.marketing_list.unsubscribe", return_value=None):
            self.assertEqual(self.client.post("/v1/public/subscribe/action",
                                              json={"token": tok, "action": "unsubscribe"}).status_code, 400)

    def test_weekly_routes(self):
        with mock.patch("api.public_recap.recap", return_value=None):
            self.assertEqual(self.client.get("/v1/public/weekly/2026-W44").status_code, 404)
        self.assertEqual(self.client.get("/v1/public/weekly/latest").status_code, 422)

    def test_admin_marketing_needs_admin(self):
        self.assertIn(self.client.get("/v1/admin/marketing").status_code, (401, 403))
        self.assertIn(self.client.get("/v1/admin/marketing/subscribers.csv").status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
