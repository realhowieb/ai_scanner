"""Weekly social posts sent to Buffer as drafts (services/social_posts.py, db/social_posts.py)."""
import datetime as dt
import unittest
from unittest import mock

from tests.test_marketing import FakeDB

UTC = dt.timezone.utc


def _base():
    mon = dt.date(2026, 11, 9)
    rows = lambda *t: [{"ticker": x, "time": "amc", "last": 50.0, "dollar_volume": 1e9 - i, "rank": None, "scan_score": None}
                       for i, x in enumerate(t)]
    days = [{"date": mon, "rows": rows("AAPL", "PLTR", "VRTX", "TINY") + [{"ticker": "PENY", "time": None, "last": 1.0,
                                                                         "dollar_volume": 2e9, "rank": None, "scan_score": None}]},
            {"date": mon + dt.timedelta(days=1), "rows": rows(*[f"T{i}" for i in range(120)])},
            {"date": mon + dt.timedelta(days=2), "rows": []}]
    return {"start": mon, "end": mon + dt.timedelta(days=4), "days": days, "logos": {}}


FOUNDING = {"open": True, "tiers": [
    {"tier": "pro", "label": "Founder Pro", "price_cents": 49900, "capacity": 75, "remaining": 70, "on_sale": True, "five_year": True},
    {"tier": "elite", "label": "Founder Elite", "price_cents": 99900, "capacity": 25, "remaining": 25, "on_sale": True, "five_year": True}]}
RECAP = {"week": "2026-W45", "start": "2026-11-02", "end": "2026-11-06",
         "ranked": [{"ticker": "NVDA", "name": "NVIDIA", "days": 4, "best_score": 88, "setup": "breakout"},
                    {"ticker": "AMD", "days": 3, "best_score": 70}, {"ticker": "BEAT", "days": 2, "best_score": 60},
                    {"ticker": "GME", "days": 1, "best_score": 50}]}


class ComposeTests(unittest.TestCase):
    def setUp(self):
        from services import social_posts as sp

        self.sp = sp
        mock.patch("services.social_posts._base_url", return_value="https://app.hsfinest.com").start()
        self.addCleanup(mock.patch.stopall)

    def test_earnings_post(self):
        p = self.sp.compose_earnings("2026-W46", base=_base())
        self.assertIn("Mon: $AAPL $PLTR $VRTX\nTue: $T0 $T1 $T2", p["body"])
        self.assertNotIn("PENY", p["body"])   # sub-$5 names never shown
        self.assertNotIn("Wed:", p["body"])    # empty days left out
        self.assertIn("100+ companies report this week.", p["body"])
        self.assertEqual([d["day"] for d in p["image"]["days"]], ["Mon", "Tue"])
        self.assertEqual(p["image"]["kicker"], "Earnings · Nov 9–13")
        self.assertIsNone(self.sp.compose_earnings("2026-W46", base={**_base(), "days": []}))

    def test_founding_and_recap(self):
        f = self.sp.compose_founding("2026-W45", status=FOUNDING)
        self.assertIn("Founding 100 is open: 100 numbered memberships, one payment, 5 years.", f["body"])
        self.assertIn("Founder Pro $499: 70 of 75 left", f["body"])
        self.assertIsNone(self.sp.compose_founding("2026-W45", status={"open": False}))
        sold = {"open": True, "tiers": [{**t, "remaining": 0} for t in FOUNDING["tiers"]]}
        self.assertIsNone(self.sp.compose_founding("2026-W45", status=sold))
        r = self.sp.compose_recap("2026-W45", recap=RECAP)
        self.assertIn("(Nov 2–6): $NVDA $AMD $BEAT", r["body"])
        self.assertEqual(r["page"], "/weekly/2026-W45")
        self.assertEqual(len(r["image"]["names"]), 3)
        self.assertIsNone(self.sp.compose_recap("2026-W45", recap={"ranked": []}))

    def test_channel_text_and_claims(self):
        p = self.sp.compose_recap("2026-W45", recap=RECAP)
        t = self.sp.channel_text(p, "threads")
        self.assertTrue(t.endswith("The full recap:\nhttps://app.hsfinest.com/weekly/2026-W45?utm_source=threads"
                                   "&utm_medium=social&utm_campaign=recap-w45"))
        self.assertIn("utm_source=x&", self.sp.channel_text(p, "twitter"))
        ig = self.sp.channel_text(p, "instagram")
        self.assertIn("The full recap: link in bio.", ig)
        self.assertNotIn("https://", ig)
        self.sp.check_claims(t)  # $BEAT is a ticker, not a claim
        for bad in ("NVDA beat the market", "Up 12% this week", "Another winner", "Guaranteed setups", "Our returns"):
            with self.assertRaises(ValueError):
                self.sp.check_claims(bad)
        for kind_post in (self.sp.compose_product(f"2026-W{w}") for w in range(40, 44)):
            for s in ("threads", "linkedin", "instagram"):
                self.sp.check_claims(self.sp.channel_text(kind_post, s))
        self.sp.check_claims(self.sp.channel_text(self.sp.compose_earnings("2026-W46", base=_base()), "linkedin"))
        self.sp.check_claims(self.sp.channel_text(self.sp.compose_founding("2026-W45", status=FOUNDING), "threads"))

    def test_schedule(self):
        et = lambda *a: dt.datetime(*a, tzinfo=self.sp._et()).astimezone(UTC)
        self.assertEqual(self.sp.post_time("earnings", "2026-W46"), et(2026, 11, 9, 8, 30))
        self.assertEqual(self.sp.due("earnings", et(2026, 11, 8, 17, 59)), None)
        self.assertEqual(self.sp.due("earnings", et(2026, 11, 8, 18, 0)), "2026-W46")
        self.assertEqual(self.sp.due("earnings", et(2026, 11, 9, 6, 29)), "2026-W46")
        # no automatic draft in the last 2 hours (2026-11-02: one landed after the post was made by hand)
        self.assertEqual(self.sp.due("earnings", et(2026, 11, 9, 6, 30)), None)
        self.assertEqual(self.sp.due("earnings", et(2026, 11, 9, 8, 30)), None)
        self.assertEqual(self.sp.next_week_for("earnings", et(2026, 11, 9, 9, 0)), "2026-W47")
        self.assertEqual(self.sp.due("recap", et(2026, 11, 14, 0, 30)), None)
        self.assertEqual(self.sp.due("recap", et(2026, 11, 14, 1, 0)), "2026-W46")
        self.assertEqual(self.sp.due("founding", et(2026, 11, 12, 20, 0)), "2026-W46")
        self.assertIsNone(self.sp.due("product", et(2026, 11, 12, 20, 0)))


class _Res:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def json(self):
        return self.data


class FakeBuffer:
    def __init__(self, fail_service=None):
        self.created, self.fail_service = [], fail_service

    def post(self, url, json=None, timeout=None, headers=None):
        assert headers["Authorization"] == "Bearer key-1"
        q = json["query"]
        if "organizations" in q:
            return _Res({"data": {"account": {"organizations": [{"id": "org1"}]}}})
        if "channels" in q:
            assert '"org1"' in q
            return _Res({"data": {"channels": [{"id": "c1", "name": "hsf", "service": "threads"},
                                               {"id": "c2", "name": "hsf", "service": "instagram"},
                                               {"id": "c3", "name": "HSF", "service": "linkedin"}]}})
        inp = json["variables"]["input"]
        if self.fail_service and inp["channelId"] == self.fail_service:
            return _Res({"data": {"createPost": {"__typename": "MutationError", "message": "nope"}}})
        self.created.append(inp)
        return _Res({"data": {"createPost": {"__typename": "PostActionSuccess", "post": {"id": f"p{len(self.created)}"}}}})


class SendTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        mock.patch("db.social_posts.get_neon_conn", return_value=self.db).start()
        mock.patch.dict("os.environ", {"AI_SCANNER_SCHEMA_ONCE": "0", "BUFFER_API_KEY": "key-1"}).start()
        mock.patch("services.social_posts._base_url", return_value="https://app.hsfinest.com").start()
        self.addCleanup(mock.patch.stopall)
        from services import social_posts as sp

        self.sp = sp
        self.now = dt.datetime(2026, 11, 13, 1, 0, tzinfo=UTC)
        mock.patch.object(sp, "compose", side_effect=lambda k, w: sp.compose_founding(w, status=FOUNDING)).start()

    def test_drafts_once_per_channel(self):
        buf = FakeBuffer()
        out = self.sp.send("founding", "2026-W46", now=self.now, session=buf)
        self.assertEqual(out["sent"], 3)
        ig = next(c for c in buf.created if c["channelId"] == "c2")
        self.assertTrue(ig["saveToDraft"])
        self.assertEqual(ig["mode"], "customScheduled")
        self.assertEqual(ig["dueAt"], "2026-11-13T17:00:00.000Z")
        self.assertEqual(ig["metadata"], {"instagram": {"type": "post", "shouldShareToFeed": True}})
        self.assertEqual(ig["assets"], [{"image": {"url": "https://app.hsfinest.com/social/founding/2026-W46.png"}}])
        th = next(c for c in buf.created if c["channelId"] == "c1")
        self.assertEqual(th["metadata"]["threads"]["topic"], "Stocks")
        self.assertIn("utm_source=threads", th["text"])
        again = self.sp.send("founding", "2026-W46", now=self.now, session=buf)
        self.assertEqual((again["sent"], again["already"]), (0, 3))
        self.assertEqual(len(buf.created), 3)
        stored = self.sp.post_for("founding", "2026-W46")
        self.assertIn("70 of 75 left", stored["body"])

    def test_failed_channel_retries_then_gives_up(self):
        buf = FakeBuffer(fail_service="c3")
        for _ in range(self.sp.MAX_ATTEMPTS):
            with self.assertRaises(self.sp.BufferError):
                self.sp.send("founding", "2026-W46", now=self.now, session=buf)
        out = self.sp.send("founding", "2026-W46", now=self.now, session=buf)
        self.assertEqual((out["sent"], out["already"], out["gave_up"]), (0, 2, 1))
        from db import social_posts as db

        self.assertIn("nope", db.recent()[0]["error"] or db.recent()[1]["error"] or "")

    def test_nothing_without_key_or_post(self):
        with mock.patch.dict("os.environ", {"BUFFER_API_KEY": ""}):
            self.assertEqual(self.sp.send("founding", "2026-W46", now=self.now)["skipped"], "no_buffer_key")
        with mock.patch.object(self.sp, "compose", return_value=None):
            self.assertEqual(self.sp.send("recap", "2026-W46", now=self.now, session=FakeBuffer())["skipped"], "nothing_to_post")

    def test_refuses_claims(self):
        bad = {**self.sp.compose_founding("2026-W47", status=FOUNDING), "week": "2026-W47", "body": "Our picks beat the market"}
        buf = FakeBuffer()
        with mock.patch.object(self.sp, "compose", return_value=bad):
            out = self.sp.send("founding", "2026-W47", now=self.now, session=buf)
        self.assertEqual(out["skipped"], "claim_check")
        self.assertEqual(buf.created, [])


class WeeklyRotationTests(unittest.TestCase):
    def setUp(self):
        from services import social_posts as sp

        self.sp = sp
        mock.patch("services.social_posts._base_url", return_value="https://app.hsfinest.com").start()
        self.addCleanup(mock.patch.stopall)

    def test_spotlight_and_explainer_rotate_and_pass_claims(self):
        sp = self.sp
        for kind, posts in (("spotlight", sp.SPOTLIGHT_POSTS), ("explainer", sp.EXPLAINER_POSTS)):
            seen = set()
            for w in range(40, 40 + len(posts)):
                p = sp.compose(kind, f"2026-W{w}")
                self.assertEqual(p["kind"], kind)
                seen.add(p["image"]["headline"])
                for s in ("threads", "linkedin", "instagram"):
                    t = sp.channel_text(p, s)
                    sp.check_claims(t)
                    self.assertLessEqual(len(t), 500, (kind, w, s))  # Threads limit
                self.assertTrue(all(len(x) <= 34 for x in p["image"]["points"]), p["image"]["points"])
            self.assertEqual(len(seen), len(posts))
        self.assertEqual(sp.compose_product("2026-W45")["image"]["kicker"], "What HSF does")

    def test_new_kinds_post_times_and_start_weeks(self):
        sp = self.sp
        self.assertEqual(sp.post_time("spotlight", "2026-W48"), dt.datetime(2026, 11, 24, 17, 0, tzinfo=UTC))
        self.assertEqual(sp.post_time("explainer", "2026-W47"), dt.datetime(2026, 11, 19, 17, 0, tzinfo=UTC))
        early = dt.datetime(2026, 11, 9, 23, 30, tzinfo=UTC)  # Mon 6:30pm ET, before the first automatic week
        self.assertEqual(sp.next_week_for("spotlight", early), "2026-W48")
        self.assertIsNone(sp.due("spotlight", early))
        self.assertIsNone(sp.due("explainer", dt.datetime(2026, 11, 11, 23, 30, tzinfo=UTC)))  # W46 was hand-made
        self.assertEqual(sp.due("explainer", dt.datetime(2026, 11, 18, 23, 30, tzinfo=UTC)), "2026-W47")
        self.assertEqual(sp.due("spotlight", dt.datetime(2026, 11, 23, 23, 30, tzinfo=UTC)), "2026-W48")
        self.assertIsNone(sp.due("spotlight", dt.datetime(2026, 11, 24, 16, 0, tzinfo=UTC)))  # inside the 2h cutoff


class OpsJobTests(unittest.TestCase):
    def test_tick_runs_due_kinds_only_with_key(self):
        from api import ops_jobs

        now = dt.datetime(2026, 11, 13, 1, 0, tzinfo=UTC)  # Thu 8pm ET: Friday's Founding 100 post is due
        env = {k: "0" for k in ("HSF_LIVE_CHECK_NOTIFY", "HSF_NIGHTLY_OPS_NOTE", "HSF_WEEKLY_RECAP_ENABLED",
                                "HSF_WEEKLY_EARNINGS_ENABLED", "HSF_BUSINESS_EMAIL_ENABLED", "HSF_FOUNDER_FEEDBACK_NUDGE")}
        with mock.patch.dict("os.environ", {**env, "BUFFER_API_KEY": "k"}), \
                mock.patch("api.ops_jobs._run_once") as run:
            ops_jobs.tick(now=now)
        social = [c.args for c in run.call_args_list if c.args[0].startswith("social_")]
        self.assertEqual({a[0]: a[1] for a in social},
                         {"social_earnings": None, "social_spotlight": None, "social_product": None, "social_explainer": None,
                          "social_founding": "2026-W46", "social_recap": None})
        with mock.patch.dict("os.environ", {**env, "BUFFER_API_KEY": ""}), mock.patch("api.ops_jobs._run_once") as run:
            ops_jobs.tick(now=now)
        self.assertFalse([c for c in run.call_args_list if c.args[0].startswith("social_")])


if __name__ == "__main__":
    unittest.main()
