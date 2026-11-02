"""Earnings week at a glance (api.earnings_week), its route, and the Sunday email."""
import datetime as dt
import unittest
from unittest import mock

from api import earnings_week as ew
from services import weekly_earnings as we

SUN = dt.date(2026, 11, 1)


def _cal():
    rows = []
    for d, names in {dt.date(2026, 11, 2): ["AAPL", "PLTR", "PENNY", "O"], dt.date(2026, 11, 3): ["SHOP", "AMD"],
                     dt.date(2026, 11, 5): ["NET"], dt.date(2026, 11, 9): ["LATE"]}.items():
        for n in names:
            rows.append({"symbol": n, "earnings_date": d, "earnings_time": "amc" if n != "SHOP" else "bmo"})
    return rows


SIZES = {"AAPL": {"last": 270.0, "chg_pct": 1.0, "dollar_volume": 9e9},
         "PLTR": {"last": 180.0, "chg_pct": -2.0, "dollar_volume": 8e9},
         "PENNY": {"last": 2.0, "chg_pct": 0.0, "dollar_volume": 9.9e9},
         "O": {"last": 57.0, "chg_pct": 0.1, "dollar_volume": 1e8},
         "AMD": {"last": 250.0, "chg_pct": 3.0, "dollar_volume": 7e9},
         "SHOP": {"last": 150.0, "chg_pct": 0.5, "dollar_volume": 2e9}}


def _base():
    return ew.build_base(SUN, calendar=_cal(), sizes=lambda ts: {t: SIZES[t] for t in ts if t in SIZES},
                         scores=lambda: {"at": "2026-10-30T19:49:00Z", "ranks": {"AAPL": (3, 61), "AMD": (150, 55)}})


class WeekTests(unittest.TestCase):
    def test_weekend_starts_monday_and_skips_holidays(self):
        self.assertEqual(ew.week_days(SUN), [dt.date(2026, 11, d) for d in (2, 3, 4, 5, 6)])
        # Thanksgiving 2026-11-26 is skipped.
        self.assertNotIn(dt.date(2026, 11, 26), ew.week_days(dt.date(2026, 11, 23)))

    def test_days_sorted_by_dollar_volume_and_pennies_only_counted(self):
        out = ew.for_plan(_base(), per_day=2, cap=100)
        mon = out["days"][0]
        self.assertEqual((out["start"], out["end"]), ("2026-11-02", "2026-11-06"))
        self.assertEqual([n["ticker"] for n in mon["names"]], ["AAPL", "PLTR"])
        self.assertEqual(mon["total"], 4)
        self.assertEqual(out["days"][1]["names"][0]["ticker"], "AMD")
        self.assertEqual([d["total"] for d in out["days"]], [4, 2, 0, 1, 0])   # LATE is outside the week
        self.assertEqual(mon["names"][0]["logo_url"], "https://financialmodelingprep.com/image-stock/AAPL.png")

    def test_scores_follow_plan_and_watchlist_always_shown(self):
        base = _base()
        pro = ew.for_plan(base, per_day=1, cap=100, watched={"O"})
        mon = pro["days"][0]["names"]
        self.assertEqual([(n["ticker"], n["score"], n["on_watchlist"]) for n in mon], [("AAPL", 61, False), ("O", None, True)])
        self.assertIsNone(pro["days"][1]["names"][0]["score"])          # AMD ranked 150, beyond a 100-row cap
        prem = ew.for_plan(base, per_day=1, cap=100, auto={"AMD": {"score": 70}})
        self.assertEqual((prem["days"][1]["names"][0]["score"], prem["days"][1]["names"][0]["score_source"]), (70, "auto"))
        email = ew.for_plan(base, per_day=1, all_scores=True)
        self.assertEqual(email["days"][1]["names"][0]["score"], 55)

    def test_snapshot_size(self):
        s = ew.size_from_snapshot({"latestTrade": {"p": 11}, "dailyBar": {"c": 11, "v": 100},
                                   "prevDailyBar": {"c": 10, "v": 1000}})
        self.assertEqual((s["last"], round(s["chg_pct"], 2), s["dollar_volume"]), (11, 10.0, 10000))
        self.assertEqual(ew.size_from_snapshot({}), {"last": None, "chg_pct": None, "dollar_volume": None})


class EmailTests(unittest.TestCase):
    def test_compose_has_day_boxes_tiles_and_text(self):
        week = ew.for_plan(_base(), per_day=12, watched={"PLTR"}, all_scores=True)
        subject, text, html, glance = we.compose(week, base_url="https://app.example", logos={"AAPL": True},
                                                 watched=["PLTR"])
        self.assertEqual(subject, "Earnings this week: 7 companies report (week of Nov 2)")
        self.assertIn("Monday, Nov 2", html)
        self.assertIn("https://financialmodelingprep.com/image-stock/AAPL.png", html)
        self.assertNotIn("image-stock/PLTR.png", html)               # no logo -> letter tile, never a broken image
        self.assertIn("https://app.example/stocks/AMD", html)
        self.assertIn("+ 1 more", html)                               # PENNY is counted, not shown
        self.assertIn("No companies we track report this day.", html)
        self.assertIn("eview=calendar", html)
        self.assertIn("Monday Nov 2: AAPL (after close), *PLTR (after close), O (after close) + 1 more", text)
        self.assertIn("Busiest day: Monday", glance)
        self.assertIn("PLTR</b> Mon", glance)

    def test_logo_check_cached(self):
        calls = []
        with mock.patch("api.shared_cache.get", return_value=None), mock.patch("api.shared_cache.put"):
            self.assertTrue(we.logo_ok("AAPL", head=lambda u: calls.append(u) or 200))
            self.assertFalse(we.logo_ok("ZZZZ", head=lambda u: 404))
        self.assertEqual(calls, ["https://financialmodelingprep.com/image-stock/AAPL.png"])

    def test_due_sunday_evening_or_monday_catch_up(self):
        from api.ops_jobs import weekly_earnings_due

        utc = dt.timezone.utc
        self.assertIsNone(weekly_earnings_due(dt.datetime(2026, 11, 1, 21, 59, tzinfo=utc)))
        self.assertEqual(weekly_earnings_due(dt.datetime(2026, 11, 1, 22, 0, tzinfo=utc)), "2026-W45")
        self.assertEqual(weekly_earnings_due(dt.datetime(2026, 11, 2, 12, 59, tzinfo=utc)), "2026-W45")
        self.assertIsNone(weekly_earnings_due(dt.datetime(2026, 11, 2, 13, 0, tzinfo=utc)))
        self.assertIsNone(weekly_earnings_due(dt.datetime(2026, 11, 7, 23, 0, tzinfo=utc)))

    def test_run_sends_pro_only_once_per_week(self):
        users = {"pro@x.com": {"is_active": True}, "free@x.com": {"is_active": True}, "off@x.com": {"is_active": True}}
        sent, claimed = [], set()

        def claim(email, kind, ref):
            if (email, ref) in claimed:
                return False
            claimed.add((email, ref))
            return True

        tiers = {"pro@x.com": "pro", "free@x.com": "basic", "off@x.com": "premium"}
        patches = [
            mock.patch("api.earnings_week.build_base", return_value=_base()),
            mock.patch("api.scans.earnings_autoscore", return_value={"rows": {}}),
            mock.patch("db.users.load_users", return_value=users),
            mock.patch("scheduler.morning_digest._email_tier_key", side_effect=lambda e, *a: tiers[e]),
            mock.patch("db.referrals.grant_until", return_value=None),
            mock.patch("db.referrals.claim_email", side_effect=claim),
            mock.patch("db.referrals.release_email"),
            mock.patch("db.email_verification.is_email_verified", return_value=True),
            mock.patch("db.email_prefs.wants_email", side_effect=lambda e, k: e != "off@x.com"),
            mock.patch("db.email_prefs.unsubscribe_url", return_value="https://u"),
            mock.patch("scheduler.morning_digest.default_watchlist_tickers", return_value=["PLTR"]),
            mock.patch("scheduler.morning_digest.record_email_job"),
            mock.patch("services.weekly_earnings.logo_ok", return_value=False),
            mock.patch("ui.email_utils.send_digest_email", side_effect=lambda to, subj, *a, **k: sent.append((to, subj)) or True),
        ]
        for p in patches:
            p.start()
        try:
            out = we.run(now=dt.datetime(2026, 11, 1, 22, 5, tzinfo=dt.timezone.utc))
            again = we.run(now=dt.datetime(2026, 11, 2, 1, 0, tzinfo=dt.timezone.utc))
        finally:
            for p in patches:
                p.stop()
        self.assertEqual([t for t, _ in sent], ["pro@x.com"])
        self.assertEqual(out["skipped"], {"plan_below_pro": 1, "unsubscribed": 1})
        self.assertEqual(again["skipped"].get("already_sent"), 1)


if __name__ == "__main__":
    unittest.main()
