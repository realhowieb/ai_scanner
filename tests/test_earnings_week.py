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


LOGOS = {"AAPL": {"kind": "logo", "light": False}, "AMD": {"kind": "logo", "light": True}, "PLTR": {"kind": "none"}}


def _base():
    return ew.build_base(SUN, calendar=_cal(), sizes=lambda ts: {t: SIZES[t] for t in ts if t in SIZES},
                         scores=lambda: {"at": "2026-10-30T19:49:00Z", "ranks": {"AAPL": (3, 61), "AMD": (150, 55),
                                                                                  "SHOP": (8, 12)}},
                         logos=lambda ts: {t: LOGOS[t] for t in ts if t in LOGOS})


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
        email = ew.for_plan(base, per_day=2, all_scores=True)
        self.assertEqual(email["days"][1]["names"][0]["score"], 55)
        self.assertIsNone(email["days"][1]["names"][1]["score"])        # SHOP 12: below 40 reads as noise, hidden

    def test_logo_kinds(self):
        out = ew.for_plan(_base(), per_day=3)
        kinds = {n["ticker"]: n["logo"] for d in out["days"] for n in d["names"]}
        self.assertEqual((kinds["AAPL"], kinds["AMD"], kinds["PLTR"], kinds["O"]), ("standard", "light", "none", None))

    def test_snapshot_size(self):
        s = ew.size_from_snapshot({"latestTrade": {"p": 11}, "dailyBar": {"c": 11, "v": 100},
                                   "prevDailyBar": {"c": 10, "v": 1000}})
        self.assertEqual((s["last"], round(s["chg_pct"], 2), s["dollar_volume"]), (11, 10.0, 10000))
        self.assertEqual(ew.size_from_snapshot({}), {"last": None, "chg_pct": None, "dollar_volume": None})


class EmailTests(unittest.TestCase):
    def test_compose_has_day_boxes_tiles_and_text(self):
        week = ew.for_plan(_base(), per_day=12, watched={"PLTR"}, all_scores=True)
        subject, text, html, glance = we.compose(week, base_url="https://app.example", watched=["PLTR"])
        self.assertEqual(subject, "Earnings this week: 7 companies report (week of Nov 2)")
        self.assertIn("Monday, Nov 2", html)
        self.assertIn("https://financialmodelingprep.com/image-stock/AAPL.png", html)
        self.assertNotIn("image-stock/PLTR.png", html)               # no logo -> letter tile, never a broken image
        self.assertIn("background:#1e293b;border:1px solid #e2e8f0'><img src='https://financialmodelingprep.com/image-stock/AMD.png", html)
        self.assertIn("https://app.example/stocks/AMD", html)
        self.assertIn("+ 1 more", html)                               # PENNY is counted, not shown
        self.assertIn("No companies we track report this day.", html)
        self.assertIn("eview=calendar", html)
        self.assertIn("Monday Nov 2: AAPL (after close), *PLTR (after close), O (after close) + 1 more", text)
        self.assertIn("Busiest day: Monday", glance)
        self.assertIn("PLTR</b> Mon", glance)

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
            mock.patch("api.logo_check.logo_info", return_value={"kind": "none"}),
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


def _png(width, height, pixel, ctype=6):
    """A tiny 8-bit PNG whose pixel(x, y) returns RGBA (ctype 6) or RGB (ctype 2)."""
    import struct
    import zlib

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    rows = b""
    for y in range(height):
        line = b"".join(bytes(pixel(x, y)[:4 if ctype == 6 else 3]) for x in range(width))
        rows += b"\x00" + line
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class LogoCheckTests(unittest.TestCase):
    def test_classify(self):
        from api.logo_check import classify

        ring = lambda x, y: (255, 255, 255, 255) if 4 <= x < 12 else (0, 0, 0, 0)          # white on transparent
        dark = lambda x, y: (20, 20, 20, 255) if 4 <= x < 12 else (0, 0, 0, 0)
        boxed = lambda x, y: (255, 255, 255) if x < 12 else (200, 0, 0)                      # opaque, two colors
        photo = lambda x, y: ((x * 37) % 256, (y * 53) % 256, (x * y * 11) % 256)            # opaque, many colors
        self.assertEqual(classify(_png(16, 16, ring)), {"kind": "logo", "light": True})
        self.assertEqual(classify(_png(16, 16, dark)), {"kind": "logo", "light": False})
        self.assertEqual(classify(_png(16, 16, boxed, ctype=2)), {"kind": "logo", "light": False})
        self.assertEqual(classify(_png(64, 64, photo, ctype=2)), {"kind": "none"})
        # A detailed logo on a plain background (many colors, but most pixels share one): still a logo.
        busy = lambda x, y: photo(x, y) if 24 <= x < 40 and 24 <= y < 40 else (255, 255, 255)
        self.assertEqual(classify(_png(64, 64, busy, ctype=2)), {"kind": "logo", "light": False})
        self.assertEqual(classify(b"\xff\xd8\xff\xe0 jpeg"), {"kind": "none"})
        self.assertEqual(classify(b""), {"kind": "none"})

    def test_paeth_filtered_rows_decode(self):
        import struct
        import zlib

        from api.logo_check import decode_png

        # Two 2x1 RGBA rows; row 2 uses the Up filter (2) with zero deltas = same pixels as row 1.
        raw = b"\x00" + bytes([10, 20, 30, 255, 40, 50, 60, 0]) + b"\x02" + bytes(8)

        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        self.assertEqual(decode_png(png)["pixels"], [(10, 20, 30, 255), (40, 50, 60, 0)] * 2)

    def test_logo_info_cached_and_errors_read_as_none(self):
        from api import logo_check

        with mock.patch("api.shared_cache.get", return_value=None), mock.patch("api.shared_cache.put"):
            self.assertEqual(logo_check.logo_info("ZZZZ", fetch=lambda u: None), {"kind": "none"})
            self.assertEqual(logo_check.logo_info("BOOM", fetch=lambda u: 1 / 0), {"kind": "none"})
            got = logo_check.logo_infos(["AAPL", "AAPL"], fetch=lambda u: _png(4, 4, lambda x, y: (0, 0, 0, 255 if x else 0)))
        self.assertEqual(got, {"AAPL": {"kind": "logo", "light": False}})


if __name__ == "__main__":
    unittest.main()
