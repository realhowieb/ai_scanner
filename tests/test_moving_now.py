"""Day Trader "Moving now": the tested rule (+3% on 2x volume pace from 9:40 ET, scan
liquidity floor), first-spotted time/price kept for the day, and the Pro endpoint."""
import datetime as dt
import unittest
from unittest import mock

from api import moving_now as mn
from tests.test_api_paid_features import PaidApiTestCase

UTC = dt.timezone.utc
THU = dt.date(2026, 11, 19)
AT_1030 = dt.datetime(2026, 11, 19, 15, 30, tzinfo=UTC)   # 10:30 ET: 60 of 390 minutes gone
AT_0935 = dt.datetime(2026, 11, 19, 14, 35, tzinfo=UTC)
AT_1100 = dt.datetime(2026, 11, 19, 16, 0, tzinfo=UTC)
AT_1700 = dt.datetime(2026, 11, 19, 22, 0, tzinfo=UTC)


def row(t, chg, volume, last=103.0, prev=100.0, gap=1.0):
    return {"ticker": t, "chg_pct": chg, "volume": volume, "last": last, "previous_close": prev, "gap_pct": gap}


# avg 1.3M shares/day, $130M/day: at 10:30 the usual volume so far is 1.3M * 60/390 = 200k
STATS = {"UP": (1_300_000, 130e6), "THIN": (1_300_000, 1e6), "NEW": None}


class RuleTests(unittest.TestCase):
    def test_volume_pace_scales_to_time_of_day(self):
        self.assertAlmostEqual(mn.volume_pace(400_000, 1_300_000, AT_1030), 2.0)
        self.assertIsNone(mn.volume_pace(0, 1_300_000, AT_1030))
        self.assertIsNone(mn.volume_pace(100, None, AT_1030))

    def test_spotting_window(self):
        self.assertTrue(mn.spotting_open(AT_1030, "open"))
        self.assertFalse(mn.spotting_open(AT_0935, "open"))        # before 9:40 ET
        self.assertFalse(mn.spotting_open(AT_1700, "afterhours"))

    def test_candidates_need_the_price_side(self):
        rows = [row("UP", 3.0, 1), row("FLAT", 2.9, 1), row("SPLIT", 50.0, 1, gap=45.0), row("WILD", 150.0, 1),
                row("PENNY", 5.0, 1, last=0.9, prev=0.8), {"ticker": "NOQ", "chg_pct": None}]
        self.assertEqual([r["ticker"] for r in mn.candidates(rows)], ["UP"])

    def test_triggered_needs_volume_pace_and_liquidity(self):
        rows = [row("UP", 4.0, 400_000), row("SLOW", 4.0, 399_000), row("THIN", 4.0, 900_000), row("NEW", 4.0, 900_000)]
        stats = {**STATS, "SLOW": (1_300_000, 130e6)}
        hits = mn.triggered(rows, stats, AT_1030)
        self.assertEqual([h["ticker"] for h in hits], ["UP"])
        self.assertAlmostEqual(hits[0]["volume_pace"], 2.0)


class DailyStatsTests(unittest.TestCase):
    def setUp(self):
        mn._stats_day, mn._stats = None, {}
        self.addCleanup(lambda: setattr(mn, "_stats_day", None))

    def test_each_symbol_fetched_once_per_day(self):
        fetch = mock.Mock(side_effect=lambda syms, day: {s: (1.0, 2.0) for s in syms})
        mn.daily_stats(["A", "B"], THU, fetch=fetch)
        got = mn.daily_stats(["A", "B", "C"], THU, fetch=fetch)
        self.assertEqual(fetch.call_args_list[1].args[0], ["C"])
        self.assertEqual(got["C"], (1.0, 2.0))
        mn.daily_stats(["A"], THU + dt.timedelta(days=1), fetch=fetch)   # a new day starts over
        self.assertEqual(fetch.call_count, 3)

    def test_fetch_stats_averages_the_20_sessions_before_today(self):
        bars = [{"t": f"2026-10-{d:02d}T04:00:00Z", "v": 1000 * d, "c": 10.0} for d in range(1, 31)]
        bars.append({"t": "2026-11-19T05:00:00Z", "v": 10**9, "c": 10.0})    # today's bar is ignored
        with mock.patch("data.price_alpaca.fetch_minute_bars_multi", return_value={"A": bars, "B": bars[:3]}):
            got = mn._fetch_stats(["A", "B"], THU)
        self.assertAlmostEqual(got["A"][0], sum(1000 * d for d in range(11, 31)) / 20)
        self.assertAlmostEqual(got["A"][1], got["A"][0] * 10.0)
        self.assertIsNone(got["B"])                                        # too little history


class SpottedTests(unittest.TestCase):
    def setUp(self):
        self.store = {}
        mock.patch("api.shared_cache.get", side_effect=lambda k: self.store.get(k)).start()
        mock.patch("api.shared_cache.put", side_effect=lambda k, v, ttl: self.store.__setitem__(k, v)).start()
        self.addCleanup(mock.patch.stopall)

    def test_first_sighting_is_kept(self):
        hit = {**row("UP", 4.0, 400_000), "volume_pace": 2.0}
        mn.record(THU, [hit], AT_1030)
        mn.record(THU, [{**hit, "last": 110.0, "volume_pace": 5.0}], AT_1100)
        s = mn.spotted(THU)["UP"]
        self.assertEqual((s["at"], s["price"], s["pace"]), (AT_1030.isoformat(), 103.0, 2.0))

    def test_refresh_spots_and_list_shows_move_since(self):
        mn._stats_day, mn._stats = THU, dict(STATS)
        self.addCleanup(lambda: setattr(mn, "_stats_day", None))
        self.assertTrue(mn.refresh(AT_1030, rows=[row("UP", 3.0, 400_000), row("THIN", 9.0, 9e6)], state="open"))
        self.assertEqual(list(mn.spotted(THU)), ["UP"])
        self.assertFalse(mn.refresh(AT_1700, rows=[row("UP", 3.0, 400_000)], state="afterhours"))

        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        later = [row("UP", 8.15, 2_000_000, last=108.15)]
        with mock.patch("api.market.dt_universe_rows", return_value=later), \
                mock.patch("ui.day_trader.market_state", return_value="open"):
            out = mn.moving_now(now=AT_1100)
        self.assertTrue(out["active"])
        self.assertEqual(out["spotted_count"], 1)
        r = out["rows"][0]
        self.assertEqual((r["ticker"], r["spotted_price"], r["last"]), ("UP", 103.0, 108.15))
        self.assertAlmostEqual(r["since_spotted_pct"], 5.0)
        self.assertGreater(r["volume_pace"], 2.0)                          # live pace, not the pace when spotted


class WithSpottedTests(unittest.TestCase):
    def test_pickup_gets_same_day_sighting_only(self):
        store = {"hsf_day": {}, mn._key(THU): {"UP": {"at": AT_1030.isoformat(), "price": 103.0}}}
        with mock.patch("api.shared_cache.get", side_effect=lambda k: store.get(k)) as g:
            out = mn.with_spotted({"UP": {"at": AT_1100.isoformat(), "price": 105.0},
                                   "OLD": {"at": "2026-11-18T16:00:00+00:00", "price": 9.0},
                                   "NONE": None})
        self.assertEqual((out["UP"]["spotted_at"], out["UP"]["spotted_price"], out["UP"]["price"]),
                         (AT_1030.isoformat(), 103.0, 105.0))
        self.assertNotIn("spotted_at", out["OLD"])
        self.assertIsNone(out["NONE"])
        self.assertEqual(g.call_count, 2)                                  # one read per ET day

    def test_ask_hsf_line_mentions_the_sighting(self):
        from api.scans import pickup_text

        text = pickup_text({"at": AT_1100.isoformat(), "price": 105.0, "rank": 4,
                            "spotted_at": AT_1030.isoformat(), "spotted_price": 103.0})
        self.assertIn("Day Trader's Moving now spotted it at 10:30 AM ET at $103.00", text)


class MovingNowApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)

    def test_pro_gets_list_free_refused(self):
        body = {"active": False, "state": "closed", "as_of": AT_1700, "session_date": "2026-11-19",
                "spotted_count": 1, "rule": mn.RULE_TEXT,
                "rows": [{"ticker": "UP", "spotted_at": AT_1030.isoformat(), "spotted_price": 103.0, "spotted_chg_pct": 3.0,
                          "last": None, "chg_pct": None, "since_spotted_pct": None, "volume_pace": 2.0}]}
        with mock.patch("api.moving_now.moving_now", return_value=body) as m:
            r = self.get("pro@example.com", "/v1/day-trader/moving-now?limit=5")
            free = self.get("free@example.com", "/v1/day-trader/moving-now")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["rows"][0]["ticker"], "UP")
        m.assert_called_once_with(5)
        self.assertEqual(free.status_code, 403)


if __name__ == "__main__":
    unittest.main()
