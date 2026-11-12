"""When HSF picked each stock up: the first scan of its current unbroken run of
full-market scans, with its price and rank then (Scanner, stock page, Ask HSF)."""
import datetime as dt
import unittest
from unittest import mock

UTC = dt.timezone.utc


def run(i, hour, minute=35, day=12):
    return {"id": i, "created_at": dt.datetime(2026, 11, day, hour, minute, tzinfo=UTC), "row_count": 100}


class PickupTests(unittest.TestCase):
    def setUp(self):
        from api import scans

        self.scans = scans
        # newest first: 19:35, 18:35, 16:35, 14:35 UTC (= 2:35 PM .. 9:35 AM EST)
        self.runs = [run(4, 19), run(3, 18), run(2, 16), run(1, 14)]

    def test_first_scan_of_the_current_run(self):
        window = {
            4: {"AAA": {"rank": 3, "last": 12.0}, "BBB": {"rank": 1, "last": 50.0}, "CCC": {"rank": 9, "last": 5.0}},
            3: {"AAA": {"rank": 7, "last": 11.0}, "BBB": {"rank": 2, "last": 49.0}},
            2: {"AAA": {"rank": 20, "last": 10.5}, "CCC": {"rank": 4, "last": 6.0}},
            1: {"BBB": {"rank": 5, "last": 48.0}},
        }
        got = self.scans._pickups(self.runs, window)
        self.assertEqual(got["AAA"], {"at": "2026-11-12T16:35:00+00:00", "price": 10.5, "rank": 20, "open_ended": False})
        self.assertEqual(got["BBB"]["at"], "2026-11-12T18:35:00+00:00")  # dropped out at 16:35, back since 18:35
        self.assertEqual(got["CCC"]["at"], "2026-11-12T19:35:00+00:00")  # new this scan
        self.assertEqual(set(got), {"AAA", "BBB", "CCC"})

    def test_in_every_scan_is_open_ended(self):
        window = {i: {"AAA": {"rank": 1, "last": 1.0}} for i in (1, 2, 3, 4)}
        got = self.scans._pickups(self.runs, window)["AAA"]
        self.assertEqual((got["at"], got["open_ended"]), ("2026-11-12T14:35:00+00:00", True))

    def test_unreadable_run_is_skipped_not_a_gap(self):
        window = {4: {"AAA": {"rank": 1, "last": 1.0}}, 2: {"AAA": {"rank": 2, "last": 2.0}},
                  1: {"ZZZ": {"rank": 1, "last": 1.0}}}
        self.assertEqual(self.scans._pickups(self.runs, window)["AAA"]["at"], "2026-11-12T16:35:00+00:00")
        self.assertEqual(self.scans._pickups(self.runs, {}), {})

    def test_pickups_is_empty_when_history_fails(self):
        with mock.patch.object(self.scans, "market_runs", return_value=self.runs), \
             mock.patch.object(self.scans, "_score_window", side_effect=RuntimeError("db down")):
            self.assertEqual(self.scans.pickups(), {})
            self.assertIsNone(self.scans.pickup_for("AAA"))

    def test_latest_scan_rows_carry_their_pickup(self):
        opps = [{"ticker": "AAA", "score": 90, "signals": []}, {"ticker": "BBB", "score": 80, "signals": []}]
        got = {"AAA": {"at": "2026-11-12T14:35:00+00:00", "price": 10.0, "rank": 2, "open_ended": False}}
        with mock.patch.object(self.scans, "market_runs", return_value=[{"id": 1, "created_at": None}]), \
             mock.patch.object(self.scans, "run_opportunities", return_value=opps), \
             mock.patch.object(self.scans, "pickups", return_value=got), \
             mock.patch.object(self.scans, "scan_freshness", return_value={"stale": False}), \
             mock.patch.object(self.scans, "run_freshness", return_value={}), \
             mock.patch.object(self.scans, "_iso", return_value=None):
            r = self.scans.latest_scan({}, "premium", limit=50, offset=0)
        self.assertEqual([s["pickup"] for s in r["setups"]], [got["AAA"], None])

    def test_labels_in_eastern_time(self):
        self.assertEqual(self.scans.et_scan_label("2026-11-12T14:35:00+00:00"), "9:35 AM ET Thu Nov 12")
        self.assertEqual(self.scans.et_scan_label("2026-10-08T18:05:00Z", with_day=False), "2:05 PM ET")
        self.assertIsNone(self.scans.et_scan_label("nope"))
        p = {"at": "2026-11-12T14:35:00+00:00", "price": 142.1, "rank": 30, "open_ended": False}
        self.assertEqual(self.scans.pickup_text(p),
                         "first picked up by HSF in the 9:35 AM ET Thu Nov 12 scan at $142.10 (rank #30 then)")
        self.assertEqual(self.scans.pickup_text({**p, "open_ended": True, "price": None, "rank": None}),
                         "in every HSF scan since at least the 9:35 AM ET Thu Nov 12 scan")
        self.assertIsNone(self.scans.pickup_text(None))

    def test_ask_hsf_fact_line_names_the_pickup(self):
        from api.ai_premium import view_line

        v = {"ticker": "AMD", "present": True, "ranked": True, "hsf_score": 49, "rank": 12, "price": 150.0,
             "pickup": {"at": "2026-11-12T14:35:00+00:00", "price": 142.1, "rank": 30, "open_ended": False}}
        line = view_line(v)
        self.assertIn("first picked up by HSF in the 9:35 AM ET Thu Nov 12 scan at $142.10", line)
        self.assertNotIn("picked up", view_line({**v, "pickup": None}))


if __name__ == "__main__":
    unittest.main()
