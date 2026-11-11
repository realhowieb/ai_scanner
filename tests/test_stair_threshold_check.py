"""P2-35 step 1: Stair-stepper threshold replay (analytics/stair_step_threshold_check.py)."""
import datetime as dt
import json
import math
import unittest

from analytics import stair_step_threshold_check as tc

UTC = dt.timezone.utc
OPEN = dt.datetime(2026, 10, 5, 13, 30, tzinfo=UTC)
CLOSE = dt.datetime(2026, 10, 5, 20, 0, tzinfo=UTC)


def _bars(price_at, minutes=390, every=1):
    return [{"t": (OPEN + dt.timedelta(minutes=m)).isoformat().replace("+00:00", "Z"), "c": price_at(m)}
            for m in range(0, minutes, every)]


SMOOTH = _bars(lambda m: 100 * (1 + 0.0002 * m))                       # +1.2%/hr, straight
CHOPPY = _bars(lambda m: 100 + 2 * math.sin(m / 3))                     # no trend
LATE = _bars(lambda m: 100.0 + (m % 2) * 0.3 if m < 200 else 100 * (1 + 0.0003 * (m - 200)))  # trend after 16:50 UTC
THIN = _bars(lambda m: 50 + 0.01 * m, every=7)                          # a trade every 7 minutes


class ThresholdCheckTests(unittest.TestCase):
    def setUp(self):
        self.report = tc.build_report({"SMOO": SMOOTH, "CHOP": CHOPPY, "LATE": LATE, "THIN": THIN},
                                      (OPEN, CLOSE), window=45)

    def default_cell(self, direction="up"):
        return next(c for c in self.report["grids"][direction] if c["is_default"])

    def test_check_times(self):
        times = tc.check_times(OPEN, CLOSE, 45, 5)
        self.assertEqual(times[0], OPEN + dt.timedelta(minutes=45))
        self.assertEqual(times[-1], CLOSE)
        self.assertEqual(self.report["check_times"], len(times))

    def test_default_cell_counts_the_smooth_trend_and_holds(self):
        cell = self.default_cell()
        self.assertEqual(cell["top_symbols"], ["LATE", "SMOO"])
        self.assertGreaterEqual(cell["median_hold_min"], 100)
        self.assertEqual(self.default_cell("down")["symbols_hit"], 0)

    def test_no_lookahead(self):
        times = tc.check_times(OPEN, CLOSE, 45, 5)
        series = tc.replay_metrics({"LATE": LATE}, times, 45)["LATE"]
        early = [m for t, m in zip(times, series) if t < OPEN + dt.timedelta(minutes=200)]
        self.assertTrue(early)
        self.assertFalse(any(tc.is_stair_stepper(m) for m in early))

    def test_grid_is_monotonic_in_r2(self):
        cells = [c for c in self.report["grids"]["up"]
                 if c["max_pullback_pct"] == 1.0 and c["min_trend_pct_per_hour"] == 0.5]
        hits = [c["avg_hits_per_check"] for c in sorted(cells, key=lambda c: c["r2_min"])]
        self.assertEqual(hits, sorted(hits, reverse=True))
        self.assertEqual(len(self.report["grids"]["up"]), len(tc.R2_GRID) * len(tc.PULLBACK_GRID) * len(tc.TREND_GRID))

    def test_thin_names_are_reported(self):
        self.assertIn("THIN", self.report["data_quality"]["mostly_thin_symbols"])

    def test_report_is_strict_json_and_renders(self):
        json.dumps(self.report, allow_nan=False)
        md = tc.render_markdown(self.report)
        self.assertIn("★", md)
        self.assertIn("Direction: up", md)

    def test_streaks(self):
        self.assertEqual(tc._streaks([True, True, False, True, False, False, True]), [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
