"""P2-26 — stair-stepper (1-minute Linear Regression R²) check on Day Trader."""
import importlib.util
import math
import random
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from analytics.stair_step import (
    is_stair_stepper,
    latest_session_bars,
    stair_step_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None
START = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)   # Fri 10:00 ET


def bars(prices, start=START, step_min=1):
    return [{"t": (start + timedelta(minutes=i * step_min)).isoformat().replace("+00:00", "Z"),
             "o": p, "h": p, "l": p, "c": p, "v": 1000} for i, p in enumerate(prices)]


def stair(n=45, base=100.0, per_bar=0.02, wiggle=0.01, seed=1):
    rnd = random.Random(seed)
    return [base * (1 + per_bar / 100 * i) + rnd.uniform(-wiggle, wiggle) for i in range(n)]


class MetricsTests(unittest.TestCase):
    def test_straight_climb_scores_near_one(self):
        m = stair_step_metrics(bars(stair()))
        self.assertEqual(m["status"], "ok")
        self.assertGreater(m["r2"], 0.97)
        self.assertEqual(m["direction"], "up")
        self.assertAlmostEqual(m["trend_pct_per_hour"], 1.2, delta=0.1)   # 0.02%/bar × 60
        self.assertLess(m["max_pullback_pct"], 0.05)
        self.assertTrue(is_stair_stepper(m))

    def test_choppy_path_is_rejected(self):
        prices = [100 + 0.6 * math.sin(i / 2.0) for i in range(45)]
        m = stair_step_metrics(bars(prices))
        self.assertEqual(m["status"], "ok")
        self.assertLess(m["r2"], 0.5)
        self.assertFalse(is_stair_stepper(m, direction="either"))

    def test_down_trend_needs_down_or_either(self):
        m = stair_step_metrics(bars(stair(per_bar=-0.03)))
        self.assertEqual(m["direction"], "down")
        self.assertFalse(is_stair_stepper(m, direction="up"))
        self.assertTrue(is_stair_stepper(m, direction="down"))
        self.assertTrue(is_stair_stepper(m, direction="either"))

    def test_deep_pullback_fails_the_pullback_limit(self):
        prices = stair(per_bar=0.05)
        prices[30:34] = [p * 0.985 for p in prices[30:34]]           # 1.5% dip
        m = stair_step_metrics(bars(prices))
        self.assertGreaterEqual(m["max_pullback_pct"], 1.4)
        self.assertFalse(is_stair_stepper(m, r2_min=0.5, max_pullback_pct=1.0))
        self.assertTrue(is_stair_stepper(m, r2_min=0.5, max_pullback_pct=2.0))

    def test_straight_but_flat_line_is_not_a_trend(self):
        m = stair_step_metrics(bars(stair(per_bar=0.001, wiggle=0.0)))
        self.assertGreater(m["r2"], 0.99)
        self.assertFalse(is_stair_stepper(m))                          # 0.06%/hr < 0.5

    def test_too_few_bars_is_insufficient(self):
        m = stair_step_metrics(bars(stair(n=20)), window=45)
        self.assertEqual(m["status"], "insufficient")
        self.assertIsNone(m["r2"])
        self.assertFalse(is_stair_stepper(m))

    def test_gappy_feed_is_sparse_not_scored(self):
        m = stair_step_metrics(bars(stair(), step_min=3))              # 45 bars over 133 minutes
        self.assertEqual(m["status"], "sparse")
        self.assertIsNone(m["r2"])

    def test_gaps_use_real_time_not_bar_index(self):
        prices = stair(n=50, wiggle=0.0)
        b = bars(prices)
        del b[20:25]                                                   # 5 missing minutes
        m = stair_step_metrics(b)
        self.assertEqual(m["status"], "ok")
        self.assertGreater(m["r2"], 0.999)

    def test_window_never_bridges_the_overnight_gap(self):
        yesterday = bars(stair(n=60, base=50.0), start=START - timedelta(days=1))
        today = bars(stair(n=45, base=100.0))
        session = latest_session_bars(yesterday + today)
        self.assertEqual(len(session), 45)
        m = stair_step_metrics(yesterday + today)
        self.assertGreater(m["r2"], 0.97)

    def test_bad_rows_are_ignored(self):
        b = bars(stair()) + [{"t": None, "c": 1}, {"t": "x", "c": 1}, {"t": START.isoformat(), "c": None}]
        self.assertEqual(stair_step_metrics(b)["status"], "ok")


SCRIPT = '''
from ui.stair_stepper import render_stair_steppers
render_stair_steppers(["AAA", "BBB", "CCC"])
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class RenderTests(unittest.TestCase):
    def run_page(self, fetched, click=True):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        with mock.patch("ui.stair_stepper._cached_bars", side_effect=fetched) as fetch:
            at.run()
            if click:
                at.button(key="ss_run").click().run()
        return at, fetch

    def test_nothing_fetched_until_the_button_is_clicked(self):
        at, fetch = self.run_page(lambda s: {}, click=False)
        self.assertFalse(at.exception)
        fetch.assert_not_called()

    def test_lists_only_matching_symbols(self):
        data = {"AAA": bars(stair()),
                "BBB": bars([100 + 0.6 * math.sin(i / 2.0) for i in range(45)]),
                "CCC": bars(stair(n=10))}
        at, _ = self.run_page(lambda s: data)
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        df = at.dataframe[0].value
        self.assertEqual(list(df["Ticker"]), ["AAA"])
        self.assertIn("Not enough 1-minute data to judge (1): CCC", " ".join(c.value for c in at.caption))

    def test_provider_error_shows_a_message(self):
        def boom(s):
            raise RuntimeError("alpaca down")
        at, _ = self.run_page(boom)
        self.assertFalse(at.exception)
        self.assertIn("Couldn't load 1-minute bars", at.warning[0].value)

    def test_no_match_says_so(self):
        at, _ = self.run_page(lambda s: {"AAA": bars(stair(per_bar=-0.03))})
        self.assertIn("No symbols match", at.info[0].value)


class WiringTests(unittest.TestCase):
    def test_day_trader_renders_the_check_outside_showcase(self):
        src = (ROOT / "ui" / "day_trader.py").read_text()
        self.assertIn("from ui.stair_stepper import render_stair_steppers", src)
        self.assertIn("render_stair_steppers(symbols)", src)

    def test_descriptive_only_no_scoring_imports(self):
        for rel in ("analytics/stair_step.py", "ui/stair_stepper.py"):
            src = (ROOT / rel).read_text()
            for forbidden in ("headline_score", "scan.engine", "ml_prebreakout", "research", "signal_outcomes"):
                self.assertNotIn(forbidden, src.replace("research data", ""), f"{rel}: {forbidden}")


if __name__ == "__main__":
    unittest.main()
