"""P1 backlog — lenses, cards, new-since-visit, recap, Today, My Stocks, nav,
Ticker Intelligence v2. Presentation only; no DB, no Streamlit runtime."""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

try:  # the Streamlit app was retired; its modules are gone
    from ui import discover, last_visit, market_scans, recap
except ImportError:
    discover = last_visit = market_scans = recap = None
from ui.product_copy import find_prohibited_claims
from ui.results_intelligence import consolidate_scanner_results

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
HAS_ST = True


def _df():
    return pd.DataFrame([
        {"Ticker": "AAA", "Why": "a", "BreakoutScore": 40, "IsBreakout": True, "VolRel20": 2.4,
         "GapPct": 4.1, "PctChange": 3.0, "Trend10D%": 12, "PreBreakoutProb%": 62, "Last": 10.0},
        {"Ticker": "BBB", "Why": "b", "BreakoutScore": 30, "IsBreakout": False, "BreakoutPos20D": 0.99,
         "VolRel20": 1.1, "GapPct": 0.5, "PctChange": 0.2, "Trend10D%": 1, "Last": 20.0},
        {"Ticker": "CCC", "Why": "c", "BreakoutScore": 20, "IsBreakout": False, "VolRel20": 1.6,
         "GapPct": -2.5, "PctChange": -2.4, "Trend10D%": -3, "Last": 30.0},
    ])


class LastVisitAndTodayTests(unittest.TestCase):

    def test_watchlist_in_scan(self):
        from ui.today import watchlist_in_scan

        w = watchlist_in_scan(["AAA", "ZZZ"], _df())
        self.assertEqual([r["ticker"] for r in w["found"]], ["AAA"])
        self.assertIsInstance(w["found"][0]["score"], int)
        self.assertEqual(w["missing"], ["ZZZ"])


class PagesAndNavTests(unittest.TestCase):


    def test_mobile_css(self):
        from ui.chrome import CHROME_CSS

        self.assertIn("@media (max-width:640px)", CHROME_CSS)


class TickerIntelligenceV2Tests(unittest.TestCase):
    def _life(self):
        from ui.stock_intelligence import reconstruct_lifecycle

        return reconstruct_lifecycle([
            {"time": dt.datetime(2026, 9, 22, 14, 35, tzinfo=UTC), "score": 58, "status": "WATCH",
             "score_version": "1.0", "signals": ["breakout"]},
            {"time": dt.datetime(2026, 9, 25, 16, 35, tzinfo=UTC), "score": 71, "status": "STRONG",
             "score_version": "1.0", "signals": ["breakout", "gainer"]},
        ])

    def test_timeline_summary_and_lines(self):
        from ui.stock_intelligence import _timeline_line, timeline_summary

        life = self._life()
        self.assertEqual(timeline_summary(life),
                         "First recorded Sep 22, 10:35 AM ET at HSF 58 · now HSF 71 (+13) across 1 update")
        self.assertIn("First recorded by HSF", _timeline_line(life[0]))
        self.assertIn("WATCH → STRONG", _timeline_line(life[1]))
        self.assertIn("+gainer", _timeline_line(life[1]))

    def test_layout_order_and_headings(self):
        src = (ROOT / "ui" / "stock_intelligence.py").read_text()
        body = src[src.index("    _render_header(intel)\n"):]
        self.assertLess(body.index("_render_actions(intel"), body.index("_render_historical(intel)"))
        self.assertIn("#### Why HSF is showing this", src)
        self.assertIn("#### What changed since HSF first noticed it", src)
        self.assertNotIn("hsf_observations", src)                        # scanner history only


if __name__ == "__main__":
    unittest.main()
