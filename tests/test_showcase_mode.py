from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from analytics.day_trade_intel import day_trade_intelligence
from analytics.intelligence_view import build_intelligence_view
from ui.results_intelligence import consolidate_scanner_results
from ui.showcase import honest_display, initial_sidebar_state, screenshot_mode, select_columns


class ShowcaseConfigTests(unittest.TestCase):
    def test_defaults_off_and_sidebar_expanded(self):
        self.assertFalse(screenshot_mode({}))
        self.assertEqual(initial_sidebar_state({}), "expanded")

    def test_explicit_mode_collapses_sidebar(self):
        self.assertTrue(screenshot_mode({"HSF_SCREENSHOT_MODE": "true"}))
        self.assertEqual(initial_sidebar_state({"HSF_SCREENSHOT_MODE": "1"}), "collapsed")

    def test_unknown_values_stay_honest(self):
        self.assertEqual(honest_display(None, kind="score"), "—")
        self.assertEqual(honest_display(float("nan"), kind="percent"), "—")
        self.assertEqual(honest_display(182.341, kind="price"), "$182.34")
        self.assertEqual(honest_display(2.41, kind="percent"), "+2.41%")
        self.assertEqual(honest_display(3.84, kind="rvol"), "3.8x")


class ShowcaseIsolationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"Ticker": "AAA", "BreakoutScore": 82.0, "PreBreakoutProb": 0.68,
             "AI Confidence": 0.71, "PctChange": 2.4, "IsBreakout": True},
            {"Ticker": "BBB", "BreakoutScore": 74.0, "PreBreakoutProb": 0.61,
             "AI Confidence": 0.64, "PctChange": 1.2, "IsBreakout": False},
        ])

    def test_column_selection_preserves_order_and_values(self):
        original = self.frame.copy(deep=True)
        shaped = select_columns(self.frame, ("Ticker", "PreBreakoutProb", "BreakoutScore"))
        self.assertEqual(shaped["Ticker"].tolist(), ["AAA", "BBB"])
        self.assertEqual(shaped["PreBreakoutProb"].tolist(), [0.68, 0.61])
        pd.testing.assert_frame_equal(self.frame, original)

    def test_mode_does_not_change_scanner_ranking_or_scores(self):
        rows = self.frame.to_dict("records")
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": ""}, clear=False):
            normal = consolidate_scanner_results(rows)
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": "true"}, clear=False):
            showcase = consolidate_scanner_results(rows)
        self.assertEqual(normal, showcase)

    def test_mode_does_not_change_day_trader_values(self):
        row = {"chg_pct": 1.5, "gap_pct": 0.6, "rvol": 2.1,
               "vs_vwap_pct": 0.7, "adx": 27,
               "supertrend_direction": "green", "ewo": 2.4}
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": ""}, clear=False):
            normal = day_trade_intelligence(row)
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": "true"}, clear=False):
            showcase = day_trade_intelligence(row)
        self.assertEqual(normal, showcase)

    def test_mode_does_not_change_intelligence_view(self):
        observation = {
            "symbol": "AAA", "timestamp": "2026-09-22T14:00:00+00:00",
            "scan_timestamp": "2026-09-22T14:01:00+00:00",
            "market": {"price": 100.0},
            "indicators": {"rvol": 2.0, "vs_vwap_pct": 0.4},
            "models": {"prebreakout": {"probability": 0.68},
                       "ai_confidence": {"confidence": 0.71}},
            "scanners": [{"name": "prebreakout", "triggered": True, "direction": "long"}],
            "market_context": {}, "data_quality": {}, "versions": {},
        }
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": ""}, clear=False):
            normal = build_intelligence_view(observation, opportunity_score=82)
        with patch.dict(os.environ, {"HSF_SCREENSHOT_MODE": "true"}, clear=False):
            showcase = build_intelligence_view(observation, opportunity_score=82)
        self.assertEqual(normal, showcase)

    def test_showcase_module_has_no_trading_or_persistence_imports(self):
        source = open("ui/showcase.py", encoding="utf-8").read()
        for forbidden in ("from scan", "from ml_", "from db", "research_cohort", "save_"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
