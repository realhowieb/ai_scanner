from __future__ import annotations

import unittest
from pathlib import Path

from ui.onboarding import (
    activation_reached,
    add_first_watch_ticker,
    current_opportunity_for_ticker,
    user_is_first_run,
)

ROOT = Path(__file__).resolve().parents[1]


def _snapshot():
    return [
        {
            "snapshot_time": "2026-09-12T15:00:00+00:00",
            "opportunities": [
                {
                    "ticker": "NVDA",
                    "score": 78,
                    "status": "STRONG",
                    "signals": ["breakout", "gainer", "prebreakout"],
                }
            ],
        }
    ]


class HsfOnboardingTests(unittest.TestCase):
    def test_first_run_detection_uses_watchlist_emptiness(self):
        self.assertTrue(user_is_first_run([]))
        self.assertTrue(user_is_first_run([""]))
        self.assertFalse(user_is_first_run(["nvda"]))

    def test_activation_definition_is_watchlist_plus_personal_intel_view(self):
        self.assertFalse(activation_reached([], True))
        self.assertFalse(activation_reached(["NVDA"], False))
        self.assertTrue(activation_reached(["NVDA"], True))

    def test_first_valid_ticker_is_normalized_added_and_gets_current_context(self):
        calls = []
        result = add_first_watch_ticker(
            "USER@EXAMPLE.COM",
            " nvda ",
            existing_watchlist=[],
            add_fn=lambda user, ticker: calls.append((user, ticker)) or True,
            snapshots_loader=_snapshot,
        )
        self.assertEqual(result.status, "added")
        self.assertEqual(result.ticker, "NVDA")
        self.assertEqual(calls, [("user@example.com", "NVDA")])
        self.assertEqual(result.opportunity["status"], "STRONG")
        self.assertEqual(result.opportunity["score"], 78)

    def test_invalid_ticker_gets_friendly_message_and_no_write(self):
        calls = []
        result = add_first_watch_ticker(
            "u",
            "bad symbol",
            existing_watchlist=[],
            add_fn=lambda user, ticker: calls.append((user, ticker)) or True,
        )
        self.assertEqual(result.status, "invalid")
        self.assertIn("couldn't recognize", result.message)
        self.assertEqual(calls, [])

    def test_duplicate_ticker_is_not_error_and_does_not_write(self):
        calls = []
        result = add_first_watch_ticker(
            "u",
            "NVDA",
            existing_watchlist=["nvda"],
            add_fn=lambda user, ticker: calls.append((user, ticker)) or True,
            snapshots_loader=_snapshot,
        )
        self.assertEqual(result.status, "duplicate")
        self.assertIn("already in your watchlist", result.message)
        self.assertEqual(result.opportunity["ticker"], "NVDA")
        self.assertEqual(calls, [])

    def test_unranked_ticker_is_still_successful_first_value(self):
        result = add_first_watch_ticker(
            "u",
            "AAPL",
            existing_watchlist=[],
            add_fn=lambda user, ticker: True,
            snapshots_loader=_snapshot,
        )
        self.assertEqual(result.status, "added")
        self.assertEqual(result.ticker, "AAPL")
        self.assertIsNone(result.opportunity)

    def test_current_context_never_fabricates_missing_opportunity(self):
        self.assertIsNone(current_opportunity_for_ticker("AAPL", snapshots_loader=_snapshot))
        self.assertEqual(current_opportunity_for_ticker("nvda", snapshots_loader=_snapshot)["status"], "STRONG")

    def test_onboarding_source_is_read_only_except_watchlist_add(self):
        source = (ROOT / "ui" / "onboarding.py").read_text()
        forbidden = (
            "compute_compared_opportunities",
            "save_opportunity_snapshot",
            "freeze_opportunities",
            "mature_signal",
            "deliver_intelligence_alerts",
            "ask_claude",
            "do_scan",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("add_to_watchlist", source)

    def test_pages_have_first_run_orientation_without_changing_core_pages(self):
        app = (ROOT / "app.py").read_text()
        brief = (ROOT / "pages" / "brief.py").read_text()
        stock = (ROOT / "pages" / "stock.py").read_text()
        alerts = (ROOT / "pages" / "alerts.py").read_text()
        watchlists = (ROOT / "ui" / "personal_watchlist.py").read_text()
        self.assertIn("render_hsf_onboarding_entry", app)
        self.assertIn("render_scanner_orientation", app)
        self.assertIn("render_market_brief_orientation", brief)
        self.assertIn("render_stock_intelligence_orientation", stock)
        self.assertIn("render_alerts_orientation", alerts)
        self.assertIn("Build your personalized market view", watchlists)


if __name__ == "__main__":
    unittest.main()
