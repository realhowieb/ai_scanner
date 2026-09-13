from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

from analytics.watchlist_intelligence import (
    build_watchlist_intelligence,
    match_watchlist_opportunities,
    normalize_tickers,
)
from db.watchlists import normalize_watchlist_tickers

ROOT = Path(__file__).resolve().parents[1]


def _opp(ticker: str, score: int, status: str, *, signals=None, fading: bool = False):
    return {
        "ticker": ticker,
        "score": score,
        "status": status,
        "score_version": "1.0",
        "signals": list(signals or ["breakout", "gainer"]),
        "n_signals": len(signals or ["breakout", "gainer"]),
        "fading": fading,
    }


class PersonalWatchlistIntelligenceTests(unittest.TestCase):
    def test_watchlist_normalization_dedupes_and_rejects_bad_symbols(self):
        raw = [" aapl ", "AAPL", "ms-ft", "", "bad symbol", None, "spy"]
        self.assertEqual(normalize_watchlist_tickers(raw), ["AAPL", "MS-FT", "SPY"])
        self.assertEqual(normalize_tickers(raw), ["AAPL", "MS-FT", "SPY"])

    def test_build_is_user_scoped_and_classifies_attention(self):
        loader_calls = []

        def watchlist_loader(user: str):
            loader_calls.append(user)
            return ["aapl", "nvda", "msft", "goog"]

        now = datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)
        current = {
            "snapshot_time": datetime(2026, 9, 12, 15, 45, tzinfo=timezone.utc),
            "opportunities": [
                _opp("AAPL", 70, "WATCH", fading=True),
                _opp("NVDA", 66, "WATCH"),
                _opp("GOOG", 52, "CAUTION"),
                _opp("TSLA", 99, "STRONG"),
            ],
        }
        previous = {
            "snapshot_time": datetime(2026, 9, 12, 14, 45, tzinfo=timezone.utc),
            "opportunities": [
                _opp("AAPL", 82, "STRONG"),
                _opp("GOOG", 52, "CAUTION"),
                _opp("MSFT", 60, "WATCH"),
                _opp("TSLA", 20, "CAUTION"),
            ],
        }
        alerts = [{"ticker": "AAPL", "event_type": "FADING"}, {"ticker": "TSLA", "event_type": "NEW"}]

        intel = build_watchlist_intelligence(
            "USER@EXAMPLE.COM",
            watchlist_loader=watchlist_loader,
            current_snapshot=current,
            previous_snapshot=previous,
            recent_alerts=alerts,
            now=now,
        )

        self.assertEqual(loader_calls, ["user@example.com"])
        self.assertEqual(intel["watchlist"], ["AAPL", "GOOG", "MSFT", "NVDA"])
        self.assertEqual(intel["summary"]["tracked"], 4)
        self.assertEqual(intel["summary"]["needs_attention"], 2)
        self.assertEqual(intel["summary"]["strengthening"], 1)
        self.assertEqual(intel["summary"]["recent_alerts"], 1)
        groups = intel["groups"]
        self.assertEqual([r["ticker"] for r in groups["needs_attention"]], ["AAPL", "MSFT"])
        self.assertEqual([r["ticker"] for r in groups["improving"]], ["NVDA"])
        self.assertEqual([r["ticker"] for r in groups["stable"]], ["GOOG"])
        msft = next(row for row in intel["rows"] if row["ticker"] == "MSFT")
        self.assertFalse(msft["in_current_opportunities"])
        self.assertIn("Left the current HSF opportunity ranking", msft["attention_reason"])

    def test_unranked_watchlist_member_is_quiet_not_currently_ranked(self):
        intel = build_watchlist_intelligence(
            "u",
            watchlist=["zm"],
            current_snapshot={"snapshot_time": None, "opportunities": []},
            previous_snapshot={"snapshot_time": None, "opportunities": []},
            recent_alerts=[],
        )
        row = intel["rows"][0]
        self.assertEqual(row["group"], "quiet")
        self.assertFalse(row["in_current_opportunities"])
        self.assertEqual(row["attention_reason"], "Not currently ranked as an HSF opportunity.")

    def test_market_brief_watchlist_matching_is_exact_and_ranked(self):
        matches = match_watchlist_opportunities(
            ["aapl", "msft"],
            [_opp("TSLA", 99, "STRONG"), _opp("MSFT", 60, "WATCH"), _opp("AAPL", 75, "STRONG")],
        )
        self.assertEqual([row["ticker"] for row in matches], ["AAPL", "MSFT"])

    def test_personal_watchlist_composer_stays_read_only(self):
        source = (ROOT / "analytics" / "watchlist_intelligence.py").read_text()
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

    def test_ui_wires_personal_watchlist_without_replacing_management(self):
        page = (ROOT / "pages" / "watchlists.py").read_text()
        results = (ROOT / "ui" / "results.py").read_text()
        result_filter = (ROOT / "ui" / "result_watchlist_filter.py").read_text()
        brief = (ROOT / "ui" / "market_brief.py").read_text()
        self.assertIn("render_personal_watchlist", page)
        self.assertIn("render_watchlists_panel", page)
        self.assertIn("apply_watchlist_result_view", results)
        self.assertIn("Watchlist only", result_filter)
        self.assertIn("_render_watchlist_opportunity_matches", brief)


if __name__ == "__main__":
    unittest.main()
