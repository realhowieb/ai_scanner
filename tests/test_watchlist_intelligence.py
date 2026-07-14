import unittest
from unittest.mock import MagicMock, patch

with patch.dict("sys.modules", {"streamlit": MagicMock()}):
    from ui.watchlist_intelligence import (
        classify_watchlist_signal,
        summarize_watchlist_intelligence,
        summarize_watchlist_movers,
    )


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        for idx, row in enumerate(self.rows):
            yield idx, row


class WatchlistIntelligenceTests(unittest.TestCase):
    def test_classifies_active_breakout_and_false_string(self):
        self.assertEqual(
            classify_watchlist_signal({"Ticker": "AAA", "IsBreakout": "true"})[0],
            "Active breakout",
        )
        self.assertEqual(
            classify_watchlist_signal({"Ticker": "BBB", "IsBreakout": "false", "PreBreakoutProb%": 65})[0],
            "Heating up",
        )

    def test_summarizes_watchlist_against_latest_results(self):
        rows = summarize_watchlist_intelligence(
            ["cold", "hot", "miss"],
            FakeFrame(
                [
                    {"Ticker": "HOT", "PreBreakoutProb%": 72, "AI Confidence": 20, "BreakoutScore": 10},
                    {"Ticker": "COLD", "PreBreakoutProb%": 1, "AI Confidence": 2, "BreakoutScore": 4},
                ]
            ),
        )

        labels = {row["Ticker"]: row["Signal"] for row in rows}
        self.assertEqual(labels["HOT"], "Heating up")
        self.assertEqual(labels["COLD"], "Cooling down")
        self.assertEqual(labels["MISS"], "Not in latest scan")
        self.assertEqual(rows[0]["Ticker"], "HOT")

    def test_summarizes_watchlist_movers_since_previous_scan(self):
        rows = summarize_watchlist_movers(
            ["hot", "cold", "new", "gone"],
            FakeFrame(
                [
                    {"Ticker": "HOT", "PreBreakoutProb%": 72, "AI Confidence": 45, "BreakoutScore": 18},
                    {"Ticker": "COLD", "PreBreakoutProb%": 4, "AI Confidence": 3, "BreakoutScore": 2},
                    {"Ticker": "NEW", "PreBreakoutProb%": 30, "AI Confidence": 20, "BreakoutScore": 9},
                ]
            ),
            FakeFrame(
                [
                    {"Ticker": "HOT", "PreBreakoutProb%": 40, "AI Confidence": 25, "BreakoutScore": 10},
                    {"Ticker": "COLD", "PreBreakoutProb%": 35, "AI Confidence": 30, "BreakoutScore": 8},
                    {"Ticker": "GONE", "PreBreakoutProb%": 55, "AI Confidence": 50, "BreakoutScore": 14},
                ]
            ),
        )

        moves = {row["Ticker"]: row for row in rows}
        self.assertEqual(moves["HOT"]["Move"], "Heating up")
        self.assertEqual(moves["COLD"]["Move"], "Cooling down")
        self.assertEqual(moves["NEW"]["Move"], "New in latest scan")
        self.assertEqual(moves["GONE"]["Move"], "Dropped out")
        self.assertAlmostEqual(moves["HOT"]["Pre Δ"], 32.0)


if __name__ == "__main__":
    unittest.main()
