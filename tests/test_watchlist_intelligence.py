import unittest
from unittest.mock import MagicMock, patch

with patch.dict("sys.modules", {"streamlit": MagicMock()}):
    from ui.watchlists import classify_watchlist_signal, summarize_watchlist_intelligence


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


if __name__ == "__main__":
    unittest.main()
