import unittest
from unittest.mock import MagicMock, patch

with patch.dict("sys.modules", {"streamlit": MagicMock()}):
    from ui.watchlist_intelligence import (
        _change_summary,
        _delta_display,
        _to_bool,
        _to_float,
        classify_watchlist_signal,
        filter_watchlist_movers,
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

    def test_filters_stable_movers_by_default(self):
        rows = [
            {"Ticker": "HOT", "Move": "Heating up"},
            {"Ticker": "FLAT", "Move": "Stable"},
            {"Ticker": "GONE", "Move": "Dropped out"},
        ]

        self.assertEqual(
            [row["Ticker"] for row in filter_watchlist_movers(rows)],
            ["HOT", "GONE"],
        )
        self.assertEqual(
            [row["Ticker"] for row in filter_watchlist_movers(rows, include_stable=True)],
            ["HOT", "FLAT", "GONE"],
        )


class NanSafetyTests(unittest.TestCase):
    def test_to_float_drops_nan(self):
        self.assertIsNone(_to_float(float("nan")))
        self.assertEqual(_to_float(3.5), 3.5)
        self.assertIsNone(_to_float(None))

    def test_to_bool_treats_nan_as_false(self):
        # bool(nan) is True in Python; a missing IsBreakout flag must read False.
        self.assertFalse(_to_bool(float("nan")))
        self.assertTrue(_to_bool(True))
        self.assertFalse(_to_bool(0))

    def test_nan_is_breakout_flag_does_not_force_active_breakout(self):
        row = {
            "PreBreakoutProb%": 2.0, "AI Confidence": 1.0,
            "BreakoutScore": 5.0, "IsBreakout": float("nan"),
        }
        signal, _ = classify_watchlist_signal(row)
        self.assertEqual(signal, "Cooling down")


class StableRowDisplayTests(unittest.TestCase):
    def test_delta_display_hides_zero_and_none(self):
        self.assertEqual(_delta_display(0.0), "—")   # not "+0.0"
        self.assertEqual(_delta_display(None), "")
        self.assertEqual(_delta_display(1.24), "+1.2")
        self.assertEqual(_delta_display(-0.4), "-0.4")

    def test_change_summary_collapses_all_zero_to_one_line(self):
        self.assertEqual(
            _change_summary(0.0, 0.0, 0.0),
            "No change since the previous saved scan.",
        )

    def test_change_summary_lists_only_nonzero_moves(self):
        self.assertEqual(_change_summary(1.2, 0.0, -3.0), "PreBreakout +1.2 pts; Score -3.0")

    def test_change_summary_no_metrics(self):
        self.assertEqual(_change_summary(None, None, None), "No comparable model metrics.")


if __name__ == "__main__":
    unittest.main()
