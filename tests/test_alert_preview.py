"""Tests for smart-create alert insights (distribution + would-have-fired)."""
import datetime as dt
import unittest

from ui.alert_preview import preview_breakout_threshold, summarize_scores


def _hist(*day_scores):
    """Build history: each arg is a list of (ticker, score) for one day, newest first."""
    base = dt.date(2026, 7, 10)
    return [
        {"day": base - dt.timedelta(days=i), "scores": scores}
        for i, scores in enumerate(day_scores)
    ]


class SummarizeScoresTests(unittest.TestCase):
    def test_median_and_max_of_daily_top_scores(self):
        history = _hist(
            [("A", 62.0), ("B", 40.0)],
            [("C", 38.0)],
            [("D", 45.0)],
        )
        stats = summarize_scores(history)
        self.assertEqual(stats["snapshots"], 3)
        self.assertEqual(stats["max_top"], 62.0)
        self.assertEqual(stats["median_top"], 45.0)

    def test_empty_history(self):
        self.assertIsNone(summarize_scores([]))


class PreviewThresholdTests(unittest.TestCase):
    def test_counts_days_fired_and_reports_newest_example(self):
        history = _hist(
            [("CRNX", 61.9), ("META", 51.4)],  # fires
            [("XX", 30.0)],                     # doesn't
            [("APLS", 45.1)],                   # fires
        )
        p = preview_breakout_threshold(history, 40.0)
        self.assertEqual((p["fired"], p["total"]), (2, 3))
        self.assertEqual(p["example"][0], "CRNX")
        self.assertEqual(p["example"][1], 61.9)

    def test_dead_threshold_reports_zero(self):
        history = _hist([("A", 30.0)], [("B", 25.0)])
        p = preview_breakout_threshold(history, 90.0)
        self.assertEqual((p["fired"], p["total"]), (0, 2))
        self.assertIsNone(p["example"])

    def test_spam_threshold_fires_every_day(self):
        history = _hist([("A", 30.0)], [("B", 25.0)])
        p = preview_breakout_threshold(history, 1.0)
        self.assertEqual((p["fired"], p["total"]), (2, 2))

    def test_no_history(self):
        self.assertIsNone(preview_breakout_threshold([], 40.0))


if __name__ == "__main__":
    unittest.main()
