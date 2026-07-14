"""Tests for the score-map breadth bucketing."""
import unittest

from ui.score_map import bucket_counts


class BucketCountsTests(unittest.TestCase):
    def test_buckets_and_boundaries(self):
        counts = bucket_counts([5.0, 19.9, 20.0, 39.9, 40.0, 145.0])
        self.assertEqual(counts["Quiet (<20)"], 2)
        self.assertEqual(counts["Warm (20-40)"], 2)
        self.assertEqual(counts["Hot (40+)"], 2)

    def test_junk_ignored(self):
        counts = bucket_counts([None, "x", 25.0])
        self.assertEqual(sum(counts.values()), 1)

    def test_empty(self):
        self.assertEqual(sum(bucket_counts([]).values()), 0)


if __name__ == "__main__":
    unittest.main()


class RankHistoryTests(unittest.TestCase):
    def test_ranks_oldest_first_with_gaps(self):
        from ui.score_map import rank_history

        history = [  # newest-first, as load_score_history returns
            {"day": "d3", "scores": [("BBB", 50.0), ("AAA", 40.0)]},
            {"day": "d2", "scores": [("AAA", 45.0)]},
            {"day": "d1", "scores": [("AAA", 41.0), ("BBB", 39.0)]},
        ]
        out = rank_history(history, ["AAA", "BBB"], days=3)
        self.assertEqual(out["AAA"], [1, 1, 2])   # oldest-first
        self.assertEqual(out["BBB"], [2, None, 1])


class ScoreComponentsTests(unittest.TestCase):
    def test_parts_sum_to_score(self):
        from ui.trade_plan import score_components

        row = {
            "BreakoutScore": 90.0, "PctChange": 22.0, "Trend20D%": 30.0,
            "Trend10D%": 40.0, "BreakoutPos20D": 0.98, "GapPct": 4.0,
            "VolRel20": 3.0, "IsBreakout": True,
        }
        parts = score_components(row)
        self.assertAlmostEqual(sum(parts.values()), 90.0, places=0)
        self.assertIn("Trend 10d", parts)
        self.assertEqual(parts["Breakout bonus"], 5.0)

    def test_no_score_returns_none(self):
        from ui.trade_plan import score_components

        self.assertIsNone(score_components({}))
