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
