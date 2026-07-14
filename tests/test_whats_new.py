"""Tests for the since-yesterday snapshot diff."""
import datetime as dt
import unittest

from ui.whats_new import diff_snapshots


def _hist(*day_scores):
    base = dt.date(2026, 7, 13)
    return [
        {"day": base - dt.timedelta(days=i), "scores": scores}
        for i, scores in enumerate(day_scores)
    ]


class DiffSnapshotsTests(unittest.TestCase):
    def test_new_entrants_and_movers(self):
        diff = diff_snapshots(
            _hist(
                [("NEWCO", 55.0), ("AAA", 50.0), ("BBB", 30.0)],  # today
                [("AAA", 40.0), ("BBB", 31.0)],                    # yesterday
            )
        )
        self.assertEqual(diff["new"], ["NEWCO"])
        self.assertEqual(diff["movers"][0], ("AAA", 10.0, 50.0))
        # BBB moved only -1.0 — at the threshold edge, |d|>=1 keeps it.
        self.assertIn(("BBB", -1.0, 30.0), diff["movers"])

    def test_small_moves_suppressed(self):
        diff = diff_snapshots(
            _hist([("AAA", 40.5)], [("AAA", 40.0)])
        )
        self.assertEqual(diff["movers"], [])
        self.assertEqual(diff["new"], [])

    def test_needs_two_days(self):
        self.assertIsNone(diff_snapshots(_hist([("AAA", 40.0)])))
        self.assertIsNone(diff_snapshots([]))


if __name__ == "__main__":
    unittest.main()
