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


class BubbleColorTests(unittest.TestCase):
    def test_polarity_and_neutral(self):
        from ui.score_map import bubble_color

        fill_up, border_up = bubble_color(3.0)
        self.assertIn("22,163,74", fill_up)
        fill_dn, border_dn = bubble_color(-3.0)
        self.assertIn("220,38,38", border_dn)
        fill_na, _ = bubble_color(None)
        self.assertIn("100,116,139", fill_na)

    def test_intensity_scales(self):
        from ui.score_map import bubble_color

        _, weak = bubble_color(0.5)
        _, strong = bubble_color(10.0)
        self.assertLess(float(weak.split(",")[-1].rstrip(")")), float(strong.split(",")[-1].rstrip(")")))


class BubbleHtmlTests(unittest.TestCase):
    def _html(self):
        try:
            import circlify
        except ModuleNotFoundError:
            self.skipTest("circlify not installed")
        from ui.score_map import build_bubble_html

        data = [
            {"id": "AAA", "datum": 80.0, "chg": 5.0},
            {"id": "BBB", "datum": 40.0, "chg": -2.0},
            {"id": "CCC", "datum": 20.0, "chg": None},
        ]
        circles = circlify.circlify(
            [{"id": d["id"], "datum": d["datum"]} for d in data], show_enclosure=False
        )
        return build_bubble_html(data, circles)

    def test_bubbles_and_motion_present(self):
        html = self._html()
        for t in ("AAA", "BBB", "drift", "prefers-reduced-motion"):
            self.assertIn(t, html)
        self.assertEqual(html.count("class='bub'"), 3)

    def test_polarity_colors_in_markup(self):
        html = self._html()
        self.assertIn("22,163,74", html)   # green for +5%
        self.assertIn("220,38,38", html)   # red for -2%
