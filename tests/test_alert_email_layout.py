"""Alert email layout (sections + ticker tables) and P2-50 overlap removal."""
import unittest

from scheduler.alert_email import compose, drop_covered_breakouts, make_section, split_line


def brk(thr, tickers, **kw):
    a = {"alert_type": "breakout", "threshold": thr, **kw}
    return make_section(a, [f"{t}: BreakoutScore {s} (≥ {thr:g})" for t, s in tickers])


WATCH = make_section({"alert_type": "watchlist"}, ["SMTC: in scan results (BreakoutScore 33.4)",
                                                   "AEHR: in scan results (BreakoutScore 26.8)"])
B30 = brk(30.0, [("IOVA", 118.9), ("KOD", 69.5), ("MINT", 44.9)])
B50 = brk(50.0, [("IOVA", 118.9), ("KOD", 69.5)])


class OverlapTests(unittest.TestCase):
    def test_narrower_breakout_section_is_dropped(self):
        self.assertEqual([s["heading"] for s in drop_covered_breakouts([WATCH, B30, B50])],
                         [WATCH["heading"], B30["heading"]])
        self.assertEqual(len(drop_covered_breakouts([B50, B30])), 1)        # order doesn't matter

    def test_non_overlapping_sections_stay(self):
        other = brk(50.0, [("ZZZ", 80.0)])
        self.assertEqual(len(drop_covered_breakouts([B30, other])), 2)

    def test_watchlist_only_is_never_dropped_for_a_breakout(self):
        w = make_section({"alert_type": "watchlist"}, ["IOVA: in scan results"])
        self.assertEqual(len(drop_covered_breakouts([B30, w])), 2)


class LayoutTests(unittest.TestCase):
    def test_split_line_moves_the_threshold_to_the_heading(self):
        self.assertEqual(split_line("IOVA: BreakoutScore 118.9 (≥ 30)"), ("IOVA", "BreakoutScore 118.9"))
        self.assertEqual(split_line("KMX: BreakoutScore 47.4 (≥ 30) ⚠️ earnings in 2d"),
                         ("KMX", "BreakoutScore 47.4 ⚠️ earnings in 2d"))
        self.assertEqual(split_line("SMTC: in scan results (BreakoutScore 33.4)"), ("SMTC", "BreakoutScore 33.4"))
        self.assertEqual(split_line("DAL: in scan results ⚠️ earnings in 1d"), ("DAL", "in the latest scan ⚠️ earnings in 1d"))

    def test_html_has_headings_and_rows_not_pre(self):
        subject, text, html = compose([WATCH, B30, B50])
        self.assertEqual(subject, "📈 2 alerts: SMTC, AEHR, IOVA…")
        self.assertNotIn("<pre>", html)
        self.assertIn("<h3", html)
        self.assertIn("Breakout alert · BreakoutScore ≥ 30</h3>", html)
        self.assertIn("Watchlist alert · your watchlist names in the latest scan</h3>", html)
        self.assertEqual(html.count("<tr>"), 5)
        self.assertNotIn("(≥ 30)", html)
        self.assertIn("  IOVA   BreakoutScore 118.9", text)

    def test_long_lists_are_capped(self):
        many = brk(10.0, [(f"T{i}", 20.0) for i in range(30)])
        _s, text, html = compose([many])
        self.assertIn("…and 5 more", text)
        self.assertEqual(html.count("<tr>"), 25)

    def test_html_is_escaped(self):
        sec = make_section({"alert_type": "price", "ticker": "A<B"}, ["A<B: crossed <script>"])
        self.assertNotIn("<script>", compose([sec])[2])

    def test_watchlist_only_breakout_heading(self):
        self.assertEqual(brk(40.0, [("X", 50.0)], watchlist_only=True)["heading"],
                         "Breakout alert · BreakoutScore ≥ 40 · your watchlist")


if __name__ == "__main__":
    unittest.main()
