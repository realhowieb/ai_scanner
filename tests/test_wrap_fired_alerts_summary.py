"""Evening wrap / Brief 'Alerts that fired today': one short line per alert,
overlaps removed (owner's Sep 30 wrap listed three long run-on alerts)."""
import unittest
from unittest import mock

from scheduler.alert_email import summarize_fired

B30 = ("Breakout alert: IOVA: BreakoutScore 118.9 (≥ 30)\nKOD: BreakoutScore 69.5 (≥ 30)\n"
       "VICR: BreakoutScore 57.3 (≥ 30)\nGRAL: BreakoutScore 56.2 (≥ 30)\nMXL: BreakoutScore 50.0 (≥ 30)\n"
       "MRNA: BreakoutScore 49.6 (≥ 30)\n…and 26 more.")
B50 = ("Breakout alert: IOVA: BreakoutScore 118.9 (≥ 50)\nKOD: BreakoutScore 69.5 (≥ 50)\n"
       "VICR: BreakoutScore 57.3 (≥ 50)\nGRAL: BreakoutScore 56.2 (≥ 50)")
WATCH = ("Watchlist alert: SMTC: in scan results (BreakoutScore 33.4)\nAEHR: in scan results (BreakoutScore 26.8)\n"
         "INTC: in scan results (BreakoutScore 24.3)\nGNRC: in scan results (BreakoutScore 22.7)")


class SummaryTests(unittest.TestCase):
    def test_owner_wrap_becomes_two_short_lines(self):
        self.assertEqual(summarize_fired([B50, B30, WATCH]), [
            "Breakout alert (≥ 30) · 32 names: IOVA, KOD, VICR, GRAL, MXL…",
            "Watchlist alert · SMTC, AEHR, INTC, GNRC",
        ])

    def test_same_alert_twice_is_listed_once(self):
        self.assertEqual(len(summarize_fired([WATCH, WATCH.replace("33.4", "31.0")])), 1)

    def test_non_overlapping_breakouts_both_stay(self):
        other = "Breakout alert: ZZZ: BreakoutScore 80.0 (≥ 70)"
        self.assertEqual(summarize_fired([B50, other]), [
            "Breakout alert (≥ 50) · IOVA, KOD, VICR, GRAL", "Breakout alert (≥ 70) · ZZZ"])

    def test_unrecognised_messages_pass_through(self):
        self.assertEqual(summarize_fired(["GRAL breakout"]), ["GRAL breakout"])

    def test_price_alert(self):
        self.assertEqual(summarize_fired(["Price alert: AAPL: last 251.20 ≥ 250"]), ["Price alert · AAPL"])

    def test_wrap_uses_it_and_escapes_html(self):
        import scheduler.evening_wrap as ew

        src = open(ew.__file__).read()
        self.assertIn("return summarize_fired(seen, max_items=8)", src)
        self.assertIn("_html.escape(m)", src)


if __name__ == "__main__":
    unittest.main()
