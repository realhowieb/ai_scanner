"""P2-74 follow-up: the morning digest lists this morning's pre-market movers
(the same list as Today's Before the open card)."""
import datetime as dt
import unittest
from unittest import mock

from scheduler import morning_digest as md

MOVERS = [{"ticker": "BBB", "pct": -10.0, "last": 18.0, "score": 61},
          {"ticker": "AAA", "pct": 6.0, "last": None, "score": None}]


class SectionTests(unittest.TestCase):
    def test_section_html_and_text(self):
        html, text = md._premarket_section(MOVERS)
        self.assertIn("Pre-market movers", html)
        self.assertIn("<strong>BBB</strong>", html)
        self.assertIn("-10.00%", html)
        self.assertIn("$18.00", html)
        self.assertIn("HSF Score 61", html)
        self.assertIn("#dc2626", html)
        self.assertIn("  BBB -10.00% · $18.00 · HSF Score 61", text)
        self.assertIn("  AAA +6.00%", text)

    def test_empty_list_adds_nothing(self):
        self.assertEqual(md._premarket_section([]), ("", ""))

    def test_compose_places_it_after_gappers_and_omits_it_when_empty(self):
        with mock.patch.object(md, "_track_record_line", return_value=("", "")):
            html, text = md._compose("u@example.com", [], [], [], [], premarket=MOVERS)
            html_none, text_none = md._compose("u@example.com", [], [], [], [])
        self.assertLess(html.index("Top market gappers"), html.index("Pre-market movers"))
        self.assertIn("Pre-market movers (8:35 AM ET scan", text)
        self.assertNotIn("Pre-market movers", html_none)
        self.assertNotIn("Pre-market movers", text_none)


class LoaderTests(unittest.TestCase):
    NOW = dt.datetime(2026, 9, 29, 12, 40, tzinfo=dt.timezone.utc)   # Tue 8:40 ET

    def test_uses_this_mornings_premarket_run(self):
        runs = [{"id": 5, "label": "premarket", "created_at": "2026-09-29T12:35:00+00:00"}]
        with mock.patch("db.runs.list_runs", return_value=runs), \
                mock.patch("ui.market_scans._run_df_uncached", return_value="DF") as load, \
                mock.patch("ui.before_open.premarket_movers", return_value=MOVERS) as movers:
            self.assertEqual(md._premarket_movers(now=self.NOW), MOVERS)
        load.assert_called_once_with(5)
        movers.assert_called_once_with("DF", n=5)

    def test_no_run_or_error_is_empty(self):
        with mock.patch("db.runs.list_runs", return_value=[]):
            self.assertEqual(md._premarket_movers(now=self.NOW), [])
        with mock.patch("db.runs.list_runs", side_effect=RuntimeError("db down")):
            self.assertEqual(md._premarket_movers(now=self.NOW), [])


if __name__ == "__main__":
    unittest.main()
