"""Tests for the plain-English 'why this passed' result explanations."""
import unittest

from ui.result_explain import WHY_COL, add_why_column, explain_row

try:
    import pandas as pd
except Exception:  # pragma: no cover - core envs without pandas
    pd = None

requires_pandas = unittest.skipIf(pd is None, "pandas not installed")


class ExplainRowTests(unittest.TestCase):
    def test_strong_row_names_the_key_factors(self):
        why = explain_row(
            {
                "VolRel20": 2.4,
                "GapPct": 4.2,
                "Trend10D%": 12.0,
                "BreakoutPos20D": 0.99,
                "RSvsSPY": 8.0,
            }
        )
        self.assertIn("2.4× avg volume", why)
        self.assertIn("+4.2% gap", why)
        self.assertIn("+12% over 10d", why)
        # Capped at 4 reasons, strongest-first ordering preserved.
        self.assertLessEqual(len(why.split(" · ")), 4)

    def test_quiet_row_falls_back_to_score(self):
        self.assertEqual(explain_row({"BreakoutScore": 6.5}), "score 6.5")

    def test_empty_row_returns_empty_string(self):
        self.assertEqual(explain_row({}), "")

    def test_thresholds_suppress_weak_signals(self):
        # Below every threshold: 1.1x volume, 0.5% gap, 2% trend.
        why = explain_row(
            {"VolRel20": 1.1, "GapPct": 0.5, "Trend10D%": 2.0, "BreakoutScore": 3.0}
        )
        self.assertEqual(why, "score 3")

    def test_earnings_flag_always_kept(self):
        why = explain_row(
            {
                "VolRel20": 3.0,
                "GapPct": 5.0,
                "Trend10D%": 15.0,
                "BreakoutPos20D": 0.99,
                "RSvsSPY": 9.0,
                "📅 Earnings in X days": 2,
            }
        )
        self.assertIn("earnings in 2d ⚠️", why)
        self.assertLessEqual(len(why.split(" · ")), 4)

    def test_earnings_beyond_window_not_flagged(self):
        why = explain_row({"VolRel20": 2.0, "📅 Earnings in X days": 12})
        self.assertNotIn("earnings", why)

    def test_non_numeric_values_ignored(self):
        why = explain_row({"VolRel20": "n/a", "GapPct": None, "BreakoutScore": 5.0})
        self.assertEqual(why, "score 5")

    def test_negative_gap_formats_signed(self):
        self.assertIn("-3.5% gap", explain_row({"GapPct": -3.5}))

    def test_model_confidence_included_when_high(self):
        self.assertIn("85% model conf", explain_row({"PreBreakoutProb%": 85.0}))


@requires_pandas
class AddWhyColumnTests(unittest.TestCase):
    def test_column_inserted_after_ticker(self):
        df = pd.DataFrame(
            {"Ticker": ["AAA"], "BreakoutScore": [7.0], "VolRel20": [2.5]}
        )
        out = add_why_column(df)
        self.assertEqual(list(out.columns)[:2], ["Ticker", WHY_COL])
        self.assertIn("2.5× avg volume", out[WHY_COL].iloc[0])

    def test_idempotent(self):
        df = pd.DataFrame({"Ticker": ["AAA"], "VolRel20": [2.5]})
        once = add_why_column(df)
        twice = add_why_column(once)
        self.assertEqual(list(once.columns), list(twice.columns))

    def test_none_and_empty_are_passthrough(self):
        self.assertIsNone(add_why_column(None))
        empty = pd.DataFrame()
        self.assertTrue(add_why_column(empty).empty)


if __name__ == "__main__":
    unittest.main()
