"""P0-7 — HSF Score is the one headline number in scanner results (presentation only)."""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from ui import headline_score as hs
from ui.results_intelligence import consolidate_scanner_results

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    return [
        # strong: breakout + golden cross + gainer, high BreakoutScore
        {"Ticker": "AAA", "Why": "x", "BreakoutScore": 42.0, "PreBreakoutProb%": 61.0, "IsBreakout": True,
         "EMACross": "Golden", "PctChange": 3.5, "GapPct": 1.0, "Last": 10.0},
        # model score only
        {"Ticker": "BBB", "Why": "y", "BreakoutScore": 12.0, "PreBreakoutProb%": None, "IsBreakout": False,
         "EMACross": None, "PctChange": 0.4, "GapPct": 0.1, "Last": 20.0},
        # no model score, one signal → not an HSF opportunity
        {"Ticker": "CCC", "Why": "z", "BreakoutScore": None, "PreBreakoutProb%": None, "IsBreakout": False,
         "EMACross": None, "PctChange": 2.5, "GapPct": 0.0, "Last": 30.0},
    ]


class HsfScoreColumnTests(unittest.TestCase):
    def test_scores_match_the_canonical_intelligence_panel(self):
        df = hs.add_hsf_score_column(pd.DataFrame(_rows()))
        canonical = {o["ticker"]: o["score"] for o in consolidate_scanner_results(_rows(), top_n=None)}
        for t, s in zip(df["Ticker"], df[hs.HSF_SCORE_COL]):
            if t in canonical:
                self.assertEqual(int(s), canonical[t], t)
            else:
                self.assertTrue(pd.isna(s), t)          # non-qualifying names get no score
        self.assertIn("AAA", canonical)
        self.assertNotIn("CCC", canonical)

    def test_row_order_is_never_changed(self):
        rows = list(reversed(_rows()))                    # scanner ranking order preserved as given
        df = hs.add_hsf_score_column(pd.DataFrame(rows))
        self.assertEqual(list(df["Ticker"]), ["CCC", "BBB", "AAA"])

    def test_column_sits_after_why_and_input_is_not_mutated(self):
        src = pd.DataFrame(_rows())
        out = hs.add_hsf_score_column(src)
        cols = list(out.columns)
        self.assertEqual(cols[cols.index("Why") + 1], hs.HSF_SCORE_COL)
        self.assertNotIn(hs.HSF_SCORE_COL, src.columns)

    def test_noop_cases(self):
        self.assertIsNone(hs.add_hsf_score_column(None))
        empty = pd.DataFrame()
        self.assertIs(hs.add_hsf_score_column(empty), empty)
        no_ticker = pd.DataFrame({"x": [1]})
        self.assertIs(hs.add_hsf_score_column(no_ticker), no_ticker)
        done = hs.add_hsf_score_column(pd.DataFrame(_rows()))
        self.assertIs(hs.add_hsf_score_column(done), done)

    def test_model_details_hidden_by_default(self):
        cols = ["Ticker", "Why", "HSF Score", "BreakoutScore", "PreBreakoutProb%", "AI Confidence", "Last"]
        self.assertEqual(hs.visible_columns(cols, show_details=False), ["Ticker", "Why", "HSF Score", "Last"])
        self.assertEqual(hs.visible_columns(cols, show_details=True), cols)

    def test_toggle_label_names_only_present_columns(self):
        self.assertEqual(hs.details_toggle_label(["Ticker", "BreakoutScore"]), "Show model details (Breakout score)")
        self.assertEqual(hs.details_toggle_label(["BreakoutScore", "PreBreakoutProb%"]),
                         "Show model details (Breakout score, PreBreakout setup probability)")
        self.assertEqual(hs.details_toggle_label(["AI Confidence", "BreakoutScore"]),
                         "Show model details (Breakout score, 5D outcome probability)")

    def test_detail_card_text(self):
        df = hs.add_hsf_score_column(pd.DataFrame(_rows()))
        self.assertRegex(hs.hsf_metric_text(df.iloc[0]), r"^\d+$")
        self.assertEqual(hs.hsf_metric_text(df.iloc[2]), "—")
        self.assertIn("not a probability of profit", hs.breakout_score_help(None))
        self.assertIn("Breakout score (model input): 42.00", hs.breakout_score_help(42.0))


if __name__ == "__main__":
    unittest.main()
