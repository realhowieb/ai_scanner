"""Tests for the schema-aware AI note (breakout vs watchlist quote rows)."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_notes imports pandas")
class GenerateAiNoteTest(unittest.TestCase):
    def test_breakout_row(self):
        from ui.ai_notes import generate_ai_note
        note = generate_ai_note({
            "Ticker": "MSFT", "BreakoutScore": 21.3, "GapPct": 1.22,
            "Trend20D%": -12.65, "VolRel20": 3.0,
            "DollarVol20": 16081384840, "Volatility20D%": 2.88,
        })
        self.assertIn("MSFT", note)
        self.assertIn("breakout score of 21.3", note)
        self.assertIn("1.22%", note)            # gap no longer 0.00%
        self.assertIn("bearish", note)

    def test_watchlist_quote_row(self):
        from ui.ai_notes import generate_ai_note
        note = generate_ai_note({
            "Symbol": "FDS", "Name": "FactSet Research Systems Inc.",
            "Last": 231.74, "Change": 23.74, "% Change": 11.4135,
            "High": 232.54, "Low": 212.47,
        })
        self.assertIn("FDS", note)
        self.assertIn("231.74", note)
        self.assertIn("up", note)
        self.assertNotIn("breakout score", note)  # not the wrong template
        self.assertNotIn("?", note)               # ticker resolved from Symbol

    def test_sparse_row(self):
        from ui.ai_notes import generate_ai_note
        note = generate_ai_note({"Symbol": "XYZ"})
        self.assertIn("XYZ", note)
        self.assertIn("no scan metrics", note)


if __name__ == "__main__":
    unittest.main()
