"""Tests for ui/ai_insights.py — scan diff, watchlist digest, alert lines."""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_insights requires pandas")
class FilterTest(unittest.TestCase):
    def _df(self):
        import pandas as pd
        return pd.DataFrame([
            {"Ticker": "AAPL", "BreakoutScore": 9.0},
            {"Ticker": "NVDA", "BreakoutScore": 7.5},
            {"Ticker": "TSLA", "BreakoutScore": 5.0},
        ])

    def test_filter_to_watchlist_case_insensitive(self):
        from ui.ai_insights import _filter_to_watchlist
        sub = _filter_to_watchlist(self._df(), ["aapl", "nvda"])
        self.assertEqual(set(sub["Ticker"]), {"AAPL", "NVDA"})

    def test_filter_empty_when_no_match(self):
        from ui.ai_insights import _filter_to_watchlist
        sub = _filter_to_watchlist(self._df(), ["ZZZZ"])
        self.assertEqual(len(sub), 0)


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_insights requires pandas")
class DiffGuardTest(unittest.TestCase):
    def test_no_current_scan(self):
        import pandas as pd

        from ui.ai_insights import generate_scan_diff
        text, err = generate_scan_diff(pd.DataFrame(), pd.DataFrame())
        self.assertIsNone(text)
        self.assertIn("No current scan", err)

    def test_no_previous_snapshot(self):
        import pandas as pd

        from ui.ai_insights import generate_scan_diff
        curr = pd.DataFrame([{"Ticker": "AAPL", "BreakoutScore": 9.0}])
        text, err = generate_scan_diff(pd.DataFrame(), curr)
        self.assertIsNone(text)
        self.assertIn("previous", err.lower())

    def test_diff_calls_claude_when_both_present(self):
        import pandas as pd

        from ui import ai_insights
        prev = pd.DataFrame([{"Ticker": "AAPL", "BreakoutScore": 5.0}])
        curr = pd.DataFrame([{"Ticker": "AAPL", "BreakoutScore": 9.0}])
        with patch("ui.ai.ask_claude", return_value=("AAPL score jumped.", None)):
            text, err = ai_insights.generate_scan_diff(prev, curr)
        self.assertIsNone(err)
        self.assertIn("AAPL", text)


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_insights requires pandas")
class DigestGuardTest(unittest.TestCase):
    def test_digest_no_matching_tickers(self):
        import pandas as pd

        from ui.ai_insights import generate_watchlist_digest
        df = pd.DataFrame([{"Ticker": "AAPL", "BreakoutScore": 9.0}])
        text, err = generate_watchlist_digest(["ZZZZ"], df)
        self.assertIsNone(text)
        self.assertIn("watchlist", err.lower())


if __name__ == "__main__":
    unittest.main()
