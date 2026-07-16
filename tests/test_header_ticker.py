"""Tests for the top ticker tape defaulting to today's top scan picks."""
from __future__ import annotations

import importlib.util
import unittest

_DEPS = importlib.util.find_spec("streamlit") is not None and importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_DEPS, "header tape needs streamlit + pandas")
class TickerTapeTests(unittest.TestCase):
    def setUp(self):
        import ui.header as h

        self.h = h
        h.st.session_state = _FakeState()

    def _set_results(self, df):
        self.h.st.session_state["results_df"] = df

    def test_no_scan_falls_back_to_static_strip(self):
        self.assertEqual(self.h.default_ticker_symbols(), self.h.TICKER_STRIP)

    def test_top_scan_symbols_sorted_by_score(self):
        import pandas as pd

        self._set_results(pd.DataFrame(
            {"Ticker": ["zzz", "abc", "mmm"], "BreakoutScore": [10, 99, 55]}
        ))
        self.assertEqual(self.h.top_scan_symbols(), ["ABC", "MMM", "ZZZ"])

    def test_default_prepends_anchors_and_dedupes(self):
        import pandas as pd

        self._set_results(pd.DataFrame(
            {"Ticker": ["SPY", "abc"], "BreakoutScore": [80, 20]}
        ))
        # SPY is an anchor already, so it must not appear twice.
        self.assertEqual(self.h.default_ticker_symbols(), ["SPY", "QQQ", "ABC"])

    def test_limit_is_honored(self):
        import pandas as pd

        self._set_results(pd.DataFrame(
            {"Ticker": [f"T{i}" for i in range(20)], "BreakoutScore": list(range(20))}
        ))
        self.assertEqual(len(self.h.top_scan_symbols(limit=5)), 5)


class _FakeState(dict):
    """dict that also allows attribute-free .get like st.session_state."""


if __name__ == "__main__":
    unittest.main()
