"""Tests for watchlist Add-field symbol parsing (single or comma/space-separated)."""
from __future__ import annotations

import importlib.util
import unittest

_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(_STREAMLIT, "ui.watchlists imports streamlit")
class ParseSymbolsTest(unittest.TestCase):
    def _parse(self, text, **kw):
        from ui.watchlists import _parse_symbols

        return _parse_symbols(text, **kw)

    def test_comma_separated_splits_into_individual_tickers(self):
        # The reported bug: the whole string was stored as one "ticker".
        self.assertEqual(
            self._parse("VLO, SKHY, KLAC, ALAB, SOXL, AAOI"),
            ["VLO", "SKHY", "KLAC", "ALAB", "SOXL", "AAOI"],
        )

    def test_single_ticker(self):
        self.assertEqual(self._parse("aapl"), ["AAPL"])

    def test_whitespace_separated_and_dedup(self):
        self.assertEqual(self._parse(" msft  nvda msft "), ["MSFT", "NVDA"])

    def test_dotted_symbol_allowed(self):
        self.assertEqual(self._parse("BRK.B"), ["BRK.B"])

    def test_empty_and_none(self):
        self.assertEqual(self._parse(""), [])
        self.assertEqual(self._parse(None), [])

    def test_cap_is_honored(self):
        many = ",".join(f"T{i}" for i in range(300))
        self.assertEqual(len(self._parse(many, cap=200)), 200)


if __name__ == "__main__":
    unittest.main()
