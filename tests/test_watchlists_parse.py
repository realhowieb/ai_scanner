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

    def test_bulk_edit_with_duplicates_is_deduped(self):
        # Save-list path: a pasted box carrying duplicate/garbage-derived tokens
        # must collapse to unique symbols, not re-store duplicates.
        box = "ABT,GE,PLD,TSM,UAL,UNH,USB,USB,PLD,ABT,GE,TSM,UNH,UAL"
        self.assertEqual(
            self._parse(box),
            ["ABT", "GE", "PLD", "TSM", "UAL", "UNH", "USB"],
        )

    def test_empty_and_none(self):
        self.assertEqual(self._parse(""), [])
        self.assertEqual(self._parse(None), [])

    def test_cap_is_honored(self):
        many = ",".join(f"T{i}" for i in range(300))
        self.assertEqual(len(self._parse(many, cap=200)), 200)


@unittest.skipUnless(_STREAMLIT, "ui.watchlists imports streamlit")
class NormalizeStoredTickersTest(unittest.TestCase):
    def test_legacy_concatenated_row_is_flattened_and_healed(self):
        import ui.watchlists as wl

        # A watchlist that already contains the pre-fix garbage row.
        stored = ["ABT", "TSM", "UNH", "GE", "PLD", "UAL", "USB",
                  "USB,PLD,ABT,GE,TSM,UNH,UAL"]
        writes = []
        orig = wl.set_watchlist_tickers
        wl.set_watchlist_tickers = lambda wid, user, tickers: writes.append(tickers)
        try:
            cleaned = wl._normalize_stored_tickers(stored, 1, "u@example.com")
        finally:
            wl.set_watchlist_tickers = orig

        # Garbage row dropped; each symbol appears once, order preserved.
        self.assertEqual(cleaned, ["ABT", "TSM", "UNH", "GE", "PLD", "UAL", "USB"])
        # Self-heal: the cleaned list was written back exactly once.
        self.assertEqual(writes, [["ABT", "TSM", "UNH", "GE", "PLD", "UAL", "USB"]])

    def test_already_clean_list_is_not_rewritten(self):
        import ui.watchlists as wl

        writes = []
        orig = wl.set_watchlist_tickers
        wl.set_watchlist_tickers = lambda wid, user, tickers: writes.append(tickers)
        try:
            cleaned = wl._normalize_stored_tickers(["AAPL", "MSFT"], 1, "u")
        finally:
            wl.set_watchlist_tickers = orig

        self.assertEqual(cleaned, ["AAPL", "MSFT"])
        self.assertEqual(writes, [])  # no needless DB write


@unittest.skipUnless(_STREAMLIT, "ui.watchlists imports streamlit")
class ScanAllPassthroughTest(unittest.TestCase):
    """The 'Score every ticker' checkbox must bypass filters on the watchlist scan."""

    def _run(self, scan_all):
        import ui.watchlists as wl

        wl.st.session_state = _FakeState({"active_watchlist_tickers": ["AAPL", "MSFT"]})
        calls = []

        def fake_do_scan(tickers, label, *, bypass_filters=False):
            calls.append({"tickers": tickers, "label": label, "bypass": bypass_filters})

        wl.handle_active_watchlist_actions(
            view_watchlist=False,
            run_watchlist=True,
            clear_watchlist=False,
            add_symbol=False,
            remove_symbol=False,
            symbol="",
            username="u",
            do_scan=fake_do_scan,
            banner=lambda *a, **k: None,
            scan_all=scan_all,
        )
        return calls

    def test_scan_all_bypasses_filters_and_tags_label(self):
        calls = self._run(scan_all=True)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["bypass"])
        self.assertTrue(calls[0]["label"].endswith("· all"))

    def test_default_scan_keeps_filters(self):
        calls = self._run(scan_all=False)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["bypass"])
        self.assertNotIn("· all", calls[0]["label"])


class _FakeState(dict):
    pass


if __name__ == "__main__":
    unittest.main()
