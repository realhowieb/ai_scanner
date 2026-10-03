"""P2-72 — import a watchlist from a broker CSV, a TradingView export or a plain list."""
import importlib.util
import unittest
from unittest import mock

from ui import watchlist_import as wi

HAS_ST = importlib.util.find_spec("streamlit") is not None


def parse(name, text):
    return wi.parse_watchlist_file(name, text.encode("utf-8"))


class ParseTests(unittest.TestCase):
    def test_broker_csv_with_title_lines_above_the_header(self):
        text = ('"Positions for account Individual ...123 as of 10/02/2026"\n\n'
                '"Symbol","Description","Quantity","Price"\n'
                '"AAPL","APPLE INC","10","255.00"\n"msft","MICROSOFT","5","500.00"\n'
                '"Cash & Cash Investments","--","--","--"\n"Account Total","--","--","--"\n')
        r = parse("positions.csv", text)
        self.assertEqual(r["tickers"], ["AAPL", "MSFT"])
        self.assertEqual(r["skipped"], [])  # "Cash & Cash Investments", "Account Total" are labels

    def test_ticker_column_in_any_position(self):
        r = parse("export.csv", "Name,Ticker,Shares\nNvidia,NVDA,1\nAMD,amd,2\n")
        self.assertEqual(r["tickers"], ["NVDA", "AMD"])

    def test_csv_without_a_symbol_header_reads_every_cell(self):
        r = parse("list.csv", "AAPL,MSFT\nNVDA\n")
        self.assertEqual(r["tickers"], ["AAPL", "MSFT", "NVDA"])

    def test_tradingview_export(self):
        r = parse("tech.txt", "###Tech,NASDAQ:AAPL,NASDAQ:MSFT,###Energy,NYSE:XOM,AMEX:SPY")
        self.assertEqual(r["tickers"], ["AAPL", "MSFT", "XOM", "SPY"])

    def test_plain_list_any_separator_dedupes_and_strips_dollar(self):
        r = parse("x.txt", "aapl, $TSLA\nMSFT\tNVDA; AAPL  BRK-B")
        self.assertEqual(r["tickers"], ["AAPL", "TSLA", "MSFT", "NVDA", "BRK-B"])

    def test_invalid_tokens_are_reported_not_imported(self):
        r = parse("x.txt", "AAPL, not-a-ticker!, TOOLONGTICKER, MSFT")
        self.assertEqual(r["tickers"], ["AAPL", "MSFT"])
        self.assertIn("TOOLONGTICKER", r["skipped"])

    def test_cap_and_size_limit(self):
        many = ",".join(f"T{i}" for i in range(250))
        r = parse("x.txt", many)
        self.assertEqual(len(r["tickers"]), wi.MAX_TICKERS)
        self.assertEqual(r["over_cap"], 50)
        with self.assertRaises(ValueError):
            wi.parse_watchlist_file("big.csv", b"A," * (wi.MAX_FILE_BYTES // 2 + 1))

    def test_latin1_and_bom(self):
        self.assertEqual(wi.parse_watchlist_file("x.csv", "﻿Symbol\nAAPL\n".encode("utf-8"))["tickers"], ["AAPL"])
        self.assertEqual(wi.parse_watchlist_file("x.txt", "AAPL caf\xe9".encode("latin-1"))["tickers"], ["AAPL"])


SCRIPT = '''
import streamlit as st
from ui.watchlist_import import render_watchlist_import
render_watchlist_import("alice", st.session_state.get("active_id"), "Main")
'''


class _Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


@unittest.skipUnless(HAS_ST, "needs streamlit")
class UiTests(unittest.TestCase):
    def run_app(self, active_id=7, click=False, target=None):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["active_id"] = active_id
        add = mock.patch("db.watchlists.add_tickers_to_watchlist",
                         return_value={"added": ["AAPL"], "already_present": ["MSFT"], "invalid": []})
        create = mock.patch("db.watchlists.create_watchlist", return_value=42)
        upload = mock.patch("streamlit.file_uploader", return_value=_Upload("tech.txt", b"NASDAQ:AAPL,NASDAQ:MSFT"))
        m_add, m_create, _ = add.start(), create.start(), upload.start()
        self.addCleanup(mock.patch.stopall)
        at.run()
        if target:
            at.radio(key="wl_import_target").set_value(target).run()
        if click:
            at.button(key="wl_import_btn").click().run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at, m_add, m_create

    def test_preview_then_add_to_active_list(self):
        at, m_add, m_create = self.run_app()
        self.assertIn("Found **2** tickers: AAPL, MSFT", " ".join(m.value for m in at.markdown))
        m_add.assert_not_called()
        at, m_add, m_create = self.run_app(click=True)
        m_add.assert_called_once_with("alice", ["AAPL", "MSFT"], 7)
        m_create.assert_not_called()
        self.assertEqual(at.session_state["active_watchlist_id"], 7)

    def test_new_watchlist_named_after_the_file(self):
        at, m_add, m_create = self.run_app(click=True, target="new")
        m_create.assert_called_once_with("alice", "tech")
        m_add.assert_called_once_with("alice", ["AAPL", "MSFT"], 42)

    def test_without_a_list_only_new_is_offered(self):
        at, _, _ = self.run_app(active_id=None)
        self.assertEqual(at.radio(key="wl_import_target").options, ["Create a new watchlist"])


class WiringTests(unittest.TestCase):
    def test_manage_watchlist_renders_import(self):
        src = open("ui/watchlists.py").read()
        self.assertIn("render_watchlist_import(username, active_id, active_name)", src)


if __name__ == "__main__":
    unittest.main()
