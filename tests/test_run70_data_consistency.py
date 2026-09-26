"""Run 70 — data-consistency hardening.

Every screen agrees on the same market facts: one canonical latest scheduled
full-market scan (cron · US_MARKET) and one canonical opportunity handed between
screens. Includes a headless run of the real Scanner (app.py main()) with a
signed-in session, sample market data and no database or network.
"""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import results_empty, stock_handoff
from ui.results_intelligence import consolidate_scanner_results

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
HAS_ST = importlib.util.find_spec("streamlit") is not None


def _scan():
    return pd.DataFrame([
        {"Ticker": "AAA", "BreakoutScore": 40, "IsBreakout": True, "EMACross": "Golden", "VolRel20": 2.4,
         "GapPct": 4.1, "PctChange": 3.0, "Last": 10.0},
        {"Ticker": "BBB", "BreakoutScore": 30, "BreakoutPos20D": 0.99, "VolRel20": 1.1, "PctChange": 0.2,
         "Last": 20.0},
        {"Ticker": "ZZZ", "PctChange": 2.5, "Last": 5.0},        # no model score, one signal → not an opportunity
    ])


class HandoffTests(unittest.TestCase):
    """P0-9: Stock Intelligence receives the opportunity the user saw."""

    def test_canonical_opportunity_matches_the_scanner_column(self):
        df = _scan()
        canonical = {o["ticker"]: o for o in consolidate_scanner_results(df.to_dict("records"), top_n=None)}
        opp = stock_handoff.canonical_opportunity("aaa", df)
        self.assertEqual(opp["score"], canonical["AAA"]["score"])
        self.assertEqual(opp["primary_setup"], canonical["AAA"]["primary_setup"])
        self.assertIsNone(stock_handoff.canonical_opportunity("ZZZ", df))   # non-qualifying, as in the column

    def test_handoff_state_carries_opp_and_row(self):
        state = stock_handoff.handoff_state(" bbb ", scan_df=_scan())
        self.assertEqual(state[stock_handoff.TICKER_KEY], "BBB")
        self.assertEqual(state[stock_handoff.OPP_KEY]["ticker"], "BBB")
        self.assertEqual(state[stock_handoff.ROW_KEY]["Last"], 20.0)
        explicit = {"ticker": "BBB", "score": 99}
        self.assertIs(stock_handoff.handoff_state("BBB", scan_df=_scan(), opp=explicit)[stock_handoff.OPP_KEY],
                      explicit)

    def test_stock_intelligence_uses_the_handed_over_state(self):
        from ui.stock_intelligence import build_stock_intelligence

        state = stock_handoff.handoff_state("AAA", scan_df=_scan())
        intel = build_stock_intelligence("AAA", current_opp=state[stock_handoff.OPP_KEY],
                                         current_row=state[stock_handoff.ROW_KEY], history=[])
        self.assertTrue(intel["has_opportunity"])
        self.assertEqual(intel["hsf_score"], stock_handoff.canonical_opportunity("AAA", _scan())["score"])
        self.assertFalse(intel.get("from_history"))

    def test_session_context_rejects_stale_values(self):
        fake = mock.MagicMock()
        fake.session_state = {stock_handoff.OPP_KEY: {"ticker": "AAA"}, stock_handoff.ROW_KEY: {"Ticker": "AAA"}}
        with mock.patch.object(stock_handoff, "st", fake):
            self.assertEqual(stock_handoff.session_context("aaa")[0], {"ticker": "AAA"})
            self.assertEqual(stock_handoff.session_context("BBB"), (None, None))

    def test_typed_or_shared_ticker_uses_the_latest_market_scan(self):
        runs = [{"id": 5, "username": "cron", "label": "US_MARKET", "created_at": dt.datetime(2026, 9, 28, tzinfo=UTC)}]
        with mock.patch("ui.market_scans.safe_recent_runs", return_value=runs), \
                mock.patch("ui.market_scans.safe_run_df", return_value=_scan()) as load:
            opp, row = stock_handoff.latest_market_context("AAA")
        load.assert_called_once_with(5)
        self.assertEqual(opp["ticker"], "AAA")
        self.assertEqual(row["Ticker"], "AAA")

    def test_every_open_path_uses_the_handoff(self):
        today = (ROOT / "ui" / "today.py").read_text()
        self.assertIn("open_in_stock_intelligence(ticker, scan_df=scan_df, opp=opp)", today)
        self.assertIn('scan_df=scan_df, opp=o)', today)
        cards = (ROOT / "ui" / "result_cards.py").read_text()
        self.assertIn("open_in_stock_intelligence(m[\"ticker\"], scan_df=df)", cards)
        self.assertNotIn('st.session_state.pop("hsf_stock_opp"', today + cards)
        stock = (ROOT / "pages" / "stock.py").read_text()
        self.assertIn("session_context(_ticker)", stock)
        self.assertIn("latest_market_context(_ticker)", stock)
        self.assertIn("current_row=_row", stock)


class CanonicalScanSourceTests(unittest.TestCase):
    """P0-10: Market Brief and the Scanner header read cron · US_MARKET."""

    RUNS = [{"id": 9, "username": "cron", "label": "US_MARKET", "created_at": dt.datetime(2026, 9, 28, tzinfo=UTC)}]

    def test_market_brief_reads_the_latest_market_run(self):
        import ui.market_brief as mb

        with mock.patch("ui.market_scans.safe_recent_runs", return_value=self.RUNS), \
                mock.patch("ui.market_scans.safe_run_df", return_value=_scan()) as load:
            df = mb._brief_scan_df()
        load.assert_called_once_with(9)
        self.assertEqual(list(df["Ticker"]), ["AAA", "BBB", "ZZZ"])
        with mock.patch("ui.market_scans.safe_recent_runs", return_value=[]):
            self.assertIsNone(mb._brief_scan_df())

    def test_market_brief_no_longer_uses_any_user_loader(self):
        src = (ROOT / "ui" / "market_brief.py").read_text()
        compute = src[src.index("def _compute_brief"):src.index("def _compute_brief") + 1500]
        self.assertNotIn("_latest_snapshot_df", compute)
        self.assertIn("_brief_scan_df()", compute)

    @unittest.skipUnless(HAS_ST, "needs streamlit")
    def test_header_snapshot_reads_the_latest_market_run(self):
        import ui.app_user_profile as up

        fn = getattr(up._load_saved_snapshot_df, "__wrapped__", up._load_saved_snapshot_df)
        with mock.patch("ui.market_scans.safe_recent_runs", return_value=self.RUNS), \
                mock.patch("ui.market_scans.safe_run_df", return_value=_scan()) as load:
            df = fn()
        load.assert_called_once_with(9)
        self.assertEqual(len(df), 3)
        src = (ROOT / "ui" / "app_user_profile.py").read_text()
        self.assertNotIn("list_runs(limit=10)", src)


class SessionAndEmptyStateTests(unittest.TestCase):
    @unittest.skipUnless(HAS_ST, "ui.app_runtime needs streamlit")
    def test_market_session_is_holiday_aware(self):
        from zoneinfo import ZoneInfo

        from ui.app_runtime import get_market_session

        et = ZoneInfo("America/New_York")
        self.assertEqual(get_market_session(dt.datetime(2026, 11, 26, 11, 0, tzinfo=et)), "closed")      # Thanksgiving
        self.assertEqual(get_market_session(dt.datetime(2026, 11, 27, 14, 0, tzinfo=et)), "afterhours")  # 1pm close
        self.assertEqual(get_market_session(dt.datetime(2026, 11, 27, 12, 0, tzinfo=et)), "regular")
        self.assertEqual(get_market_session(dt.datetime(2026, 9, 28, 10, 0, tzinfo=et)), "regular")

    def test_three_distinct_empty_states(self):
        self.assertEqual(results_empty.results_empty_message(None), results_empty.NO_SESSION_SCAN_MESSAGE)
        self.assertEqual(results_empty.results_empty_message(None, market_unavailable=True),
                         results_empty.MARKET_UNAVAILABLE_MESSAGE)
        self.assertEqual(results_empty.results_empty_message(pd.DataFrame(), market_unavailable=True),
                         results_empty.NO_MATCHES_MESSAGE)
        self.assertEqual(len({results_empty.NO_SESSION_SCAN_MESSAGE, results_empty.MARKET_UNAVAILABLE_MESSAGE,
                              results_empty.NO_MATCHES_MESSAGE}), 3)

    def test_copy_points_to_scan_tools_below_the_results(self):
        from ui.market_default import REPLACE_HINT

        self.assertIn("below", REPLACE_HINT)
        self.assertIn("below", results_empty.NO_SESSION_SCAN_MESSAGE)


class PricingCopyTests(unittest.TestCase):
    """P1-13: pricing and sign-up copy match FEATURE_MIN_TIER."""

    def test_no_claims_for_features_customers_do_not_get(self):
        billing = (ROOT / "pages" / "billing.py").read_text()
        auth = (ROOT / "ui" / "auth.py").read_text()
        for gone in ("Diagnostics / Retrain", "diagnostics", "Full Universe Mode", "full-universe mode",
                     "get earlier signals", "Curated Breakout"):
            self.assertNotIn(gone, billing, gone)
        self.assertNotIn("AI-powered rankings", auth)
        self.assertNotIn("curated breakout scans", auth)

    def test_listed_paid_features_exist_in_the_entitlement_map(self):
        from ui.app_session import FEATURE_MIN_TIER

        self.assertEqual(FEATURE_MIN_TIER["can_track_record"], "pro")       # "historical research" row
        self.assertEqual(FEATURE_MIN_TIER["can_paper_trade"], "premium")    # "Paper trading" row
        self.assertEqual(FEATURE_MIN_TIER["can_diagnostics"], "admin")      # so never sold as Premium
        billing = (ROOT / "pages" / "billing.py").read_text()
        self.assertIn("Scan history & historical research | ❌ | ✅ | ✅", billing)
        self.assertIn("Paper trading (Alpaca) | ❌ | ❌ | ✅", billing)


SCANNER_SCRIPT = '''
import datetime as dt, runpy, sys
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None            # AppTest has no multipage registry
DeltaGenerator.page_link = lambda self, *a, **k: None
UTC = dt.timezone.utc
DF = pd.DataFrame([
    {"Ticker": "AAA", "BreakoutScore": 40, "IsBreakout": True, "VolRel20": 2.4, "GapPct": 4.1,
     "PctChange": 3.0, "Last": 10.0},
    {"Ticker": "BBB", "BreakoutScore": 30, "BreakoutPos20D": 0.99, "VolRel20": 1.1, "PctChange": 0.2, "Last": 20.0},
])
RUNS = [{"id": 7, "username": "cron", "label": "US_MARKET",
         "created_at": dt.datetime(2026, 9, 28, 19, 35, tzinfo=UTC)}]
NO_MARKET = bool(st.session_state.get("_test_no_market"))
import ui.market_default as md
md._load_cached = (lambda: None) if NO_MARKET else (
    lambda: (DF.copy(), {"run_id": 7, "created_at": RUNS[0]["created_at"].isoformat(), "rows": 2}))
import ui.market_scans as ms
ms.safe_recent_runs = lambda: [] if NO_MARKET else RUNS
ms.safe_run_df = lambda rid: None if NO_MARKET else DF.copy()
import ui.header as hd
hd.fetch_ticker_quotes = lambda *a, **k: []
hd.fetch_index_snapshot = lambda *a, **k: {}
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "app.py")


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ScannerEndToEndTests(unittest.TestCase):
    """The real app.py main() with a signed-in session, sample data, no DB/network."""

    def _run(self, **state):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCANNER_SCRIPT, default_timeout=120)
        at.session_state["username"] = "tester@example.com"
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at

    def test_market_default_renders_results_first(self):
        at = self._run()
        self.assertIn("📊 Latest market scan (2 setups)", [t.label for t in at.tabs])
        heads = [m.value for m in at.markdown if m.value.startswith("## ")]
        self.assertLess(heads.index("## Scanner"), heads.index("## Run your own scan"))
        self.assertTrue(any(c.value.startswith("Latest full-market scan") for c in at.caption))
        self.assertFalse(any(b.label.startswith("↩ Back") for b in at.button))    # already on the market view

    def test_back_to_latest_market_scan_after_own_scan(self):
        at = self._run(results_df=pd.DataFrame())        # the user's own scan matched nothing
        self.assertIn("📊 Your scan results (no matches)", [t.label for t in at.tabs])
        back = [b for b in at.button if b.label == "↩ Back to the latest market scan"]
        self.assertEqual(len(back), 1)
        back[0].click().run()
        self.assertFalse(at.exception)
        self.assertIn("📊 Latest market scan (2 setups)", [t.label for t in at.tabs])

    def test_market_unavailable_is_named(self):
        at = self._run(_test_no_market=True)
        self.assertIn(results_empty.MARKET_UNAVAILABLE_MESSAGE, [i.value for i in at.info])


if __name__ == "__main__":
    unittest.main()
