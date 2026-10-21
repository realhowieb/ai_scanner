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

try:  # the Streamlit app was retired; its modules are gone
    from ui import results_empty, stock_handoff
except ImportError:
    results_empty = stock_handoff = None
from ui.results_intelligence import consolidate_scanner_results

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
HAS_ST = True


def _scan():
    return pd.DataFrame([
        {"Ticker": "AAA", "BreakoutScore": 40, "IsBreakout": True, "EMACross": "Golden", "VolRel20": 2.4,
         "GapPct": 4.1, "PctChange": 3.0, "Last": 10.0},
        {"Ticker": "BBB", "BreakoutScore": 30, "BreakoutPos20D": 0.99, "VolRel20": 1.1, "PctChange": 0.2,
         "Last": 20.0},
        {"Ticker": "ZZZ", "PctChange": 2.5, "Last": 5.0},        # no model score, one signal → not an opportunity
    ])


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


class PricingCopyTests(unittest.TestCase):
    """P1-13: pricing and sign-up copy match FEATURE_MIN_TIER."""


    def test_listed_paid_features_exist_in_the_entitlement_map(self):
        from ui.app_session import FEATURE_MIN_TIER

        self.assertEqual(FEATURE_MIN_TIER["can_track_record"], "pro")       # "historical research" row
        self.assertEqual(FEATURE_MIN_TIER["can_paper_trade"], "premium")    # "Paper trading" row
        self.assertEqual(FEATURE_MIN_TIER["can_diagnostics"], "admin")      # so never sold as Premium
        from ui.pricing import pricing_markdown

        table = pricing_markdown()          # Run 72: generated from FEATURE_MIN_TIER
        self.assertIn("Scan history & historical research | ❌ | ✅ | ✅", table)
        self.assertIn("Alpaca paper-trading workflow | ❌ | ❌ | ✅", table)


SCANNER_SCRIPT = '''
import datetime as dt, runpy, sys
import pandas as pd
from utils.st_compat import st
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


if __name__ == "__main__":
    unittest.main()
