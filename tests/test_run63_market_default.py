"""Run 63 — latest scheduled full-market scan as the Scanner's default view.

No database or Streamlit runtime: the loader is patched and `st` is faked.
"""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import market_default as md
from ui import results_empty

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 28, 16, 40, tzinfo=UTC)


class _FakeSt:
    def __init__(self):
        self.session_state = {}


class PickAndCaptionTests(unittest.TestCase):
    def test_picks_newest_cron_full_market_run_only(self):
        runs = [
            {"id": 1, "username": "cron", "label": "US_MARKET", "created_at": NOW - dt.timedelta(hours=3)},
            {"id": 2, "username": "cron", "label": "US_MARKET", "created_at": NOW - dt.timedelta(minutes=5)},
            {"id": 3, "username": "alice", "label": "US_MARKET", "created_at": NOW},
            {"id": 4, "username": "cron", "label": "premarket", "created_at": NOW},
        ]
        self.assertEqual(md.pick_market_run(runs)["id"], 2)
        self.assertIsNone(md.pick_market_run([]))

    def test_caption_uses_the_scans_own_time(self):
        meta = {"created_at": (NOW - dt.timedelta(minutes=5)).isoformat(), "rows": 100}
        cap = md.market_view_caption(meta, NOW)
        self.assertEqual(cap, "Latest full-market scan · 12:35 PM ET (5 min ago) · 100 ranked setups. "
                              + md.REPLACE_HINT)

    def test_caption_without_rows_or_time_stays_honest(self):
        self.assertEqual(md.market_view_caption({}, NOW), "Latest full-market scan. " + md.REPLACE_HINT)


class DefaultResultsTests(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeSt()
        p = mock.patch.object(md, "st", self.fake)
        p.start()
        self.addCleanup(p.stop)

    def test_session_scan_always_wins(self):
        self.fake.session_state[md.MARKET_VIEW_KEY] = {"rows": 5}
        own = pd.DataFrame({"Ticker": ["AAA"]})
        with mock.patch.object(md, "_load_cached") as loader:
            out = md.default_results(own)
        self.assertIs(out, own)
        loader.assert_not_called()
        self.assertNotIn(md.MARKET_VIEW_KEY, self.fake.session_state)

    def test_empty_session_scan_is_not_replaced(self):
        # A scan that matched nothing is the user's result, not "no scan".
        own = pd.DataFrame()
        with mock.patch.object(md, "_load_cached") as loader:
            self.assertIs(md.default_results(own), own)
        loader.assert_not_called()

    def test_no_session_scan_shows_latest_market_scan(self):
        cached = pd.DataFrame({"Ticker": ["AAA", "BBB"], "BreakoutScore": [9.1, 8.2]})
        meta = {"run_id": 7, "created_at": NOW.isoformat(), "rows": 2}
        with mock.patch.object(md, "_load_cached", return_value=(cached, meta)):
            out = md.default_results(None)
        self.assertEqual(list(out["Ticker"]), ["AAA", "BBB"])
        self.assertIsNot(out, cached)                       # a copy: enrichment can't mutate the cache
        self.assertEqual(self.fake.session_state[md.MARKET_VIEW_KEY]["run_id"], 7)

    def test_no_market_scan_falls_back_to_empty_state(self):
        self.fake.session_state[md.MARKET_VIEW_KEY] = {"rows": 5}
        with mock.patch.object(md, "_load_cached", return_value=None):
            self.assertIsNone(md.default_results(None))
        self.assertNotIn(md.MARKET_VIEW_KEY, self.fake.session_state)

    def test_loader_failure_is_logged_not_raised(self):
        with mock.patch.object(md, "_load_cached", side_effect=RuntimeError("db down")), \
                self.assertLogs("hsf.ui", level="ERROR"):
            self.assertIsNone(md.default_results(None))


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "loader normalizes via ui.app_runtime (streamlit)")
class LoaderTests(unittest.TestCase):
    def test_loader_reads_saved_run_and_normalizes(self):
        runs = [{"id": 11, "username": "cron", "label": "US_MARKET", "created_at": NOW, "row_count": 2}]
        with mock.patch("db.runs.list_runs", return_value=runs) as lr, \
                mock.patch("db.runs.load_run_results", return_value='[{"Ticker":"AAA"},{"Ticker":"BBB"}]') as rr:
            df, meta = md._load_latest_market_results()
        lr.assert_called_once_with(limit=25, include_snapshots=True, username="cron")
        rr.assert_called_once_with(11)
        self.assertEqual(len(df), 2)
        self.assertEqual(meta, {"run_id": 11, "created_at": NOW.isoformat(), "rows": 2})

    def test_loader_none_when_no_run_or_empty_results(self):
        with mock.patch("db.runs.list_runs", return_value=[]):
            self.assertIsNone(md._load_latest_market_results())
        runs = [{"id": 11, "username": "cron", "label": "US_MARKET", "created_at": NOW}]
        with mock.patch("db.runs.list_runs", return_value=runs), \
                mock.patch("db.runs.load_run_results", return_value="[]"):
            self.assertIsNone(md._load_latest_market_results())


class PresentationTests(unittest.TestCase):
    def test_tab_label_names_the_market_view(self):
        df = pd.DataFrame({"Ticker": ["A"] * 100})
        self.assertEqual(results_empty.results_tab_label(df, {"rows": 100}), "📊 Latest market scan (100 setups)")
        self.assertEqual(results_empty.results_tab_label(df), "📊 Latest scan results (100 rows)")
        self.assertEqual(results_empty.results_tab_label(None, {"rows": 1}), "📊 Your scan results")

    def test_app_uses_default_results_and_tabs_show_caption(self):
        app = (ROOT / "app.py").read_text()
        self.assertIn("df = default_results(get_results_df())", app)
        tabs = (ROOT / "ui" / "results_tabs.py").read_text()
        self.assertIn("market_view_caption(market_view)", tabs)
        self.assertIn("if not market_view:", tabs)          # no "vs your last scan" diff on the market view

    def test_session_is_not_preseeded_with_an_empty_table(self):
        # Regression: scans.py used to set results_df = pd.DataFrame() on every
        # session start, so "no scan yet" looked like "no matches" and the
        # market default never showed.
        src = (ROOT / "ui" / "scans.py").read_text()
        self.assertNotIn('if "results_df" not in st.session_state:', src)

    def test_module_is_read_only(self):
        src = (ROOT / "ui" / "market_default.py").read_text()
        for write in ("save_run", "save_daily_snapshot", "INSERT", "UPDATE", "DELETE", "results_df\"] ="):
            self.assertNotIn(write, src)
        self.assertNotIn('session_state["results_df"]', src)   # never overwrites a user's session scan


if __name__ == "__main__":
    unittest.main()
