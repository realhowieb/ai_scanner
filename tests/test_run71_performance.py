"""Run 71 — per-click database work on the Scanner.

A headless run of the real app.py main() counts the DB-backed reads per rerun:
the second rerun must make no watchlist or tier reads, a watchlist write must
be visible on the very next rerun, and the returning-user intelligence summary
must not run on the Scanner at all.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None


class TierStateTests(unittest.TestCase):
    def _fresh(self, cached, **kw):
        from ui.user_lookup import tier_state_is_fresh

        args = dict(username="a", now=100.0, session_tier="pro", billing_return=False)
        args.update(kw)
        return tier_state_is_fresh(cached, **args)

    def test_reuse_rules(self):
        c = {"user": "a", "at": 90.0, "state": {"tier_key": "pro"}}
        self.assertTrue(self._fresh(c))
        self.assertFalse(self._fresh(c, now=121.0))                 # older than 30 s
        self.assertFalse(self._fresh(c, username="b"))              # another user
        self.assertFalse(self._fresh(c, billing_return=True))       # back from Stripe
        self.assertFalse(self._fresh(c, session_tier="premium"))    # billing page refreshed the plan
        self.assertFalse(self._fresh(None))


class WatchlistVersionTests(unittest.TestCase):
    def test_every_write_bumps_the_users_version(self):
        import db.watchlists as w

        before = w.data_version("Alice@X.com")
        with mock.patch.object(w, "_get_conn", side_effect=RuntimeError("no db")):
            for name, args in (("set_watchlist_tickers", (1, "alice@x.com", ["A"])),
                               ("add_to_watchlist", ("ALICE@x.com", "A")),
                               ("delete_watchlist", (1, "alice@x.com"))):
                try:
                    getattr(w, name)(*args)
                except Exception:
                    pass
        self.assertGreaterEqual(w.data_version("alice@x.com"), before + 3)
        self.assertEqual(w.data_version("someone-else"), 0)

    def test_wrapped_write_functions_keep_their_signatures(self):
        import inspect

        import db.watchlists as w

        for name in w._WRITE_FUNCTIONS:
            self.assertIn("user_id", inspect.signature(getattr(w, name)).parameters, name)


class SummaryAndCacheWiringTests(unittest.TestCase):
    def test_summary_line(self):
        from ui.user_cache import summary_line

        self.assertIsNone(summary_line({"tracked": 0}))
        self.assertEqual(summary_line({"tracked": 4, "needs_attention": 1, "strengthening": 2, "fading": 1}),
                         "4 watched · 1 need attention · 2 strengthening · 1 fading")

    def test_returning_user_block_is_off_the_scanner_and_on_today(self):
        app = (ROOT / "app.py").read_text()
        self.assertIn("render_hsf_onboarding_entry(username, tier_name=tier_name, show_returning=False)", app)
        self.assertIn("watchlist_summary(username)", (ROOT / "ui" / "today.py").read_text())

    @unittest.skipUnless(HAS_ST, "needs streamlit")
    def test_stock_intelligence_inputs_are_cached(self):
        from ui import stock_intelligence as si

        self.assertTrue(hasattr(si._history_cached, "clear"))
        self.assertTrue(hasattr(si._earnings_days_cached, "clear"))
        si._history_cached.clear()
        with mock.patch("db.signal_outcomes.fetch_ticker_opportunity_history", return_value=[{"score": 1}]) as f:
            si._history_cached("ZZZ")
            si._history_cached("ZZZ")
        self.assertEqual(f.call_count, 1)
        si._history_cached.clear()


SCRIPT = '''
import datetime as dt, runpy
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
# AppTest can't serialise a single-select segmented_control on rerun (harness
# limitation, not app behaviour): stand in with its current/default value.
def _seg(*a, key=None, default=None, **k):
    return st.session_state.get(key, default)
DeltaGenerator.segmented_control = lambda self, *a, **k: _seg(*a, **k)
st.segmented_control = _seg
if not st.session_state.get("_cleared"):
    st.cache_data.clear()
    st.session_state["_cleared"] = True
counts = st.session_state.setdefault("_counts", {})
def counted(name, value):
    def fn(*a, **k):
        counts[name] = counts.get(name, 0) + 1
        return value() if callable(value) else value
    return fn
import db.watchlists as w
w.list_watchlists = counted("list_watchlists", [{"id": 1, "name": "Main", "is_default": True, "symbol_count": 2}])
w.get_watchlist_tickers = counted("get_watchlist_tickers", ["AAA", "BBB"])
w.get_user_watchlist = counted("get_user_watchlist", ["AAA", "BBB"])
w.get_default_watchlist_id = counted("get_default_watchlist_id", 1)
import analytics.watchlist_intelligence as wi
wi.build_watchlist_intelligence = counted("build_watchlist_intelligence", {"summary": {"tracked": 2}})
import auth.tier_sync as ts
ts.resolve_user_tier = counted("resolve_user_tier", {"tier_key": "basic"})
if st.session_state.pop("_simulate_write", False):
    w.bump_data_version("tester@example.com")
UTC = dt.timezone.utc
DF = pd.DataFrame([{"Ticker": "AAA", "BreakoutScore": 40, "IsBreakout": True, "PctChange": 3.0, "Last": 10.0}])
import ui.market_default as md
md._load_cached = lambda: (DF.copy(), {"run_id": 7, "created_at": "2026-09-28T19:35:00+00:00", "rows": 1})
import ui.market_scans as ms
ms.safe_recent_runs = lambda: [{"id": 7, "username": "cron", "label": "US_MARKET",
                                "created_at": dt.datetime(2026, 9, 28, 19, 35, tzinfo=UTC)}]
ms.safe_run_df = lambda rid: DF.copy()
import ui.header as hd
hd.fetch_ticker_quotes = lambda *a, **k: []
hd.fetch_index_snapshot = lambda *a, **k: {}
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "app.py")


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ScannerRerunReadCountTests(unittest.TestCase):
    PER_RERUN = ("list_watchlists", "get_watchlist_tickers", "resolve_user_tier")

    def test_reruns_skip_db_reads_until_a_write(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=120)
        at.session_state["username"] = "tester@example.com"
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        first = dict(at.session_state["_counts"])
        for name in self.PER_RERUN:
            self.assertEqual(first.get(name), 1, (name, first))
        self.assertNotIn("build_watchlist_intelligence", first)          # summary no longer on the Scanner

        at.run()                                                          # an ordinary click/rerun
        second = dict(at.session_state["_counts"])
        for name in self.PER_RERUN:
            self.assertEqual(second.get(name), first.get(name), (name, first, second))

        at.session_state["_simulate_write"] = True                       # e.g. a ticker added elsewhere
        at.run()
        third = dict(at.session_state["_counts"])
        self.assertEqual(third["list_watchlists"], first["list_watchlists"] + 1)
        self.assertEqual(third["resolve_user_tier"], first["resolve_user_tier"])   # tier still within 30 s


if __name__ == "__main__":
    unittest.main()
