"""Run 83B — Scanner always opens on the canonical market results (B3).

Manual testing after Run 83 saw a Premium account click Scanner and get the
"My Watchlists" card wall as the primary content. Root cause: the Scanner's
results slot sat at the top but was filled at the END of the run, after the
watchlist panel (live quote fetch for every watched ticker) and Custom scan. A
populated watchlist (not the tier) therefore showed "Scanner → My Watchlists"
until the quotes returned, and any failure after the panel left the results
empty. These tests run the real app.py headlessly per tier and state.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None
USER = "tester@example.com"
MARKET = ["AAA", "BBB", "CCC"]

SCRIPT = '''
import datetime as dt, runpy, time
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
def _seg(*a, key=None, default=None, **k):
    return st.session_state.get(key, default)
DeltaGenerator.segmented_control = lambda self, *a, **k: _seg(*a, **k)
st.segmented_control = _seg
if not st.session_state.get("_cleared"):          # per-user caches must not leak between tests
    st.cache_data.clear()
    st.session_state["_cleared"] = True
CFG = st.session_state["_cfg"]
marks = st.session_state.setdefault("_marks", [])
import db.watchlists as w
lists = [{"id": 1, "name": "Main", "is_default": True, "symbol_count": len(CFG["watch"])},
         {"id": 2, "name": "Other", "is_default": False, "symbol_count": 1}] if CFG["watch"] else []
w.list_watchlists = lambda *a, **k: lists
def _tickers(wid, *a, **k):
    if CFG.get("fail_watchlist"):
        raise RuntimeError("watchlist backend down")
    return list(CFG["watch"]) if wid == 1 else ["OTHER1"]
w.get_watchlist_tickers = _tickers
w.get_user_watchlist = lambda *a, **k: list(CFG["watch"])
w.get_default_watchlist_id = lambda *a, **k: 1 if CFG["watch"] else None
import market_data as md
def _quotes(tickers, **k):
    marks.append("watchlist quotes")
    time.sleep(CFG.get("slow", 0))
    return [{"ticker": t, "last": 10.0, "chg_pct": 1.0} for t in tickers]
md.build_day_trader_metrics = _quotes
import analytics.watchlist_intelligence as wi
wi.build_watchlist_intelligence = lambda *a, **k: {"summary": {"tracked": len(CFG["watch"])}}
import auth.tier_sync as ts
ts.resolve_user_tier = lambda *a, **k: {"tier_key": CFG["tier"]}
UTC = dt.timezone.utc
DF = pd.DataFrame([
    {"Ticker": "AAA", "BreakoutScore": 20, "IsBreakout": False, "PctChange": 1.0, "Last": 10.0},
    {"Ticker": "BBB", "BreakoutScore": 55, "IsBreakout": True, "PctChange": 6.0, "Last": 20.0, "VolRel20": 3.0},
    {"Ticker": "CCC", "BreakoutScore": 40, "IsBreakout": True, "PctChange": 3.0, "Last": 30.0},
])
import ui.market_default as mdef
mdef._load_cached = lambda: (DF.copy(), {"run_id": 7, "created_at": "2026-09-28T19:35:00+00:00", "rows": 3})
import ui.market_scans as ms
ms.safe_recent_runs = lambda: [{"id": 7, "username": "cron", "label": "US_MARKET",
                                "created_at": dt.datetime(2026, 9, 28, 19, 35, tzinfo=UTC)}]
ms.safe_run_df = lambda rid: DF.copy()
import ui.header as hd
hd.fetch_ticker_quotes = lambda *a, **k: []
hd.fetch_index_snapshot = lambda *a, **k: {}
import ui.headline_score as hs
_rank = hs.rank_hsf_opportunities
def _ranked(df):
    marks.append("results")
    out = _rank(df)
    st.session_state["_ranked"] = [str(t) for t in out["Ticker"]]
    return out
hs.rank_hsf_opportunities = _ranked
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "app.py")


# Module attributes the script patches; restored after every run so the stubs
# never leak into other tests sharing this process.
PATCHED = {
    "db.watchlists": ("list_watchlists", "get_watchlist_tickers", "get_user_watchlist", "get_default_watchlist_id"),
    "market_data": ("build_day_trader_metrics",),
    "analytics.watchlist_intelligence": ("build_watchlist_intelligence",),
    "auth.tier_sync": ("resolve_user_tier",),
    "ui.market_default": ("_load_cached",),
    "ui.market_scans": ("safe_recent_runs", "safe_run_df"),
    "ui.header": ("fetch_ticker_quotes", "fetch_index_snapshot"),
    "ui.headline_score": ("rank_hsf_opportunities",),
}


def run_app(*, tier, watch=("WATCH1", "WATCH2"), session=None, runs=1, **cfg):
    import importlib

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    saved = []
    for mod_name, attrs in PATCHED.items():
        mod = importlib.import_module(mod_name)
        saved += [(mod, a, getattr(mod, a)) for a in attrs if hasattr(mod, a)]
    try:
        at = AppTest.from_string(SCRIPT, default_timeout=180)
        at.session_state["_cfg"] = {"tier": tier, "watch": list(watch), **cfg}
        at.session_state["username"] = USER
        at.session_state["tier"] = tier
        at.session_state["hsf_today_landed_for"] = USER      # already landed on Today; now opens Scanner
        for k, v in (session or {}).items():
            at.session_state[k] = v
        for _ in range(runs):
            at.run()
        return at
    finally:
        for mod, a, value in saved:
            setattr(mod, a, value)
        st.cache_data.clear()


def headings(at):
    return [m.value.strip() for m in at.markdown if m.value.strip().startswith("#")]


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ScannerEntryTests(unittest.TestCase):
    def assert_canonical(self, at, msg=""):
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        hs = headings(at)
        self.assertIn("## Scanner", hs, msg)
        after = hs[hs.index("## Scanner") + 1]
        self.assertEqual(after, "### HSF Opportunities", (msg, hs[:6]))   # results, not My Watchlists
        if "### My Watchlists" in hs:
            self.assertLess(hs.index("### HSF Opportunities"), hs.index("### My Watchlists"), msg)
        self.assertEqual(sorted(at.session_state["_ranked"]), MARKET, msg)   # the market dataset, whole
        self.assertLess(at.session_state["_marks"].index("results"),
                        at.session_state["_marks"].index("watchlist quotes")
                        if "watchlist quotes" in at.session_state["_marks"] else 99, msg)

    def test_every_tier_opens_the_canonical_scanner_with_a_populated_watchlist(self):
        orders = {}
        for tier in ("basic", "pro", "premium"):
            at = run_app(tier=tier)
            self.assert_canonical(at, tier)
            orders[tier] = at.session_state["_ranked"]
        self.assertEqual(orders["basic"], orders["pro"])                     # same HSF ranking for all
        self.assertEqual(orders["basic"], orders["premium"])

    def test_empty_watchlist(self):
        for tier in ("basic", "premium"):
            self.assert_canonical(run_app(tier=tier, watch=()), tier)

    def test_slow_watchlist_quotes_cannot_delay_results(self):
        at = run_app(tier="premium", slow=1.0)
        self.assertEqual(at.session_state["_marks"][:2], ["results", "watchlist quotes"])

    def test_failing_watchlist_panel_cannot_blank_results(self):
        at = run_app(tier="premium", fail_watchlist=True)
        self.assert_canonical(at, "watchlist backend down")
        self.assertTrue(any("watchlist" in (i.value or "").lower() for i in [*at.info, *at.warning, *at.error]))

    def test_after_my_stocks_selection_scanner_is_still_canonical(self):
        # My Stocks left a non-default list selected; Scanner must not become that list.
        at = run_app(tier="premium", session={"active_watchlist_id": 2, "active_watchlist_tickers": ["OTHER1"],
                                              "hsf_stock_ticker": "OTHER1", "hsf_stock_opp": {"ticker": "OTHER1"}})
        self.assert_canonical(at, "after My Stocks / Stock Intelligence")

    def test_repeated_navigation_and_reruns(self):
        at = run_app(tier="premium", runs=3)                                 # Scanner → elsewhere → Scanner
        self.assert_canonical(at, "reruns")

    def test_persisted_view_preferences_keep_the_dataset(self):
        at = run_app(tier="premium", session={"hsf_results_view": "Cards"})
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertEqual(sorted(at.session_state["_ranked"]), MARKET)

    def test_badges_seeded_from_the_current_users_default_list(self):
        at = run_app(tier="premium", watch=("BBB",))
        self.assert_canonical(at)
        self.assertEqual(list(at.session_state["active_watchlist_tickers"]), ["BBB"])

    def test_account_transition_does_not_carry_the_previous_watchlist(self):
        from ui.app_session import ACCOUNT_OWNER_KEY, _owner_tag

        for prev_tier, tier in (("premium", "basic"), ("basic", "premium"), ("premium", "premium")):
            # First run after the identity change lands on Today (Run 75); the
            # second is the user opening Scanner.
            at = run_app(tier=tier, watch=("BOBPICK",), runs=2, session={
                ACCOUNT_OWNER_KEY: _owner_tag("alice@example.com"),          # state belonged to Alice
                "active_watchlist_id": 99, "active_watchlist_tickers": ["ALICEPICK"],
                "active_watchlist_quote_rows": [{"ticker": "ALICEPICK"}]})
            self.assert_canonical(at, f"{prev_tier}->{tier}")
            self.assertNotIn("ALICEPICK", list(at.session_state["active_watchlist_tickers"] or []))
            self.assertEqual(list(at.session_state["active_watchlist_tickers"]), ["BOBPICK"])


class SourceContractTests(unittest.TestCase):
    def test_results_render_before_the_watchlist_panel(self):
        app = (ROOT / "app.py").read_text()
        fill = app.index("with results_slot:")
        self.assertLess(app.index('st.markdown("## Scanner")'), fill)
        self.assertLess(fill, app.index("render_watchlists_panel(username)"))
        self.assertLess(fill, app.index('st.expander("Custom scan"'))
        self.assertLess(fill, app.index("render_three_step_scanner(container=custom_scan_box)"))

    def test_three_step_scan_reruns_to_show_results_at_the_top(self):
        src = (ROOT / "ui" / "three_step_scanner.py").read_text()
        after = src[src.index("_persist_three_step_run(df, duration_sec=duration_sec)"):]
        self.assertIn('st.session_state["_three_step_flash"]', after)
        self.assertIn("st.rerun()", after)
        self.assertIn('st.session_state.pop("_three_step_flash", None)', src)


if __name__ == "__main__":
    unittest.main()
