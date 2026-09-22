"""Run 42 — Watchlist Intelligence 2.0 tests (deterministic, Streamlit-free)."""
import unittest

from analytics import watchlist_view as wv


def _row(ticker="NVDA", **kw):
    base = {"ticker": ticker, "last": 131.0, "chg_pct": 1.5, "gap_pct": 0.5,
            "rvol": 3.1, "vs_vwap_pct": 0.8, "adx": 28}
    base.update(kw)
    return base


class AdapterTests(unittest.TestCase):
    def test_row_to_observation_and_triggers(self):
        obs = wv.observation_from_market_row(_row())
        self.assertEqual(obs["symbol"], "NVDA")
        names = {s["name"] for s in obs["scanners"]}
        self.assertIn("unusual_vol", names)   # rvol 3.1 >= 2
        self.assertIn("momentum", names)      # chg > 0
        self.assertEqual(obs["indicators"]["vs_vwap_pct"], 0.8)

    def test_no_trigger_row(self):
        obs = wv.observation_from_market_row(_row(rvol=0.9, chg_pct=-0.2, gap_pct=0.0))
        self.assertEqual(obs["scanners"], [])


class SymbolViewTests(unittest.TestCase):
    def test_all_symbols_visible_including_no_setup(self):
        views = wv.build_watchlist_views([_row("NVDA"), _row("AAPL", rvol=0.9,
                                          chg_pct=0.4, gap_pct=0.0)])
        syms = {v["symbol"] for v in views}
        self.assertEqual(syms, {"NVDA", "AAPL"})
        aapl = next(v for v in views if v["symbol"] == "AAPL")
        self.assertTrue(aapl["no_active_setup"])
        self.assertTrue(any("+0.4% today" in f for f in aapl["no_setup_facts"]))

    def test_watchlist_identity_and_flag(self):
        v = wv.build_watchlist_symbol_view(_row(), watchlist_id=7, watchlist_name="Tech")
        self.assertEqual(v["watchlist_id"], 7)
        self.assertEqual(v["watchlist_name"], "Tech")
        self.assertTrue(v["is_watchlist"])
        self.assertEqual(v["schema_version"], wv.SCHEMA_VERSION)

    def test_why_and_caution_reuse_run40(self):
        v = wv.build_watchlist_symbol_view(_row())
        self.assertTrue(any("RVOL 3.1x" in r for r in v["positive_reasons"]))
        weak = wv.build_watchlist_symbol_view(_row(vs_vwap_pct=-0.5, rvol=0.6, adx=12))
        self.assertTrue(any("Below VWAP" in r for r in weak["risk_reasons"]))


class ChangeLifecycleTests(unittest.TestCase):
    def test_new_when_no_prior(self):
        v = wv.build_watchlist_symbol_view(_row())
        self.assertEqual(v["lifecycle_state"], "NEW")

    def test_strengthening_on_new_scanner(self):
        prior = _row(rvol=1.0, gap_pct=0.0)   # only momentum
        cur = _row(rvol=3.2, gap_pct=0.0)     # + unusual_vol
        v = wv.build_watchlist_symbol_view(cur, prior_row=prior)
        self.assertTrue(any("RVOL" in c for c in v["changes_since_prior"]))
        self.assertEqual(v["lifecycle_state"], "STRENGTHENING")

    def test_weakening_on_lost_vwap(self):
        prior = _row(vs_vwap_pct=0.5)
        cur = _row(vs_vwap_pct=-0.5)
        v = wv.build_watchlist_symbol_view(cur, prior_row=prior)
        self.assertTrue(any("Lost VWAP" in c for c in v["changes_since_prior"]))
        self.assertEqual(v["lifecycle_state"], "WEAKENING")

    def test_no_setup_new_becomes_active(self):
        v = wv.build_watchlist_symbol_view(_row("AAPL", rvol=0.9, chg_pct=0.1, gap_pct=0.0))
        self.assertEqual(v["lifecycle_state"], "ACTIVE")  # quiet, present, not an alert


class SortingTests(unittest.TestCase):
    def _mixed(self):
        return wv.build_watchlist_views([
            _row("HIGH", rvol=3.5, gap_pct=1.0),           # 3 scanners → HIGH
            _row("QUIET", rvol=0.9, chg_pct=0.2, gap_pct=0.0),  # no setup
            _row("MED", rvol=2.1, chg_pct=0.5, gap_pct=0.0),    # unusual_vol+momentum
        ])

    def test_attention_puts_high_first_quiet_last(self):
        ranked = wv.attention_sort(self._mixed())
        self.assertEqual(ranked[0]["symbol"], "HIGH")
        self.assertEqual(ranked[-1]["symbol"], "QUIET")

    def test_ticker_and_rvol_sorts(self):
        views = self._mixed()
        self.assertEqual([v["symbol"] for v in wv.sort_views(views, "Ticker")],
                         ["HIGH", "MED", "QUIET"])
        self.assertEqual(wv.sort_views(views, "RVOL")[0]["symbol"], "HIGH")


class SummaryFilterTests(unittest.TestCase):
    def _views(self):
        return wv.build_watchlist_views([
            _row("A", rvol=3.5, gap_pct=1.0), _row("B", rvol=2.1, gap_pct=0.0),
            _row("C", rvol=0.8, chg_pct=0.1, gap_pct=0.0)])

    def test_summary_counts(self):
        s = wv.watchlist_summary(self._views())
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["no_active_setup"], 1)
        self.assertGreaterEqual(s["needs_attention"], 1)

    def test_filters(self):
        views = self._views()
        self.assertEqual([v["symbol"] for v in wv.filter_watchlist(views, "No Setup")], ["C"])
        self.assertTrue(all(not v["no_active_setup"]
                            for v in wv.filter_watchlist(views, "Active Setups")))
        self.assertEqual([v["symbol"] for v in wv.filter_watchlist(views, "High Priority")], ["A"])


class ActivityFeedTests(unittest.TestCase):
    def test_feed_from_changes_only(self):
        prior = {"NVDA": _row("NVDA", rvol=1.0, gap_pct=0.0)}
        views = wv.build_watchlist_views([_row("NVDA", rvol=3.2, gap_pct=0.0)],
                                         prior_by_symbol=prior)
        feed = wv.activity_feed(views)
        self.assertTrue(feed)
        self.assertEqual(feed[0]["symbol"], "NVDA")

    def test_no_changes_no_events(self):
        views = wv.build_watchlist_views([_row()])  # no prior → no changes
        self.assertEqual(wv.activity_feed(views), [])


class RobustnessTests(unittest.TestCase):
    def test_one_bad_row_skipped(self):
        views = wv.build_watchlist_views([_row("GOOD"), None, {"no_ticker": 1}])
        syms = {v["symbol"] for v in views}
        self.assertIn("GOOD", syms)  # bad rows skipped, good survives

    def test_no_pii(self):
        v = wv.build_watchlist_symbol_view(_row())
        blob = str(v).lower()
        for pii in ("email", "@", "password", "user_id", "billing"):
            self.assertNotIn(pii, blob)

    def test_deterministic(self):
        self.assertEqual(wv.build_watchlist_symbol_view(_row()),
                         wv.build_watchlist_symbol_view(_row()))

    def test_empty_message(self):
        self.assertIn("empty", wv.empty_watchlist_message().lower())


if __name__ == "__main__":
    unittest.main()
