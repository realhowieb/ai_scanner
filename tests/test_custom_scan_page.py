"""Custom Scan page (owner, 2026-10-06): scan filters and buttons moved off the
Scanner onto their own page; a finished scan opens the Scanner with its results."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None

PAGE_SCRIPT = '''
import runpy
from types import SimpleNamespace
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None            # AppTest has no multipage registry
DeltaGenerator.page_link = lambda self, *a, **k: None
SWITCHED = []
def _switch(page):
    st.session_state["_test_switched_to"] = page
    st.stop()
st.switch_page = _switch
if st.session_state.get("_test_plan"):
    from ui.app_session import compute_entitlements
    plan = st.session_state["_test_plan"]
    tier = SimpleNamespace(key=plan, name=plan.upper())
    st.session_state["tier"] = tier
    st.session_state["tier_key"] = plan
    st.session_state["is_admin"] = False
    st.session_state["entitlements"] = dict(compute_entitlements(tier_obj=tier, is_admin=False))
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "pages" / "custom_scan.py")


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CustomScanPageTests(unittest.TestCase):
    def _run(self, **state):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(PAGE_SCRIPT, default_timeout=120)
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at

    def test_signed_out_is_sent_to_login(self):
        at = self._run()
        self.assertTrue(any("log in" in i.value for i in at.info))
        self.assertFalse(any(b.label.startswith("Run SP500") for b in at.button))

    def test_plan_not_loaded_points_to_the_scanner(self):
        at = self._run(username="tester@example.com")
        self.assertTrue(any("Open the Scanner once" in i.value for i in at.info))

    def test_renders_filters_and_scan_buttons(self):
        at = self._run(username="tester@example.com", _test_plan="basic")
        labels = [b.label for b in at.button]
        self.assertIn("Run SP500 Scan", labels)
        sp500 = next(b for b in at.button if b.label == "Run SP500 Scan")
        nasdaq = next(b for b in at.button if b.label == "Run NASDAQ Scan")
        self.assertFalse(sp500.disabled)            # Free can scan the S&P 500
        self.assertTrue(nasdaq.disabled)            # NASDAQ is Pro+
        self.assertTrue(any(m.value == "#### Scan filters" for m in at.markdown))
        self.assertNotIn("_test_switched_to", at.session_state)

    def test_pro_can_scan_nasdaq(self):
        at = self._run(username="tester@example.com", _test_plan="pro")
        self.assertFalse(next(b for b in at.button if b.label == "Run NASDAQ Scan").disabled)

    def test_finished_scan_opens_the_scanner(self):
        at = self._run(username="tester@example.com", _test_plan="basic", force_results_refresh=True)
        self.assertEqual(at.session_state["_test_switched_to"], "app.py")
        self.assertTrue(at.session_state["hsf_scan_just_ran"])
        self.assertNotIn("force_results_refresh", at.session_state)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ScannerAfterCustomScanTests(unittest.TestCase):
    def test_scanner_confirms_the_scan_and_shows_its_results(self):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        from tests.test_run70_data_consistency import SCANNER_SCRIPT

        at = AppTest.from_string(SCANNER_SCRIPT, default_timeout=120)
        at.session_state["username"] = "tester@example.com"
        at.session_state["hsf_today_landed_for"] = "tester@example.com"
        at.session_state["results_df"] = pd.DataFrame()
        at.session_state["hsf_scan_just_ran"] = True
        at.session_state["_three_step_flash"] = "Scan complete in 3.0s — 0 rows."
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertIn("Scan complete in 3.0s — 0 rows.", [x.value for x in at.success])
        self.assertIn("📊 Your scan results (no matches)", [t.label for t in at.tabs])
        self.assertTrue(any(b.label == "↩ Back to the latest market scan" for b in at.button))


class _State(dict):
    def __getattr__(self, k):
        return self[k]


class WatchlistHandoffTests(unittest.TestCase):
    def _fake_st(self, tools):
        fake = mock.MagicMock()
        fake.session_state = _State(_wl_tools_state=tools)
        fake.switch_page.side_effect = lambda page: fake.session_state.__setitem__("_switched", page)
        return fake

    def test_run_watchlist_scan_hands_off_to_custom_scan(self):
        from ui import custom_scan

        fake = self._fake_st((False, True, False, False, False, "", True))
        with mock.patch.object(custom_scan, "st", fake):
            custom_scan.handle_watchlist_tools("tester@example.com")
        self.assertEqual(fake.session_state["_switched"], "pages/custom_scan.py")
        self.assertEqual(fake.session_state["_wl_pending_scan"], "run")
        self.assertTrue(fake.session_state["_wl_pending_scan_all"])
        self.assertEqual(fake.session_state["_wl_tools_state"][1], False)   # one click acts once

    def test_view_as_table_hands_off(self):
        from ui import custom_scan

        fake = self._fake_st((True, False, False, False, False, "", False))
        with mock.patch.object(custom_scan, "st", fake):
            custom_scan.handle_watchlist_tools("tester@example.com")
        self.assertEqual(fake.session_state["_wl_pending_scan"], "view")

    def test_add_symbol_stays_on_the_page(self):
        from ui import custom_scan

        fake = self._fake_st((False, False, False, True, False, "AAPL", False))
        with mock.patch.object(custom_scan, "st", fake), \
                mock.patch("ui.watchlists.handle_active_watchlist_actions") as act:
            custom_scan.handle_watchlist_tools("tester@example.com")
        self.assertTrue(act.call_args.kwargs["add_symbol"])
        self.assertEqual(act.call_args.kwargs["symbol"], "AAPL")
        fake.rerun.assert_called_once()
        self.assertNotIn("_switched", fake.session_state)

    def test_nothing_pressed_does_nothing(self):
        from ui import custom_scan

        fake = self._fake_st((False, False, False, False, False, "", False))
        with mock.patch.object(custom_scan, "st", fake):
            custom_scan.handle_watchlist_tools("tester@example.com")
        fake.switch_page.assert_not_called()
        fake.rerun.assert_not_called()

    def test_handoff_is_consumed_by_the_scan_controls(self):
        src = (ROOT / "ui" / "scans.py").read_text()
        self.assertIn('st.session_state.pop("_wl_pending_scan", None)', src)
        page = (ROOT / "pages" / "watchlists.py").read_text()
        self.assertIn("handle_watchlist_tools(_username)", page)
        self.assertNotIn('st.switch_page("app.py")', page)


class ScannerLinksTests(unittest.TestCase):
    def test_nav_and_scanner_link_to_the_page(self):
        from ui.nav import _NAV

        self.assertIn(("pages/custom_scan.py", "Custom Scan", "🧪"), _NAV)
        app = (ROOT / "app.py").read_text()
        self.assertIn('st.page_link("pages/custom_scan.py"', app)
        self.assertIn("handle_watchlist_tools(username)", app)
        self.assertIn('st.session_state.pop("hsf_scan_just_ran", False)', app)


if __name__ == "__main__":
    unittest.main()
