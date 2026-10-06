"""P0-6 — Scanner declutter: results are the hero; other tools live on their pages."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _main_src() -> str:
    src = (ROOT / "app.py").read_text()
    return src[src.index("def main():"):src.index('if __name__ == "__main__":')]


class ScannerLayoutTests(unittest.TestCase):
    def test_results_slot_is_placed_above_the_scan_tools(self):
        m = _main_src()
        slot = m.index("results_slot = st.container()")
        self.assertLess(m.index("render_market_snapshot("), slot)
        for later in ("render_watchlists_panel(", "pages/custom_scan.py"):
            self.assertLess(slot, m.index(later), later)
        for gone in ('st.expander("Custom scan"', "render_scan_controls(", "render_three_step_scanner("):
            self.assertNotIn(gone, m, gone)                      # scan tools live on the Custom Scan page

    def test_results_fill_before_the_scan_tools_and_scans_rerun_to_show(self):
        # Run 83B (B3): filling the slot LAST let the watchlist panel's live-quote
        # fetch (and any failure after it) leave "Scanner" showing My Watchlists.
        # Results now fill first; a scan started below still shows its results
        # at the top because both scan paths rerun afterwards.
        m = _main_src()
        fill = m.index("with results_slot:")
        self.assertLess(fill, m.index("render_watchlists_panel("))
        self.assertLess(fill, m.index("render_results_tabs("))
        self.assertLess(fill, m.index("default_results(get_results_df())"))
        refresh = m[m.index('st.session_state.pop("force_results_refresh", False)'):]
        self.assertIn("st.rerun()", refresh[:600])                # manual scans rerun into the results
        three_step = (ROOT / "ui" / "three_step_scanner.py").read_text()
        after = three_step[three_step.index("_persist_three_step_run(df,"):]
        self.assertIn("st.rerun()", after)
        self.assertIn('st.session_state["force_results_refresh"] = True', after)   # opens the Scanner
        page = (ROOT / "ui" / "custom_scan.py").read_text()
        self.assertIn('st.session_state.pop("force_results_refresh", False)', page)
        self.assertIn("st.switch_page(SCANNER_PAGE)", page)

    def test_secondary_tools_are_off_the_scanner(self):
        m = _main_src()
        for gone in ("render_connect_panel(", "render_activity_feed(", "render_journal_panel("):
            self.assertNotIn(gone, m, gone)
        for link in ("pages/day_trader.py", "pages/alerts.py", "pages/journal.py"):
            self.assertIn(link, m)                                # one compact "more tools" row

    def test_alert_bell_stays_near_the_top(self):
        m = _main_src()
        self.assertLess(m.index("render_alert_bell(username)"), m.index("results_slot = st.container()"))

    def test_moved_panels_still_have_a_home(self):
        journal = (ROOT / "pages" / "journal.py").read_text()
        self.assertIn("render_activity_feed(_username)", journal)
        self.assertIn("pages/settings.py", journal)
        self.assertIn("render_connect_panel", (ROOT / "pages" / "settings.py").read_text())


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "needs streamlit")
class SlotOrderingBehaviourTests(unittest.TestCase):
    """The pattern app.py relies on: a container created early renders at its
    creation position even when filled after later elements."""

    def test_container_filled_late_renders_first(self):
        from streamlit.testing.v1 import AppTest

        script = (
            "import streamlit as st\n"
            "slot = st.container()\n"
            "st.markdown('tools')\n"
            "with slot:\n"
            "    st.markdown('results')\n"
        )
        at = AppTest.from_string(script).run()
        texts = [m.value for m in at.markdown]
        self.assertEqual(texts, ["results", "tools"])


if __name__ == "__main__":
    unittest.main()
