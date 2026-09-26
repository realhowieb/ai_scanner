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
        for later in ("render_watchlists_panel(", 'st.markdown("## Run your own scan")',
                      "render_earnings_controls(", "render_scan_controls(", "render_three_step_scanner()"):
            self.assertLess(slot, m.index(later), later)

    def test_results_fill_after_the_scan_tools_run(self):
        # Filling the slot last means a scan started below still shows its
        # results in the same run (the slot renders at its top position).
        m = _main_src()
        fill = m.index("with results_slot:")
        self.assertLess(m.index("render_scan_controls("), fill)
        self.assertLess(m.index("render_three_step_scanner()"), fill)
        self.assertLess(fill, m.index("render_results_tabs("))
        self.assertLess(fill, m.index("default_results(get_results_df())"))
        self.assertLess(m.index("force_results_refresh"), fill)   # rerun check still precedes rendering

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
