"""Runs 67/68 leftovers — filters in a popover, phone top menu, lazy panels."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None


class FiltersPopoverTests(unittest.TestCase):
    def test_filter_helpers_render_into_a_given_container(self):
        for rel, fn in (("ui/filters.py", "render_filters"), ("ui/user_settings.py", "render_user_settings_footer")):
            src = (ROOT / rel).read_text()
            with self.subTest(rel=rel):
                self.assertIn("container: Any = None", src)
                # the sidebar is only the default, never written to directly
                self.assertEqual(src.count("st.sidebar"), 1 + (1 if rel.endswith("user_settings.py") else 0))

    def test_custom_scan_page_renders_filters_in_its_box(self):
        src = (ROOT / "ui" / "custom_scan.py").read_text()
        main = src[src.index("def render_custom_scan("):]
        pop = main.index("filters_box = st.container(border=True)")
        call = main.index("render_filters(tier, container=filters_box)")
        self.assertLess(pop, call)
        self.assertLess(call, main.index("render_scan_controls("))      # values exist before scans use them
        self.assertIn("container=filters_box,", main)                  # save/reset defaults move too
        self.assertNotIn("st.sidebar", main)


class TopMenuTests(unittest.TestCase):
    def test_top_menu_uses_the_grouped_sections_and_is_phone_only(self):
        from ui.chrome import CHROME_CSS
        from ui.nav import TOP_MENU_KEY

        nav = (ROOT / "ui" / "nav.py").read_text()
        self.assertIn('with st.container(key=TOP_MENU_KEY):', nav)
        self.assertIn('st.popover("☰ Menu")', nav)
        self.assertIn("render_top_menu()", nav[nav.index("def render_sidebar_nav"):])
        hide = f".st-key-{TOP_MENU_KEY}{{display:none !important;}}"
        show = f".st-key-{TOP_MENU_KEY}{{display:block !important;}}"
        self.assertIn(hide, CHROME_CSS)
        media = CHROME_CSS.index("@media (max-width:640px)")
        self.assertLess(CHROME_CSS.index(hide), media)
        self.assertGreater(CHROME_CSS.index(show), media)


class LazyPanelTests(unittest.TestCase):
    def test_secondary_tabs_load_on_demand(self):
        src = (ROOT / "ui" / "results_tabs.py").read_text()
        for key in ('"research"', '"early"', '"history"'):
            self.assertIn(f"lazy_open({key}", src)

    def test_lazy_open_defaults_to_rendering_without_streamlit(self):
        from ui import lazy_panel

        orig = lazy_panel.st
        try:
            lazy_panel.st = None
            self.assertTrue(lazy_panel.lazy_open("x", "Load x"))
        finally:
            lazy_panel.st = orig


@unittest.skipUnless(HAS_ST, "needs streamlit")
class BehaviourTests(unittest.TestCase):
    SCRIPT = (
        "import streamlit as st\n"
        "st.page_link = lambda *a, **k: None\n"
        "from auth.tiering import get_user_tier\n"
        "from ui.filters import render_filters\n"
        "from ui.lazy_panel import lazy_open\n"
        "from ui.nav import render_top_menu\n"
        "render_top_menu()\n"
        "box = st.popover('Scan filters')\n"
        "vals = render_filters(get_user_tier('t', {'t': {'tier': 'premium'}}), container=box)\n"
        "st.write('FILTERS', len(vals))\n"
        "if lazy_open('research', 'Load historical research'):\n"
        "    st.write('HEAVY')\n"
    )

    def test_filters_in_popover_and_lazy_switch(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self.SCRIPT, default_timeout=60).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("FILTERS" in m.value for m in at.markdown))
        self.assertEqual(len(at.sidebar.slider), 0)                  # nothing left in the sidebar
        self.assertIn("Min Gap %", [s.label for s in at.slider])
        self.assertFalse(any("HEAVY" in m.value for m in at.markdown))
        at.toggle(key="hsf_lazy_research").set_value(True).run()
        self.assertTrue(any("HEAVY" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
