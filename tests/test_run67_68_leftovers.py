"""Runs 67/68 leftovers — filters in a popover, phone top menu, lazy panels."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True


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


if __name__ == "__main__":
    unittest.main()
