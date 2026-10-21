"""P2-25 (owner decision 2026-09-29): Labs / Kalshi BTC is admin-only."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True

PAGE = '''
import runpy
from utils.st_compat import st
st.page_link = lambda *a, **k: None
runpy.run_path(%r, run_name="__main__")
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class NavTests(unittest.TestCase):
    def sections(self, is_admin):
        from ui import nav
        from utils.st_compat import st

        with mock.patch.object(st, "session_state", {"is_admin": is_admin}):
            return [s for s, _items in nav._visible_sections()]

    def test_labs_only_in_admin_menu(self):
        self.assertNotIn("Labs", self.sections(False))
        self.assertNotIn("Admin", self.sections(False))
        self.assertIn("Labs", self.sections(True))
        self.assertIn("Admin", self.sections(True))
        self.assertEqual(self.sections(False), ["Today", "Discover", "Research", "My Stocks", "Account"])

    def test_both_menus_use_the_filter(self):
        src = (ROOT / "ui" / "nav.py").read_text()
        self.assertEqual(src.count("for section, items in _visible_sections():"), 2)
        self.assertNotIn("for section, items in _NAV_SECTIONS:", src)


if __name__ == "__main__":
    unittest.main()
