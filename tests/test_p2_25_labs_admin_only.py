"""P2-25 (owner decision 2026-09-29): Labs / Kalshi BTC is admin-only."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None

PAGE = '''
import runpy
import streamlit as st
st.page_link = lambda *a, **k: None
runpy.run_path(%r, run_name="__main__")
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class KalshiPageTests(unittest.TestCase):
    def render(self, db_admin, session_admin=False):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(PAGE % str(ROOT / "pages" / "kalshi.py"), default_timeout=60)
        at.session_state["username"] = "someone@example.com"
        at.session_state["is_admin"] = session_admin
        with mock.patch("db.users.is_admin_from_db", return_value=db_admin) as check, \
                mock.patch("ui.kalshi_scanner.render_kalshi_scanner") as scanner:
            at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at, scanner, check

    def test_non_admin_is_turned_away(self):
        at, scanner, _ = self.render(db_admin=False)
        self.assertIn("admin-only Labs tool", " ".join(i.value for i in at.info))
        scanner.assert_not_called()

    def test_session_flag_alone_is_not_enough(self):
        _, scanner, _ = self.render(db_admin=False, session_admin=True)
        scanner.assert_not_called()

    def test_admin_sees_the_monitor(self):
        at, scanner, check = self.render(db_admin=True)
        scanner.assert_called_once()
        check.assert_called_once_with("someone@example.com")
        self.assertNotIn("admin-only", " ".join(i.value for i in at.info))


@unittest.skipUnless(HAS_ST, "needs streamlit")
class NavTests(unittest.TestCase):
    def sections(self, is_admin):
        import streamlit as st

        from ui import nav

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
