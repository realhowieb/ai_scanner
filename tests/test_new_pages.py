"""New sidebar pages: login-gated and wired to the right existing panels."""
from __future__ import annotations

import unittest
from pathlib import Path


class NewPagesTests(unittest.TestCase):
    def _src(self, name: str) -> str:
        return Path(f"pages/{name}").read_text()

    def test_all_new_pages_login_gated(self):
        for name in ("journal.py", "watchlists.py", "settings.py"):
            src = self._src(name)
            self.assertIn('st.session_state.get("username")', src, name)
            self.assertIn("st.stop()", src, name)
            self.assertIn('st.page_link("app.py"', src, name)

    def test_journal_reuses_panel(self):
        src = self._src("journal.py")
        self.assertIn("render_journal_panel", src)
        self.assertIn("list_trades", src)          # empty-state check

    def test_watchlists_reuses_panel(self):
        self.assertIn("render_watchlists_panel", self._src("watchlists.py"))

    def test_settings_surfaces_account_and_connections(self):
        src = self._src("settings.py")
        self.assertIn("render_connect_panel", src)
        self.assertIn("pages/billing.py", src)
        self.assertIn("tier_key", src)


if __name__ == "__main__":
    unittest.main()
