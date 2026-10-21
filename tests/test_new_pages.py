"""New sidebar pages: login-gated and wired to the right existing panels."""
from __future__ import annotations

import unittest
from pathlib import Path


class NewPagesTests(unittest.TestCase):
    def _src(self, name: str) -> str:
        return Path(f"pages/{name}").read_text()


    def test_custom_nav_excludes_auth_utility_pages(self):
        from ui.nav import _NAV

        paths = [p for p, _l, _i in _NAV]
        self.assertIn("pages/settings.py", paths)
        self.assertNotIn("pages/reset_password.py", paths)  # reachable via URL, not nav
        self.assertNotIn("pages/verify_email.py", paths)

    def test_custom_nav_follows_hsf_product_flow(self):
        # P1-1 plus the admin-only operational destination.
        from ui.nav import _NAV, _NAV_SECTIONS

        self.assertEqual([s for s, _items in _NAV_SECTIONS],
                         ["Today", "Discover", "Research", "My Stocks", "Account", "Admin", "Labs"])
        labels = [label for _p, label, _i in _NAV[:6]]
        self.assertEqual(labels, ["Today", "Market Brief", "Scanner", "Custom Scan", "Day Trader",
                                  "Stock Intelligence"])
        self.assertIn("My Stocks", [label for _p, label, _i in _NAV])
        self.assertNotIn("pages/alerts.py", [p for p, _l, _i in _NAV])   # alerts live in My Stocks


if __name__ == "__main__":
    unittest.main()
