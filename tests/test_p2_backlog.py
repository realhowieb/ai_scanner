"""P2 backlog — design pass, tour, saved screens, share links, home screen."""
import importlib.util
import re
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

try:  # the Streamlit app was retired; its modules are gone
    from ui import browser_prefs, discover, tour
except ImportError:
    browser_prefs = discover = tour = None

ROOT = Path(__file__).resolve().parents[1]
GATED_PAGES = ("alerts", "brief", "day_trader", "journal", "kalshi", "settings", "stock", "today", "watchlists")


class ShareAndHomeScreenTests(unittest.TestCase):
    def test_share_url(self):
        from ui.stock_intelligence import share_url

        with mock.patch("config.APP_BASE_URL", "https://example.test/"):
            self.assertEqual(share_url(" nvda "), "https://example.test/stocks/NVDA")


if __name__ == "__main__":
    unittest.main()
