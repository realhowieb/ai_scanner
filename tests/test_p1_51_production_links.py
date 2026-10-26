"""P1-51: links default to the production web app (app.hsfinest.com), not Streamlit.

The scheduled digest/wrap/alert emails are built by the GitHub job, which sets
no APP_BASE_URL, so the code default is what their links use.
"""
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROD = "https://app.hsfinest.com"


class ProductionLinkTests(unittest.TestCase):
    def test_no_code_defaults_to_beta(self):
        for rel in ("config.py", "ui/stock_intelligence.py", "db/email_prefs.py",
                    "billing_service/realtime_alerts.py", "billing_service/main.py"):
            src = (ROOT / rel).read_text()
            self.assertNotIn("hsf-beta.streamlit.app", src, rel)
            self.assertNotIn("hsfinestai.streamlit.app", src.replace("# e.g. https://hsfinestai.streamlit.app", ""), rel)
        self.assertIn(f'_get("APP_BASE_URL", "{PROD}")', (ROOT / "config.py").read_text())

    def test_config_default_without_setting(self):
        import importlib

        import config

        env = {k: v for k, v in os.environ.items() if k != "APP_BASE_URL"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(importlib.reload(config).APP_BASE_URL, PROD)
        importlib.reload(config)

    def test_setting_still_overrides(self):
        import importlib

        import config

        with mock.patch.dict(os.environ, {"APP_BASE_URL": "https://hsf-beta.streamlit.app"}):
            self.assertEqual(importlib.reload(config).APP_BASE_URL, "https://hsf-beta.streamlit.app")
        importlib.reload(config)

    def test_share_url_matches_the_app(self):
        from ui import stock_intelligence as si

        with mock.patch("config.APP_BASE_URL", PROD):
            self.assertEqual(si.share_url("nvda"), f"{PROD}/stocks/NVDA")
        with mock.patch("config.APP_BASE_URL", "https://hsfinestai.streamlit.app"):
            self.assertEqual(si.share_url("nvda"), "https://hsfinestai.streamlit.app/stock?ticker=NVDA")


if __name__ == "__main__":
    unittest.main()
