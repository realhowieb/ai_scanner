"""The API must boot even when the retired UI dependencies are unavailable."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "needs API runtime dependencies")
class ApiWithoutUITests(unittest.TestCase):
    def test_app_and_shared_scanner_helpers_import_without_ui_packages(self):
        # A fresh interpreter prevents the developer environment's installed UI
        # packages, or another test's module cache, from hiding a missing dependency.
        code = '''
import importlib.abc
import sys

blocked = {"streamlit", "streamlit_authenticator", "streamlit_autorefresh",
           "streamlit_cookies_manager", "extra_streamlit_components",
           "plotly", "altair", "pyarrow", "PIL"}

class NoUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoUI())
from api.main import create_app
from api.settings import Settings
import api.today
import ui.stock_intelligence
import ui.market_scans
import scan.engine

app = create_app(Settings(jwt_secret="test-" + "x" * 40,
                          access_ttl_s=900, refresh_ttl_s=3600, cors_origins=()))
assert "/v1/today" in app.openapi()["paths"]
assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
