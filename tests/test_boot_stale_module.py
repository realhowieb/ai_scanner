"""Streamlit Cloud redeploy: a stale cached first-party module must self-heal.

After a redeploy the process can still hold a pre-redeploy copy of a module
(e.g. ui.app_session) that lacks a name app.py imports. The boot guard must
drop that stale module and rerun, so the fresh file on disk is imported
instead of every visitor getting an ImportError.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None


class BootGuardSourceTests(unittest.TestCase):
    def test_guard_covers_stale_import_errors(self):
        src = (ROOT / "app.py").read_text()
        self.assertIn("except (KeyError, ImportError) as _boot_err:", src)
        self.assertIn("sys.modules.pop(_stale, None)", src)


SCRIPT = '''
import sys, time, types
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
time.sleep = lambda *_: None
if not st.session_state.get("_planted"):
    st.session_state["_planted"] = True
    stale = types.ModuleType("ui.app_session")      # pre-redeploy copy: missing names
    sys.modules["ui.app_session"] = stale
import runpy
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "app.py")


@unittest.skipUnless(HAS_ST, "needs streamlit")
class StaleModuleRecoveryTests(unittest.TestCase):
    def test_stale_module_is_replaced_on_rerun(self):
        import sys

        from streamlit.testing.v1 import AppTest

        real = sys.modules.get("ui.app_session")
        self.addCleanup(lambda: sys.modules.__setitem__("ui.app_session", real) if real else None)
        at = AppTest.from_string(SCRIPT, default_timeout=120)
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertTrue(hasattr(sys.modules["ui.app_session"], "should_land_on_today"))
        self.assertNotIn("_boot_import_retries", at.session_state)   # recovered, counter cleared


if __name__ == "__main__":
    unittest.main()
