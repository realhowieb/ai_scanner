"""Run 83B follow-up — Streamlit Cloud stale modules can't take the app down.

Production after the Run 83B deploy showed:
  "Startup problem: Import error: ImportError: cannot import name
   'ensure_active_watchlist_state' from 'ui.watchlists'"
The running process still held the pre-deploy ui.watchlists. These tests plant
stale modules and run the real app.py headlessly.
"""
import importlib.util
import sys
import types
import unittest

from ui import boot_recovery as br

HAS_ST = importlib.util.find_spec("streamlit") is not None


class RetryTests(unittest.TestCase):
    def test_first_party_stale_module_is_dropped_and_retried_boundedly(self):
        state = {}
        sys.modules["ui._stale_probe"] = types.ModuleType("ui._stale_probe")
        self.addCleanup(sys.modules.pop, "ui._stale_probe", None)
        self.assertTrue(br.retry_stale_import("ui._stale_probe", state))
        self.assertNotIn("ui._stale_probe", sys.modules)
        self.assertTrue(br.retry_stale_import("ui._stale_probe", state))
        self.assertTrue(br.retry_stale_import("ui._stale_probe", state))
        self.assertFalse(br.retry_stale_import("ui._stale_probe", state))   # bounded: real error surfaces
        br.clear_retries(state)
        self.assertEqual(state, {})

    def test_third_party_and_unknown_errors_are_not_retried(self):
        for name in ("pandas", "requests", None, "", "streamlit.runtime"):
            self.assertFalse(br.retry_stale_import(name, {}), name)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class StaleModuleAppTests(unittest.TestCase):
    def _run_with_stale(self, module_name, drop):
        import test_run83b_scanner_state as t

        real = sys.modules.get(module_name) or importlib.import_module(module_name)
        stale = types.ModuleType(module_name)
        stale.__dict__.update({k: v for k, v in vars(real).items() if k not in drop})
        sys.modules[module_name] = stale                           # pre-deploy copy, still cached
        self.addCleanup(sys.modules.__setitem__, module_name, real)
        return t.run_app(tier="premium")

    def test_stale_watchlists_without_the_new_helper_does_not_break_startup(self):
        # The exact production failure: the helper added in 83B is missing.
        at = self._run_with_stale("ui.watchlists", {"ensure_active_watchlist_state"})
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        text = " ".join(e.value for e in [*at.error, *at.warning])
        self.assertNotIn("Startup problem", text)
        self.assertIn("### HSF Opportunities", [m.value.strip() for m in at.markdown])

    def test_stale_module_missing_a_feature_import_self_heals(self):
        # A name the feature-import block needs is missing from the cached module:
        # the app drops it, reruns, and imports the file on disk.
        at = self._run_with_stale("ui.watchlists", {"render_watchlists_panel"})
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertIn("### HSF Opportunities", [m.value.strip() for m in at.markdown])
        self.assertTrue(hasattr(sys.modules["ui.watchlists"], "render_watchlists_panel"))


if __name__ == "__main__":
    unittest.main()
