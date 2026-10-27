"""Day Trader is a Pro feature (FEATURE_MIN_TIER["can_day_trader"] = "pro").

Free accounts see an upgrade note instead of the live monitor; Pro, Premium and
Admin get the monitor. Sessions whose entitlements predate the flag fall back to
the resolved tier, so a redeploy can't lock Pro users out.
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True

SCRIPT = '''
import runpy, streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "pages" / "day_trader.py")


def _flags(tier, is_admin=False):
    from ui.app_session import compute_entitlements

    return dict(compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier), is_admin=is_admin))


class EntitlementTests(unittest.TestCase):
    def test_day_trader_is_pro_and_up(self):
        from ui.app_session import FEATURE_MIN_TIER

        self.assertEqual(FEATURE_MIN_TIER["can_day_trader"], "pro")
        self.assertFalse(_flags("basic")["can_day_trader"])
        for tier in ("pro", "premium"):
            self.assertTrue(_flags(tier)["can_day_trader"], tier)
        self.assertTrue(_flags("basic", is_admin=True)["can_day_trader"])

    def test_pricing_and_upgrade_copy(self):
        from ui.pricing import included, upgrade_message

        self.assertFalse(included("can_day_trader", "basic"))
        self.assertTrue(included("can_day_trader", "pro"))
        self.assertIn("Pro adds the live Day Trader monitor", upgrade_message("can_day_trader"))


if __name__ == "__main__":
    unittest.main()
