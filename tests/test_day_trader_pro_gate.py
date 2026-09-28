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
HAS_ST = importlib.util.find_spec("streamlit") is not None

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


@unittest.skipUnless(HAS_ST, "needs streamlit")
class PageGateTests(unittest.TestCase):
    def render(self, state):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        for k, v in {"username": "u@example.com", **state}.items():
            at.session_state[k] = v
        with mock.patch("ui.day_trader.render_day_trader_panel") as panel, \
             mock.patch("ui.app_runtime._upgrade_button") as upgrade:
            at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        page_cta = [c for c in upgrade.call_args_list if c.args[2] == "upgrade_to_pro_day_trader"]
        return at, panel, page_cta

    def test_free_sees_upgrade_note_not_the_monitor(self):
        at, panel, upgrade = self.render({"tier_key": "basic", "entitlements": _flags("basic")})
        panel.assert_not_called()
        self.assertEqual(len(upgrade), 1)
        self.assertIn("Pro adds the live Day Trader monitor", at.info[0].value)

    def test_paid_tiers_and_admin_get_the_monitor(self):
        for state in ({"tier_key": "pro", "entitlements": _flags("pro")},
                      {"tier_key": "premium", "entitlements": _flags("premium")},
                      {"tier_key": "basic", "is_admin": True, "entitlements": _flags("basic", True)}):
            _, panel, upgrade = self.render(state)
            panel.assert_called_once()
            self.assertEqual(upgrade, [])

    def test_session_from_before_the_flag_falls_back_to_tier(self):
        old = {k: v for k, v in _flags("pro").items() if k != "can_day_trader"}
        _, panel, _ = self.render({"tier_key": "pro", "entitlements": old})
        panel.assert_called_once()
        old_free = {k: v for k, v in _flags("basic").items() if k != "can_day_trader"}
        _, panel, _ = self.render({"tier_key": "basic", "entitlements": old_free})
        panel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
