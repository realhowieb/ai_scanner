"""Returning from the Stripe portal shows the current plan instead of waiting for
an upgrade (P1-27 finding 2026-09-29: a portal downgrade to Free showed
"Activating your plan upgrade…" and then "Upgrade is still processing")."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None

SCRIPT = '''
import streamlit as st
from ui.billing_return import handle_portal_return
st.query_params["portal"] = "return"
st.query_params["rt"] = "tok"
handle_portal_return("a@example.com", lambda u: st.session_state["db_tier"])
st.session_state["params_left"] = sorted(st.query_params.keys())
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class PortalReturnTests(unittest.TestCase):
    def run_with(self, db_tier, session_tier):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=30)
        at.session_state["db_tier"] = db_tier
        at.session_state["tier"] = session_tier
        at.session_state["_tier_poll_attempt"] = 3
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at

    def test_downgrade_to_free_is_shown_at_once(self):
        at = self.run_with("basic", "premium")
        self.assertEqual(at.session_state["tier"], "basic")
        text = " ".join(i.value for i in at.info)
        self.assertIn("Your plan: **Free**", text)
        self.assertNotIn("upgrade", text.lower())
        self.assertEqual(at.session_state["params_left"], [])
        self.assertNotIn("_tier_poll_attempt", at.session_state)
        self.assertEqual(len(at.spinner) if hasattr(at, "spinner") else 0, 0)

    def test_switch_to_premium(self):
        at = self.run_with("premium", "pro")
        self.assertEqual(at.session_state["plan"], "premium")
        self.assertIn("Your plan: **Premium**", " ".join(i.value for i in at.info))

    def test_database_unavailable_keeps_the_session_plan(self):
        at = self.run_with(None, "pro")
        self.assertEqual(at.session_state["tier"], "pro")


class WiringTests(unittest.TestCase):
    def test_only_checkout_waits_for_an_upgrade(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertIn('if checkout_flag == "success" and "username" in st.session_state:\n'
                      '        _poll_for_tier_upgrade(', src)
        self.assertIn("handle_portal_return(st.session_state[\"username\"], _resolve_tier_key)", src)
        self.assertNotIn("if _stripe_return and \"username\" in st.session_state:", src)


if __name__ == "__main__":
    unittest.main()
