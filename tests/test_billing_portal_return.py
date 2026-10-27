"""Returning from the Stripe portal shows the current plan instead of waiting for
an upgrade (P1-27 finding 2026-09-29: a portal downgrade to Free showed
"Activating your plan upgrade…" and then "Upgrade is still processing")."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True

SCRIPT = '''
from utils.st_compat import st
from ui.billing_return import handle_portal_return
st.query_params["portal"] = "return"
st.query_params["rt"] = "tok"
handle_portal_return("a@example.com", lambda u: st.session_state["db_tier"])
st.session_state["params_left"] = sorted(st.query_params.keys())
'''


class WiringTests(unittest.TestCase):
    def test_only_checkout_waits_for_an_upgrade(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertIn('if checkout_flag == "success" and "username" in st.session_state:\n'
                      '        _poll_for_tier_upgrade(', src)
        self.assertIn("handle_portal_return(st.session_state[\"username\"], _resolve_tier_key)", src)
        self.assertNotIn("if _stripe_return and \"username\" in st.session_state:", src)


if __name__ == "__main__":
    unittest.main()
