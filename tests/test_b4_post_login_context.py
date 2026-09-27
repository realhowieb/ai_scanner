"""B4 — the first page after sign-in must already have the account's context.

Manual testing: every account type (Basic, Pro, Premium, Admin) signed in,
landed on Today with the wrong plan / locked features, and only became correct
after visiting Scanner. Root cause: app.py's main() redirected to Today (and to
a shared-link destination) BEFORE resolving tier, admin status and entitlements
into the session, so the first page rendered with that context missing; Scanner
is app.py itself and wrote it on the next visit.

These tests run the real app.py headlessly as a freshly signed-in user and
capture the session at the moment of the redirect — no Scanner visit, no rerun.
"""
import importlib.util
import unittest

HAS_ST = importlib.util.find_spec("streamlit") is not None

EXPECTED = {
    # tier: (tier_key, is_admin, entitlement that must be on, entitlement that must be off)
    "basic": ("basic", False, "can_scan_sp500", "can_email_alerts"),
    "pro": ("pro", False, "can_email_alerts", "can_ai_notes"),
    "premium": ("premium", False, "can_ai_notes", "can_admin_panel"),
    "admin": ("admin", True, "can_admin_panel", None),
}

CAPTURE = '''
import streamlit as _st
def _capture_switch(page, *a, **k):
    keys = ("username", "tier_key", "is_admin", "entitlements", "hsf_today_landed_for")
    _st.session_state["_redirect"] = {"page": page, **{k: _st.session_state.get(k) for k in keys}}
    _st.stop()
_st.switch_page = _capture_switch
'''


def sign_in_and_capture(tier, *, user="tester@example.com", session=None):
    """Fresh sign-in (identity just established, no account context yet)."""
    import importlib

    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from test_run83b_scanner_state import PATCHED, SCRIPT

    saved = []
    for mod_name, attrs in PATCHED.items():
        mod = importlib.import_module(mod_name)
        saved += [(mod, a, getattr(mod, a)) for a in attrs if hasattr(mod, a)]
    tiering = importlib.import_module("auth.tiering")
    saved.append((tiering, "ADMIN_USERS", tiering.ADMIN_USERS))
    if tier == "admin":                      # admins come from ADMIN_USERS (production secrets)
        tiering.ADMIN_USERS = set(tiering.ADMIN_USERS) | {user}
    try:
        # The 83B script pins the user to tester@example.com; B4 sessions start
        # without the "already landed on Today" marker so the real redirect runs.
        script = CAPTURE + SCRIPT.replace('USER = "tester@example.com"', "")
        at = AppTest.from_string(script, default_timeout=180)
        at.session_state["_cfg"] = {"tier": tier, "watch": ["WATCH1"]}
        at.session_state["username"] = user          # what every sign-in path sets
        for k, v in (session or {}).items():
            at.session_state[k] = v
        at.run()
        return at
    finally:
        for mod, a, value in saved:
            setattr(mod, a, value)
        st.cache_data.clear()


@unittest.skipUnless(HAS_ST, "needs streamlit")
class FirstRenderContextTests(unittest.TestCase):
    def assert_ready(self, at, tier, user="tester@example.com"):
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        red = at.session_state["_redirect"]
        tier_key, is_admin, on, off = EXPECTED[tier]
        self.assertEqual(red["username"], user, tier)
        self.assertEqual(red["tier_key"], tier_key, (tier, red))           # not "Basic" by default
        self.assertEqual(bool(red["is_admin"]), is_admin, tier)
        ent = red["entitlements"] or {}
        self.assertTrue(ent.get(on), (tier, on, ent))                     # entitlements resolved
        if off:
            self.assertFalse(ent.get(off), (tier, off))                    # and not elevated
        return red

    def test_every_account_type_is_ready_when_today_first_renders(self):
        for tier in ("basic", "pro", "premium", "admin"):
            red = self.assert_ready(sign_in_and_capture(tier), tier)
            self.assertEqual(red["page"], "pages/today.py", tier)

    def test_shared_link_destination_also_gets_the_ready_context(self):
        for tier in ("basic", "premium"):
            at = sign_in_and_capture(tier, session={"hsf_after_login_page": "pages/stock.py"})
            red = self.assert_ready(at, tier)
            self.assertEqual(red["page"], "pages/stock.py", tier)          # deep link still wins

    def test_account_transitions_render_the_new_account_first(self):
        from ui.app_session import ACCOUNT_OWNER_KEY, _owner_tag, clear_account_session_state

        pairs = [("basic", "pro"), ("basic", "premium"), ("basic", "admin"),
                 ("admin", "basic"), ("premium", "basic"), ("pro", "admin")]
        for prev, new in pairs:
            prev_flags = {"can_admin_panel": prev == "admin", "can_ai_notes": prev in ("premium", "admin")}
            session = {ACCOUNT_OWNER_KEY: _owner_tag("previous@example.com"),
                       "tier_key": prev, "is_admin": prev == "admin", "entitlements": prev_flags}
            clear_account_session_state(session)                             # what logout does
            session[ACCOUNT_OWNER_KEY] = _owner_tag("previous@example.com")
            at = sign_in_and_capture(new, session=session)
            self.assert_ready(at, new)


if __name__ == "__main__":
    unittest.main()
