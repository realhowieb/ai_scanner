"""Sub-pages restore login from the cookie (deep-link / refresh no longer 401s)."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class _SS(dict):
    pass


@unittest.skipUnless(_STREAMLIT, "ui.auth imports streamlit")
class RestoreSessionFromCookieTests(unittest.TestCase):
    def test_returns_existing_session_without_cookie_lookup(self):
        import ui.auth as a

        a.st.session_state = _SS({"username": "bob@x.com"})
        with mock.patch.object(a, "_cookies_ready_or_stop") as ready:
            self.assertEqual(a.restore_session_from_cookie(), "bob@x.com")
            ready.assert_not_called()  # short-circuits when already logged in

    def test_restores_username_and_tier_from_cookie(self):
        import ui.auth as a

        a.st.session_state = _SS()
        cookies = mock.MagicMock()
        cookies.get.return_value = "sid123"
        with mock.patch.object(a, "_cookies_ready_or_stop", return_value=cookies), \
             mock.patch.object(a, "_get_username_for_session", return_value="Alice@X.com"), \
             mock.patch.object(a, "_resolve_tier_key", return_value="pro"):
            u = a.restore_session_from_cookie()
        self.assertEqual(u, "alice@x.com")             # normalized
        self.assertEqual(a.st.session_state["tier"], "pro")
        self.assertEqual(a.st.session_state["plan"], "pro")

    def test_no_cookie_returns_none(self):
        import ui.auth as a

        a.st.session_state = _SS()
        cookies = mock.MagicMock()
        cookies.get.return_value = None
        with mock.patch.object(a, "_cookies_ready_or_stop", return_value=cookies):
            self.assertIsNone(a.restore_session_from_cookie())

    def test_expired_session_id_returns_none(self):
        import ui.auth as a

        a.st.session_state = _SS()
        cookies = mock.MagicMock()
        cookies.get.return_value = "stale-sid"
        with mock.patch.object(a, "_cookies_ready_or_stop", return_value=cookies), \
             mock.patch.object(a, "_get_username_for_session", return_value=None):
            self.assertIsNone(a.restore_session_from_cookie())
        self.assertNotIn("username", a.st.session_state)


if __name__ == "__main__":
    unittest.main()
