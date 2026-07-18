"""Billing resolves the account email so username-login users can still upgrade."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class _SS(dict):
    pass


@unittest.skipUnless(_STREAMLIT, "pages.billing imports streamlit")
class LoggedInEmailTests(unittest.TestCase):
    def test_email_identifier_returned_as_is(self):
        import pages.billing as b

        b.st.session_state = _SS({"username": "alice@x.com"})
        with mock.patch("db.users.find_username_by_display_name") as f:
            self.assertEqual(b._logged_in_email(), "alice@x.com")
            f.assert_not_called()  # already an email, no DB lookup

    def test_display_name_resolves_to_email_and_caches(self):
        import pages.billing as b

        b.st.session_state = _SS({"username": "pro user 3"})
        with mock.patch("db.users.find_username_by_display_name",
                        return_value="pro3@x.com"):
            self.assertEqual(b._logged_in_email(), "pro3@x.com")
        self.assertEqual(b.st.session_state.get("_billing_email"), "pro3@x.com")

    def test_unresolvable_identifier_falls_back(self):
        import pages.billing as b

        b.st.session_state = _SS({"username": "ghost"})
        with mock.patch("db.users.find_username_by_display_name", return_value=None):
            self.assertEqual(b._logged_in_email(), "ghost")  # still gated on '@'

    def test_empty_identifier(self):
        import pages.billing as b

        b.st.session_state = _SS()
        self.assertEqual(b._logged_in_email(), "")


if __name__ == "__main__":
    unittest.main()
