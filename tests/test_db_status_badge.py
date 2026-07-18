"""The DB status badge is admin-gated; status is still computed for everyone."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(_STREAMLIT, "db_status imports streamlit")
class DbStatusBadgeTests(unittest.TestCase):
    def _run(self, show_badge):
        import ui.db_status as d

        fake_sidebar = mock.MagicMock()
        with mock.patch.object(d, "get_db_status", return_value="neon"), \
             mock.patch.object(d.st, "sidebar", fake_sidebar):
            status = d.render_db_status_badge(show_badge=show_badge)
        return status, fake_sidebar

    def test_non_admin_gets_status_but_no_badge(self):
        status, sidebar = self._run(show_badge=False)
        self.assertEqual(status, "neon")           # still computed
        sidebar.markdown.assert_not_called()       # but not shown

    def test_admin_sees_badge(self):
        status, sidebar = self._run(show_badge=True)
        self.assertEqual(status, "neon")
        sidebar.markdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
