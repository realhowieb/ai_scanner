""""New since your last visit" survives page refreshes and fresh sign-ins.

Each refresh or sign-in is a new Streamlit session. The marker used to advance
to the latest scan at the start of every session, so any second session in the
same scan window compared the latest scan with itself and showed nothing.
"""
import sys
import types
import unittest
from unittest import mock

from ui import last_visit
from ui.last_visit import next_marker


class NextMarkerTests(unittest.TestCase):
    def test_first_visit_marks_nothing_and_records_seen(self):
        self.assertEqual(next_marker(None, 100), (None, "100:"))

    def test_newer_scan_compares_against_the_last_seen_one(self):
        self.assertEqual(next_marker("100:", 105), (100, "105:100"))
        self.assertEqual(next_marker("100:90", 105), (100, "105:100"))

    def test_refresh_in_the_same_scan_window_keeps_the_baseline(self):
        base, value = next_marker("105:100", 105)
        self.assertEqual((base, value), (100, "105:100"))
        self.assertEqual(next_marker(value, 105), (100, "105:100"))    # and again

    def test_legacy_single_value_cookie(self):
        self.assertEqual(next_marker("100", 105), (100, "105:100"))
        self.assertEqual(next_marker("100", 100), (None, "100:"))      # no baseline known yet

    def test_runs_unavailable_changes_nothing(self):
        self.assertEqual(next_marker("105:100", None), (100, "105:100"))
        self.assertEqual(next_marker(None, None), (None, ""))

    def test_garbage_cookie_is_treated_as_first_visit(self):
        self.assertEqual(next_marker("abc:x", 7), (None, "7:"))


class SessionFlowTests(unittest.TestCase):
    """Simulate sessions against a fake cookie jar."""

    def session(self, jar, latest):
        fake_st = mock.MagicMock()
        fake_st.session_state = {}
        fake_sessions = types.SimpleNamespace(cookies_ready_or_stop=lambda: jar, save_cookies=lambda c: None)
        with mock.patch.object(last_visit, "st", fake_st), \
             mock.patch.object(last_visit, "_latest_run_id", return_value=latest), \
             mock.patch.dict(sys.modules, {"ui.auth_sessions": fake_sessions}):
            return last_visit.baseline_run_id()

    def test_refresh_and_relogin_keep_showing_new_names(self):
        jar = {}
        self.assertIsNone(self.session(jar, 100))      # first visit
        self.assertEqual(self.session(jar, 105), 100)  # next visit after a new scan
        self.assertEqual(self.session(jar, 105), 100)  # refresh: same baseline (was None before the fix)
        self.assertEqual(self.session(jar, 105), 100)  # sign in again: same
        self.assertEqual(self.session(jar, 110), 105)  # next scan: moves forward
        self.assertEqual(jar[last_visit.COOKIE_KEY], "110:105")


if __name__ == "__main__":
    unittest.main()
