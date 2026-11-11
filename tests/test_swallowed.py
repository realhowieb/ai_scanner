"""utils.swallowed: ignored errors leave one redacted warning per call site, never raise."""
import logging
import unittest
from unittest import mock

from utils import swallowed as sw


class SwallowedTests(unittest.TestCase):
    def setUp(self):
        sw.reset()
        self.addCleanup(sw.reset)

    def test_logs_once_per_site_then_counts_the_rest(self):
        with self.assertLogs("hsf.swallowed", level=logging.WARNING) as logs:
            sw.swallowed("api.x.f", ValueError("boom"))
            sw.swallowed("api.x.f", ValueError("boom"))
            sw.swallowed("api.x.g", KeyError("k"))
        self.assertEqual(len(logs.output), 2)
        self.assertIn("api.x.f: ValueError: boom", logs.output[0])
        with mock.patch.object(sw.time, "monotonic", return_value=sw.time.monotonic() + sw.INTERVAL_S + 1):
            with self.assertLogs("hsf.swallowed", level=logging.WARNING) as logs:
                sw.swallowed("api.x.f", ValueError("again"))
        self.assertIn("(+1 more since last note)", logs.output[0])

    def test_redacts_emails_and_url_passwords(self):
        with self.assertLogs("hsf.swallowed", level=logging.WARNING) as logs:
            sw.swallowed("a", RuntimeError("send to jane.doe@example.com via postgresql://u:secret@host/db failed"))
        self.assertNotIn("jane.doe", logs.output[0])
        self.assertNotIn("secret", logs.output[0])

    def test_never_raises(self):
        class Bad(Exception):
            def __str__(self):
                raise RuntimeError("no str")

        sw.swallowed("a", Bad())
        sw.swallowed("b", None)


if __name__ == "__main__":
    unittest.main()
