"""Tests for the soft email-verification upgrade gate."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class EmailIsVerifiedTest(unittest.TestCase):
    def test_passes_through_verified(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.is_email_verified", return_value=True):
            self.assertTrue(g.email_is_verified("u@x.com"))

    def test_passes_through_unverified(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.is_email_verified", return_value=False):
            self.assertFalse(g.email_is_verified("u@x.com"))

    def test_fail_open_on_error(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.is_email_verified", side_effect=RuntimeError):
            self.assertTrue(g.email_is_verified("u@x.com"))


if __name__ == "__main__":
    unittest.main()
