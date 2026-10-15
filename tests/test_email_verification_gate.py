"""Tests for the email-verification upgrade gate (fails closed on lookup outages)."""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock, patch

HAS_ST = importlib.util.find_spec("streamlit") is not None


class EmailIsVerifiedTest(unittest.TestCase):
    def test_passes_through_verified(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.email_verification_status", return_value=True):
            self.assertTrue(g.email_is_verified("u@x.com"))

    def test_passes_through_unverified(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.email_verification_status", return_value=False):
            self.assertFalse(g.email_is_verified("u@x.com"))

    def test_fails_closed_on_error(self):
        from ui import email_verification_gate as g
        with patch("db.email_verification.email_verification_status", side_effect=RuntimeError):
            self.assertFalse(g.email_is_verified("u@x.com"))
            with self.assertRaises(g.VerificationUnavailable):
                g.verification_status("u@x.com")


@unittest.skipUnless(HAS_ST, "needs streamlit")
class UpgradeGateTest(unittest.TestCase):
    def _gate(self, **status):
        import streamlit as st

        from ui import email_verification_gate as g
        with patch("db.email_verification.email_verification_status", **status), \
                patch.object(st, "warning") as warn, patch.object(st, "button", return_value=False) as button:
            allowed = g.require_verified_for_upgrade("u@x.com", key_suffix="t")
        return allowed, " ".join(str(c.args[0]) for c in warn.call_args_list), button.called

    def test_verified_may_upgrade(self):
        self.assertEqual(self._gate(return_value=True), (True, "", False))

    def test_unverified_is_told_to_verify_and_offered_a_resend(self):
        allowed, warning, offered_resend = self._gate(return_value=False)
        self.assertFalse(allowed)
        self.assertIn("verify your email", warning)
        self.assertTrue(offered_resend)

    def test_outage_blocks_with_a_temporary_message_not_unverified(self):
        allowed, warning, offered_resend = self._gate(side_effect=ConnectionError("db down"))
        self.assertFalse(allowed)
        self.assertIn("can't confirm your email verification right now", warning)
        self.assertNotIn("verify your email before", warning)
        self.assertFalse(offered_resend)


class DbVerificationStatusTest(unittest.TestCase):
    """db.email_verification tells "not verified" apart from "can't tell"."""

    def _conn(self, row=None, error=None):
        cur = MagicMock()
        cur.fetchone.return_value = row
        if error:
            cur.execute.side_effect = error
        conn = MagicMock()
        conn.cursor.return_value = cur
        return conn

    def _status(self, conn=None, conn_error=None):
        from db import email_verification as ev
        factory = MagicMock(return_value=conn, side_effect=conn_error)
        with patch.object(ev, "get_neon_conn", factory), patch.object(ev, "_ensure_schema"):
            try:
                return ev.email_verification_status("U@X.com"), ev.is_email_verified("U@X.com")
            except ev.VerificationUnavailable:
                return "unavailable", ev.is_email_verified("U@X.com")

    def test_verified_and_unverified(self):
        self.assertEqual(self._status(self._conn(row=(True,))), (True, True))
        self.assertEqual(self._status(self._conn(row=(False,))), (False, False))
        self.assertEqual(self._status(self._conn(row={"email_verified": True})), (True, True))

    def test_unknown_account_is_not_verified(self):
        self.assertEqual(self._status(self._conn(row=None)), (False, False))

    def test_outages_are_unavailable_and_never_verified(self):
        self.assertEqual(self._status(None), ("unavailable", False))                      # no connection
        self.assertEqual(self._status(conn_error=OSError("refused")), ("unavailable", False))
        self.assertEqual(self._status(self._conn(error=RuntimeError("query failed"))), ("unavailable", False))

    def test_no_driver_is_unavailable(self):
        from db import email_verification as ev
        with patch.object(ev, "get_neon_conn", None):
            with self.assertRaises(ev.VerificationUnavailable):
                ev.email_verification_status("u@x.com")
            self.assertFalse(ev.is_email_verified("u@x.com"))


if __name__ == "__main__":
    unittest.main()
