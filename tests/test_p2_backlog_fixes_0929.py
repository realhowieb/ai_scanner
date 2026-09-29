"""2026-09-29 small backlog fixes: P2-20, P2-40, P2-22, P2-34, P2-32, P2-23."""
import importlib.util
import smtplib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None
HAS_BCRYPT = importlib.util.find_spec("bcrypt") is not None


class FakeCookieManager:
    """Mimics streamlit-cookies-manager: deleting is a no-op when a prefix is set."""

    def __init__(self, sid):
        self.stored = {"ai_scanner" + "ai_scanner_sid": sid}
        self.queue = {}

    def get(self, k, default=None):
        if k in self.queue:
            return self.queue[k]
        return self.stored.get("ai_scanner" + k, default)

    def __setitem__(self, k, v):
        self.queue[k] = v

    def pop(self, k, default=None):  # the library's broken delete
        return self.get(k, default)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class P2_20_SessionCookieTests(unittest.TestCase):
    def test_clearing_blanks_the_cookie_so_the_next_visit_has_no_session(self):
        from ui import auth

        jar = FakeCookieManager("stale-sid")
        jar.pop(auth.COOKIE_NAME, None)
        self.assertEqual(jar.get(auth.COOKIE_NAME), "stale-sid")   # the old way left it
        auth._clear_session_cookie(jar)
        self.assertFalse(jar.get(auth.COOKIE_NAME))                # empty sid = no session, no notice

    def test_restore_and_logout_use_the_clear_helper(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertNotIn("cookies.pop(COOKIE_NAME", src)
        self.assertEqual(src.count("_clear_session_cookie(cookies)"), 2)


class P2_40_DeactivatedTests(unittest.TestCase):
    def test_plain_text_match(self):
        from db.inactive_login import password_matches

        self.assertTrue(password_matches("secret", ("secret", " secret ")))
        self.assertFalse(password_matches("secret", ("wrong", "")))
        self.assertFalse(password_matches(None, ("secret",)))

    @unittest.skipUnless(HAS_BCRYPT, "needs bcrypt")
    def test_bcrypt_match(self):
        import bcrypt

        from db.inactive_login import password_matches

        h = bcrypt.hashpw(b"pw123", bcrypt.gensalt(4)).decode()
        self.assertTrue(password_matches(h, ("pw123",)))
        self.assertFalse(password_matches(h, ("nope",)))

    def test_lookup_reads_inactive_rows_only(self):
        from db import inactive_login

        cur = mock.MagicMock()
        cur.fetchone.return_value = ("hash",)
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch("db.engine.get_neon_conn", return_value=conn):
            self.assertEqual(inactive_login.inactive_password(" A@X.com "), "hash")
        sql, params = cur.execute.call_args[0]
        self.assertIn("is_active = FALSE", sql)
        self.assertEqual(params, ("a@x.com",))
        conn.close.assert_called_once()

    def test_lookup_fails_closed(self):
        from db import inactive_login

        with mock.patch("db.engine.get_neon_conn", side_effect=RuntimeError("down")):
            self.assertIsNone(inactive_login.inactive_password("a@x.com"))
        self.assertIsNone(inactive_login.inactive_password(""))

    @unittest.skipUnless(HAS_ST, "needs streamlit")
    def test_message_only_after_the_right_password(self):
        from ui import auth

        with mock.patch("db.inactive_login.inactive_password", return_value="pw"):
            self.assertTrue(auth._deactivated_with_password("a@x.com", ("pw", "pw")))
            self.assertFalse(auth._deactivated_with_password("a@x.com", ("bad", "bad")))
        src = (ROOT / "ui" / "auth.py").read_text()
        i = src.index("This account is deactivated. Contact support.")
        self.assertLess(src.index("_deactivated_with_password(login_key"), i)
        self.assertLess(i, src.index('"User not found. Please use the email'))


class P2_22_LandingTests(unittest.TestCase):
    def test_score_sentence_and_methodology_link(self):
        from ui import landing, product_copy

        d = landing.details_html()
        self.assertIn(product_copy.HSF_SCORE_ONE_LINE, d)
        self.assertIn('href="/methodology" target="_self"', d)
        self.assertEqual(product_copy.find_prohibited_claims(product_copy.HSF_SCORE_ONE_LINE), [])


class P2_34_SmokeConcurrencyTests(unittest.TestCase):
    def test_one_run_per_branch_and_commit(self):
        src = (ROOT / ".github" / "workflows" / "smoke.yml").read_text()
        self.assertIn("group: smoke-${{ github.ref }}-${{ github.sha }}", src)
        self.assertIn("cancel-in-progress: true", src)
        self.assertLess(src.index("concurrency:"), src.index("jobs:"))


class P2_32_FailureReasonTests(unittest.TestCase):
    def setUp(self):
        from ui import email_failure

        self.ef = email_failure
        email_failure.clear()

    def test_classify(self):
        c = self.ef.classify
        self.assertEqual(c(smtplib.SMTPAuthenticationError(535, b"bad"))[0], "auth_failed")
        self.assertEqual(c(smtplib.SMTPSenderRefused(550, b"x", "a@b.c"))[0], "sender_rejected")
        r = c(smtplib.SMTPRecipientsRefused({"a@b.c": (550, b"test mode")}))
        self.assertEqual(r, ("recipient_rejected", "SMTPRecipientsRefused 550"))
        self.assertEqual(c(TimeoutError())[0], "network")
        self.assertEqual(c(smtplib.SMTPDataError(554, b"x"))[0], "provider_error")

    def test_reason_never_contains_the_address(self):
        reason = self.ef.classify(smtplib.SMTPRecipientsRefused({"cust@example.com": (550, b"cust@example.com")}))
        self.assertNotIn("cust@example.com", self.ef.describe(reason))

    def test_send_smtp_notes_missing_config_and_provider_errors(self):
        from ui import email_utils

        with mock.patch("config.SMTP_HOST", ""), mock.patch("config.SMTP_USER", "u"), \
                mock.patch("config.SMTP_PASS", ""), mock.patch("builtins.print"):
            self.assertFalse(email_utils._send_smtp("a@x.com", "s", "t", "h"))
        self.assertEqual(self.ef.last(), ("not_configured", "SMTP_HOST, SMTP_PASS"))

        server = mock.MagicMock()
        server.__enter__.return_value.sendmail.side_effect = smtplib.SMTPSenderRefused(550, b"x", "f")
        with mock.patch("config.SMTP_HOST", "h"), mock.patch("config.SMTP_USER", "u"), \
                mock.patch("config.SMTP_PASS", "p"), mock.patch("smtplib.SMTP", return_value=server), \
                mock.patch("builtins.print"), mock.patch.object(email_utils, "_capture"):
            self.assertFalse(email_utils._send_smtp("a@x.com", "s", "t", "h"))
        self.assertEqual(self.ef.last()[0], "sender_rejected")

    @unittest.skipUnless(HAS_ST, "needs streamlit")
    def test_only_admins_see_the_reason(self):
        import streamlit as st

        from ui import email_verification_gate as gate

        with mock.patch("db.email_verification.create_verification_token", return_value=None):
            self.assertFalse(gate._resend_verification("a@x.com"))
        with mock.patch.object(st, "session_state", {"is_admin": True}):
            self.assertIn("No account uses this address", gate.last_failure_for_admin())
        with mock.patch.object(st, "session_state", {}):
            self.assertIsNone(gate.last_failure_for_admin())


class P2_23_LifespanTests(unittest.TestCase):
    def test_billing_uses_lifespan_not_on_event(self):
        src = (ROOT / "billing_service" / "main.py").read_text()
        self.assertNotIn("@app.on_event", src)
        self.assertIn("app = FastAPI(lifespan=_lifespan)", src)
        self.assertLess(src.index("async def _lifespan"), src.index("app = FastAPI("))

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "needs fastapi")
    def test_lifespan_starts_the_alert_worker(self):
        import asyncio

        from tests.test_billing_service import _load_billing_module

        bm = _load_billing_module({"DATABASE_URL": "postgresql://u@h/db"}, db_reachable=True)
        with mock.patch.object(bm, "_start_realtime_alerts") as start:
            async def run():
                async with bm._lifespan(bm.app):
                    pass
            asyncio.run(run())
        start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
