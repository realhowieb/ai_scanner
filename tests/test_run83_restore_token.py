"""Run 83 — single-use, purpose-bound restore/billing tokens (ui/auth_tokens).

Runs in the lightweight CI env (no streamlit, no Postgres, no network): the real
SQL executes against an in-memory SQLite stand-in.
"""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _token_db import TokenDB

from ui import auth_tokens as at

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)


class RestoreTokenTests(unittest.TestCase):
    def setUp(self):
        self.db = TokenDB()
        self.addCleanup(self.db.shutdown)

    def issue(self, user="alice@example.com", purpose="restore", now=T0):
        return at.issue_token(user, purpose, conn=self.db, now=now)

    def consume(self, token, purpose="restore", now=T0 + timedelta(minutes=1)):
        return at.consume_token(token, purpose, conn=self.db, now=now)

    def test_valid_token_restores_exactly_once(self):
        tok = self.issue()
        self.assertEqual(self.consume(tok), "alice@example.com")
        self.assertIsNone(self.consume(tok))                       # replay rejected
        self.assertEqual(self.db.rows(), [])                        # invalidated, not just ignored

    def test_expired_token_rejected(self):
        tok = self.issue()
        self.assertIsNone(self.consume(tok, now=T0 + at.TTL["restore"] + timedelta(seconds=1)))

    def test_invalid_and_empty_tokens_rejected(self):
        self.issue()
        for bad in ("", "   ", "not-a-token", "x" * 300, None):
            self.assertIsNone(self.consume(bad))
        self.assertEqual(len(self.db.rows()), 1)                    # a failed attempt consumes nothing

    def test_wrong_purpose_rejected_and_not_consumed(self):
        billing = self.issue(purpose="billing")
        self.assertIsNone(self.consume(billing, purpose="restore"))  # billing token can't sign you in
        self.assertEqual(self.consume(billing, purpose="billing"), "alice@example.com")
        restore = self.issue()
        self.assertIsNone(self.consume(restore, purpose="billing"))  # restore token can't open billing
        self.assertEqual(self.consume(restore), "alice@example.com")

    def test_token_is_bound_to_its_own_user(self):
        a, b = self.issue("alice@example.com"), self.issue("bob@example.com")
        self.assertEqual(self.consume(b), "bob@example.com")
        self.assertEqual(self.consume(a), "alice@example.com")

    def test_tokens_are_unpredictable_and_stored_hashed(self):
        toks = {self.issue() for _ in range(20)}
        self.assertEqual(len(toks), 20)
        self.assertTrue(all(len(t) >= 40 for t in toks))
        stored = {h for _, _, h in self.db.rows()}
        self.assertFalse(stored & toks)                             # raw tokens never stored
        self.assertEqual(stored, {at.token_hash(t) for t in toks})

    def test_unknown_purpose_and_blank_user_are_refused(self):
        self.assertIsNone(at.issue_token("alice@example.com", "login", conn=self.db))
        self.assertIsNone(at.issue_token("  ", "restore", conn=self.db))

    def test_database_unavailable_fails_closed(self):
        class Broken:
            def cursor(self):
                raise RuntimeError("db down")
        self.assertIsNone(at.issue_token("alice@example.com", "restore", conn=Broken()))
        self.assertIsNone(at.consume_token("anything", "restore", conn=Broken()))


class AppCheckoutCallTests(unittest.TestCase):
    """ui/checkout.py: the app proves the account to the billing service."""

    def _run(self, tokens):
        import json
        from unittest import mock

        import ui.checkout as co

        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"checkout_url": "https://checkout.test/s"}).encode()

        def fake_urlopen(req, timeout=None):
            seen["headers"] = {k.lower(): v for k, v in req.header_items()}
            seen["body"] = json.loads(req.data)
            return Resp()

        it = iter(tokens)
        with mock.patch("ui.auth_tokens.issue_token", side_effect=lambda u, p, **k: next(it)), \
             mock.patch("config.BILLING_API_BASE", "https://billing.test"), \
             mock.patch("config.APP_BASE_URL", "https://app.test"), \
             mock.patch.object(co.urllib.request, "urlopen", side_effect=fake_urlopen) as uo:
            url, err = co.create_checkout_url("alice@example.com", "pro")
        return url, err, seen, uo

    def test_checkout_sends_account_token_and_distinct_restore_tokens(self):
        url, err, seen, _ = self._run(["BILLTOK", "RT_SUCCESS", "RT_PORTAL"])
        self.assertEqual((url, err), ("https://checkout.test/s", None))
        self.assertEqual(seen["headers"]["x-hsf-auth"], "BILLTOK")
        self.assertTrue(seen["body"]["success_url"].endswith("rt=RT_SUCCESS"))
        self.assertTrue(seen["body"]["return_url"].endswith("rt=RT_PORTAL"))

    def test_checkout_fails_closed_without_an_account_token(self):
        url, err, _, uo = self._run([None])
        self.assertIsNone(url)
        self.assertIn("verify your account", err)
        uo.assert_not_called()                                      # billing service never contacted


class WiringTests(unittest.TestCase):
    def test_stripe_return_links_use_single_use_restore_tokens(self):
        checkout = (ROOT / "ui" / "checkout.py").read_text()
        billing = (ROOT / "pages" / "billing.py").read_text()
        auth = (ROOT / "ui" / "auth.py").read_text()
        for src in (checkout, billing):
            self.assertIn('issue_token(', src)
            self.assertNotIn("create_session", src)                 # no reusable login session in URLs
        self.assertIn('consume_token(rt, "restore")', auth)
        self.assertNotIn("_get_username_for_session(rt)", auth)
        self.assertIn('success_rt = issue_token(username, "restore")', checkout)   # one token per link
        self.assertIn('portal_rt = issue_token(username, "restore")', checkout)

    def test_billing_page_mints_a_fresh_token_for_every_attempt(self):
        billing = (ROOT / "pages" / "billing.py").read_text()
        loop = billing[billing.index("for attempt in range(POST_RETRIES + 1):"):]
        loop = loop[: loop.index("raise RuntimeError(\n        \"Billing service is taking")]
        self.assertIn("auth = billing_auth_headers(username)", loop)
        self.assertIn("headers=auth", loop)
        self.assertIn('_billing_post_json("/create-portal-session", body, username=email)', billing)

    def test_tokens_are_never_logged(self):
        for rel in ("ui/auth_tokens.py", "ui/checkout.py", "pages/billing.py", "billing_service/main.py"):
            src = (ROOT / rel).read_text()
            for bad in ("print(token", "log.info(token", "%s\", token", "print(rt", "print(sid"):
                self.assertNotIn(bad, src, rel)


if __name__ == "__main__":
    unittest.main()
