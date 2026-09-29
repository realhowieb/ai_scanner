"""P1-39 — the billing service's /health reports its email settings without secrets.

Runs in the billing-contract CI job (FastAPI installed); skipped elsewhere.
"""
import importlib.util
import json
import unittest

from tests.test_billing_service import _load_billing_module

_FASTAPI = importlib.util.find_spec("fastapi") is not None
FULL = {"APP_SUCCESS_URL": "https://x/ok", "APP_CANCEL_URL": "https://x/cancel",
        "DATABASE_URL": "postgresql://u@h/db", "SMTP_HOST": "smtp.resend.com",
        "SMTP_USER": "resend-user-xyz", "SMTP_PASS": "re_live_secret_key"}


@unittest.skipUnless(_FASTAPI, "needs fastapi")
class BillingEmailHealthTests(unittest.TestCase):
    def health(self, env):
        bm = _load_billing_module(env, db_reachable=True)
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, env):
            body = bm.health()
        return body if isinstance(body, dict) else json.loads(body.body)

    def test_verified_sender_reported_by_domain_only(self):
        body = self.health({**FULL, "SMTP_FROM": "alerts@ai.hsfinest.com"})
        self.assertEqual(body["email"], {"configured": True, "missing": [], "sender_domain": "ai.hsfinest.com"})
        dumped = json.dumps(body)
        for secret in ("re_live_secret_key", "resend-user-xyz", "alerts@ai.hsfinest.com"):
            self.assertNotIn(secret, dumped)

    def test_missing_sender_is_reported(self):
        body = self.health({**FULL, "SMTP_FROM": ""})
        self.assertFalse(body["email"]["configured"])
        self.assertEqual(body["email"]["missing"], ["SMTP_FROM"])


if __name__ == "__main__":
    unittest.main()
