"""P1-39 — Admin email setup per environment, never exposing secrets."""
import importlib.util
import json
import unittest
from unittest import mock

HAS_ST = True
SECRETS = ("re_live_secret_key", "resend-user-xyz")


class DescribeTests(unittest.TestCase):
    def test_verified_domain_is_ok_and_secrets_never_appear(self):
        from ui.email_setup import describe_smtp, explain

        d = describe_smtp("smtp.resend.com", SECRETS[1], SECRETS[0], "alerts@ai.hsfinest.com")
        self.assertEqual(d["status"], "OK")
        self.assertEqual(d["sender_masked"], "al***@ai.hsfinest.com")
        self.assertEqual(d["display_name"], "HSF Alerts")
        dumped = json.dumps(d) + explain(d)
        for secret in SECRETS + ("alerts@ai.hsfinest.com",):
            self.assertNotIn(secret, dumped)

    def test_unverified_default_sender_warns(self):
        from ui.email_setup import describe_smtp, explain

        d = describe_smtp("smtp.resend.com", "u", "p", "noreply@ai-scanner.app")   # config.py default
        self.assertEqual(d["status"], "WARNING")
        self.assertIn("Resend will reject", explain(d))

    def test_missing_settings(self):
        from ui.email_setup import describe_smtp, explain

        d = describe_smtp("smtp.resend.com", "u", "", "")
        self.assertEqual(d["status"], "MISSING")
        self.assertEqual(d["missing"], ["SMTP_PASS", "SMTP_FROM"])
        self.assertIn("will not send", explain(d))

    def test_display_name_sources(self):
        from ui.email_setup import describe_smtp

        self.assertEqual(describe_smtp("h", "u", "p", "HSF Desk <a@ai.hsfinest.com>")["display_name"], "HSF Desk")
        self.assertEqual(describe_smtp("h", "u", "p", "a@ai.hsfinest.com", "Team")["display_name"], "Team")


class JobRecordTests(unittest.TestCase):
    def test_email_jobs_record_their_settings_summary(self):
        import scheduler.morning_digest as md

        with mock.patch("config.SMTP_HOST", "smtp.resend.com"), mock.patch("config.SMTP_USER", SECRETS[1]), \
             mock.patch("config.SMTP_PASS", SECRETS[0]), mock.patch("config.SMTP_FROM", "alerts@ai.hsfinest.com"), \
             mock.patch("db.email_job_runs.record_email_run") as rec:
            md.record_email_job("digest", {"sent": 1, "skipped": {}})
        stats = rec.call_args.args[1]
        self.assertEqual(stats["sent"], 1)
        self.assertEqual(stats["smtp"]["status"], "OK")
        for secret in SECRETS:
            self.assertNotIn(secret, json.dumps(stats))


SCRIPT = '''
from utils.st_compat import st
from ui.email_setup_view import render_email_setup
render_email_setup()
'''


if __name__ == "__main__":
    unittest.main()
