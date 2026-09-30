"""P1-39 — Admin email setup per environment, never exposing secrets."""
import importlib.util
import json
import unittest
from unittest import mock

HAS_ST = importlib.util.find_spec("streamlit") is not None
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

    def test_latest_job_setup_picks_newest_with_settings(self):
        from ui.email_setup_view import latest_job_setup

        runs = [{"job": "alerts", "at": "t3", "stats": {"fired": 1}},                     # recorded before P1-39
                {"job": "digest", "at": "t2", "stats": {"smtp": {"status": "OK", "sender_masked": "al***@x"}}},
                {"job": "digest", "at": "t1", "stats": {"smtp": {"status": "WARNING"}}}]
        got = latest_job_setup(runs)
        self.assertEqual((got["status"], got["at"], got["job"]), ("OK", "t2", "digest"))
        self.assertIsNone(latest_job_setup([{"job": "alerts", "at": "t", "stats": {}}]))

    def test_billing_health_block_to_status(self):
        from ui.email_setup_view import billing_setup

        self.assertEqual(billing_setup({"email": {"configured": True, "sender_domain": "ai.hsfinest.com"}})["status"], "OK")
        self.assertEqual(billing_setup({"email": {"configured": True, "sender_domain": "ai-scanner.app"}})["status"],
                         "WARNING")
        self.assertEqual(billing_setup({"email": {"configured": False, "missing": ["SMTP_FROM"]}})["status"], "MISSING")
        self.assertEqual(billing_setup({"ok": True})["status"], "UNKNOWN")                # older billing version


SCRIPT = '''
import streamlit as st
from ui.email_setup_view import render_email_setup
render_email_setup()
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CardTests(unittest.TestCase):
    def render(self, username="me@example.com", runs=None, sent=True, billing=None):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["username"] = username
        patches = [mock.patch("config.SMTP_HOST", "smtp.resend.com"), mock.patch("config.SMTP_USER", SECRETS[1]),
                   mock.patch("config.SMTP_PASS", SECRETS[0]), mock.patch("config.SMTP_FROM", "alerts@ai.hsfinest.com"),
                   mock.patch("db.email_job_runs.recent_email_runs", return_value=runs or []),
                   mock.patch("ui.email_utils.send_alert_email", return_value=sent),
                   mock.patch("ui.email_setup_view._fetch_billing_health",
                              return_value=billing or {"email": {"configured": True,
                                                                 "sender_domain": "ai.hsfinest.com"}})]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at

    def text(self, at):
        return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)

    def test_rows_and_no_secrets(self):
        runs = [{"job": "digest", "at": "2026-09-29 12:35:00", "stats": {"smtp": {
            "status": "OK", "display_name": "HSF Alerts", "sender_masked": "al***@ai.hsfinest.com"}}}]
        at = self.render(runs=runs)
        at.button(key="admin_email_setup_billing").click().run()
        t = self.text(at)
        self.assertIn("Website (Streamlit Cloud)", t)
        self.assertIn("Scheduled emails (GitHub Actions)", t)
        self.assertIn("as of the digest run at 2026-09-29 12:35", t)
        self.assertIn("Live alerts (Render billing)", t)
        self.assertIn("🟢", t)
        for secret in SECRETS + ("alerts@ai.hsfinest.com",):
            self.assertNotIn(secret, t)

    def test_test_email_goes_only_to_the_signed_in_admin(self):
        at = self.render()
        with mock.patch("ui.email_utils.send_alert_email", return_value=True) as send:
            at.button(key="admin_email_setup_test").click().run()
        self.assertEqual(send.call_args.args[0], "me@example.com")
        self.assertIn("Sent.", at.success[0].value)

    def test_failed_test_email_says_so(self):
        at = self.render()
        with mock.patch("ui.email_utils.send_alert_email", return_value=False):
            at.button(key="admin_email_setup_test").click().run()
        self.assertIn("refused", at.error[0].value)

    def test_no_job_run_yet(self):
        self.assertIn("no run recorded yet", self.text(self.render()))

    def test_admin_without_email_username_gets_no_button(self):
        at = self.render(username="howard")
        self.assertNotIn("admin_email_setup_test", [b.key for b in at.button])


if __name__ == "__main__":
    unittest.main()
