"""Customer email addresses never appear unmasked in log output.

Scheduled jobs run in GitHub Actions on a public repository, so anything they
print is public.
"""
import io
import re
import smtplib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADDR = "sample.customer@gmail.com"


class MaskingTests(unittest.TestCase):
    def test_mask_email(self):
        from ui.log_privacy import mask_email

        self.assertEqual(mask_email(ADDR), "sa***@gmail.com")
        self.assertEqual(mask_email("ab@x.io"), "a***@x.io")
        self.assertEqual(mask_email("howard"), "ho***")
        self.assertEqual(mask_email(""), "")
        self.assertEqual(mask_email(None), "")

    def test_redact_masks_every_address_in_text(self):
        from ui.log_privacy import redact

        text = f"{{'{ADDR}': (550, b'bad')}} and Jane.Doe+x@Example.co.uk"
        out = redact(text)
        self.assertNotIn(ADDR, out)
        self.assertNotIn("Jane.Doe+x", out)
        self.assertIn("sa***@gmail.com", out)
        self.assertIn("Ja***@Example.co.uk", out)

    def test_billing_service_copy_matches(self):
        from ui.log_privacy import redact

        src = (ROOT / "billing_service" / "realtime_alerts.py").read_text()
        ns: dict = {}
        exec(compile("import re\nfrom typing import Any\n" + re.search(
            r"_EMAIL_RE = .*?\n\n\ndef _redact.*?\n    \)\n", src, re.S).group(0), "x", "exec"), ns)
        sample = f"to {ADDR}: refused ab@x.io"
        self.assertEqual(ns["_redact"](sample), redact(sample))


class SendFailureLogTests(unittest.TestCase):
    def test_smtp_failure_log_has_no_address(self):
        import ui.email_utils as eu

        server = mock.MagicMock()
        server.__enter__.return_value = server
        server.sendmail.side_effect = smtplib.SMTPRecipientsRefused({ADDR: (550, b"no")})
        with mock.patch("config.SMTP_HOST", "smtp.test"), mock.patch("config.SMTP_USER", "u"), \
             mock.patch("config.SMTP_PASS", "p"), mock.patch("smtplib.SMTP", return_value=server), \
             mock.patch.object(eu, "_capture"), redirect_stdout(io.StringIO()) as out:
            self.assertFalse(eu.send_digest_email(ADDR, "s", "h", "t"))
        self.assertIn("SEND FAILED to sa***@gmail.com", out.getvalue())
        self.assertNotIn(ADDR, out.getvalue())

    def test_success_log_has_no_address(self):
        import ui.email_utils as eu

        server = mock.MagicMock()
        server.__enter__.return_value = server
        with mock.patch("config.SMTP_HOST", "smtp.test"), mock.patch("config.SMTP_USER", "u"), \
             mock.patch("config.SMTP_PASS", "p"), mock.patch("smtplib.SMTP", return_value=server), \
             redirect_stdout(io.StringIO()) as out:
            self.assertTrue(eu.send_alert_email(ADDR, "s", "b"))
        self.assertIn("sa***@gmail.com", out.getvalue())
        self.assertNotIn(ADDR, out.getvalue())


class SourceSweepTests(unittest.TestCase):
    FILES = ("ui/email_utils.py", "scheduler/morning_digest.py", "scheduler/evening_wrap.py",
             "scheduler/alert_runner.py", "billing_service/realtime_alerts.py", "telemetry.py")
    RAW = re.compile(r"(print|_log)\(f?[\"'].*\{(to_address|email|user_id|username)(!r)?\}")

    def test_user_data_job_errors_are_redacted(self):
        # Errors from cron jobs that read user rows (alerts, emails, credential
        # purges) can carry row values, so they must go through redaction.
        cron = (ROOT / "scheduler" / "cron_runner.py").read_text()
        for label in ("alert evaluation", "intelligence alert evaluation", "evening wrap",
                      "morning digest", "login purge", "credential purge"):
            self.assertIn(f'print(f"[cron] {label} failed: {{_redact(e)}}")', cron)
        runner = (ROOT / "scheduler" / "alert_runner.py").read_text()
        self.assertIn('print(f"[alert_runner] could not load alerts: {redact(e)}")', runner)

    def test_no_log_line_prints_a_raw_address(self):
        for rel in self.FILES:
            for n, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
                self.assertIsNone(self.RAW.search(line), f"{rel}:{n}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
