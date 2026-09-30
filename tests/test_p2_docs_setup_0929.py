"""P2-24 secrets template, P2-38 local setup script, P2-43 failover/restore doc."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SecretsTemplateTests(unittest.TestCase):
    TEXT = (ROOT / ".streamlit" / "secrets.toml.example").read_text()

    def test_lists_every_secret_the_app_reads(self):
        for key in ("COOKIE_PASSWORD", "APP_ENCRYPTION_KEY", "APP_BASE_URL", "ANTHROPIC_API_KEY",
                    "ANTHROPIC_MODEL", "AI_ENABLED", "SMTP_HOST", "SMTP_PORT", "SMTP_USER",
                    "SMTP_PASS", "SMTP_FROM", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY",
                    "STRIPE_SECRET_KEY", "database_url"):
            self.assertIn(key, self.TEXT)
        self.assertIn("Sending access", self.TEXT)

    def test_only_placeholders(self):
        for line in self.TEXT.splitlines():
            m = re.match(r'\s*#?\s*([A-Za-z_]+)\s*=\s*"([^"]*)"', line)
            if not m:
                continue
            value = m.group(2)
            self.assertTrue(
                value == "" or value.endswith("...") or "your-app" in value or "user:password@host" in value
                or value in ("smtp.resend.com", "587", "resend", "alerts@ai.hsfinest.com", "HSF Alerts",
                             "0", "1", "25"),
                f"{m.group(1)} looks like a real value",
            )


class SetupScriptTests(unittest.TestCase):
    def test_script_uses_the_lock_minus_linux_only_wheels(self):
        src = (ROOT / "scripts" / "setup_local_env.sh").read_text()
        self.assertIn("requirements.lock", src)
        self.assertIn("^(nvidia-|triton==)", src)
        self.assertIn("pip check", src)
        self.assertNotIn("> requirements.lock", src)   # never rewrites the lock
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.13")

    def test_lock_matches_python_version(self):
        self.assertIn("Python 3.13", (ROOT / "requirements.lock").read_text().splitlines()[0])


class DevBranchTests(unittest.TestCase):
    """P2-37: the reset script can only reset dev-local, from live, after a prompt."""

    def test_reset_script_is_pinned_to_dev_local(self):
        src = (ROOT / "scripts" / "reset_dev_branch.sh").read_text()
        self.assertIn('BRANCH="dev-local"', src)
        self.assertIn('"$parent_name" != "live"', src)
        self.assertIn("read -r -p", src)
        self.assertIn('branches reset "$BRANCH" --parent', src)
        self.assertNotIn("falling-snow", src)          # project id comes from the environment

    def test_local_secrets_are_never_committed(self):
        self.assertIn(".streamlit/secrets.toml", (ROOT / ".gitignore").read_text())


class FailoverDocTests(unittest.TestCase):
    def test_covers_providers_and_safe_restore(self):
        doc = (ROOT / "docs" / "PROVIDER_FAILOVER_AND_RESTORE.md").read_text()
        for section in ("Alpaca", "Resend", "cron-job.org", "Render", "Streamlit", "Anthropic", "Stripe",
                        "Neon: restoring the live database"):
            self.assertIn(section, doc)
        self.assertIn('Never use "Reset from parent" on `live`', doc)
        self.assertIn("--preserve-under-name", doc)
        self.assertNotRegex(doc, r"br-[a-z]+-[a-z]+-[a-z0-9]{8}")   # no Neon ids in the public repo
        self.assertNotRegex(doc, r"ep-[a-z]+-[a-z]+-[a-z0-9]{8}")


if __name__ == "__main__":
    unittest.main()
