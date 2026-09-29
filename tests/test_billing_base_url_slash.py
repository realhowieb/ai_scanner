"""A BILLING_API_BASE with a trailing slash must not turn /health into //health (404)."""
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class BillingBaseUrlTests(unittest.TestCase):
    def test_every_reader_strips_the_trailing_slash(self):
        for rel in ("pages/billing.py", "ui/admin_results_tab.py", "ui/email_setup_view.py"):
            src = (ROOT / rel).read_text()
            self.assertIn('"https://ai-scanner-h2c8.onrender.com").strip().rstrip("/")', src, rel)

    def test_not_ready_message_names_the_url_it_called(self):
        src = (ROOT / "pages" / "billing.py").read_text()
        self.assertIn("(HTTP {r.status_code} from {BILLING_API_BASE}/health)", src)

    def test_expression_normalizes(self):
        with mock.patch.dict(os.environ, {"BILLING_API_BASE": " https://billing.example.com/ "}):
            base = (os.getenv("BILLING_API_BASE") or "https://ai-scanner-h2c8.onrender.com").strip().rstrip("/")
        self.assertEqual(f"{base}/health", "https://billing.example.com/health")


if __name__ == "__main__":
    unittest.main()
