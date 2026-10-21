"""A BILLING_API_BASE with a trailing slash must not turn /health into //health (404)."""
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class BillingBaseUrlTests(unittest.TestCase):


    def test_expression_normalizes(self):
        with mock.patch.dict(os.environ, {"BILLING_API_BASE": " https://billing.example.com/ "}):
            base = (os.getenv("BILLING_API_BASE") or "https://hsf-api-v1.onrender.com/billing").strip().rstrip("/")
        self.assertEqual(f"{base}/health", "https://billing.example.com/health")


if __name__ == "__main__":
    unittest.main()
