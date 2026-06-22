import unittest

from data.provider_diagnostics import (
    classify_skip_reason,
    format_skip_examples,
    summarize_provider_skips,
)


class ProviderDiagnosticsTests(unittest.TestCase):
    def test_classifies_common_provider_failures(self):
        cases = {
            "yf_not_installed": "provider_missing",
            "error_download:RateLimit:429 Too Many Requests": "rate_limited",
            "error_download:TimeoutError:timed out": "timeout",
            "empty_single": "empty_response",
            "duplicate_frame_like:AAPL": "duplicate_data",
            "error_normalize:TypeError:bad frame": "invalid_data",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(classify_skip_reason(reason), expected)

    def test_summary_marks_zero_return_provider_failure_as_severe(self):
        summary = summarize_provider_skips(
            requested=2,
            returned=0,
            skipped=[("AAPL", "empty_single"), ("MSFT", "missing_final")],
        )

        self.assertTrue(summary.severe)
        self.assertEqual(summary.categories["empty_response"], 2)
        self.assertIn("No price data returned", summary.message)

    def test_summary_reports_partial_success(self):
        summary = summarize_provider_skips(
            requested=3,
            returned=2,
            skipped=[("BAD", "error_download:TimeoutError")],
        )

        self.assertFalse(summary.severe)
        self.assertIn("2/3", summary.message)
        self.assertEqual(summary.categories["timeout"], 1)

    def test_formats_skip_examples(self):
        examples = format_skip_examples(
            [("AAPL", "empty_single"), ("MSFT", "missing_final")],
            limit=1,
        )

        self.assertEqual(examples, "AAPL: empty_single")


if __name__ == "__main__":
    unittest.main()
