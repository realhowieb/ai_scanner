"""Tests for ui/ai_summary.py — the Claude-backed scan summary."""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock, patch

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_summary requires pandas")
class GenerateScanSummaryTest(unittest.TestCase):
    def _df(self):
        import pandas as pd
        return pd.DataFrame(
            [{"Ticker": "AAPL", "BreakoutScore": 9.1, "Gap%": 3.2, "VolRel20": 2.4}]
        )

    def test_empty_df_returns_error(self):
        import pandas as pd

        from ui.ai_summary import generate_scan_summary
        summary, err = generate_scan_summary(pd.DataFrame())
        self.assertIsNone(summary)
        self.assertIn("No scan results", err)

    def test_missing_api_key_returns_error(self):
        from ui import ai_summary
        with patch("config.ANTHROPIC_API_KEY", None):
            summary, err = ai_summary.generate_scan_summary(self._df())
        self.assertIsNone(summary)
        self.assertIn("not configured", err)

    def test_successful_summary(self):
        from ui import ai_summary
        fake_block = MagicMock()
        fake_block.type = "text"
        fake_block.text = "AAPL is the strongest setup."
        fake_resp = MagicMock()
        fake_resp.content = [fake_block]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_resp

        fake_anthropic = MagicMock()
        fake_anthropic.Anthropic.return_value = fake_client

        with patch("config.ANTHROPIC_API_KEY", "sk-test"):
            with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
                summary, err = ai_summary.generate_scan_summary(self._df())

        self.assertIsNone(err)
        self.assertIn("AAPL", summary)

    def test_fingerprint_is_stable(self):
        from ui.ai_summary import _results_fingerprint
        df = self._df()
        self.assertEqual(_results_fingerprint(df), _results_fingerprint(df.copy()))


if __name__ == "__main__":
    unittest.main()
