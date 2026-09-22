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

    def test_render_contexts_do_not_collide_on_keys(self):
        # Regression: rendering the same df from two call sites in one run must
        # not raise StreamlitDuplicateElementKey (button keys namespaced by
        # context). Emulate Streamlit's per-run key registry.
        from ui import ai_summary

        seen_keys: set[str] = set()

        class _DupKey(Exception):
            pass

        fake_st = MagicMock()

        def _button(_label, key=None, **_kw):
            if key in seen_keys:
                raise _DupKey(key)
            seen_keys.add(key)
            return False  # not clicked

        fake_st.button.side_effect = _button
        fake_st.session_state = {}

        df = self._df()
        with patch.dict("sys.modules", {"streamlit": fake_st}):
            ai_summary.render_ai_summary(df, context="latest_results")
            ai_summary.render_ai_summary(df, context="three_step")  # must not raise

        gen_keys = [k for k in seen_keys if k.startswith("gen_")]
        self.assertEqual(len(gen_keys), 2)  # two distinct generate-button keys


if __name__ == "__main__":
    unittest.main()
