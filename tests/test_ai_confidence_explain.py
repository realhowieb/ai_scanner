import unittest
from unittest.mock import patch

import ui.ai_confidence_explain as ex


class AiConfidenceExplainTests(unittest.TestCase):
    def test_confidence_tier_thresholds(self):
        self.assertEqual(ex.confidence_tier(45), "High")
        self.assertEqual(ex.confidence_tier(30), "Medium")
        self.assertEqual(ex.confidence_tier(10), "Low")
        self.assertEqual(ex.confidence_tier(None), "unknown")
        self.assertEqual(ex.confidence_tier("nan-ish"), "unknown")

    def test_explain_returns_error_when_no_features(self):
        with (
            patch("ui.ai.is_configured", return_value=True),
            patch("ui.ai.ask_claude") as ask,
        ):
            text, err = ex.explain_confidence({"Ticker": "AMD"}, username="u")
        self.assertIsNone(text)
        self.assertIsNotNone(err)
        self.assertFalse(ask.called)

    def test_explain_grounds_prompt_and_never_invents(self):
        captured = {}

        def fake_ask(*, system, user, max_tokens, username=None, feature=None):
            captured["system"] = system
            captured["user"] = user
            return ("AMD scores high on strong relative volume.", None)

        row = {
            "Ticker": "AMD",
            "AI Confidence": 45.0,
            "VolRel20": 2.4,
            "Trend10D%": 3.1,
            "BreakoutScore": 7.0,
        }
        with (
            patch("ui.ai.is_configured", return_value=True),
            patch("ui.ai.ask_claude", side_effect=fake_ask),
        ):
            text, err = ex.explain_confidence(row, earnings_days=2, username="u")

        self.assertIsNone(err)
        self.assertTrue(text.startswith("AMD"))
        # Grounded on real values + real catalyst, and told not to fabricate.
        self.assertIn("relative volume", captured["user"])
        self.assertIn("earnings in 2", captured["user"])
        self.assertIn("45.0% (tier: High)", captured["user"])
        self.assertIn("Do NOT invent news", captured["system"])

    def test_explain_graceful_when_ai_off(self):
        with patch("ui.ai.is_configured", return_value=False):
            text, err = ex.explain_confidence({"Ticker": "AMD", "VolRel20": 2.4})
        self.assertIsNone(text)
        self.assertEqual(err, "AI is not configured.")


if __name__ == "__main__":
    unittest.main()
