"""Tests for ui/ai.py cost guardrails: kill switch, daily cap, timeout."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class KillSwitchTest(unittest.TestCase):
    def test_disabled_blocks_call(self):
        from ui import ai
        with patch("config.AI_ENABLED", False):
            with patch("config.ANTHROPIC_API_KEY", "sk-test"):
                text, err = ai.ask_claude(system="s", user="u")
        self.assertIsNone(text)
        self.assertIn("disabled", err)

    def test_is_configured_requires_enabled_and_key(self):
        from ui import ai
        with patch("config.AI_ENABLED", True), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            self.assertTrue(ai.is_configured())
        with patch("config.AI_ENABLED", False), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            self.assertFalse(ai.is_configured())
        with patch("config.AI_ENABLED", True), patch("config.ANTHROPIC_API_KEY", None):
            self.assertFalse(ai.is_configured())


class DailyLimitTest(unittest.TestCase):
    def test_over_limit_blocks_call(self):
        from ui import ai
        with patch("config.AI_ENABLED", True), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            with patch("config.AI_DAILY_LIMIT", 5):
                with patch("db.ai_usage.ai_calls_today", return_value=5):
                    text, err = ai.ask_claude(system="s", user="u", username="bob@x.com")
        self.assertIsNone(text)
        self.assertIn("usage limit", err)

    def test_zero_limit_means_unlimited(self):
        from ui import ai
        # AI_DAILY_LIMIT=0 -> never over limit regardless of count
        with patch("config.AI_DAILY_LIMIT", 0):
            with patch("db.ai_usage.ai_calls_today", return_value=9999):
                self.assertFalse(ai._over_daily_limit("bob@x.com"))

    def test_no_username_skips_limit(self):
        from ui import ai
        with patch("config.AI_DAILY_LIMIT", 1):
            self.assertFalse(ai._over_daily_limit(None))


if __name__ == "__main__":
    unittest.main()
