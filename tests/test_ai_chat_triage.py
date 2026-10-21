"""Tests for the Q&A chat helper and admin error triage."""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class AskClaudeChatGuardTest(unittest.TestCase):
    def test_disabled_blocks_chat(self):
        from ui import ai
        with patch("config.AI_ENABLED", False), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            text, err = ai.ask_claude_chat(system="s", messages=[{"role": "user", "content": "hi"}])
        self.assertIsNone(text)
        self.assertIn("disabled", err)

    def test_empty_messages(self):
        from ui import ai
        with patch("config.AI_ENABLED", True), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            text, err = ai.ask_claude_chat(system="s", messages=[])
        self.assertIsNone(text)
        self.assertIn("No question", err)

    def test_over_limit_blocks_chat(self):
        from ui import ai
        with patch("config.AI_ENABLED", True), patch("config.ANTHROPIC_API_KEY", "sk-test"):
            with patch("config.AI_DAILY_LIMIT", 3):
                with patch("db.ai_usage.ai_calls_today", return_value=3):
                    text, err = ai.ask_claude_chat(
                        system="s",
                        messages=[{"role": "user", "content": "hi"}],
                        username="bob@x.com",
                    )
        self.assertIsNone(text)
        self.assertIn("usage limit", err)


if __name__ == "__main__":
    unittest.main()
