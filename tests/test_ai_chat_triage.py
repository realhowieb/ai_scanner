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


@unittest.skipUnless(_PANDAS_AVAILABLE, "ai_insights requires pandas")
class TriageTest(unittest.TestCase):
    def test_no_rows(self):
        from ui.ai_insights import triage_scan_errors
        text, err = triage_scan_errors([])
        self.assertIsNone(text)
        self.assertIn("No errors", err)

    def test_triage_calls_claude(self):
        from ui import ai_insights
        rows = [("2026-06-25", "scan", "u@x.com", 5, "TimeoutError", "provider timed out")]
        with patch("ui.ai.ask_claude", return_value=("Likely provider timeouts.", None)):
            text, err = ai_insights.triage_scan_errors(rows)
        self.assertIsNone(err)
        self.assertIn("timeout", text.lower())


if __name__ == "__main__":
    unittest.main()
