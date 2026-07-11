"""Tests for the single-source Alpaca config resolver (env-first, headless-safe)."""
import unittest
from unittest import mock

from data.alpaca_config import (
    DEFAULT_BASE_URL,
    DEFAULT_DATA_URL,
    alpaca_secret,
    get_alpaca_config,
    get_alpaca_headers,
)

_CREDS = {
    "ALPACA_API_KEY_ID": "test-key",
    "ALPACA_API_SECRET_KEY": "test-secret",
}


class AlpacaConfigTests(unittest.TestCase):
    def test_env_is_resolved_headless(self):
        """Env vars alone (the cron context) must fully configure Alpaca."""
        with mock.patch.dict("os.environ", _CREDS, clear=False):
            cfg = get_alpaca_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["api_key"], "test-key")
        self.assertEqual(cfg["data_url"], DEFAULT_DATA_URL)
        self.assertEqual(cfg["base_url"], DEFAULT_BASE_URL)

    def test_missing_credentials_return_none(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(get_alpaca_config())
            self.assertIsNone(get_alpaca_headers())

    def test_custom_urls_normalized_without_trailing_slash(self):
        env = dict(_CREDS, ALPACA_DATA_URL="https://example.test/data/")
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = get_alpaca_config()
        self.assertEqual(cfg["data_url"], "https://example.test/data")

    def test_headers_shape(self):
        with mock.patch.dict("os.environ", _CREDS, clear=False):
            headers = get_alpaca_headers()
        self.assertEqual(headers["APCA-API-KEY-ID"], "test-key")
        self.assertEqual(headers["APCA-API-SECRET-KEY"], "test-secret")
        self.assertEqual(headers["Accept"], "application/json")

    def test_env_wins_over_default(self):
        with mock.patch.dict(
            "os.environ", {"ALPACA_DATA_URL": "https://env.test"}, clear=False
        ):
            self.assertEqual(alpaca_secret("ALPACA_DATA_URL", "fallback"), "https://env.test")

    def test_default_when_unset(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(alpaca_secret("ALPACA_DATA_URL", "fallback"), "fallback")

    def test_all_readers_share_the_resolver(self):
        """The historical duplicate readers must delegate, not re-implement."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in ("market_data.py", "data/price_alpaca.py", "ui/scan_providers.py"):
            source = (root / rel).read_text()
            self.assertIn("data.alpaca_config", source, f"{rel} must delegate")
            # No independent secret reads left behind.
            self.assertNotIn('st.secrets["ALPACA_API_KEY_ID"]', source, rel)
            self.assertNotIn('os.getenv("ALPACA_API_KEY_ID")', source, rel)


if __name__ == "__main__":
    unittest.main()
