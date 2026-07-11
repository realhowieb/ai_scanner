import unittest
from unittest.mock import MagicMock, patch

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in minimal envs
    pd = None


@unittest.skipIf(pd is None, "pandas is required for price utility tests")
class PriceUtilsTests(unittest.TestCase):
    def test_normalize_price_frame_canonicalizes_columns_and_types(self):
        from data.price_utils import normalize_price_frame

        df = pd.DataFrame(
            {
                "open": ["10.1"],
                "HIGH": ["11.2"],
                "low": ["9.9"],
                "close": ["10.5"],
                "adj_close": ["10.4"],
                "vol": ["1000"],
            }
        )

        out = normalize_price_frame(df)

        self.assertIn("Open", out.columns)
        self.assertIn("High", out.columns)
        self.assertIn("Low", out.columns)
        self.assertIn("Close", out.columns)
        self.assertIn("Adj Close", out.columns)
        self.assertIn("Volume", out.columns)
        self.assertAlmostEqual(float(out.loc[0, "Open"]), 10.1, places=5)
        self.assertEqual(int(out.loc[0, "Volume"]), 1000)

    def test_frame_fingerprint_changes_with_price_data(self):
        from data.price_utils import frame_fingerprint

        first = pd.DataFrame({"Open": [1.0], "Close": [2.0], "Volume": [100]})
        second = pd.DataFrame({"Open": [1.0], "Close": [3.0], "Volume": [100]})

        self.assertIsNotNone(frame_fingerprint(first))
        self.assertNotEqual(frame_fingerprint(first), frame_fingerprint(second))

    def test_alpaca_config_reads_runtime_env_names(self):
        import data.price_alpaca as price_alpaca
        import data.prices as prices

        with patch.object(price_alpaca, "requests", object()):
            with patch.dict(
                "os.environ",
                {
                    "ALPACA_API_KEY_ID": "key-id",
                    "ALPACA_API_SECRET_KEY": "secret-key",
                    "ALPACA_DATA_URL": "https://example.test/",
                },
                clear=False,
            ):
                cfg = prices._get_alpaca_config()

        self.assertEqual(cfg["api_key"], "key-id")
        self.assertEqual(cfg["api_secret"], "secret-key")
        self.assertEqual(cfg["data_url"], "https://example.test")

    def test_alpaca_config_ignores_missing_streamlit_secrets(self):
        # Secret resolution now lives in data.alpaca_config; a raising secrets
        # object (no secrets file, e.g. the cron) must resolve to the default.
        from data.alpaca_config import alpaca_secret

        class MissingSecrets:
            def get(self, _key):
                raise RuntimeError("No secrets found")

        fake_st = type("FakeSt", (), {"secrets": MissingSecrets()})
        with patch.dict("os.environ", {}, clear=True), patch.dict(
            "sys.modules", {"streamlit": fake_st}
        ):
            self.assertIsNone(alpaca_secret("ALPACA_API_KEY_ID"))

    def test_alpaca_only_mode_skips_yfinance_when_alpaca_returns_no_bars(self):
        import data.prices as prices

        fake_yf = MagicMock()
        prices.clear_price_cache()

        cfg = prices.PriceFetchConfig(tickers=["DAY", "FI", "K", "MMC"])
        with (
            patch.object(prices, "_get_alpaca_config", return_value={"api_key": "key"}),
            patch.object(prices, "_download_multi_alpaca", return_value={}),
            patch.object(prices, "_yf", fake_yf),
            patch.dict("os.environ", {"PRICE_SKIP_YF_FALLBACK": "1"}, clear=False),
        ):
            data, skipped = prices._download_batch(["DAY", "FI", "K", "MMC"], cfg)

        self.assertEqual(data, {})
        fake_yf.Ticker.assert_not_called()
        self.assertEqual(
            skipped,
            [
                ("DAY", "skipped_yf_fallback"),
                ("FI", "skipped_yf_fallback"),
                ("K", "skipped_yf_fallback"),
                ("MMC", "skipped_yf_fallback"),
            ],
        )
        prices.clear_price_cache()


if __name__ == "__main__":
    unittest.main()
