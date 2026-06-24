import importlib.util
import unittest
from unittest.mock import patch

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from scan.engine import run_breakout_scan


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for scan engine provider tests")
class ScanEngineFakeProviderTests(unittest.TestCase):
    def _ohlcv(
        self,
        *,
        start_close=10.0,
        final_open=11.0,
        final_close=12.0,
        final_high=12.0,
        base_volume=1000,
        final_volume=2500,
        days=25,
    ):
        dates = pd.date_range("2026-01-01", periods=days, freq="D")
        closes = [start_close + i * 0.05 for i in range(days - 1)] + [final_close]
        highs = [close + 0.1 for close in closes]
        highs[-1] = final_high
        opens = list(closes)
        opens[-1] = final_open
        volumes = [base_volume] * (days - 1) + [final_volume]
        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": [close - 0.1 for close in closes],
                "Close": closes,
                "Volume": volumes,
            },
            index=dates,
        )

    def test_run_breakout_scan_completes_with_fake_provider_and_skips(self):
        requested_symbols = []

        def fake_parallel(tickers, **_kwargs):
            requested_symbols.extend(tickers)
            return (
                {
                    "BOOM": self._ohlcv(),
                    "SPY": self._ohlcv(
                        start_close=400.0,
                        final_open=405.0,
                        final_close=406.0,
                        final_high=406.0,
                        base_volume=10_000_000,
                        final_volume=11_000_000,
                    ),
                },
                [("MISS", "empty_single")],
            )

        with (
            patch("scan.engine.st.session_state", {"show_scan_progress": False}),
            patch("data.prices.fetch_price_data_parallel", side_effect=fake_parallel) as parallel_fetch,
            patch("data.prices.fetch_price_data_batch", side_effect=AssertionError("batch fallback should not run")),
        ):
            result = run_breakout_scan(
                ["BOOM", "MISS"],
                premarket=False,
                afterhours=False,
                unusual_volume=False,
                min_gap=0.0,
                min_price=1.0,
                max_price=100.0,
                top_n=10,
                diagnostics=False,
                use_cache=False,
            )

        self.assertEqual(parallel_fetch.call_count, 1)
        self.assertEqual(set(requested_symbols), {"BOOM", "MISS", "SPY"})
        self.assertEqual(list(result["Ticker"]), ["BOOM"])
        self.assertTrue(bool(result.iloc[0]["IsBreakout"]))
        self.assertIsNotNone(result.iloc[0]["RSvsSPY"])


if __name__ == "__main__":
    unittest.main()
