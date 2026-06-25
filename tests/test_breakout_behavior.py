import importlib.util
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from scan.breakout import run_breakout_scan


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for breakout behavior tests")
class BreakoutBehaviorTests(unittest.TestCase):
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

    def _scan(self, price_data, **overrides):
        params = {
            "price_data": price_data,
            "spy_df": None,
            "premarket": False,
            "afterhours": False,
            "unusual_volume": False,
            "min_gap": 0.0,
            "min_price": 1.0,
            "max_price": 1000.0,
            "top_n": 20,
            "diagnostics": False,
        }
        params.update(overrides)
        return run_breakout_scan(**params)

    def test_breakout_ticker_passes_and_gets_metrics(self):
        result = self._scan({"BOOM": self._ohlcv()})

        self.assertEqual(list(result["Ticker"]), ["BOOM"])
        row = result.iloc[0]
        self.assertTrue(bool(row["IsBreakout"]))
        self.assertGreater(row["BreakoutScore"], 0)
        self.assertGreaterEqual(row["VolRel20"], 1.5)

    def test_unusual_volume_filter_removes_low_relative_volume(self):
        low_volume = self._ohlcv(final_volume=1100)

        result = self._scan({"SLOW": low_volume}, unusual_volume=True)

        self.assertTrue(result.empty)

    def test_price_filter_removes_out_of_range_ticker(self):
        pricey = self._ohlcv(final_close=250.0, final_open=240.0, final_high=250.0)

        result = self._scan({"HIGH": pricey}, max_price=100.0)

        self.assertTrue(result.empty)

    def test_gap_filter_uses_latest_open_vs_previous_close(self):
        no_gap = self._ohlcv(final_open=11.2, final_close=12.0)
        with_gap = self._ohlcv(final_open=12.0, final_close=12.5, final_high=12.5)

        result = self._scan({"NOGAP": no_gap, "GAPPER": with_gap}, min_gap=5.0)

        self.assertEqual(set(result["Ticker"]), {"GAPPER"})

    def test_missing_spy_does_not_crash_relative_strength(self):
        result = self._scan({"BOOM": self._ohlcv()}, spy_df=None)

        self.assertEqual(list(result["Ticker"]), ["BOOM"])
        self.assertIsNone(result.iloc[0]["RSvsSPY"])

    def test_progress_callback_starts_and_finishes(self):
        events = []

        result = self._scan(
            {"BOOM": self._ohlcv()},
            progress_cb=lambda i, n, symbol: events.append((i, n, symbol)),
        )

        self.assertFalse(result.empty)
        self.assertEqual(events[0], (0, 1, "starting"))
        self.assertEqual(events[-1], (1, 1, "done"))


if __name__ == "__main__":
    unittest.main()
