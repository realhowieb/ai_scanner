"""Kalshi BTC signal engine: direction, confidence, probability, entry/exit."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None

if _PANDAS:
    import numpy as np
    import pandas as pd


def _frame(closes, vols=None):
    """OHLCV frame from a close path (H/L bracket the close by ~0.2%)."""
    closes = [float(c) for c in closes]
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    high = [c * 1.002 for c in closes]
    low = [c * 0.998 for c in closes]
    opens = [closes[0]] + closes[:-1]
    volume = vols if vols is not None else [100.0] * n
    return pd.DataFrame(
        {"Open": opens, "High": high, "Low": low, "Close": closes, "Volume": volume},
        index=idx,
    )


@unittest.skipUnless(_PANDAS, "pandas required")
class KalshiSignalTests(unittest.TestCase):
    def test_rising_series_reads_buy_up(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(60000, 66000, 60))  # steady uptrend
        sig = compute_kalshi_signal(_frame(closes))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "up")
        self.assertEqual(sig["action"], "Buy Up")
        self.assertGreater(sig["confidence"], 0)
        # entry zone sits at/below price for an up call; target above price
        self.assertLessEqual(sig["entry_zone"][0], sig["price"])
        self.assertGreater(sig["target"], sig["price"])
        self.assertLess(sig["stop"], sig["price"])

    def test_falling_series_reads_buy_down(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(66000, 60000, 60))  # steady downtrend
        sig = compute_kalshi_signal(_frame(closes))
        self.assertIsNotNone(sig)
        self.assertEqual(sig["direction"], "down")
        self.assertEqual(sig["action"], "Buy Down")
        self.assertGreater(sig["target"], 0)
        self.assertLess(sig["target"], sig["price"])   # cash-out below for a down call
        self.assertGreater(sig["stop"], sig["price"])

    def test_probability_is_clamped_heuristic(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(60000, 70000, 60))
        sig = compute_kalshi_signal(_frame(closes))
        self.assertGreaterEqual(sig["probability"], 50)
        self.assertLessEqual(sig["probability"], 85)     # never reads as certainty
        self.assertLessEqual(sig["confidence"], 100)

    def test_volume_spike_flagged(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(60000, 66000, 60))
        vols = [100.0] * 59 + [400.0]                    # last bar 4x avg
        sig = compute_kalshi_signal(_frame(closes, vols))
        self.assertTrue(sig["volume_spike"])
        self.assertGreaterEqual(sig["rvol"], 1.5)

    def test_insufficient_bars_returns_none(self):
        from scan.kalshi_signal import compute_kalshi_signal

        self.assertIsNone(compute_kalshi_signal(_frame([60000.0] * 10)))
        self.assertIsNone(compute_kalshi_signal(None))


if __name__ == "__main__":
    unittest.main()
