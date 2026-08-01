"""Kalshi BTC signal engine: direction, confidence, probability, entry/exit."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None

if _PANDAS:
    import numpy as np
    import pandas as pd


def _frame(closes, vols=None, band=0.002):
    """OHLCV frame from a close path (H/L bracket the close by ±band)."""
    closes = [float(c) for c in closes]
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    high = [c * (1 + band) for c in closes]
    low = [c * (1 - band) for c in closes]
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

    def test_thin_volume_forces_no_trade(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(60000, 66000, 60))     # clean uptrend…
        vols = [100.0] * 59 + [10.0]                      # …but last bar 0.1× avg
        sig = compute_kalshi_signal(_frame(closes, vols))
        self.assertEqual(sig["recommendation"], "No Trade")
        self.assertFalse(sig["tradeable"])
        self.assertFalse(sig["volume_ok"])
        self.assertIsNone(sig["entry_zone"])
        self.assertTrue(any("Volume" in r for r in sig["gate_reasons"]))

    def test_flat_choppy_market_is_no_trade(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = [64000.0 + (5.0 if i % 2 else 0.0) for i in range(60)]  # dead flat
        sig = compute_kalshi_signal(_frame(closes))
        self.assertEqual(sig["recommendation"], "No Trade")
        self.assertIsNone(sig["win_probability"])

    def test_higher_timeframe_disagreement_lowers_confidence(self):
        from scan.kalshi_signal import compute_kalshi_signal

        closes = list(np.linspace(60000, 66000, 60))      # 15m uptrend
        agree = compute_kalshi_signal(_frame(closes), htf_df=_frame(list(np.linspace(58000, 66000, 40))))
        disagree = compute_kalshi_signal(_frame(closes), htf_df=_frame(list(np.linspace(68000, 60000, 40))))
        self.assertTrue(agree["htf_agree"])
        self.assertFalse(disagree["htf_agree"])
        self.assertLess(disagree["confidence"], agree["confidence"])

    def test_strong_trend_is_tradeable_with_size(self):
        from scan.kalshi_signal import compute_kalshi_signal

        sig = compute_kalshi_signal(_frame(list(np.linspace(60000, 70000, 60))))
        self.assertTrue(sig["tradeable"])
        self.assertIn(sig["position_size"], ("Small", "Medium", "Large"))
        self.assertIn("Buy Up", sig["recommendation"])

    def test_frozen_tape_fails_adaptive_volatility(self):
        from scan.kalshi_signal import compute_kalshi_signal

        # Gentle uptrend but a near-zero high/low band → ATR% under the floor.
        closes = list(np.linspace(64000, 64050, 60))
        sig = compute_kalshi_signal(_frame(closes, band=0.00002))
        self.assertFalse(sig["volatility_ok"])
        self.assertEqual(sig["recommendation"], "No Trade")
        self.assertTrue(any("Volatility" in r for r in sig["gate_reasons"]))


@unittest.skipUnless(_PANDAS, "pandas required")
class EvaluateEvTests(unittest.TestCase):
    def test_up_edge_recommends_buy(self):
        from scan.kalshi_signal import evaluate_ev

        ev = evaluate_ev("up", win_prob_pct=85, yes_price_pct=52)
        self.assertEqual(ev["side"], "YES")
        self.assertEqual(ev["entry_price_pct"], 52.0)
        self.assertEqual(ev["edge_pts"], 33.0)            # 85 − 52
        self.assertTrue(ev["recommend"])

    def test_marginal_edge_passes(self):
        from scan.kalshi_signal import evaluate_ev

        ev = evaluate_ev("up", win_prob_pct=53, yes_price_pct=54)
        self.assertEqual(ev["edge_pts"], -1.0)
        self.assertFalse(ev["recommend"])                 # market already prices it

    def test_down_buys_no_side(self):
        from scan.kalshi_signal import evaluate_ev

        ev = evaluate_ev("down", win_prob_pct=70, yes_price_pct=40)
        self.assertEqual(ev["side"], "NO")
        self.assertEqual(ev["entry_price_pct"], 60.0)     # NO = 1 − YES = 0.60
        self.assertEqual(ev["edge_pts"], 10.0)            # 70 − 60
        self.assertTrue(ev["recommend"])

    def test_missing_inputs_returns_none(self):
        from scan.kalshi_signal import evaluate_ev

        self.assertIsNone(evaluate_ev("up", None, 52))
        self.assertIsNone(evaluate_ev("none", 80, 52))


if __name__ == "__main__":
    unittest.main()
