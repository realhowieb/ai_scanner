"""Run 34 — DT historical↔live parity regression tests.

Locks the semantic parity between the live Day Trader pipeline and the historical
reconstruction/validation pipeline. The scoring stack is shared code
(analytics.day_trade_intel), so the tests focus on (a) reconstruction producing
the same feature *semantics* live produces, and (b) the shared engine returning
identical outputs for identical inputs regardless of which path assembled them.
"""
import unittest

import pandas as pd

from analytics import day_trade_intel as di
from analytics import day_trade_parity as par
from analytics import day_trade_reconstruct as rc


def _daily_df(n=60, start_close=100.0):
    idx = pd.date_range("2026-06-01", periods=n, freq="B")
    closes = [start_close + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "Open": [c - 0.2 for c in closes],
        "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i * 1000 for i in range(n)],
    }, index=idx)


def _minute_bars(day="2026-08-25", base=130.0, n=40, rising=True):
    bars = []
    for i in range(n):
        c = base + (i * 0.05 if rising else -i * 0.05)
        t = f"{day}T{13 + i // 60:02d}:{35 + i % 60:02d}:00Z" if i < 25 else f"{day}T14:{i:02d}:00Z"
        bars.append({"t": t, "o": base if i == 0 else c - 0.02, "h": c + 0.1,
                     "l": c - 0.1, "c": c, "v": 5000})
    return bars


class ReconstructionSemanticsTests(unittest.TestCase):
    """Tasks 1–3: reconstruction reproduces live feature semantics."""

    def _obs0(self):
        daily = _daily_df()
        prior_close = float(daily["Close"].iloc[-1])
        bars = _minute_bars(base=prior_close + 1.0, rising=True)
        obs = rc.reconstruct_observations("NVDA", daily, bars, sample_every=5)
        return obs, prior_close, bars

    def test_1_chg_pct_uses_prior_close(self):
        obs, prior_close, _ = self._obs0()
        o = obs[0]
        expected = (o["price_at_signal"] - prior_close) / prior_close * 100
        self.assertAlmostEqual(o["diagnostic_inputs"]["chg_pct"], expected, places=6)

    def test_2_gap_pct_uses_open_over_prior_close(self):
        obs, prior_close, bars = self._obs0()
        day_open = bars[0]["o"]
        expected = (day_open - prior_close) / prior_close * 100
        self.assertAlmostEqual(obs[0]["diagnostic_inputs"]["gap_pct"], expected, places=6)

    def test_3_complete_observation_produces_expected_votes(self):
        # Rising day on a prior uptrend: momentum/vwap/supertrend/ewo should agree
        # bullish, giving a full directional vote set.
        obs, _, _ = self._obs0()
        o = obs[-1]  # later bar, more accumulation
        rec = par.parity_record(o)
        self.assertEqual(rec["fallback_status"], "full_feature")
        self.assertGreaterEqual(rec["directional_vote_count"], 3)
        self.assertEqual(rec["direction"], "bullish")


class EngineParityTests(unittest.TestCase):
    """Tasks 6–8, 10: identical inputs → identical outputs on the shared engine,
    regardless of which pipeline assembled the row."""

    FEAT = {"chg_pct": 1.5, "gap_pct": 0.8, "rvol": 2.4, "vs_vwap_pct": 0.8,
            "adx": 31, "supertrend_direction": "green", "ewo": 12.4}

    def test_6_7_8_10_live_row_and_historical_feat_score_identically(self):
        historical_feat = dict(self.FEAT)
        # A live row carries the same 7 inputs plus live-only keys that must not
        # affect scoring.
        live_row = dict(self.FEAT, ticker="NVDA", last=131.0, vwap=130.0,
                        volume=5_000_000, open=129.5, previous_close=129.0,
                        ema_cross="Golden", atr_pct=3.1)
        h = di.day_trade_intelligence(historical_feat)
        live = di.day_trade_intelligence(live_row)
        self.assertEqual(h["direction"], live["direction"])          # direction
        self.assertEqual(h["score"], live["score"])                  # DT Score/strength
        self.assertEqual(h["quality"], live["quality"])              # quality tier
        self.assertEqual(h["conflicts"], live["conflicts"])          # conflict rules
        # And the score is the coherent-strong value we expect.
        self.assertEqual(h["quality"], "strong")


class FallbackGuardTests(unittest.TestCase):
    """Tasks 4, 5, 9: incomplete data is flagged and cannot masquerade as full."""

    def test_4_missing_indicators_do_not_create_high_confidence(self):
        # Only VWAP + momentum present (the intraday-only fallback shape). It may
        # still score, but must NOT reach the Strong tier (no confirmation).
        feat = {"chg_pct": 3.0, "gap_pct": None, "rvol": None, "vs_vwap_pct": 1.5,
                "adx": None, "supertrend_direction": None, "ewo": None}
        intel = di.day_trade_intelligence(feat)
        self.assertNotEqual(intel["quality"], "strong")
        rec = par.parity_record({"diagnostic_inputs": feat, "score": intel["score"],
                                 "direction": intel["direction"],
                                 "setup_quality": intel["quality"]})
        self.assertEqual(rec["confirmation"], False)
        self.assertEqual(rec["confirmation_count"], 0)

    def test_5_partial_cannot_masquerade_as_full(self):
        full = {f: 1 for f in par.FEATURE_FIELDS}
        full["supertrend_direction"] = "green"
        fallback = {"chg_pct": 1.2, "vs_vwap_pct": 0.8, "gap_pct": None,
                    "rvol": None, "adx": None, "supertrend_direction": None, "ewo": None}
        partial = dict(full, ewo=None)  # one input missing
        self.assertEqual(par.classify_fallback(full)[0], "full_feature")
        self.assertEqual(par.classify_fallback(fallback)[0], "fallback")
        self.assertEqual(par.classify_fallback(partial)[0], "partial")

    def test_9_fallback_is_explicitly_flagged(self):
        fallback = {"chg_pct": 1.2, "vs_vwap_pct": 0.8, "gap_pct": None,
                    "rvol": None, "adx": None, "supertrend_direction": None, "ewo": None}
        rec = par.parity_record({"diagnostic_inputs": fallback,
                                 "feature_source": "intraday_fallback"})
        self.assertEqual(rec["fallback_status"], "fallback")
        self.assertIn("daily-derived", rec["fallback_reason"])
        self.assertEqual(rec["feature_source"], "intraday_fallback")

    def test_summary_counts_and_distributions(self):
        full = {"chg_pct": 1.5, "gap_pct": 0.8, "rvol": 2.4, "vs_vwap_pct": 0.8,
                "adx": 31, "supertrend_direction": "green", "ewo": 12.4}
        fb = {"chg_pct": 3.0, "gap_pct": None, "rvol": None, "vs_vwap_pct": 1.5,
              "adx": None, "supertrend_direction": None, "ewo": None}

        def _mk(inp, src):
            intel = di.day_trade_intelligence(inp)
            return {"ticker": "X", "timestamp": "T", "diagnostic_inputs": inp,
                    "feature_source": src, "score": intel["score"],
                    "direction": intel["direction"], "setup_quality": intel["quality"]}

        obs = [_mk(full, "reconstruct")] * 3 + [_mk(fb, "intraday_fallback")] * 2
        s = par.parity_summary(obs)
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["coverage"]["full_feature"]["count"], 3)
        self.assertEqual(s["coverage"]["fallback"]["count"], 2)
        self.assertEqual(s["coverage"]["by_source"],
                         {"reconstruct": 3, "intraday_fallback": 2})
        self.assertIn("bullish", s["direction"])


if __name__ == "__main__":
    unittest.main()
