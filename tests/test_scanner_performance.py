"""Run 38 — scanner-performance analytics tests."""
import unittest

from analytics import scanner_performance as sp


def rec(scanner, ret5d, *, symbol="AAA", ts="2026-08-01T14:00:00Z", direction="long",
        mfe=0.03, mae=-0.02, regime=None, session=None, liquidity=None, r1d=None):
    returns = {"5d": ret5d}
    if r1d is not None:
        returns["1d"] = r1d
    return {"scanner": scanner, "symbol": symbol, "timestamp": ts,
            "direction": direction, "returns": returns, "mfe": mfe, "mae": mae,
            "regime": regime, "session": session, "liquidity": liquidity}


class WilsonTests(unittest.TestCase):
    def test_interval_bounds(self):
        lo, hi = sp.wilson_interval(80, 100)
        self.assertTrue(0 < lo < 0.8 < hi < 1)

    def test_zero_n(self):
        self.assertEqual(sp.wilson_interval(0, 0), (None, None))


class DirectionTests(unittest.TestCase):
    def test_short_inverts(self):
        self.assertEqual(sp.directional_return("short", -0.02), 0.02)
        self.assertEqual(sp.directional_return("long", 0.02), 0.02)

    def test_missing(self):
        self.assertIsNone(sp.directional_return("long", None))


class ScoreboardTests(unittest.TestCase):
    def test_grouping_and_outcome_calc(self):
        records = ([rec("A", 0.02) for _ in range(40)]
                   + [rec("B", -0.01) for _ in range(40)])
        board = sp.scanner_scoreboard(records, primary_horizon="5d")["scanners"]
        self.assertEqual(set(board), {"A", "B"})
        self.assertEqual(board["A"]["signals"], 40)
        self.assertEqual(board["A"]["horizons"]["5d"]["hit_rate"], 1.0)
        self.assertEqual(board["B"]["horizons"]["5d"]["hit_rate"], 0.0)

    def test_missing_outcomes_report_none_not_zero(self):
        records = [{"scanner": "A", "symbol": "X", "timestamp": "t",
                    "direction": "long", "returns": {}, "mfe": None, "mae": None}]
        h = sp.scanner_scoreboard(records, horizons=["5d"])["scanners"]["A"]
        self.assertEqual(h["horizons"]["5d"]["n"], 0)
        self.assertIsNone(h["horizons"]["5d"]["hit_rate"])

    def test_sample_size_protection(self):
        # 4 great observations must NOT rank as PROVEN
        board = sp.scanner_scoreboard([rec("Tiny", 0.05) for _ in range(4)],
                                      primary_horizon="5d")["scanners"]["Tiny"]
        self.assertTrue(board["insufficient"])
        self.assertEqual(board["classification"], "INSUFFICIENT_DATA")

    def test_classification_proven_promising_weak(self):
        proven = sp.scanner_scoreboard([rec("P", 0.02) for _ in range(120)],
                                       primary_horizon="5d")["scanners"]["P"]
        self.assertEqual(proven["classification"], "PROVEN")
        # small-but-positive edge with wide CI → PROMISING
        mixed = [rec("M", 0.01) for _ in range(20)] + [rec("M", -0.01) for _ in range(15)]
        promising = sp.scanner_scoreboard(mixed, primary_horizon="5d")["scanners"]["M"]
        self.assertIn(promising["classification"], ("PROMISING", "NEUTRAL"))
        weak = sp.scanner_scoreboard([rec("W", -0.02) for _ in range(60)],
                                     primary_horizon="5d")["scanners"]["W"]
        self.assertEqual(weak["classification"], "WEAK")

    def test_risk_reward(self):
        board = sp.scanner_scoreboard([rec("A", 0.02, mfe=0.04, mae=-0.02) for _ in range(30)],
                                      primary_horizon="5d")["scanners"]["A"]
        self.assertEqual(board["risk_reward"], 2.0)


class SegmentationTests(unittest.TestCase):
    def test_regime_segmentation(self):
        records = ([rec("A", 0.02, regime="bullish") for _ in range(40)]
                   + [rec("A", -0.02, regime="bearish") for _ in range(40)])
        seg = sp.segment_scoreboard(records, key="regime", primary_horizon="5d")
        self.assertEqual(seg["bullish"]["scanners"]["A"]["horizons"]["5d"]["hit_rate"], 1.0)
        self.assertEqual(seg["bearish"]["scanners"]["A"]["horizons"]["5d"]["hit_rate"], 0.0)

    def test_session_segmentation(self):
        records = [rec("A", 0.01, session="morning") for _ in range(35)]
        seg = sp.segment_scoreboard(records, key="session", primary_horizon="5d")
        self.assertIn("morning", seg)
        self.assertEqual(seg["morning"]["scanners"]["A"]["signals"], 35)


class OverlapTests(unittest.TestCase):
    def test_agreement_buckets_and_min_sample(self):
        records = []
        # 40 symbols where A and B co-fire (positive), 40 where only A fires (flat)
        for i in range(40):
            records.append(rec("A", 0.03, symbol=f"CO{i}", ts=f"2026-08-01T1{i%9}:00:00Z"))
            records.append(rec("B", 0.03, symbol=f"CO{i}", ts=f"2026-08-01T1{i%9}:00:00Z"))
        for i in range(40):
            records.append(rec("A", 0.0, symbol=f"SOLO{i}", ts=f"2026-08-02T1{i%9}:00:00Z"))
        ov = sp.overlap_analysis(records, horizon="5d")
        self.assertTrue(ov["multi_scanner"]["sufficient"])
        self.assertGreater(ov["multi_scanner"]["avg_return"],
                           ov["single_scanner"]["avg_return"])
        self.assertIn("2", ov["by_agreement_count"])

    def test_small_combos_excluded(self):
        records = [rec("A", 0.02, symbol="X"), rec("B", 0.02, symbol="X")]
        ov = sp.overlap_analysis(records, horizon="5d")
        self.assertEqual(ov["by_combination"], {})  # under min_sample


class AdapterTests(unittest.TestCase):
    def test_signal_outcomes_adapter(self):
        rows = [{"source": "opportunity", "signal_type": "hsf_opportunity",
                 "ticker": "NVDA", "fired_at": "2026-08-01", "return_1d": 0.01,
                 "return_5d": 0.03, "mfe_5d": 0.05, "mae_5d": -0.02}]
        recs = sp.from_signal_outcomes_rows(rows)
        self.assertEqual(recs[0]["scanner"], "opportunity:hsf_opportunity")
        self.assertEqual(recs[0]["returns"]["5d"], 0.03)
        self.assertEqual(recs[0]["mfe"], 0.05)

    def test_canonical_adapter_multi_scanner(self):
        obs = [{"symbol": "NVDA", "timestamp": "t", "session": "morning",
                "market_context": {"market_regime": "bullish"},
                "scanners": [{"name": "momentum", "direction": "long"},
                             {"name": "unusual_vol", "direction": "long"}],
                "outcomes": {"+15m": {"raw_return": 0.004, "mfe": 0.01, "mae": -0.002}}}]
        recs = sp.from_canonical_observations(obs)
        self.assertEqual(len(recs), 2)  # one per scanner
        self.assertEqual(recs[0]["returns"]["15m"], 0.004)
        self.assertEqual(recs[0]["regime"], "bullish")

    def test_report_shape(self):
        records = [rec("A", 0.02, regime="bullish", session="morning", liquidity="large")
                   for _ in range(30)]
        rep = sp.build_scoreboard_report(records, primary_horizon="5d")
        self.assertEqual(rep["schema"], "hsf-scanner-scoreboard-1.0")
        self.assertIn("A", rep["overall"]["scanners"])
        self.assertIn("bullish", rep["by_regime"])
        self.assertIsNotNone(rep["overlap"])


if __name__ == "__main__":
    unittest.main()
