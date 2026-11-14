"""Research feature schema v2: prediction-record repair of core inputs + RSI/regime."""
from __future__ import annotations

import unittest

from analytics import research_dataset as rd
from analytics import research_schema as rs
from tests.research_fixtures import opp, scan, ts

NAMES = ["Trend10D%", "Trend20D%", "VolRel20", "DollarVol20", "BreakoutScore", "GapPct", "PriceReturn1D"]


def with_record(row, *, values=(4.0, 6.0, 1.5, 9e6, 22.0, 1.2, 0.1), mask=(False,) * 7,
                inferred="2026-10-14T15:28:00", scan_id="2026-10-14T15:20:00", status="captured", clip=None):
    rec = {"status": status, "feature_names": NAMES, "feature_values": list(values), "default_mask": list(mask),
           "inferred_at": ts(inferred).isoformat() if inferred else None,
           "source_scan_id": ts(scan_id).isoformat() if scan_id else None,
           "preprocessing": {"clip_thresholds": clip or {}}}
    row["raw_signal"]["models"] = {"prebreakout": rec}
    return row


def meta_scan(symbol, at, *, rsi=61.5, label="SPY_BELOW_SMA20", pct=-0.8, version="spy-sma20-v1"):
    s = scan(symbol, at)
    m = s["record"]["research_metadata"]
    m.update(rsi_14=rsi, market_regime_at_observation=label, market_regime_version=version,
             market_regime_inputs={"spy_vs_sma20_pct": pct, "spy_chg_pct": 0.1})
    return s


def snap(row, scans=(), version=2):
    return rd.build_records([row], list(scans), feature_schema_version=version)[0]["features"].values


class SchemaV2Test(unittest.TestCase):
    def test_v1_unchanged_and_v2_extends_it(self):
        v1, v2 = rs.feature_names(1), rs.feature_names(2)
        self.assertEqual(len(v1), 33)
        self.assertEqual(v2[:len(v1)], v1)
        self.assertEqual(set(v2) - set(v1), {"core_inputs_source", "rsi_14", "spy_vs_sma20_pct", "spy_above_sma20"})
        self.assertEqual(rs.FEATURE_SCHEMA_VERSION, 1)

    def test_v1_ignores_prediction_record(self):
        row = with_record(opp(1, "AAA", "2026-10-14T15:30:00"))
        v = snap(row, version=1)
        self.assertIsNone(v["trend_10d_pct"])
        self.assertNotIn("core_inputs_source", v)

    def test_fills_only_empty_inputs_when_no_scan_matched(self):
        row = with_record(opp(1, "AAA", "2026-10-14T15:30:00"))
        v = snap(row)
        self.assertEqual(v["trend_10d_pct"], 4.0)
        self.assertEqual(v["dollar_vol_20"], 9e6)
        self.assertEqual(v["scan_gap_pct"], 1.2)
        self.assertEqual(v["core_inputs_source"], rs.CORE_FROM_PREDICTION)
        self.assertIsNone(v["rsi_14"])  # no scan record -> no RSI, never guessed

    def test_scan_join_value_wins_when_present(self):
        row = with_record(opp(1, "AAA", "2026-10-14T15:30:00"))
        v = snap(row, [scan("AAA", "2026-10-14T15:00:00", rvol=2.0, trend20=5.0)])
        self.assertEqual(v["rvol_20"], 2.0)        # canonical, not the record's 1.5
        self.assertEqual(v["trend_20d_pct"], 5.0)
        self.assertEqual(v["dollar_vol_20"], 9e6)  # was empty in the scan record -> filled
        self.assertEqual(v["core_inputs_source"], rs.CORE_FROM_PREDICTION)

    def test_scan_only_row_is_flagged_scan_join(self):
        row = opp(1, "AAA", "2026-10-14T15:30:00")
        v = snap(row, [scan("AAA", "2026-10-14T15:00:00")])
        self.assertEqual(v["core_inputs_source"], rs.CORE_FROM_SCAN)

    def test_no_source_stays_missing(self):
        v = snap(opp(1, "AAA", "2026-10-14T15:30:00"))
        self.assertEqual(v["core_inputs_source"], rs.CORE_NONE)
        self.assertIsNone(v["trend_10d_pct"])

    def test_rejects_unsafe_records(self):
        cases = {
            "model ran after pick": dict(inferred="2026-10-14T15:31:00"),
            "scan after pick": dict(scan_id="2026-10-14T15:35:00"),
            "scan too old": dict(scan_id="2026-10-14T12:00:00"),
            "no scan identity": dict(scan_id=None),
            "no inference time": dict(inferred=None),
            "not captured": dict(status="unavailable"),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                v = snap(with_record(opp(1, "AAA", "2026-10-14T15:30:00"), **kw))
                self.assertIsNone(v["trend_10d_pct"])
                self.assertEqual(v["core_inputs_source"], rs.CORE_NONE)

    def test_defaulted_and_clipped_inputs_not_used(self):
        row = with_record(opp(1, "AAA", "2026-10-14T15:30:00"), mask=(True, False, False, False, False, False, False),
                          clip={"VolRel20": {"lower": 0, "upper": 5}})
        v = snap(row)
        self.assertIsNone(v["trend_10d_pct"])  # defaulted
        self.assertIsNone(v["rvol_20"])        # clip rule
        self.assertEqual(v["trend_20d_pct"], 6.0)

    def test_rsi_and_regime_from_joined_scan(self):
        row = opp(1, "AAA", "2026-11-04T15:30:00")
        v = snap(row, [meta_scan("AAA", "2026-11-04T15:00:00")])
        self.assertEqual(v["rsi_14"], 61.5)
        self.assertEqual(v["spy_vs_sma20_pct"], -0.8)
        self.assertIs(v["spy_above_sma20"], False)

    def test_unknown_regime_version_or_bad_rsi_left_empty(self):
        row = opp(1, "AAA", "2026-11-04T15:30:00")
        v = snap(row, [meta_scan("AAA", "2026-11-04T15:00:00", rsi=140.0, version="other")])
        self.assertIsNone(v["rsi_14"])
        self.assertIsNone(v["spy_above_sma20"])
        self.assertIsNone(v["spy_vs_sma20_pct"])

    def test_v2_dataset_fingerprint_deterministic_and_distinct(self):
        rows = [with_record(opp(1, "AAA", "2026-10-14T15:30:00")), opp(2, "BBB", "2026-11-04T15:30:00")]
        scans = [meta_scan("BBB", "2026-11-04T15:00:00")]
        a = rd.build_dataset(rows, scans, feature_schema_version=2)
        b = rd.build_dataset(rows, scans, feature_schema_version=2)
        c = rd.build_dataset(rows, scans, feature_schema_version=1)
        self.assertEqual(a["metadata"]["fingerprint"], b["metadata"]["fingerprint"])
        self.assertNotEqual(a["metadata"]["fingerprint"], c["metadata"]["fingerprint"])
        self.assertEqual(a["feature_columns"], list(rs.feature_names(2)))


class ProviderGuardTest(unittest.TestCase):
    def test_v2_feature_set_refuses_v1_rows(self):
        from analytics import ml_v3_audit as A
        from analytics import ml_v4_providers as P

        recs = rd.build_records([opp(i, f"T{i}", "2026-10-14T15:30:00") for i in range(4)], [],
                                feature_schema_version=1)
        rows = A.build_audit_rows(recs, [], unit="observation")
        with self.assertRaises(ValueError):
            P.ProviderModel("lightgbm", "core_v2").fit(rows, [0, 1, 0, 1])
        self.assertTrue(P.feature_schema_version(P.FEATURE_SETS["core_rsi_v2"]).startswith("research-v2/"))
        self.assertTrue(P.feature_schema_version(P.FEATURE_SETS["production"]).startswith("research-v1/"))


if __name__ == "__main__":
    unittest.main()
