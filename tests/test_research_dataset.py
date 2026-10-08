"""Point-in-time research dataset: leakage contract, joins, versioning, coverage."""
import datetime as dt
import random
import re
import sqlite3
import unittest
from unittest import mock

from analytics import research_dataset as rd
from analytics import research_schema as rs
from tests.research_fixtures import opp, scan, ts

LABEL_LIKE = re.compile(r"return|mfe|mae|benchmark|excess|matur|outcome|label|certified", re.I)


def _keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}"
            yield from _keys(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _keys(v, f"{path}[{i}]")


class CriticalLeakageTest(unittest.TestCase):
    """observed_at = T; the store holds values at T and different values at T+1."""

    def test_snapshot_returns_values_known_at_T_not_T_plus_1(self):
        T = "2026-09-15T16:40:00"
        row = opp(1, "AAA", T)
        at_t = scan("AAA", "2026-09-15T16:35:00", written="2026-09-15T16:38:00", price=10.0, rvol=1.5, trend20=4.0)
        at_t1 = scan("AAA", "2026-09-16T16:35:00", price=99.0, rvol=9.9, trend20=-50.0)       # next day
        later_same_hour = scan("AAA", "2026-09-15T16:39:00", written="2026-09-15T16:45:00",  # written after T
                               price=55.0, context="scheduled:sp500")
        recs = rd.build_records([row], [at_t, at_t1, later_same_hour])
        f = recs[0]["features"].values
        self.assertEqual((f["price"], f["rvol_20"], f["trend_20d_pct"]), (10.0, 1.5, 4.0))
        for bad in (99.0, 9.9, -50.0, 55.0):
            self.assertNotIn(bad, f.values(), "a value from after observed_at leaked into the snapshot")
        self.assertEqual(recs[0]["features"].join["status"], rd.JOIN_MATCHED)
        self.assertEqual(recs[0]["features"].join["scan_timestamp"], ts("2026-09-15T16:35:00").isoformat())

    def test_only_later_scans_gives_missing_not_current_values(self):
        row = opp(1, "AAA", "2026-09-15T16:40:00")
        f = rd.build_records([row], [scan("AAA", "2026-09-17T13:35:00", price=42.0)])[0]["features"]
        self.assertEqual(f.join["status"], rd.JOIN_FUTURE_ONLY)
        self.assertIsNone(f.values["price"])
        self.assertIsNone(f.values["rvol_20"])

    def test_hsf_state_comes_from_frozen_payload_never_the_scan(self):
        row = opp(1, "AAA", "2026-09-15T16:40:00", score=61.0, chg_pct=1.25)
        s = scan("AAA", "2026-09-15T16:35:00")
        s["record"]["indicators"]["chg_pct"] = 7.7   # scan's own field stays separate
        f = rd.build_records([row], [s])[0]["features"].values
        self.assertEqual(f["hsf_score"], 61.0)
        self.assertEqual(f["chg_pct"], 1.25)
        self.assertEqual(f["scan_chg_pct"], 7.7)


class TemporalJoinTest(unittest.TestCase):
    def _join(self, observed, scans, **kw):
        return rd.match_scan_observation("AAA", ts(observed), rd.index_scan_records(scans), **kw)

    def test_boundary_written_exactly_at_T_is_included(self):
        best, info = self._join("2026-09-15T16:40:00",
                                [scan("AAA", "2026-09-15T16:35:00", written="2026-09-15T16:40:00")])
        self.assertIsNotNone(best)
        self.assertTrue(info["known_at_verified"])

    def test_one_microsecond_after_T_is_excluded(self):
        s = scan("AAA", "2026-09-15T16:35:00")
        s["created_at"] = ts("2026-09-15T16:40:00") + dt.timedelta(microseconds=1)
        best, info = self._join("2026-09-15T16:40:00", [s])
        self.assertIsNone(best)
        self.assertEqual(info["status"], rd.JOIN_FUTURE_ONLY)

    def test_max_lag_boundary(self):
        exact = scan("AAA", "2026-09-15T13:40:00")
        best, info = self._join("2026-09-15T16:40:00", [exact])
        self.assertIsNotNone(best)
        self.assertEqual(info["lag_seconds"], 3 * 3600)
        over = scan("AAA", "2026-09-15T13:39:59")
        best, info = self._join("2026-09-15T16:40:00", [over])
        self.assertIsNone(best)
        self.assertEqual(info["status"], rd.JOIN_STALE)

    def test_latest_eligible_wins_and_us_market_breaks_ties(self):
        a = scan("AAA", "2026-09-15T13:35:00", price=1.0)
        b = scan("AAA", "2026-09-15T16:35:00", price=2.0, context="scheduled:sp500")
        c = scan("AAA", "2026-09-15T16:35:00", price=3.0)
        best, info = self._join("2026-09-15T16:50:00", [a, b, c])
        self.assertEqual(best["record"]["market"]["price"], 3.0)
        self.assertEqual(info["scan_context"], "scheduled:us_market")

    def test_no_record_at_all(self):
        best, info = self._join("2026-09-15T16:50:00", [scan("BBB", "2026-09-15T16:35:00")])
        self.assertIsNone(best)
        self.assertEqual(info["status"], rd.JOIN_MISSING)

    def test_missing_write_time_is_flagged(self):
        s = scan("AAA", "2026-09-15T16:35:00")
        s["created_at"] = None
        best, info = self._join("2026-09-15T16:50:00", [s])
        self.assertIsNotNone(best)
        self.assertFalse(info["known_at_verified"])


class LeakageContractTest(unittest.TestCase):
    def setUp(self):
        self.row = opp(1, "AAA", "2026-09-15T16:40:00", scored=True, r1=0.01, r3=0.02, r5=0.05, mfe=0.08,
                       mae=-0.03, b1=0.001, b3=0.002, b5=0.004)
        self.rec = rd.build_records([self.row], [scan("AAA", "2026-09-15T16:35:00")])[0]

    def test_feature_payload_has_no_outcome_fields(self):
        payload = self.rec["features"].to_dict()
        bad = [k for k in _keys(payload) if LABEL_LIKE.search(k.rsplit(".", 1)[-1])]
        self.assertEqual(bad, [])
        for v in (0.01, 0.02, 0.05, 0.08, -0.03, 0.004):
            self.assertNotIn(v, payload["features"].values())

    def test_no_feature_name_looks_like_an_outcome(self):
        for version in rs.FEATURE_SCHEMAS:
            for name in rs.feature_names(version):
                self.assertFalse(rs.looks_like_outcome(name), name)

    def test_snapshot_refuses_outcome_and_unknown_keys(self):
        base = dict(observation_id=1, ticker="AAA", observed_at="x", feature_schema_version=1)
        for key in ("return_20", "return_5d", "future_price", "mfe_5d", "mae_5d", "benchmark_return_5d"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                rs.FeatureSnapshot(values={key: 0.1}, **base)
        with self.assertRaises(ValueError):
            rs.FeatureSnapshot(values={"hsf_score": 1.0}, join={"mfe": 1}, **base)
        with self.assertRaises(ValueError):
            rs.FeatureSnapshot(values={"rsi_14": 50.0}, **base)   # not in schema v1: never silently added

    def test_snapshot_is_immutable(self):
        with self.assertRaises(TypeError):
            self.rec["features"].values["hsf_score"] = 99
        with self.assertRaises(Exception):
            self.rec["features"].ticker = "ZZZ"

    def test_outcome_record_is_separate_and_carries_labels(self):
        o = self.rec["outcome"]
        self.assertEqual(o.values["return_5d"], 0.05)
        self.assertAlmostEqual(o.values["excess_return_5d"], 0.05 - 0.004)
        self.assertNotIn("hsf_score", o.values)
        self.assertNotIn("return_5d", self.rec["observation"])

    def test_explicit_join_is_the_only_pairing(self):
        pairs = rs.join_features_labels([self.rec["features"]], [self.rec["outcome"]])
        self.assertIs(pairs[0][1], self.rec["outcome"])
        self.assertIsInstance(pairs[0][0], rs.FeatureSnapshot)

    def test_feature_matrix_columns_never_include_labels(self):
        ds = rd.build_dataset([self.row], [scan("AAA", "2026-09-15T16:35:00")])
        self.assertFalse(set(ds["feature_columns"]) & set(ds["label_columns"]))
        self.assertFalse([c for c in ds["feature_columns"] if LABEL_LIKE.search(c)])


class SchemaVersionTest(unittest.TestCase):
    def test_versions_are_explicit(self):
        self.assertEqual(rs.FEATURE_SCHEMA_VERSION, 1)
        self.assertEqual(rs.LABEL_SCHEMA_VERSION, 1)
        with self.assertRaises(ValueError):
            rs.feature_schema(2)
        with self.assertRaises(ValueError):
            rs.label_schema(99)
        with self.assertRaises(ValueError):
            rd.build_dataset([], [], feature_schema_version=2)

    def test_v1_feature_set_is_frozen(self):
        """Changing schema v1 must be deliberate: bump the version instead."""
        self.assertEqual(len(rs.feature_names(1)), 33)
        self.assertEqual(rs.feature_names(1)[:3], ("hsf_score", "hsf_score_version", "hsf_signals_component"))
        for f in rs.feature_schema(1):
            self.assertIn(f.pit, (rs.SAFE_STORED, rs.SAFE_RECONSTRUCTABLE))
            self.assertTrue(f.point_in_time)

    def test_labels_are_existing_definitions_only(self):
        names = rs.label_names(1)
        self.assertNotIn("return_10", " ".join(names))
        self.assertEqual({lab.horizon_days for lab in rs.label_schema(1)}, {1, 3, 5})

    def test_unavailable_features_are_not_columns(self):
        cols = set(rs.feature_names(1))
        self.assertNotIn("rsi_14", cols)
        self.assertIn("rsi_14", [u["name"] for u in rs.UNAVAILABLE_FEATURES])


class MissingnessAndMaturityTest(unittest.TestCase):
    def test_missing_features_stay_null(self):
        row = opp(1, "AAA", "2026-09-15T16:40:00", chg_pct=None, gap_pct=None, prob=None, breakout_score=None)
        f = rd.build_records([row], [scan("AAA", "2026-09-15T16:35:00", meta=False)])[0]["features"].values
        for k in ("chg_pct", "gap_pct", "prebreakout_prob", "breakout_score", "trend_20d_pct", "scanner_rank"):
            self.assertIsNone(f[k], k)
        self.assertEqual(f["price"], 10.0)

    def test_pending_unavailable_matured(self):
        pend = opp(1, "AAA", "2026-09-15T16:40:00")
        unav = opp(2, "BBB", "2026-09-15T16:40:00", scored=True)                     # scored, window incomplete
        mat = opp(3, "CCC", "2026-09-15T16:40:00", scored=True, r1=0.0, r3=0.01, r5=-0.02, mfe=0.01, mae=-0.04)
        recs = {r["observation"]["observation_id"]: r for r in rd.build_records([pend, unav, mat], [])}
        self.assertEqual(recs[1]["outcome"].maturity["5d"], rd.PENDING)
        self.assertEqual(recs[2]["outcome"].maturity["5d"], rd.UNAVAILABLE)
        self.assertEqual(recs[3]["outcome"].maturity["5d"], rd.MATURED)
        self.assertEqual(recs[3]["outcome"].values["return_1d"], 0.0)      # a real zero stays zero
        self.assertIsNone(recs[3]["outcome"].values["benchmark_return_5d"])
        self.assertIsNone(recs[3]["outcome"].values["excess_return_5d"])   # never zero-filled
        self.assertTrue(recs[3]["outcome"].certified)
        self.assertFalse(recs[2]["outcome"].certified)

    def test_unknown_score_version_is_not_certified(self):
        row = opp(1, "AAA", "2026-09-15T16:40:00", version="0.9", scored=True, r1=0.1, r3=0.1, r5=0.1)
        rec = rd.build_records([row], [])[0]
        self.assertEqual(rec["outcome"].maturity["5d"], rd.MATURED)
        self.assertFalse(rec["outcome"].certified)

    def test_model_version_unknown_not_inferred(self):
        rec = rd.build_records([opp(1, "AAA", "2026-09-15T16:40:00")], [scan("AAA", "2026-09-15T16:35:00")])[0]
        prov = rec["observation"]["provenance"]
        self.assertIsNone(prov["model_version"])
        self.assertEqual(prov["prebreakout_model_code_constant"], "prebreakout-xgb-v16")
        self.assertIsNone(prov["run_id"])
        f = rd.normalize_filters(model_version="UNKNOWN")
        self.assertEqual(len(rd.apply_filters([rec], f)), 1)
        self.assertEqual(rd.apply_filters([rec], rd.normalize_filters(model_version="prebreakout-xgb-v16")), [])


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            opp(1, "AAA", "2026-09-15T16:40:00", score=90, scored=True, r1=0.1, r3=0.1, r5=0.1),
            opp(2, "BBB", "2026-09-15T16:40:00", score=55, setup="gapper"),
            opp(3, "AAA", "2026-09-22T16:40:00", score=70, scored=True, r1=0.1, r3=None, r5=None),
        ]
        self.recs = rd.build_records(self.rows, [])

    def ids(self, **kw):
        return [r["observation"]["observation_id"] for r in rd.apply_filters(self.recs, rd.normalize_filters(**kw))]

    def test_filters(self):
        self.assertEqual(self.ids(), [1, 2, 3])
        self.assertEqual(self.ids(ticker="aaa"), [1, 3])
        self.assertEqual(self.ids(setup="GAPPER"), [2])
        self.assertEqual(self.ids(min_score=60, max_score=80), [3])
        self.assertEqual(self.ids(start_date="2026-09-20"), [3])
        self.assertEqual(self.ids(end_date="2026-09-15"), [1, 2])
        self.assertEqual(self.ids(matured_only=True), [1])
        self.assertEqual(self.ids(matured_only=True, horizon=1), [1, 3])
        # Outcome Intelligence's rule: matured (1d return present) and canonically eligible.
        self.assertEqual(self.ids(certified_only=True), [1, 3])
        self.assertEqual(self.ids(scoring_version="1.0"), [1, 2, 3])

    def test_bad_filters(self):
        for kw in ({"horizon": 10}, {"horizon": 20}, {"start_date": "2026-10-02", "end_date": "2026-10-01"},
                   {"start_date": "not-a-date"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                rd.normalize_filters(**kw)

    def test_rank_is_per_snapshot_and_unchanged_by_filters(self):
        r = {x["observation"]["observation_id"]: x for x in self.recs}
        self.assertEqual((r[1]["features"].values["snapshot_rank"], r[1]["features"].values["snapshot_size"]), (1, 2))
        self.assertEqual(r[2]["features"].values["snapshot_rank"], 2)
        self.assertEqual(r[3]["features"].values["snapshot_rank"], 1)
        only_b = rd.apply_filters(self.recs, rd.normalize_filters(ticker="BBB"))
        self.assertEqual(only_b[0]["features"].values["snapshot_rank"], 2)


class DatasetDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.rows = [opp(i, t, f"2026-09-{d:02d}T16:40:00", score=50 + i, scored=i % 2 == 0, r1=0.01 * i,
                         r3=0.02, r5=0.03, mfe=0.05, mae=-0.01)
                     for i, (t, d) in enumerate([("AAA", 1), ("BBB", 1), ("AAA", 2), ("CCC", 3), ("DDD", 4)], 1)]
        self.scans = [scan(t, f"2026-09-{d:02d}T16:35:00") for t, d in [("AAA", 1), ("BBB", 1), ("AAA", 2)]]

    def build(self, rows=None, scans=None, **kw):
        return rd.build_dataset(rows if rows is not None else self.rows,
                                scans if scans is not None else self.scans, **kw)

    def test_same_inputs_same_dataset_and_fingerprint(self):
        a, b = self.build(), self.build()
        self.assertEqual(a["metadata"]["fingerprint"], b["metadata"]["fingerprint"])
        self.assertEqual(a["observation_ids"], b["observation_ids"])
        self.assertEqual(a["features"], b["features"])
        self.assertEqual(a["labels"], b["labels"])

    def test_input_order_does_not_matter(self):
        rows, scans = list(self.rows), list(self.scans)
        random.Random(7).shuffle(rows)
        random.Random(3).shuffle(scans)
        self.assertEqual(self.build()["metadata"]["fingerprint"], self.build(rows, scans)["metadata"]["fingerprint"])
        self.assertEqual(self.build(rows, scans)["observation_ids"], [1, 2, 3, 4, 5])

    def test_any_change_changes_the_fingerprint(self):
        base = self.build()["metadata"]["fingerprint"]
        changed = [dict(r) for r in self.rows]
        changed[1] = {**changed[1], "return_5d": 0.031, "outcome_computed_at": ts("2026-09-10T00:00:00")}
        self.assertNotEqual(base, self.build(changed)["metadata"]["fingerprint"])
        self.assertNotEqual(base, self.build(filters={"min_score": 52})["metadata"]["fingerprint"])
        self.assertNotEqual(base, self.build(scans=self.scans[:1])["metadata"]["fingerprint"])

    def test_created_at_and_revision_are_not_in_the_fingerprint(self):
        self.assertEqual(self.build(code_revision="abc1234")["metadata"]["fingerprint"],
                         self.build(code_revision="def5678")["metadata"]["fingerprint"])

    def test_members_restrict_a_finalized_version(self):
        ds = self.build(members=[2, 4])
        self.assertEqual(ds["observation_ids"], [2, 4])
        self.assertEqual(ds["metadata"]["observation_count"], 2)

    def test_metadata_contents(self):
        m = self.build(filters={"start_date": "2026-09-01", "end_date": "2026-09-30"})["metadata"]
        for k in ("feature_schema_version", "label_schema_version", "filters", "observation_count", "matured_count",
                  "observation_range", "scoring_versions", "model_versions", "data_quality", "fingerprint"):
            self.assertIn(k, m)
        self.assertEqual(m["filters"]["start_date"], "2026-09-01")
        self.assertIsNone(m["filters"]["ticker"])

    def test_empty_dataset(self):
        ds = rd.build_dataset([], [])
        self.assertEqual(ds["observation_ids"], [])
        self.assertEqual(ds["metadata"]["observation_count"], 0)
        self.assertTrue(ds["metadata"]["fingerprint"].startswith("sha256:"))
        self.assertEqual(rd.build_dataset([], [])["metadata"]["fingerprint"], ds["metadata"]["fingerprint"])
        cov = rd.coverage([])
        self.assertEqual(cov["total_observations"], 0)
        self.assertIsNone(cov["features"]["hsf_score"])

    def test_version_names(self):
        day = dt.date(2026, 10, 8)
        self.assertEqual(rd.version_name(day, []), "hsf-ml-2026-10-08-v1")
        self.assertEqual(rd.version_name(day, ["hsf-ml-2026-10-08-v1", "hsf-ml-2026-10-07-v4"]),
                         "hsf-ml-2026-10-08-v2")


class CoverageAndQualityTest(unittest.TestCase):
    def test_coverage_is_computed_from_records(self):
        rows = [opp(1, "AAA", "2026-09-15T16:40:00", scored=True, r1=0.1, r3=0.1, r5=0.1, mfe=0.2, mae=-0.1,
                    b1=0.0, b3=0.01, b5=None),
                opp(2, "BBB", "2026-09-15T16:40:00", chg_pct=None),
                opp(3, "CCC", "2026-09-16T16:40:00", scored=True)]
        recs = rd.build_records(rows, [scan("AAA", "2026-09-15T16:35:00")])
        cov = rd.coverage(recs)
        self.assertEqual(cov["total_observations"], 3)
        self.assertEqual((cov["matured_observations"], cov["pending_observations"],
                          cov["unavailable_observations"]), (1, 1, 1))
        self.assertEqual(cov["certified_observations"], 1)
        self.assertEqual(cov["features"]["hsf_score"], 1.0)
        self.assertEqual(cov["features"]["chg_pct"], round(2 / 3, 4))
        self.assertEqual(cov["features"]["price"], round(1 / 3, 4))
        self.assertEqual(cov["horizons"]["5d"], {"present": 1, "of_scored": 2, "rate": 0.5})
        self.assertEqual(cov["benchmark"]["3d"]["present"], 1)
        self.assertEqual(cov["benchmark"]["5d"]["present"], 0)
        self.assertEqual(cov["scan_feature_join_rate"], round(1 / 3, 4))
        self.assertEqual(cov["earliest_observation"], ts("2026-09-15T16:40:00").isoformat())
        self.assertEqual(cov["model_versions"], {"UNKNOWN": 3})
        self.assertIsNone(cov["unsupported_horizons"]["20_bar"])

    def test_duplicates_and_overlap_are_reported_not_removed(self):
        rows = [opp(1, "AAA", "2026-09-15T13:40:00"), opp(2, "AAA", "2026-09-15T16:40:00"),
                opp(3, "AAA", "2026-09-15T16:40:00"),            # same ticker + same instant: duplicate
                opp(4, "AAA", "2026-09-17T16:40:00"), opp(5, "BBB", "2026-09-15T16:40:00", score=150)]
        recs = rd.build_records(rows, [])
        q = rd.quality_report(rows, recs)
        self.assertEqual(len(recs), 5)
        self.assertEqual(q["duplicate_observations"], 1)
        self.assertEqual(q["invalid_scores"], 1)
        self.assertEqual(q["overlap"]["rows_in_multi_row_ticker_day_groups"], 3)
        by = {r["observation"]["observation_id"]: r["observation"]["overlap"] for r in recs}
        self.assertEqual(by[1]["group"], "AAA|2026-09-15")
        self.assertEqual(by[1]["group_size"], 3)
        self.assertTrue(by[1]["first_in_group"])
        self.assertFalse(by[2]["first_in_group"])
        self.assertEqual(by[4]["overlapping_entry_days"], 1)     # 09-17 window overlaps 09-15's
        self.assertEqual(by[5]["overlapping_entry_days"], 0)

    def test_label_window_uses_trading_days(self):
        self.assertEqual(rd.entry_day(ts("2026-09-05T15:00:00")), dt.date(2026, 9, 8))  # Sat -> Tue (Mon holiday)
        self.assertEqual(rd.label_window_end(dt.date(2026, 9, 8)), dt.date(2026, 9, 15))


class ScanRecordReadTest(unittest.TestCase):
    """fetch_scan_records against the real hsf_observations schema (SQLite path)."""

    def test_reads_scheduled_records_in_window(self):
        from analytics.observation_capture import build_scan_observations
        from db import hsf_observations as store
        from db.research_datasets import fetch_scan_records

        conn = sqlite3.connect(":memory:")
        rows = [{"Ticker": "AAA", "Last": 12.5, "VolRel20": 3.0, "BreakoutScore": 9}]
        obs = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=ts("2026-09-15T16:35:00"))
        obs += build_scan_observations([{"Ticker": "AAA", "Last": 1.0}], universe="US_MARKET",
                                       scan_timestamp=ts("2026-08-01T16:35:00"))
        store.save_observations_batch(obs, conn=conn)
        got = fetch_scan_records(["aaa"], ts("2026-09-15T00:00:00"), ts("2026-09-16T00:00:00"), conn=conn)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["record"]["market"]["price"], 12.5)
        # SQLite created_at is "now", i.e. written after this past observation: the join must refuse it
        # even though the scan_timestamp alone would qualify.
        rec = rd.build_records([opp(1, "AAA", "2026-09-15T16:50:00")], got)[0]
        self.assertEqual(rec["features"].join["status"], rd.JOIN_FUTURE_ONLY)
        self.assertIsNone(rec["features"].values["price"])


class RegistryImmutabilityTest(unittest.TestCase):
    def test_insert_once_never_updates(self):
        from db import research_datasets as rdb

        cur = mock.MagicMock()
        cur.rowcount = 0
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(rdb, "_ensure_registry"):
            ok = rdb.save_dataset_version({"dataset_version": "hsf-ml-2026-10-08-v1", "feature_schema_version": 1,
                                           "label_schema_version": 1, "fingerprint": "sha256:x",
                                           "observation_count": 0, "observation_ids": [], "metadata": {}}, conn=conn)
        self.assertFalse(ok)
        sql = cur.execute.call_args[0][0]
        self.assertIn("ON CONFLICT (dataset_version) DO NOTHING", sql)
        self.assertNotIn("UPDATE", sql.upper())
        src = open(rdb.__file__).read().upper()
        self.assertNotIn("UPDATE RESEARCH_DATASET_VERSIONS", src)
        self.assertNotIn("DELETE FROM RESEARCH_DATASET_VERSIONS", src)


class AuditReportTest(unittest.TestCase):
    def test_audit_report_covers_every_feature(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / "ml" / "reports" / "point_in_time_feature_audit.md").read_text()
        for name in rs.feature_names(1):
            self.assertIn(f"`{name}`", text, name)
        for u in rs.UNAVAILABLE_FEATURES:
            self.assertIn(u["name"], text)
        for cls in rs.PIT_CLASSES:
            self.assertIn(cls, text)


if __name__ == "__main__":
    unittest.main()
