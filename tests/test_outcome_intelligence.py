"""Outcome Intelligence (analytics.outcome_intelligence): canonical records, the one
metric function, maturity, filters, units and anti-cherry-picking defaults.

Every expected number below is worked out by hand from the fixture."""
import datetime as dt
import unittest

from analytics import outcome_intelligence as oi

UTC = dt.timezone.utc


def T(day, hour=14, minute=0):
    return dt.datetime(2026, 9, day, hour, minute, tzinfo=UTC)


def row(id, ticker, fired, *, score=75.0, setup="breakout", signals=("breakout",), version="1.0",
        r1=None, r3=None, r5=None, b1=None, b3=None, b5=None, mfe=None, mae=None, computed=True, status="STRONG"):
    return {
        "id": id, "ticker": ticker, "fired_at": fired, "setup_score": 60.0, "prebreakout_prob": 20.0,
        "indicators": {"signals": list(signals), "n_signals": len(signals), "primary_setup": setup, "status": status},
        "raw_signal": {"hsf_score": score, "score_version": version, "primary_setup": setup, "status": status,
                       "score_components": {}},
        "return_1d": r1, "return_3d": r3, "return_5d": r5, "mfe_5d": mfe, "mae_5d": mae,
        "outcome_computed_at": (fired + dt.timedelta(days=8)) if computed is True else computed,
        "benchmark_symbol": "SPY" if b5 is not None else None,
        "benchmark_return_1d": b1, "benchmark_return_3d": b3, "benchmark_return_5d": b5,
    }


def fixture():
    return [
        # AAA: two observations on 2026-09-01 (same entry day) -> one signal_day record
        row(1, "AAA", T(1, 13), score=82, r1=0.01, r3=0.02, r5=0.04, b1=0.0, b3=0.01, b5=0.01, mfe=0.06, mae=-0.02),
        row(2, "AAA", T(1, 15), score=90, r1=0.01, r3=0.02, r5=0.04, b1=0.0, b3=0.01, b5=0.01, mfe=0.06, mae=-0.02),
        row(3, "BBB", T(2), score=65, setup="gapper", signals=("gapper",),
            r1=-0.01, r3=-0.02, r5=-0.03, b1=0.0, b3=0.0, b5=0.02, mfe=0.01, mae=-0.05),
        row(4, "CCC", T(3), score=55, setup="prebreakout", signals=("prebreakout", "breakout"),
            r1=0.0, r3=0.01, r5=0.02, mfe=0.03, mae=-0.01),                      # no benchmark yet
        row(5, "DDD", T(4), score=72, r1=0.02, r3=0.03, r5=0.05, b1=0.01, b3=0.02, b5=0.03, mfe=None, mae=None),
        row(6, "EEE", T(20), score=88, computed=None, r5=None),                     # pending
        row(7, "FFF", T(5), score=45, computed=True),                                # computed, no prices -> unavailable
    ]


class RecordTests(unittest.TestCase):
    def test_point_in_time_fields_come_from_the_frozen_payload(self):
        r = oi.canonical_record(row(9, "zzz", T(1), score=81.5, version="1.0", setup="gapper",
                                    signals=("Gapper", " breakout "), r5=0.01))
        self.assertEqual(r["ticker"], "ZZZ")
        self.assertEqual(r["hsf_score"], 81.5)
        self.assertEqual(r["score_bucket"], "80-89")
        self.assertEqual(r["setup"], "gapper")
        self.assertEqual(list(r["signals"]), ["breakout", "gapper"])
        self.assertEqual(r["entry_day"], dt.date(2026, 9, 1))
        self.assertIsNone(r["entry_price"])          # not stored: never invented
        self.assertIsNone(r["outcome_price"])

    def test_maturity_states(self):
        recs = {r["observation_id"]: r for r in oi.build_records(fixture())}
        self.assertEqual(recs["1"]["horizons"][5]["status"], oi.MATURED)
        self.assertEqual(recs["6"]["horizons"][5]["status"], oi.PENDING)
        self.assertEqual(recs["7"]["horizons"][5]["status"], oi.UNAVAILABLE)
        # computed at/before the observation = lookahead -> invalid, never matured
        bad = oi.canonical_record(row(8, "X", T(3), r5=0.5, r1=0.1, computed=T(3)))
        self.assertEqual(bad["horizons"][5]["status"], oi.INVALID)
        self.assertIsNone(bad["horizons"][5]["raw_return"])
        just_after = oi.canonical_record(row(8, "X", T(3), r5=0.5, r1=0.1, computed=T(3) + dt.timedelta(seconds=1)))
        self.assertEqual(just_after["horizons"][5]["status"], oi.MATURED)

    def test_pending_values_are_never_exposed(self):
        r = oi.canonical_record(row(8, "X", T(3), r1=0.3, r5=0.9, b5=0.1, mfe=0.9, computed=None))
        for h in oi.HORIZONS:
            self.assertEqual(r["horizons"][h]["status"], oi.PENDING)
            self.assertIsNone(r["horizons"][h]["raw_return"])

    def test_mfe_mae_only_for_the_stored_window(self):
        r = oi.canonical_record(fixture()[0])
        self.assertEqual(r["horizons"][5]["mfe"], 0.06)
        self.assertIsNone(r["horizons"][1]["mfe"])
        self.assertIsNone(r["horizons"][3]["mae"])

    def test_certified_requires_maturity_and_known_version(self):
        self.assertTrue(oi.canonical_record(row(1, "A", T(1), r1=0.1, r5=0.1))["certified"])
        self.assertFalse(oi.canonical_record(row(1, "A", T(1), r1=0.1, computed=None))["certified"])
        self.assertFalse(oi.canonical_record(row(1, "A", T(1), r1=0.1, version="0.9"))["certified"])
        self.assertFalse(oi.canonical_record(row(1, "A", T(1)))["certified"])          # unavailable


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.records = oi.build_records(fixture())

    def test_signal_day_metrics_by_hand(self):
        chosen = oi.select(self.records, oi.normalize_filters())
        self.assertEqual(len(chosen), 6)              # AAA x2 collapsed
        aaa = next(r for r in chosen if r["ticker"] == "AAA")
        self.assertEqual(aaa["hsf_score"], 82)        # the day's FIRST observation, not the higher score
        self.assertEqual(aaa["observations_that_day"], 2)
        m = oi.metrics(chosen, 5)
        # matured 5d returns: AAA .04, BBB -.03, CCC .02, DDD .05
        self.assertEqual((m["sample_size"], m["matured_count"], m["pending_count"], m["unavailable_count"]), (6, 4, 1, 1))
        self.assertAlmostEqual(m["average_return"], 0.02)
        self.assertAlmostEqual(m["median_return"], 0.03)
        self.assertEqual(m["win_count"], 3)
        self.assertAlmostEqual(m["win_rate"], 0.75)
        # benchmark present for AAA .01, BBB .02, DDD .03 -> excess .03, -.05, .02
        self.assertEqual(m["benchmark_count"], 3)
        self.assertAlmostEqual(m["average_benchmark_return"], 0.02)
        self.assertAlmostEqual(m["median_benchmark_return"], 0.02)
        self.assertAlmostEqual(m["average_excess_return"], 0.0)
        self.assertAlmostEqual(m["median_excess_return"], 0.02)
        self.assertAlmostEqual(m["benchmark_beat_rate"], 0.6667)
        # MFE AAA .06 BBB .01 CCC .03 (DDD missing); MAE -.02 -.05 -.01
        self.assertEqual(m["mfe_count"], 3)
        self.assertAlmostEqual(m["average_mfe"], 0.033333)
        self.assertAlmostEqual(m["average_mae"], -0.026667)
        self.assertEqual(m["distinct_days"], 4)
        self.assertEqual(m["evidence_quality"], "INSUFFICIENT")
        self.assertIsNone(m["average_return_ci95"])   # under 30: no normal-approx interval

    def test_observation_unit_counts_every_row(self):
        m = oi.metrics(oi.select(self.records, oi.normalize_filters(), unit="observation"), 5)
        self.assertEqual(m["matured_count"], 5)
        self.assertAlmostEqual(m["average_return"], (0.04 + 0.04 - 0.03 + 0.02 + 0.05) / 5)

    def test_one_day_horizon(self):
        m = oi.metrics(oi.select(self.records, oi.normalize_filters()), 1)
        self.assertAlmostEqual(m["average_return"], (0.01 - 0.01 + 0.0 + 0.02) / 4)
        self.assertEqual(m["win_count"], 2)                       # 0.0 is not a win
        self.assertIsNone(m["average_mfe"])

    def test_pending_never_contaminates(self):
        poisoned = fixture() + [row(99, "ZZZ", T(6), score=99, r1=9.0, r3=9.0, r5=9.0, b5=0.0, mfe=9.0, computed=None)]
        m1 = oi.metrics(oi.select(oi.build_records(poisoned), oi.normalize_filters()), 5)
        m0 = oi.metrics(oi.select(self.records, oi.normalize_filters()), 5)
        for k in ("average_return", "median_return", "win_rate", "average_excess_return", "average_mfe"):
            self.assertEqual(m1[k], m0[k], k)
        self.assertEqual(m1["pending_count"], m0["pending_count"] + 1)

    def test_empty_dataset(self):
        m = oi.metrics([], 5)
        self.assertEqual(m["sample_size"], 0)
        self.assertIsNone(m["average_return"])
        self.assertIsNone(m["win_rate"])
        self.assertEqual(m["evidence_quality"], "INSUFFICIENT")
        s = oi.summary([], oi.normalize_filters())
        self.assertEqual(s["total_observations"], 0)
        self.assertEqual(s["date_range"], {"start": None, "end": None})

    def test_evidence_quality_thresholds(self):
        self.assertEqual([oi.evidence_quality(n) for n in (0, 9, 10, 29, 30, 99, 100)],
                         ["INSUFFICIENT", "INSUFFICIENT", "LIMITED", "LIMITED", "MODERATE", "MODERATE", "STRONG"])

    def test_intervals(self):
        lo, hi = oi.wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        self.assertIsNone(oi.wilson_interval(0, 0))
        xs = [0.01 * (i % 5) for i in range(40)]
        lo, hi = oi.mean_interval(xs)
        self.assertLess(lo, sum(xs) / 40)
        self.assertGreater(hi, sum(xs) / 40)

    def test_coverage(self):
        chosen = oi.select(self.records, oi.normalize_filters())
        c = oi.coverage(chosen, 5)
        self.assertEqual((c["matured"], c["missing_benchmark"], c["missing_mfe_mae"]), (4, 1, 1))
        self.assertAlmostEqual(c["benchmark_coverage"], 0.75)
        c1 = oi.coverage(chosen, 1)
        self.assertFalse(c1["mfe_mae_available_for_horizon"])


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.records = oi.build_records(fixture())

    def test_defaults_are_the_complete_dataset(self):
        f = oi.normalize_filters()
        self.assertTrue(all(v in (None, False) for v in f.values()))
        s = oi.summary(self.records, f)
        self.assertEqual(s["filters"], f)
        self.assertEqual(s["horizon"], 5)
        self.assertEqual(s["unit"], "signal_day")
        self.assertEqual(s["raw_observations"], 7)

    def test_score_and_setup_filters(self):
        f = oi.normalize_filters(min_score=80)
        chosen = oi.select(self.records, f)
        self.assertEqual(sorted(r["ticker"] for r in chosen), ["AAA", "EEE"])
        f = oi.normalize_filters(min_score=85, max_score=95)
        aaa = [r for r in oi.select(self.records, f) if r["ticker"] == "AAA"]
        self.assertEqual(aaa[0]["hsf_score"], 90)   # filter first, then the day's first MATCHING observation
        self.assertEqual([r["ticker"] for r in oi.select(self.records, oi.normalize_filters(setup="GAPPER"))], ["BBB"])
        self.assertEqual(sorted(r["ticker"] for r in oi.select(self.records, oi.normalize_filters(signal="breakout"))),
                         ["AAA", "CCC", "DDD", "EEE", "FFF"])
        self.assertEqual([r["ticker"] for r in oi.select(self.records, oi.normalize_filters(score_bucket="60-69"))],
                         ["BBB"])

    def test_date_filters_are_inclusive(self):
        f = oi.normalize_filters(start_date="2026-09-02", end_date="2026-09-04")
        self.assertEqual([r["ticker"] for r in oi.select(self.records, f)], ["BBB", "CCC", "DDD"])

    def test_certified_and_matured_only(self):
        cert = oi.select(self.records, oi.normalize_filters(certified_only=True))
        self.assertEqual(sorted(r["ticker"] for r in cert), ["AAA", "BBB", "CCC", "DDD"])
        mat = oi.select(self.records, oi.normalize_filters(matured_only=True), horizon=5)
        self.assertNotIn("EEE", [r["ticker"] for r in mat])
        self.assertNotIn("FFF", [r["ticker"] for r in mat])

    def test_invalid_filters(self):
        for kw in ({"min_score": 101}, {"min_score": 80, "max_score": 70}, {"start_date": "09/01/2026"},
                   {"start_date": "2026-09-05", "end_date": "2026-09-01"}, {"score_bucket": "90-80"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                oi.normalize_filters(**kw)
        with self.assertRaises(ValueError):
            oi.select(self.records, oi.normalize_filters(), unit="best")
        with self.assertRaises(ValueError):
            oi.metrics([], 20)

    def test_buckets(self):
        self.assertEqual(oi.parse_buckets("50-59, 40-49"), [(40, 49), (50, 59)])
        for bad in ("40-60,50-70", "x-1", "90-110", "60"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                oi.parse_buckets(bad)


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.records = oi.build_records(fixture())
        self.f = oi.normalize_filters()

    def test_every_score_bucket_is_returned_in_order(self):
        out = oi.by_score(self.records, self.f)
        self.assertEqual([b["bucket"] for b in out["buckets"]], ["0-49", "50-59", "60-69", "70-79", "80-89", "90-100"])
        b60 = next(b for b in out["buckets"] if b["bucket"] == "60-69")
        self.assertEqual(b60["win_rate"], 0.0)          # a losing bucket is shown, not hidden
        custom = oi.by_score(self.records, self.f, buckets=oi.parse_buckets("40-49,50-59"))
        self.assertEqual([b["bucket"] for b in custom["buckets"]], ["40-49", "50-59"])
        self.assertEqual(custom["unbucketed_count"], 4)

    def test_calibration_lists_inversions(self):
        rows = [{"bucket": "70-79", "win_rate": 0.6, "matured_count": 40, "median_return": 0.02},
                {"bucket": "80-89", "win_rate": 0.5, "matured_count": 40, "median_return": 0.03},
                {"bucket": "90-100", "win_rate": 0.7, "matured_count": 5, "median_return": 0.0}]
        cal = oi.calibration_view(rows)
        wr = cal["metrics"]["win_rate"]
        self.assertFalse(wr["monotonic"])
        self.assertEqual(wr["buckets_compared"], ["70-79", "80-89"])    # 90-100 below LIMITED evidence
        self.assertEqual(wr["inversions"][0]["lower_bucket"], "70-79")
        self.assertTrue(cal["metrics"]["median_return"]["monotonic"])
        self.assertIsNone(cal["metrics"]["benchmark_beat_rate"]["monotonic"])

    def test_horizons_all_present_none_promoted(self):
        out = oi.by_horizon(self.records, self.f)
        self.assertEqual([h["horizon"] for h in out["horizons"]], [1, 3, 5])
        self.assertIsNone(out["horizon"])
        self.assertNotIn("best", str(out).lower())

    def test_setups_ordered_by_size_with_small_samples_visible(self):
        out = oi.by_group(self.records, self.f)
        self.assertEqual(out["groups"][0]["name"], "breakout")
        names = [g["name"] for g in out["groups"]]
        self.assertIn("gapper", names)
        g = next(g for g in out["groups"] if g["name"] == "gapper")
        self.assertEqual((g["matured_count"], g["evidence_quality"]), (1, "INSUFFICIENT"))
        self.assertEqual(g["supported_horizons"], [1, 3, 5])
        sig = oi.by_group(self.records, self.f, group_by="signal")
        self.assertTrue(sig["groups_overlap"])

    def test_timeseries(self):
        out = oi.timeseries(self.records, self.f, period="week")
        self.assertEqual([p["period_start"] for p in out["points"]], ["2026-08-31", "2026-09-14"])
        self.assertEqual(out["points"][0]["observation_count"], 5)
        days = oi.timeseries(self.records, self.f, period="day")
        self.assertEqual(len(days["points"]), 6)
        months = oi.timeseries(self.records, self.f, period="month")
        self.assertEqual(months["points"][0]["period_start"], "2026-09-01")
        with self.assertRaises(ValueError):
            oi.timeseries(self.records, self.f, period="year")

    def test_symbol_and_pagination(self):
        out = oi.symbol(self.records, "aaa", self.f, unit="observation", page=1, page_size=1)
        self.assertEqual(out["ticker"], "AAA")
        self.assertEqual(out["filters"]["ticker"], "AAA")
        self.assertEqual(out["observations"]["total"], 2)
        self.assertEqual(out["observations"]["items"][0]["observation_id"], "2")   # newest first
        p2 = oi.symbol(self.records, "AAA", self.f, unit="observation", page=2, page_size=1)
        self.assertEqual(p2["observations"]["items"][0]["observation_id"], "1")
        self.assertEqual([h["horizon"] for h in out["horizons"]], [1, 3, 5])
        none = oi.symbol(self.records, "NOPE", self.f)
        self.assertEqual(none["observations"]["total"], 0)

    def test_mixed_versions_warn(self):
        recs = oi.build_records(fixture() + [row(50, "VVV", T(7), version="2.0", r1=0.01, r5=0.01)])
        self.assertTrue(oi.summary(recs, self.f)["warnings"])
        self.assertFalse(oi.summary(recs, oi.normalize_filters(score_version="1.0"))["warnings"])

    def test_ai_text_has_every_horizon_and_sample_sizes(self):
        text = oi.ai_evidence_text(oi.symbol(self.records, "AAA", self.f))
        for h in ("1 trading days", "3 trading days", "5 trading days"):
            self.assertIn(h, text)
        self.assertIn("1 matured", text)
        self.assertIn("not a forecast or guarantee", text)
        self.assertIsNone(oi.ai_evidence_text(oi.symbol(self.records, "EEE", self.f)))

    def test_does_not_mutate_input_rows(self):
        rows = fixture()
        before = repr(rows)
        oi.summary(oi.build_records(rows), self.f)
        self.assertEqual(repr(rows), before)


if __name__ == "__main__":
    unittest.main()
