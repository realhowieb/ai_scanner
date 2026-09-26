"""Run 55 — research evidence & signal effectiveness (pure analysis tests)."""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analytics import signal_evidence as se
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS

NOW = dt.datetime(2026, 9, 26, 12, 0, tzinfo=dt.timezone.utc)
HMIN = {"+5m": 5, "+15m": 15, "+30m": 30, "+60m": 60}


def _obs(oid, symbol, cohort, run, *, ts="2026-09-24T14:05:00+00:00", score=None,
         direction="long", explicit=True, rvol=None, price=10.0):
    scanners = []
    if cohort in (CANDIDATE, NEAR_MISS):
        scanners = [{"name": "breakout", "score": score, "direction": direction,
                     "triggered": True, "meta": {"is_breakout": bool(score and score > 20)}}]
    rec = {"observation_id": oid, "symbol": symbol, "timestamp": ts, "scan_timestamp": ts,
           "market": {"price": price, "volume": 1e5},
           "indicators": {"rvol": rvol} if rvol is not None else {},
           "scanners": scanners, "market_context": {"scan_id": run}}
    if explicit:
        rec["research_cohort"] = cohort
        rec["market_context"]["research_cohort"] = cohort
    return rec


def _oc(oid, h, raw, *, ts="2026-09-24T14:05:00+00:00", direction="LONG", stored=True,
        mfe=0.01, mae=-0.01):
    a = dt.datetime.fromisoformat(ts)
    dr = (-raw if direction == "SHORT" else raw) if stored else None
    return {"observation_id": oid, "horizon": h, "raw_return": raw, "directional_return": dr,
            "mfe": mfe, "mae": mae, "data_status": "MATURED",
            "evaluation_time": (a + dt.timedelta(minutes=HMIN[h])).isoformat()}


def _dataset(edge=0.004, runs=30, n_cand=20, n_ctrl=20, n_nm=10, sd=0.004, seed=1,
             score_signal=0.0):
    """Synthetic production-shaped data with a shared per-run market shock."""
    rng = np.random.default_rng(seed)
    obs, outs = [], {}
    for r in range(runs):
        run = f"run{r:03d}"
        ts = (dt.datetime(2026, 9, 1, 14, 5, tzinfo=dt.timezone.utc)
              + dt.timedelta(days=r)).isoformat()
        shock = rng.normal(0, 0.003)
        for cohort, n, mu in ((CANDIDATE, n_cand, edge), (NEAR_MISS, n_nm, edge / 2),
                              (CONTROL, n_ctrl, 0.0)):
            for i in range(n):
                oid = f"{run}-{cohort}-{i}"
                score = float(rng.uniform(10, 95)) if cohort != CONTROL else None
                o = _obs(oid, f"{cohort[:2]}{i}", cohort, run, ts=ts, score=score,
                         rvol=float(rng.uniform(0.5, 5)))
                obs.append(o)
                bonus = score_signal * ((score or 50) - 50) / 50
                outs[oid] = [_oc(oid, h, float(shock + mu + bonus + rng.normal(0, sd)), ts=ts)
                             for h in se.HORIZONS]
    return obs, outs


class RecordBuildingTests(unittest.TestCase):
    def test_legacy_rows_excluded_from_all_analysis(self):
        obs = [_obs("a", "AAA", CANDIDATE, "r1"),
               _obs("b", "BBB", CANDIDATE, "r1", explicit=False)]
        outs = {k: [_oc(k, "+60m", 0.01)] for k in ("a", "b")}
        built = se.build_records(obs, outs)
        self.assertEqual({r["observation_id"] for r in built["records"]}, {"a"})
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["readiness"]["legacy_inferred"], 1)
        self.assertEqual(rep["readiness"]["observations_by_cohort_all"].get("LEGACY_INFERRED"), 1)

    def test_cohort_isolation_and_overlap_dedupe(self):
        obs = [_obs("c", "XYZ", CONTROL, "r1"), _obs("n", "XYZ", NEAR_MISS, "r1"),
               _obs("k", "ABC", CONTROL, "r1"), _obs("z", "XYZ", CONTROL, "r2")]
        outs = {k: [_oc(k, "+60m", 0.01)] for k in ("c", "n", "k", "z")}
        built = se.build_records(obs, outs)
        by = {r["observation_id"]: r["cohort"] for r in built["records"]}
        self.assertEqual(by, {"n": NEAR_MISS, "k": CONTROL, "z": CONTROL})
        self.assertEqual(built["dropped_overlap"], 1)

    def test_only_matured_known_horizons_count(self):
        obs = [_obs("a", "AAA", CANDIDATE, "r1")]
        oc = _oc("a", "+60m", 0.01)
        outs = {"a": [dict(oc, data_status="PENDING"), dict(oc, horizon="EOD"),
                      dict(_oc("a", "+5m", 0.01), raw_return=None)]}
        self.assertEqual(se.build_records(obs, outs)["records"], [])


class DirectionTests(unittest.TestCase):
    def test_short_derived_from_raw_when_not_stored(self):
        obs = [_obs("s", "SSS", CANDIDATE, "r1", direction="short"),
               _obs("l", "LLL", CANDIDATE, "r1", direction="long")]
        outs = {"s": [_oc("s", "+60m", 0.02, direction="SHORT", stored=False)],
                "l": [_oc("l", "+60m", 0.02, stored=False)]}
        recs = {r["observation_id"]: r for r in se.build_records(obs, outs)["records"]}
        self.assertEqual(recs["s"]["direction"], "SHORT")
        self.assertAlmostEqual(recs["s"]["dr"], -0.02)
        self.assertAlmostEqual(recs["l"]["dr"], 0.02)
        self.assertEqual(recs["s"]["dr_source"], "derived")

    def test_stored_transform_verified(self):
        obs = [_obs("s", "SSS", CANDIDATE, "r1", direction="bearish")]
        good = se.build_records(obs, {"s": [_oc("s", "+60m", 0.02, direction="SHORT")]})
        self.assertEqual(good["transform"], {"checked": 1, "mismatches": 0, "stored": 1, "derived": 0})
        bad_oc = dict(_oc("s", "+60m", 0.02, direction="SHORT"), directional_return=0.02)
        bad = se.build_records(obs, {"s": [bad_oc]})
        self.assertEqual(bad["transform"]["mismatches"], 1)

    def test_control_without_direction_is_long_convention(self):
        obs = [_obs("c", "CCC", CONTROL, "r1")]
        rec = se.build_records(obs, {"c": [_oc("c", "+60m", -0.01, stored=False)]})["records"][0]
        self.assertEqual((rec["direction"], rec["direction_known"], rec["dr"]), ("LONG", False, -0.01))

    def test_short_insufficient_label(self):
        obs, outs = _dataset(runs=3, n_cand=5, n_ctrl=5, n_nm=2)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["direction"]["verdict"], "INSUFFICIENT")


class BucketTests(unittest.TestCase):
    def _scored(self, scores):
        return [{"horizon": "+60m", "features": {"score": s}} for s in scores]

    def test_assign_bucket(self):
        b = se.FIXED_SCORE_BUCKETS
        self.assertEqual(se.assign_bucket(49.99, b), "<50")
        self.assertEqual(se.assign_bucket(50.0, b), "50-59")
        self.assertEqual(se.assign_bucket(95.0, b), "90-100")
        self.assertEqual(se.assign_bucket(120.0, b), "90-100")
        self.assertIsNone(se.assign_bucket(None, b))

    def test_small_buckets_merge_into_neighbour(self):
        scores = [40] * 40 + [55] * 5 + [65] * 40 + [75] * 40 + [85] * 40 + [95] * 10
        spec = se.score_buckets(self._scored(scores))
        self.assertEqual(spec["mode"], "fixed")
        labels = [b[2] for b in spec["buckets"]]
        self.assertEqual(labels, ["<50", "50-69", "70-79", "80-100"])

    def test_quintile_fallback_when_fixed_collapses(self):
        scores = [float(x) for x in np.linspace(0, 40, 200)]  # all below 50
        spec = se.score_buckets(self._scored(scores))
        self.assertEqual(spec["mode"], "quintile")
        self.assertEqual(len(spec["buckets"]), 5)
        self.assertTrue(all(isinstance(b[2], str) for b in spec["buckets"]))


class StatisticsTests(unittest.TestCase):
    def _rows(self, vals, run="r"):
        return [{"drw": v, "dr": v, "scan_run": f"{run}{i % 10}", "mfe": None, "mae": None}
                for i, v in enumerate(vals)]

    def test_summary_and_wilson_ci(self):
        s = se.summarize(self._rows([0.01, -0.01, 0.02, 0.03]))
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["win_rate"], 0.75)
        lo, hi = s["win_rate_ci"]
        self.assertTrue(0 < lo < 0.75 < hi <= 1)
        self.assertAlmostEqual(s["mean"], 0.0125)
        self.assertAlmostEqual(s["payoff_ratio"], 2.0)

    def test_zero_and_one_observation(self):
        s0 = se.summarize([])
        self.assertEqual((s0["n"], s0["mean"], s0["win_rate"]), (0, None, None))
        s1 = se.summarize(self._rows([0.01]))
        self.assertEqual((s1["n"], s1["mean"], s1["sd"]), (1, 0.01, None))
        self.assertEqual(s1["mean_ci"], [None, None])
        c = se.compare([], self._rows([0.01]))
        self.assertEqual((c["evidence"], c["powered"], c["diff_mean"]), ("INSUFFICIENT", False, None))
        self.assertEqual(se.correlation_with_ci([], lambda r: 1)["spearman"], None)

    def test_cluster_bootstrap_is_deterministic_and_covers_truth(self):
        rng = np.random.default_rng(3)
        a = self._rows(list(rng.normal(0.005, 0.01, 400)), "x")
        b = self._rows(list(rng.normal(0.0, 0.01, 400)), "x")
        c1, c2 = se.cluster_bootstrap_diff(a, b), se.cluster_bootstrap_diff(a, b)
        self.assertEqual(c1, c2)
        lo, hi = c1["diff_mean_ci"]
        self.assertLess(lo, 0.005)
        self.assertGreater(hi, 0.005)
        self.assertGreater(lo, 0)

    def test_spearman_matches_rank_definition(self):
        self.assertAlmostEqual(se.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(se.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(se.spearman([1, 1, 1], [1, 2, 3]))

    def test_evidence_labels(self):
        self.assertEqual(se.evidence_label(20, 50, "POSITIVE"), "INSUFFICIENT")
        self.assertEqual(se.evidence_label(500, 5, "POSITIVE"), "INSUFFICIENT")
        self.assertEqual(se.evidence_label(50, 50, "POSITIVE"), "WEAK_EVIDENCE")
        self.assertEqual(se.evidence_label(150, 12, "NEGATIVE"), "MODERATE_EVIDENCE")
        self.assertEqual(se.evidence_label(400, 25, "POSITIVE"), "STRONG_EVIDENCE")
        self.assertEqual(se.evidence_label(400, 25, "NONE"), "WEAK_EVIDENCE")


class VerdictTests(unittest.TestCase):
    def _cmp(self, sign, evidence="STRONG_EVIDENCE", powered=True):
        return {"sign": sign, "evidence": evidence, "powered": powered}

    def _coh(self, cc_signs, cn_signs=("NONE",) * 4):
        return {"by_horizon": {h: {"candidate_minus_control": self._cmp(cc_signs[i]),
                                   "candidate_minus_near_miss": self._cmp(cn_signs[i])}
                               for i, h in enumerate(se.HORIZONS)}}

    READY = {"verdict": "READY", "failed_gates": []}
    MONO = {"verdict": "INCONCLUSIVE"}
    TIERS = {"verdict": "INCONCLUSIVE"}

    def test_single_lucky_horizon_is_not_edge(self):
        v = se.primary_verdict(self.READY, self._coh(["NONE", "NONE", "NONE", "POSITIVE"]),
                               self.MONO, self.TIERS)
        self.assertEqual(v["verdict"], "NO_EDGE_DETECTED")

    def test_consistent_edge(self):
        v = se.primary_verdict(self.READY, self._coh(["POSITIVE", "POSITIVE", "POSITIVE", "NONE"]),
                               self.MONO, self.TIERS)
        self.assertEqual(v["verdict"], "EDGE_DETECTED")
        self.assertEqual(v["evidence_quality"], "STRONG_EVIDENCE")

    def test_edge_contradicted_by_near_miss(self):
        v = se.primary_verdict(self.READY, self._coh(["POSITIVE"] * 4, ["NEGATIVE"] * 2 + ["NONE"] * 2),
                               self.MONO, self.TIERS)
        self.assertEqual(v["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_unpowered_null_is_insufficient_not_no_edge(self):
        coh = self._coh(["NONE"] * 4)
        for h in se.HORIZONS:
            coh["by_horizon"][h]["candidate_minus_control"]["powered"] = False
        v = se.primary_verdict(self.READY, coh, self.MONO, self.TIERS)
        self.assertEqual(v["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_readiness_insufficient_blocks_everything(self):
        v = se.primary_verdict({"verdict": "INSUFFICIENT", "failed_gates": ["point_in_time"]},
                               self._coh(["POSITIVE"] * 4), self.MONO, self.TIERS)
        self.assertEqual(v["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_run56_matrix(self):
        feats = {"numeric": {"rvol": {"classification": "USEFUL"}}, "flags": {}}
        integ = se.run56_recommendation({"failed_gates": ["point_in_time"]},
                                        {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "x"},
                                        self.MONO, self.TIERS, feats)
        self.assertEqual(integ["action"], "INVESTIGATE_DATA_QUALITY")
        more = se.run56_recommendation({"failed_gates": ["scan_runs"]},
                                       {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "x"},
                                       self.MONO, self.TIERS, feats)
        self.assertEqual(more["action"], "COLLECT_MORE_DATA")
        noedge = se.run56_recommendation({"failed_gates": []}, {"verdict": "NO_EDGE_DETECTED"},
                                         self.MONO, self.TIERS, feats)
        self.assertEqual((noedge["action"], noedge["feature_leads"]), ("IMPROVE_FEATURE_SET", ["rvol"]))
        edge_no_mono = se.run56_recommendation({"failed_gates": []}, {"verdict": "EDGE_DETECTED"},
                                               {"verdict": "NO"}, self.TIERS, feats)
        self.assertEqual(edge_no_mono["action"], "RECALIBRATE_SCORE")


class EndToEndTests(unittest.TestCase):
    def test_planted_edge_is_detected(self):
        obs, outs = _dataset(edge=0.004)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertNotEqual(rep["readiness"]["verdict"], "INSUFFICIENT")
        self.assertEqual(rep["verdict"]["verdict"], "EDGE_DETECTED")
        cc = rep["cohorts"]["by_horizon"]["+60m"]["candidate_minus_control"]
        self.assertLess(cc["diff_mean_ci"][0], 0.004)
        self.assertGreater(cc["diff_mean_ci"][1], 0.004)

    def test_null_with_power_is_no_edge(self):
        obs, outs = _dataset(edge=0.0, seed=7)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["verdict"]["verdict"], "NO_EDGE_DETECTED")
        self.assertEqual(rep["run56"]["code"], "E")

    def test_small_sample_is_insufficient(self):
        obs, outs = _dataset(edge=0.01, runs=3, n_cand=5, n_ctrl=5, n_nm=2)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["readiness"]["verdict"], "INSUFFICIENT")
        self.assertEqual(rep["verdict"]["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(rep["run56"]["action"], "COLLECT_MORE_DATA")

    def test_score_monotonicity_detects_planted_ordering(self):
        obs, outs = _dataset(edge=0.0, score_signal=0.006, seed=5)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["score_monotonicity"]["verdict"], "YES")

    def test_empty_and_single_observation(self):
        for obs, outs in (([], {}), ([_obs("a", "AAA", CANDIDATE, "r1")], {"a": [_oc("a", "+60m", 0.01)]})):
            rep = se.analyze(obs, outs, now=NOW)
            self.assertEqual(rep["verdict"]["verdict"], "INSUFFICIENT_EVIDENCE")
            json.dumps(rep, default=str, allow_nan=False)

    def test_missing_optional_fields_and_unavailable_sections(self):
        obs, outs = _dataset(runs=12, n_cand=12, n_ctrl=12, n_nm=6)
        for o in obs:
            o.pop("indicators", None)
            o["market"].pop("volume", None)
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["features"]["numeric"]["rvol"]["classification"], "INSUFFICIENT")
        self.assertEqual(rep["tiers"]["verdict"], "INCONCLUSIVE")
        self.assertFalse(rep["tiers"]["available"])
        self.assertEqual(rep["regime"]["verdict"], "REGIME ANALYSIS UNAVAILABLE")
        self.assertIn("conflict_count / conflict flags", rep["features"]["unavailable"])

    def test_non_regular_session_excluded_from_primary_but_in_time_of_day(self):
        obs = [_obs("pre", "PPP", CANDIDATE, "r1", ts="2026-09-24T12:35:00+00:00")]
        outs = {"pre": [_oc("pre", "+60m", 0.01, ts="2026-09-24T12:35:00+00:00")]}
        rep = se.analyze(obs, outs, now=NOW)
        self.assertEqual(rep["primary_population"]["records"], 0)
        self.assertEqual(rep["time_of_day"]["n_by_bucket_primary"]["PRE (<09:30)"], 1)

    def test_point_in_time_violation_blocks_readiness(self):
        obs, outs = _dataset(runs=25)
        first = next(iter(outs))
        outs[first][0]["evaluation_time"] = "2020-01-01T00:00:00+00:00"
        rep = se.analyze(obs, outs, now=NOW)
        self.assertIn("point_in_time", rep["readiness"]["failed_gates"])
        self.assertEqual(rep["run56"]["action"], "INVESTIGATE_DATA_QUALITY")

    def test_analysis_never_mutates_inputs(self):
        obs, outs = _dataset(runs=4, n_cand=4, n_ctrl=4, n_nm=2)
        before = json.dumps([obs, outs], sort_keys=True, default=str)
        se.analyze(obs, outs, now=NOW)
        self.assertEqual(json.dumps([obs, outs], sort_keys=True, default=str), before)


class ScriptTests(unittest.TestCase):
    def test_replay_writes_json_and_markdown(self):
        from scripts import analyze_signal_effectiveness as script
        obs, outs = _dataset(runs=6, n_cand=6, n_ctrl=6, n_nm=3)
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "snap.json"
            inp.write_text(json.dumps({"observations": obs, "outcomes_by_id": outs}))
            with mock.patch("sys.argv", ["x", "--input", str(inp), "--out", d]):
                self.assertEqual(script.main(), 0)
            rep = json.loads((Path(d) / "run55_signal_effectiveness.json").read_text())
            md = (Path(d) / "run55_signal_effectiveness.md").read_text()
        self.assertIn(rep["verdict"]["verdict"], ("EDGE_DETECTED", "NO_EDGE_DETECTED", "INSUFFICIENT_EVIDENCE"))
        for n in range(1, 15):
            self.assertIn(f"## {n}. ", md)
        self.assertNotIn("nan%", md.lower())


if __name__ == "__main__":
    unittest.main()
