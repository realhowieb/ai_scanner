"""Run 56 — forward-evidence readiness monitor (deterministic, anti-peeking)."""
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analytics import forward_readiness as fr
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS

UTC = dt.timezone.utc
EPOCH = fr.epoch_start()
SLOTS = ((13, 35), (16, 35), (19, 35))  # 09:35 / 12:35 / 15:35 EDT


def trading_days(n, start=dt.date(2026, 9, 28)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _obs(oid, sym, cohort, run, ts, direction="long", versions=True):
    scanners = ([{"name": "breakout", "version": "breakout-1", "score": 20.0, "direction": direction}]
                if cohort != CONTROL else [])
    rec = {"observation_id": oid, "symbol": sym, "timestamp": ts, "scan_timestamp": ts,
           "research_cohort": cohort, "market": {"price": 10.0, "volume": 1000},
           "scanners": scanners, "market_context": {"scan_id": run, "research_cohort": cohort}}
    if versions:
        rec["versions"] = {"hsf_score": "1.0", "prebreakout_model": "prebreakout-xgb-v16"}
    return rec


def _oc(oid, h, ts, raw, *, direction="long", full=True):
    a = dt.datetime.fromisoformat(ts)
    dr = -raw if direction == "short" else raw
    return {"observation_id": oid, "horizon": h, "observation_timestamp": ts,
            "evaluation_time": (a + dt.timedelta(minutes=fr.HORIZON_MIN[h])).isoformat(),
            "raw_return": raw, "directional_return": dr if full else None,
            "mfe": 0.01 if full else None, "mae": -0.01 if full else None, "data_status": "MATURED"}


def dataset(n_days, *, per=(10, 5, 10), mature=(1.0, 1.0, 1.0), short_every=0, seed_ret=0.003):
    """n_days of forward trading days, 3 runs/day; maturation share per cohort."""
    obs, outs = [], {}
    k = 0
    for day in trading_days(n_days):
        for hh, mm in SLOTS:
            ts = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=UTC).isoformat()
            run = f"{day.isoformat()}T{hh:02d}"
            for cohort, n, share in zip((CANDIDATE, NEAR_MISS, CONTROL), per, mature):
                for i in range(n):
                    k += 1
                    oid = f"{run}-{cohort}-{i}"
                    direction = "short" if short_every and cohort != CONTROL and i % short_every == 0 else "long"
                    obs.append(_obs(oid, f"S{cohort[:2]}{i}", cohort, run, ts, direction))
                    if i < round(n * share):
                        outs[oid] = [_oc(oid, h, ts, seed_ret * ((k % 7) - 3), direction=direction)
                                     for h in fr.HORIZONS]
    return obs, outs


def now_after(n_days):
    last = trading_days(n_days)[-1]
    return dt.datetime(last.year, last.month, last.day, 23, 30, tzinfo=UTC)


class EpochTests(unittest.TestCase):
    def test_epoch_boundary_and_pre_epoch_excluded(self):
        at = EPOCH.isoformat()
        before = (EPOCH - dt.timedelta(seconds=1)).isoformat()
        obs = [_obs("a", "AAA", CANDIDATE, "r1", at), _obs("b", "BBB", CANDIDATE, "r0", before),
               {**_obs("c", "CCC", CANDIDATE, "r1", at), "research_cohort": None,
                "market_context": {"scan_id": "r1"}}]
        sel = fr.select_forward(obs)
        self.assertEqual([o["observation_id"] for o in sel["forward"]], ["a"])
        self.assertEqual((sel["pre_epoch_excluded"], sel["legacy_excluded"]), (1, 1))

    def test_epoch_metadata(self):
        r = fr.monitor([], {}, now=now_after(1))
        e = r["epoch"]
        self.assertEqual(e["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")
        self.assertEqual(e["run55_evaluation_commit"], "284e2ac8ec8640d73679c55555772eeda2505485")
        self.assertEqual((e["research_schema_version"], e["outcome_schema_version"]),
                         ("hsf-obs-1.0", "hsf-outcome-1.0"))
        self.assertIsNone(e["forward_epoch_start_scan_run"])


class CountingTests(unittest.TestCase):
    def test_trading_day_counting(self):
        obs, outs = dataset(6)  # Mon 9/28 .. Mon 10/5, weekend skipped
        r = fr.monitor(obs, outs, now=now_after(6))
        t = r["time_coverage"]
        self.assertEqual((t["forward_trading_days"], t["completed_forward_trading_days"]), (6, 6))
        self.assertEqual(t["calendar_days"], 8)
        mid = dt.datetime(2026, 10, 5, 18, 0, tzinfo=UTC)  # 14:00 ET on the last day
        self.assertEqual(fr.monitor(obs, outs, now=mid)["time_coverage"]["completed_forward_trading_days"], 5)

    def test_cohort_and_run_counting(self):
        obs, outs = dataset(2, per=(4, 2, 3), mature=(1.0, 0.5, 0.0))
        r = fr.monitor(obs, outs, now=now_after(2))
        self.assertEqual(r["scan_coverage"]["regular_session_scan_runs"], 6)
        self.assertEqual(r["scan_coverage"]["successful_scan_runs"], 6)
        c = r["cohorts"]
        self.assertEqual((c[CANDIDATE]["observations"], c[CANDIDATE]["matured_observations"]), (24, 24))
        self.assertEqual((c[NEAR_MISS]["matured_observations"], c[NEAR_MISS]["unmatured_observations"]), (6, 6))
        self.assertEqual(c[CONTROL]["matured_observations"], 0)

    def test_horizon_coverage_respects_eligibility(self):
        ts = "2026-09-28T13:35:00+00:00"
        obs = [_obs("a", "AAA", CANDIDATE, "r1", ts)]
        outs = {"a": [_oc("a", "+5m", ts, 0.01)]}
        # settled = anchor + h + 15m slack + 45m grace: +5m 14:40, +15m 14:50, +30m 15:05
        now = dt.datetime(2026, 9, 28, 14, 55, tzinfo=UTC)
        r = fr.monitor(obs, outs, now=now)
        h = r["horizons"]
        self.assertEqual((h["+5m"][CANDIDATE]["eligible_observations"], h["+5m"][CANDIDATE]["maturation_pct"]), (1, 100.0))
        self.assertEqual(h["+15m"][CANDIDATE]["maturation_pct"], 0.0)
        self.assertEqual(h["+15m"][CANDIDATE]["unmatured_reasons"]["INSUFFICIENT_FUTURE_BARS"], 1)
        self.assertEqual(h["+30m"][CANDIDATE]["unmatured_reasons"]["NOT_YET_ELIGIBLE"], 1)
        self.assertEqual(h["+30m"][CANDIDATE]["eligible_observations"], 0)
        self.assertIsNone(h["+60m"][CANDIDATE]["maturation_pct"])

    def test_maturation_parity_gap(self):
        obs, outs = dataset(3, per=(20, 20, 20), mature=(0.9, 0.8, 0.5))  # >= 100 settled each
        p = fr.monitor(obs, outs, now=now_after(3))["maturation_parity"]["+60m"]
        self.assertEqual((p["candidate_maturation_pct"], p["near_miss_maturation_pct"],
                          p["control_maturation_pct"], p["maturation_parity_gap"]), (90.0, 80.0, 50.0, 40.0))
        self.assertTrue(p["measurable"])

    def test_retired_and_policy_reasons(self):
        ts = "2026-09-28T13:35:00+00:00"
        obs = [_obs("a", "AAA", CONTROL, "r1", ts), _obs("p", "PSA.PRF", CONTROL, "r1", ts)]
        r = fr.monitor(obs, {}, now=dt.datetime(2026, 10, 6, 12, 0, tzinfo=UTC))
        reasons = r["horizons"]["+60m"][CONTROL]["unmatured_reasons"]
        self.assertEqual((reasons["RETIRED"], reasons["FILTERED_BY_POLICY"]), (1, 1))
        self.assertEqual(r["horizons"]["+60m"][CONTROL]["eligible_observations"], 2)


class DirectionTests(unittest.TestCase):
    def test_zero_short_is_insufficient_and_does_not_block_long(self):
        obs, outs = dataset(20)
        r = fr.monitor(obs, outs, now=now_after(20))
        self.assertEqual(r["directions"]["SHORT"]["observations"], 0)
        self.assertEqual((r["long_readiness"], r["short_readiness"]), ("READY", "INSUFFICIENT"))
        self.assertEqual(r["state"], "READY_FOR_RUN55_RERUN")

    def test_some_short_is_collecting(self):
        obs, outs = dataset(3, short_every=5)
        r = fr.monitor(obs, outs, now=now_after(3))
        self.assertGreater(r["directions"]["SHORT"]["observations"], 0)
        self.assertEqual(r["short_readiness"], "COLLECTING")
        self.assertEqual(r["data_quality"]["direction_transform_mismatches"], 0)


class StateTests(unittest.TestCase):
    def test_zero_observations(self):
        r = fr.monitor([], {}, now=now_after(1))
        self.assertEqual((r["state"], r["limiting_factor"]), ("COLLECTING", "NO_FORWARD_DATA"))
        self.assertEqual(r["estimated_trading_days_until_ready"], "UNKNOWN")
        self.assertFalse(r["RUN55_RERUN_RECOMMENDED"])
        json.dumps(r, allow_nan=False, default=str)

    def test_healthy_but_insufficient_time(self):
        obs, outs = dataset(5)
        r = fr.monitor(obs, outs, now=now_after(5))
        self.assertEqual((r["state"], r["limiting_factor"]), ("COLLECTING", "NOT_ENOUGH_TIME"))
        failed = {k for k, g in r["gates"].items() if g["status"] == "FAIL"}
        self.assertEqual(failed, {"A_trading_days", "B_scan_runs", "C_cohort_clusters", "G_effective_clusters"})
        # A: 5 more days; B: (50-15)/3 -> 12; C: (30-15)/3 -> 5; G: (20-15)/3 -> 2
        self.assertEqual(r["estimated_trading_days_until_ready"], 12)

    def test_insufficient_clusters(self):
        obs, outs = dataset(1, per=(200, 100, 200))  # lots of rows, only 3 runs
        r = fr.monitor(obs, outs, now=now_after(1))
        self.assertEqual(r["gates"]["C_cohort_clusters"]["value"], 3)
        self.assertEqual(r["gates"]["G_effective_clusters"]["status"], "FAIL")
        self.assertEqual(r["estimated_trading_days_until_ready"], "UNKNOWN")  # 1 day: no rate yet

    def test_data_quality_blocked_by_parity(self):
        obs, outs = dataset(5, per=(10, 10, 10), mature=(0.9, 0.9, 0.1))
        r = fr.monitor(obs, outs, now=now_after(5))
        self.assertEqual((r["state"], r["limiting_factor"]), ("DATA_QUALITY_BLOCKED", "DATA_PIPELINE_BIAS"))
        self.assertEqual(r["gates"]["E_maturation_parity"]["status"], "FAIL")
        self.assertEqual(r["estimated_trading_days_until_ready"], "UNKNOWN")
        self.assertEqual(r["long_readiness"], "INSUFFICIENT")

    def test_early_imbalance_not_measurable_is_not_blocked(self):
        obs, outs = dataset(1, per=(10, 5, 10), mature=(1.0, 1.0, 0.0))
        obs = [o for o in obs if o["market_context"]["scan_id"].endswith("T13")]
        r = fr.monitor(obs, outs, now=now_after(1))
        self.assertFalse(r["maturation_parity"]["+60m"]["measurable"])
        self.assertEqual(r["state"], "COLLECTING")

    def test_data_quality_blocked_by_integrity(self):
        obs, outs = dataset(3)
        first = next(iter(outs))
        outs[first][0]["evaluation_time"] = "2020-01-01T00:00:00+00:00"
        r = fr.monitor(obs, outs, now=now_after(3))
        self.assertEqual(r["gates"]["H_research_integrity"]["status"], "FAIL")
        self.assertEqual((r["state"], r["limiting_factor"]), ("DATA_QUALITY_BLOCKED", "DATA_INTEGRITY"))

    def test_approaching_ready(self):
        obs, outs = dataset(13)  # 39 runs: B fails at 78% progress, everything else passes
        r = fr.monitor(obs, outs, now=now_after(13))
        self.assertEqual(r["state"], "APPROACHING_READY")
        self.assertEqual([k for k, g in r["gates"].items() if g["status"] == "FAIL"], ["B_scan_runs"])
        self.assertEqual(r["estimated_trading_days_until_ready"], 4)  # (50-39)/3

    def test_fully_ready(self):
        obs, outs = dataset(20)
        r = fr.monitor(obs, outs, now=now_after(20))
        self.assertEqual(r["state"], "READY_FOR_RUN55_RERUN")
        self.assertTrue(r["RUN55_RERUN_RECOMMENDED"])
        self.assertEqual(r["gates"]["B_scan_runs"]["status"], "WARN")  # 60 < preferred 100
        self.assertEqual(r["estimated_trading_days_until_ready"], 0)

    def test_directional_coverage_gate(self):
        obs, outs = dataset(5)
        for ocs in outs.values():
            for oc in ocs:
                oc["mfe"] = None
        r = fr.monitor(obs, outs, now=now_after(5))
        self.assertEqual(r["gates"]["F_directional_integrity"]["status"], "FAIL")
        self.assertEqual(r["state"], "DATA_QUALITY_BLOCKED")


class HygieneTests(unittest.TestCase):
    def test_duplicates(self):
        obs, outs = dataset(2)
        oid = obs[0]["observation_id"]
        obs.append(copy.deepcopy(obs[0]))
        outs[oid].append(dict(outs[oid][0]))  # identical duplicate outcome
        r = fr.monitor(obs, outs, now=now_after(2))
        dq = r["data_quality"]
        self.assertEqual((dq["duplicate_observation_ids"], dq["conflicting_observation_ids"]), (1, 0))
        self.assertEqual((dq["duplicate_outcomes"], dq["conflicting_outcomes"]), (1, 0))
        self.assertEqual(r["gates"]["H_research_integrity"]["status"], "WARN")
        outs[oid].append(dict(outs[oid][0], raw_return=0.5))  # conflicting
        r2 = fr.monitor(obs, outs, now=now_after(2))
        self.assertEqual(r2["gates"]["H_research_integrity"]["status"], "FAIL")

    def test_missing_optional_metadata(self):
        obs, outs = dataset(2)
        for o in obs:
            o.pop("versions", None)
            o.pop("market", None)
            o["market_context"].pop("scan_id", None)
        r = fr.monitor(obs, outs, now=now_after(2))
        self.assertEqual(r["scan_coverage"]["regular_session_scan_runs"], 6)  # falls back to scan_timestamp
        self.assertFalse(r["data_quality"]["scoring_version_drift"])

    def test_scoring_version_drift_warns(self):
        obs, outs = dataset(2)
        obs[0]["scanners"][0]["version"] = "breakout-2"
        r = fr.monitor(obs, outs, now=now_after(2))
        self.assertTrue(r["data_quality"]["scoring_version_drift"])
        self.assertIn("scoring version changed", r["gates"]["H_research_integrity"]["detail"])

    def test_inputs_not_mutated(self):
        obs, outs = dataset(2)
        before = json.dumps([obs, outs], sort_keys=True)
        fr.monitor(obs, outs, now=now_after(2))
        self.assertEqual(json.dumps([obs, outs], sort_keys=True), before)


class AntiPeekingTests(unittest.TestCase):
    def test_output_is_invariant_to_returns(self):
        obs, outs = dataset(6)
        flipped = copy.deepcopy(outs)
        for ocs in flipped.values():
            for oc in ocs:
                oc["raw_return"] = -3.0 * oc["raw_return"] + 0.02
                oc["directional_return"] = oc["raw_return"]
                oc["mfe"], oc["mae"] = 0.2, -0.0001
        a = fr.monitor(obs, outs, now=now_after(6))
        b = fr.monitor(obs, flipped, now=now_after(6))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_no_prohibited_keys_anywhere(self):
        obs, outs = dataset(20, short_every=4)
        for r in (fr.monitor(obs, outs, now=now_after(20)), fr.monitor([], {}, now=now_after(1))):
            self.assertEqual(fr.forbidden_keys(r), [])
            text = json.dumps(r).lower()
            for bad in ("win_rate", "mean_return", "spearman", "payoff", "optimal", "best_"):
                self.assertNotIn(bad, text)

    def test_guard_rejects_effectiveness_keys(self):
        for key in ("win_rate", "candidate_mean", "spearman", "return_diff", "best_bucket",
                    "optimal_threshold", "payoff_ratio", "directional_return"):
            with self.assertRaises(ValueError, msg=key):
                fr.assert_no_effectiveness_metrics({"horizons": {"+60m": {key: 0.1}}})
        fr.assert_no_effectiveness_metrics({"directional_field_coverage_pct": {"directional_return": 99.0}})


class ScriptTests(unittest.TestCase):
    def test_replay_writes_artifacts_and_github_output(self):
        from scripts import forward_evidence_readiness as script
        obs, outs = dataset(20)
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "snap.json"
            snap.write_text(json.dumps({"observations": obs, "outcomes_by_id": outs}))
            gh = Path(d) / "gh_out"
            with mock.patch("sys.argv", ["x", "--input", str(snap), "--out", d]), \
                    mock.patch.dict("os.environ", {"GITHUB_OUTPUT": str(gh)}), \
                    mock.patch.object(script, "monitor",
                                      side_effect=lambda o, c, **kw: fr.monitor(o, c, now=now_after(20), **kw)):
                self.assertEqual(script.main(), 0)
            rep = json.loads((Path(d) / "forward_evidence_readiness.json").read_text())
            md = (Path(d) / "forward_evidence_readiness.md").read_text()
            self.assertIn("RUN55_RERUN_RECOMMENDED=true", gh.read_text())
        self.assertEqual(rep["schema"], "hsf-forward-readiness-1.0")
        self.assertIn("READY_FOR_RUN55_RERUN", md)
        self.assertEqual(fr.forbidden_keys(rep), [])


if __name__ == "__main__":
    unittest.main()
