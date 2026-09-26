"""Run 58 — cohort maturation parity & missingness audit (deterministic)."""
import copy
import datetime as dt
import json
import sqlite3
import unittest
from unittest import mock

from analytics import forward_readiness as fr
from analytics import maturation_parity as mp
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS
from scripts import mature_observations as worker

UTC = dt.timezone.utc
HIST_NOW = dt.datetime(2026, 9, 25, 23, 0, tzinfo=UTC)   # before the Run 56 epoch
DAY = dt.datetime(2026, 9, 24, tzinfo=UTC)


def _obs(oid, sym, cohort, *, hhmm=(13, 36), day=DAY, run=None, price=10.0, volume=1e6):
    ts = day.replace(hour=hhmm[0], minute=hhmm[1]).isoformat()
    scanners = ([{"name": "breakout", "version": "breakout-1", "score": 20.0, "direction": "long"}]
                if cohort != CONTROL else [])
    return {"observation_id": oid, "symbol": sym, "timestamp": ts, "scan_timestamp": ts,
            "research_cohort": cohort, "scanners": scanners,
            "market": {"price": price, "volume": volume},
            "market_context": {"scan_id": run or ts, "research_cohort": cohort}}


def _oc(oid, h, raw=0.01):
    return {"observation_id": oid, "horizon": h, "raw_return": raw, "directional_return": raw,
            "mfe": 0.02, "mae": -0.01, "data_status": "MATURED",
            "evaluation_time": "2026-09-24T15:00:00+00:00"}


def cohort_set(n, share, *, cohort, prefix, runs=4, **kw):
    """n observations per run for `runs` runs; `share` of each run matured at all horizons."""
    obs, outs = [], {}
    for r in range(runs):
        hh = 13 + 3 * (r % 3)
        for i in range(n):
            oid = f"{prefix}{r}-{i}"
            obs.append(_obs(oid, f"{prefix}{r}{i}", cohort, hhmm=(hh, 36), run=f"run{r}", **kw))
            if i < round(n * share):
                outs[oid] = [_oc(oid, h) for h in mp.HORIZONS]
    return obs, outs


def dataset(shares=(0.5, 0.5, 0.5), n=(40, 30, 40), **kw):
    obs, outs = [], {}
    for c, s, k, p in zip(mp.COHORTS, shares, n, ("CA", "NM", "CT")):
        o, x = cohort_set(k, s, cohort=c, prefix=p, **kw)
        obs += o
        outs.update(x)
    return obs, outs


class ParityClassTests(unittest.TestCase):
    def test_classes(self):
        self.assertEqual([mp.parity_class(g) for g in (0, 5, 5.01, 10, 10.5, 20, 20.1, None)],
                         ["HEALTHY", "HEALTHY", "ACCEPTABLE", "ACCEPTABLE", "WARNING", "WARNING",
                          "CRITICAL", "NOT_MEASURABLE"])

    def test_equal_maturation_is_healthy(self):
        obs, outs = dataset()
        r = mp.audit_scope(obs, outs, scope="historical", now=HIST_NOW)
        p = r["parity"]["+60m"]
        self.assertEqual((p["candidate_coverage"], p["control_coverage"], p["parity_gap"]), (50.0, 50.0, 0.0))
        self.assertEqual(p["classification"], "HEALTHY")

    def test_gap_calculation(self):
        obs, outs = dataset(shares=(0.75, 0.5, 0.1))
        p = mp.audit_scope(obs, outs, scope="historical", now=HIST_NOW)["parity"]["+5m"]
        self.assertEqual((p["candidate_coverage"], p["near_miss_coverage"], p["control_coverage"]), (75.0, 50.0, 10.0))
        self.assertEqual((p["parity_gap"], p["classification"]), (65.0, "CRITICAL"))


class ScopeTests(unittest.TestCase):
    def test_historical_forward_split(self):
        hist, _ = dataset(runs=1)
        fwd_day = dt.datetime(2026, 9, 28, tzinfo=UTC)
        fwd, _ = dataset(runs=1, day=fwd_day)
        for o in fwd:
            o["observation_id"] = "f" + o["observation_id"]
        allobs = hist + fwd
        self.assertEqual(len(mp.population(allobs, "historical")), len(hist))
        self.assertEqual(len(mp.population(allobs, "forward")), len(fwd))

    def test_zero_forward(self):
        obs, outs = dataset()
        r = mp.audit_scope(obs, outs, scope="forward", now=HIST_NOW)
        self.assertEqual(r, {"status": "NO_FORWARD_DATA", "observations": 0})


class MissingnessTests(unittest.TestCase):
    def _one(self, sym="AAA", matured=(), trace_status=None, now=HIST_NOW):
        o = _obs("x", sym, CONTROL)
        m = {"x": set(matured)}
        tr = {("x", "+60m"): trace_status} if trace_status else None
        return mp.missing_reason(o, "+60m", m, now, tr)

    def test_traced_reasons(self):
        for st, want in (("PRICE_DATA_UNAVAILABLE", "PRICE_DATA_UNAVAILABLE"),
                         ("INSUFFICIENT_FUTURE_BARS", "INSUFFICIENT_FUTURE_BARS"),
                         ("RATE_LIMITED", "RATE_LIMITED"), ("DEFERRED", "SCHEDULER_DEFERRED"),
                         ("MATURED", "PENDING_BACKLOG"), ("RETIRED", "RETIRED_WINDOW_CLOSED")):
            self.assertEqual(self._one(trace_status=st), (want, "traced"))

    def test_structural_and_inferred(self):
        self.assertEqual(self._one(sym="PSA.PRF"), ("INELIGIBLE_SYMBOL", "structural"))
        self.assertEqual(self._one(matured={"+5m"}), ("INSUFFICIENT_FUTURE_BARS", "inferred"))
        self.assertEqual(self._one(), ("UNKNOWN", "inferred"))
        self.assertEqual(self._one(now=HIST_NOW + dt.timedelta(days=7)), ("RETIRED_WINDOW_CLOSED", "structural"))
        o = _obs("y", "AAA", CONTROL)
        o["scan_timestamp"] = o["timestamp"] = "garbage"
        self.assertEqual(mp.missing_reason(o, "+60m", {}, HIST_NOW), ("INVALID_ANCHOR", "structural"))

    def test_not_yet_eligible_excluded_from_denominator(self):
        # settled = anchor + h + 15m slack + 45m grace: +5m at 22:55 (<= 23:00), +60m at 23:50
        o = _obs("z", "AAA", CANDIDATE, hhmm=(21, 50), day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        cov = mp.coverage([o], {}, HIST_NOW)
        self.assertEqual((cov["+60m"][CANDIDATE]["eligible"], cov["+60m"][CANDIDATE]["not_yet_eligible"]), (0, 1))
        self.assertEqual(cov["+5m"][CANDIDATE]["eligible"], 1)


class TimingTests(unittest.TestCase):
    def test_late_day_plus60_structural(self):
        early = _obs("a", "AAA", CONTROL, hhmm=(14, 0))    # 10:00 ET
        late = _obs("b", "BBB", CONTROL, hhmm=(19, 36))    # 15:36 ET
        t = mp.timing([early, late], {}, HIST_NOW)["by_cohort"][CONTROL]
        self.assertEqual((t["plus60_crosses_close_pct"], t["plus30_crosses_close_pct"]), (50.0, 50.0))
        self.assertEqual(t["bucket_counts"]["15:00-16:00"], 1)
        self.assertEqual(t["median_time_et"] in ("10:00", "15:36"), True)


class SchedulerTests(unittest.TestCase):
    def test_cap_ordering_starves_newest_cohort(self):
        # controls old, candidates new -> a tight cap reaches controls first
        ctrl, _ = cohort_set(10, 0.0, cohort=CONTROL, prefix="CT", runs=1)
        cand, _ = cohort_set(10, 0.0, cohort=CANDIDATE, prefix="CA", runs=1,
                             day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        alloc = mp.cap_allocation(ctrl + cand, {}, HIST_NOW, caps=(10, 100))
        self.assertEqual(alloc["10"]["by_cohort"][CONTROL]["reached_pct"], 100.0)
        self.assertEqual(alloc["10"]["by_cohort"][CANDIDATE]["reached_pct"], 0.0)   # candidate starvation
        self.assertEqual(alloc["100"]["reach_gap_pp"], 0.0)

    def test_control_starvation_when_controls_are_newer(self):
        cand, _ = cohort_set(10, 0.0, cohort=CANDIDATE, prefix="CA", runs=1)
        ctrl, _ = cohort_set(10, 0.0, cohort=CONTROL, prefix="CT", runs=1,
                             day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        alloc = mp.cap_allocation(cand + ctrl, {}, HIST_NOW, caps=(10,))
        self.assertEqual(alloc["10"]["by_cohort"][CONTROL]["reached_pct"], 0.0)

    def test_repeated_symbol_rides_along(self):
        # AAA (candidate) has an old pending obs and a new one; unique control ZZZ is also new.
        old = _obs("a1", "AAA", CANDIDATE, day=dt.datetime(2026, 9, 23, tzinfo=UTC))
        new_c = _obs("a2", "AAA", CANDIDATE, day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        new_k = _obs("z1", "ZZZ", CONTROL, day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        tr = []
        worker.mature_observations(mp.with_outcomes([old, new_c, new_k], {}), now=HIST_NOW, dry_run=True,
                                   max_symbols=1, fetch_bars_batch=lambda s, a, b: {}, trace=tr,
                                   retire_after=None, exclusion_reason=lambda s: None)
        st = {(t["observation_id"], t["horizon"]): t["status"] for t in tr}
        self.assertEqual(st[("a2", "+60m")], "PRICE_DATA_UNAVAILABLE")  # reached via its old sibling
        self.assertEqual(st[("z1", "+60m")], "DEFERRED")                # same-age control deferred

    def test_replay_counts_never_attempted(self):
        ctrl, _ = cohort_set(5, 0.0, cohort=CONTROL, prefix="CT", runs=1,
                             day=dt.datetime(2026, 9, 25, tzinfo=UTC))
        cand, outs = cohort_set(5, 1.0, cohort=CANDIDATE, prefix="CA", runs=1)
        runs = [dt.datetime(2026, 9, 24, 22, 0, tzinfo=UTC)]  # before controls existed
        rp = mp.replay(cand + ctrl, mp.matured_index(outs), runs, cap=400)
        self.assertEqual(rp["by_horizon"]["+60m"][CONTROL]["unmatured_eligible"], 0)
        rp2 = mp.replay(cand + ctrl, mp.matured_index(outs), runs + [HIST_NOW], cap=1)
        self.assertEqual(rp2["by_horizon"]["+60m"][CONTROL]["never_attempted"], 5)

    def test_static_audit_flags_loader_window(self):
        items = {x["item"]: x for x in mp.static_audit(1400.0)}
        self.assertIn("3.6 days", items["loader query"]["risk"])


class SharingTests(unittest.TestCase):
    def test_shared_bars_serve_both_cohorts(self):
        a = _obs("c1", "AAA", CANDIDATE)
        b = _obs("k1", "AAA", CONTROL)
        t0 = dt.datetime.fromisoformat(a["scan_timestamp"])
        bars = [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "c": 10 + i * 0.01} for i in range(70)]
        tr = []
        worker.mature_observations(mp.with_outcomes([a, b], {}), now=HIST_NOW, dry_run=True,
                                   fetch_bars_batch=lambda s, x, y: {k: bars for k in s}, trace=tr,
                                   retire_after=None, exclusion_reason=lambda s: None)
        st = {(t["observation_id"], t["horizon"]): t["status"] for t in tr}
        self.assertEqual(st[("c1", "+60m")], "MATURED")
        self.assertEqual(st[("k1", "+60m")], "MATURED")

    def test_same_anchor_mismatch_is_flagged_as_bug(self):
        a, b = _obs("c1", "AAA", CANDIDATE), _obs("k1", "AAA", CONTROL)
        share = mp.symbol_sharing([a, b], {"c1": {"+60m"}}, HIST_NOW)
        self.assertEqual(share["mismatch_cases"], {"SAME_ANCHOR_MISMATCH": 1})
        other = _obs("k2", "AAA", CONTROL, day=dt.datetime(2026, 9, 23, tzinfo=UTC))
        self.assertEqual(mp.symbol_sharing([a, other], {"c1": {"+60m"}}, HIST_NOW)["mismatch_cases"],
                         {"DIFFERENT_SCAN_RUN": 1})


class RetirementTests(unittest.TestCase):
    def test_retirement_is_time_based_and_cohort_neutral(self):
        obs, outs = dataset(shares=(0.5, 0.5, 0.5), runs=1)
        r = mp.retirement(obs, mp.matured_index(outs), HIST_NOW + dt.timedelta(days=7))
        self.assertEqual({c: r[c]["retirement_rate_pct"] for c in mp.COHORTS},
                         {CANDIDATE: 50.0, NEAR_MISS: 50.0, CONTROL: 50.0})
        self.assertEqual(r["retirement_rate_spread_pp"], 0.0)
        self.assertTrue(r["active"])
        self.assertFalse(mp.retirement(obs, mp.matured_index(outs), HIST_NOW)["active"])


class RootCauseTests(unittest.TestCase):
    def _run(self, ctrl_status, cand_status="MATURED", n=150):
        cand, _ = cohort_set(n, 0.0, cohort=CANDIDATE, prefix="CA", runs=1)
        nm, _ = cohort_set(n, 0.0, cohort=NEAR_MISS, prefix="NM", runs=1)
        ctrl, _ = cohort_set(n, 0.0, cohort=CONTROL, prefix="CT", runs=1)
        trace = []
        for o in cand + nm:
            trace += [{"observation_id": o["observation_id"], "horizon": h, "status": cand_status} for h in mp.HORIZONS]
        for o in ctrl:
            trace += [{"observation_id": o["observation_id"], "horizon": h, "status": ctrl_status} for h in mp.HORIZONS]
        # candidates / near-misses already matured in the store; controls not
        outs = {o["observation_id"]: [_oc(o["observation_id"], h) for h in mp.HORIZONS] for o in cand + nm}
        return mp.audit_scope(cand + nm + ctrl, outs, scope="historical", now=HIST_NOW, trace=trace)

    def test_starvation(self):
        r = self._run("MATURED")  # data exists for controls -> pure backlog
        self.assertEqual(r["root_cause"]["classification"], "HISTORICAL_SCHEDULER_STARVATION")
        self.assertEqual(r["projected_parity_after_backlog"]["+60m"]["classification"], "HEALTHY")

    def test_market_data_availability(self):
        r = self._run("PRICE_DATA_UNAVAILABLE")
        self.assertEqual(r["root_cause"]["classification"], "MARKET_DATA_AVAILABILITY_EFFECT")
        self.assertEqual(r["root_cause"]["confidence"], "MODERATE")  # n=150 < 300

    def test_multiple(self):
        cand, _ = cohort_set(400, 0.0, cohort=CANDIDATE, prefix="CA", runs=1)
        nm, _ = cohort_set(400, 0.0, cohort=NEAR_MISS, prefix="NM", runs=1)
        ctrl, _ = cohort_set(400, 0.0, cohort=CONTROL, prefix="CT", runs=1)
        outs = {o["observation_id"]: [_oc(o["observation_id"], h) for h in mp.HORIZONS] for o in cand + nm}
        trace = [{"observation_id": o["observation_id"], "horizon": h,
                  "status": "DEFERRED" if i % 2 else "PRICE_DATA_UNAVAILABLE"}
                 for i, o in enumerate(ctrl) for h in mp.HORIZONS]
        r = mp.audit_scope(cand + nm + ctrl, outs, scope="historical", now=HIST_NOW, trace=trace)
        self.assertEqual(r["root_cause"]["classification"], "MULTIPLE_CAUSES")
        self.assertEqual(r["root_cause"]["confidence"], "HIGH")
        self.assertEqual(set(r["root_cause"]["material_mechanisms"]),
                         {"HISTORICAL_SCHEDULER_STARVATION", "MARKET_DATA_AVAILABILITY_EFFECT"})

    def test_untraced_low_confidence_and_small_insufficient(self):
        obs, outs = dataset(shares=(0.9, 0.9, 0.1), n=(40, 40, 40))
        rc = mp.audit_scope(obs, outs, scope="historical", now=HIST_NOW)["root_cause"]
        self.assertEqual((rc["classification"], rc["confidence"]), ("INSUFFICIENT_DATA", "LOW"))


class ContractTests(unittest.TestCase):
    def test_anti_peeking_output_and_invariance(self):
        obs, outs = dataset(shares=(0.8, 0.5, 0.2))
        a = mp.audit_scope(obs, outs, scope="historical", now=HIST_NOW)
        flipped = copy.deepcopy(outs)
        for ocs in flipped.values():
            for oc in ocs:
                oc["raw_return"] = oc["directional_return"] = -0.5
                oc["mfe"], oc["mae"] = 0.0, -0.9
        b = mp.audit_scope(obs, flipped, scope="historical", now=HIST_NOW)
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))
        self.assertEqual(mp.forbidden_keys(a), [])
        for k in ("win_rate", "mean_return", "mfe_mean", "spearman", "payoff_ratio"):
            with self.assertRaises(ValueError):
                mp.assert_clean({"x": {k: 1}})
        mp.assert_clean({"missing_reasons": {"RETIRED_WINDOW_CLOSED": 1}})

    def test_inputs_not_mutated(self):
        obs, outs = dataset()
        before = json.dumps([obs, outs], sort_keys=True)
        mp.audit_scope(obs, outs, scope="historical", now=HIST_NOW)
        self.assertEqual(json.dumps([obs, outs], sort_keys=True), before)

    def test_trace_does_not_change_worker_report_or_outcomes(self):
        obs, _ = dataset(runs=1)
        t0 = dt.datetime.fromisoformat(obs[0]["scan_timestamp"])
        bars = [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "c": 10 + i * 0.01} for i in range(200)]
        saved_a, saved_b = [], []
        kw = dict(now=HIST_NOW, fetch_bars_batch=lambda s, x, y: {k: bars for k in s},
                  retire_after=None, exclusion_reason=lambda s: None)
        ra = worker.mature_observations(copy.deepcopy(obs), save_fn=lambda o: saved_a.append(o) or True, **kw)
        rb = worker.mature_observations(copy.deepcopy(obs), save_fn=lambda o: saved_b.append(o) or True,
                                        trace=[], **kw)
        for r in (ra, rb):
            r.pop("generated_at")
        self.assertEqual(ra, rb)
        self.assertEqual(saved_a, saved_b)  # outcome formulas / values unchanged

    def test_idempotent_maturation_with_trace(self):
        from db import hsf_observations as store
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            obs, _ = dataset(runs=1, n=(3, 2, 3))
            for o in obs:
                store.save_observation(o, conn=conn)
            t0 = dt.datetime.fromisoformat(obs[0]["scan_timestamp"])
            bars = [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "c": 10 + i * 0.01} for i in range(200)]

            def run(trace):
                return worker.mature_observations(
                    store.load_recent_observations(limit=100, attach_outcomes=True, conn=conn),
                    now=HIST_NOW, fetch_bars_batch=lambda s, x, y: {k: bars for k in s}, trace=trace,
                    save_fn=lambda o: store.save_outcome(o, conn=conn), retire_after=None,
                    exclusion_reason=lambda s: None)
            t1, t2 = [], []
            self.assertEqual(run(t1)["outcomes_matured"], 32)
            self.assertEqual(run(t2)["outcomes_matured"], 0)
            self.assertTrue(t2 and all(t["status"] == "ALREADY" for t in t2))
        finally:
            conn.close()

    def test_audit_touches_no_scanner_code(self):
        import inspect
        src = inspect.getsource(mp)
        for forbidden in ("from scan", "import scan", "research_cohorts import build", "select_control_symbols"):
            self.assertNotIn(forbidden, src)

    def test_run56_surfaces_parity_class_and_capacity(self):
        from tests.test_forward_readiness import dataset as fdataset
        from tests.test_forward_readiness import now_after
        obs, outs = fdataset(3, per=(20, 20, 20), mature=(0.9, 0.9, 0.5))
        r = fr.monitor(obs, outs, now=now_after(3), maturation_report={"symbols_deferred": 12})
        self.assertEqual(r["maturation_parity"]["+60m"]["parity_classification"], "CRITICAL")
        self.assertTrue(r["data_quality"]["maturation_capacity_binding"])
        self.assertEqual(r["epoch"]["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")
        self.assertEqual(fr.forbidden_keys(r), [])


class ScriptTests(unittest.TestCase):
    def test_run55_label_and_full_report(self):
        import tempfile
        from pathlib import Path

        from scripts import audit_maturation_parity as script
        obs, outs = dataset()
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "s.json"
            snap.write_text(json.dumps({"observations": obs, "outcomes_by_id": outs}))
            with mock.patch("sys.argv", ["x", "--input", str(snap), "--label", "run55",
                                         "--as-of", HIST_NOW.isoformat(), "--out", d]):
                self.assertEqual(script.main(), 0)
            sec = Path(d) / "maturation_parity_run55_snapshot.json"
            with mock.patch("sys.argv", ["x", "--input", str(snap), "--as-of", HIST_NOW.isoformat(),
                                         "--run55-section", str(sec), "--out", d]):
                self.assertEqual(script.main(), 0)
            rep = json.loads((Path(d) / "maturation_parity_audit.json").read_text())
            md = (Path(d) / "maturation_parity_audit.md").read_text()
        self.assertEqual(rep["schema"], "hsf-maturation-parity-1.0")
        self.assertFalse(rep["fix_applied"])
        self.assertEqual(rep["forward_parity_status"], "NO_FORWARD_DATA")
        self.assertIsNotNone(rep["run55_snapshot"])
        self.assertIn("## Run 55 snapshot", md)
        self.assertEqual(mp.forbidden_keys(rep), [])


if __name__ == "__main__":
    unittest.main()
