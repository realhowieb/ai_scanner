"""Run 59 — system health control plane: failure-injection scenarios."""
import copy
import datetime as dt
import importlib.util
import json
import unittest
from unittest import mock

from analytics import market_calendar as mc
from analytics import system_health as sh

_STREAMLIT = importlib.util.find_spec("streamlit") is not None
UTC = dt.timezone.utc
WED = dt.datetime(2026, 9, 30, 23, 50, tzinfo=UTC)      # trading day, after all cycles


def _slot_runs(end: dt.datetime, days: int = 6, conclusion="success"):
    out = []
    d = (end - dt.timedelta(days=days)).date()
    while d <= end.date():
        for s in mc.expected_scan_slots(d):
            if s + dt.timedelta(minutes=5) <= end:
                out.append({"created_at": (s + dt.timedelta(seconds=10)).isoformat(),
                            "updated_at": (s + dt.timedelta(minutes=3)).isoformat(),
                            "conclusion": conclusion, "status": "completed", "event": "workflow_dispatch"})
        d += dt.timedelta(days=1)
    return out


def _daily_runs(end: dt.datetime, hh: int, mm: int, days: int = 6, event="schedule"):
    out = []
    d = (end - dt.timedelta(days=days)).date()
    while d <= end.date():
        t = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=UTC)
        if mc.is_trading_day(d) and t <= end:
            out.append({"created_at": t.isoformat(), "updated_at": (t + dt.timedelta(minutes=4)).isoformat(),
                        "conclusion": "success", "status": "completed", "event": event})
        d += dt.timedelta(days=1)
    return out


def _per_scan(end: dt.datetime, days: int = 3):
    out = []
    d = (end - dt.timedelta(days=days)).date()
    while d <= end.date():
        for h, m in ((13, 35), (16, 35), (19, 35)):
            t = dt.datetime(d.year, d.month, d.day, h, m, tzinfo=UTC) + dt.timedelta(minutes=1)
            if mc.is_trading_day(d) and t <= end:
                out.append({"scan_time": t.isoformat(), "n": 250,
                            "by_cohort": {"CANDIDATE": 100, "NEAR_MISS": 50, "CONTROL": 100}})
        d += dt.timedelta(days=1)
    return out


def _readiness(state="COLLECTING", limiting="NOT_ENOUGH_TIME", days=3, runs=9, parity=None):
    parity = parity or {h: {"candidate_maturation_pct": None, "near_miss_maturation_pct": None,
                            "control_maturation_pct": None, "maturation_parity_gap": None,
                            "measurable": False, "parity_classification": "NOT_MEASURABLE"}
                        for h in ("+5m", "+15m", "+30m", "+60m")}
    return {"generated_at": WED.isoformat(), "state": state, "limiting_factor": limiting,
            "state_reason": "test", "epoch": {"forward_epoch_start_timestamp": "2026-09-26T07:23:11+00:00"},
            "time_coverage": {"completed_forward_trading_days": days},
            "scan_coverage": {"regular_session_scan_runs": runs, "forward_observations_regular_session": 700},
            "gates": {g: {"status": "FAIL"} for g in sh._READINESS_WHITELIST_GATES},
            "horizons": {h: {c: {"maturation_pct": 90.0} for c in ("CANDIDATE", "NEAR_MISS", "CONTROL")}
                         for h in ("+5m", "+15m", "+30m", "+60m")},
            "maturation_parity": parity, "estimated_trading_days_until_ready": 12,
            "long_readiness": "COLLECTING", "short_readiness": "INSUFFICIENT"}


def baseline(now: dt.datetime = WED):
    scans = _slot_runs(now)
    mat = _daily_runs(now, 22, 30)
    return {
        "now": now.isoformat(),
        "universe": {"symbol_count": 11000, "source": "live", "generated_at": now.isoformat(),
                     "exclusions": {"preferred_share": 400, "inactive": 50, "non_tradable": 20},
                     "provider_assets": 12000, "duplicates": 0, "malformed": 0},
        "previous": {"generated_at": (now - dt.timedelta(hours=6)).isoformat(),
                     "subsystems": {"universe": {"metrics": {"symbol_count": 11050}},
                                    "maturation": {"metrics": {"ready_symbols": 1500, "retired_observations": 10}}}},
        "scan_runs": scans,
        "db_runs": [{"created_at": now.isoformat(), "label": "US_MARKET", "row_count": 100, "duration_sec": 90}],
        "observations_summary": {"observations": 2250, "by_cohort": {"CANDIDATE": 900, "NEAR_MISS": 450, "CONTROL": 900},
                                 "untagged": 0, "duplicate_ids": 0, "malformed": 0, "metadata_scope_n": 2250,
                                 "metadata_block_pct": 100.0, "per_scan": _per_scan(now),
                                 "latest_created_at": now.isoformat()},
        "maturation_report": {"schema": "hsf-maturation-1.2", "generated_at": mat[-1]["created_at"], "dry_run": False,
                              "alpaca_requests": 25, "alpaca_429_count": 0, "alpaca_retry_count": 0,
                              "rate_limited_symbols": 0, "provider_error_symbols": 0,
                              "price_data_unavailable_symbols": 100, "symbols_processed": 1500,
                              "symbols_deferred": 0, "outcomes_matured": 4000,
                              "backlog": {"ready_symbols": 1500, "deferred_symbols": 0},
                              "retired": {"observations": 12}},
        "maturation_runs": mat,
        "db_probe": {"connected": True, "latency_ms": 120, "observations_accessible": True,
                     "outcomes_accessible": True, "duplicate_outcome_keys": 0,
                     "latest_observation_created_at": now.isoformat(),
                     "null_rates_pct": {"symbol": 0.0, "timestamp": 0.0, "scan_id": 0.0, "market.price": 0.0}},
        "workflows": {
            "scheduled-scans.yml": scans,
            "mature-observations.yml": mat,
            "refresh-universe.yml": [{"created_at": "2026-09-27T10:30:00+00:00", "conclusion": "success",
                                      "status": "completed", "event": "schedule"}],
            "forward-evidence-readiness.yml": _daily_runs(now, 23, 15),
            "maturation-parity-audit.yml": [{"created_at": "2026-09-26T08:30:00+00:00", "conclusion": "success",
                                             "status": "completed", "event": "workflow_dispatch"}],
            "system-health.yml": _daily_runs(now, 17, 5),
        },
        "readiness": _readiness(),
        "parity": {"generated_at": (now - dt.timedelta(days=2)).isoformat(),
                   "current_historical": {"parity": {h: {"parity_gap": 43.7, "classification": "CRITICAL"}
                                                     for h in ("+5m", "+15m", "+30m", "+60m")}},
                   "root_cause": {"classification": "MARKET_DATA_AVAILABILITY_EFFECT", "confidence": "HIGH"}},
        "artifacts": {"forward_evidence_readiness": {"generated_at": _daily_runs(now, 23, 15)[-1]["created_at"]},
                      "maturation_parity_audit": {"generated_at": (now - dt.timedelta(days=2)).isoformat()},
                      "latest_scanner_observation": {"generated_at": now.isoformat()},
                      "previous_system_health": {"generated_at": (now - dt.timedelta(hours=6)).isoformat()}},
    }


def run(inputs):
    return sh.evaluate(inputs)


def codes(r):
    return {i["incident_id"].split(":")[1] for i in r["incidents"]}


class CalendarTests(unittest.TestCase):
    def test_calendar(self):
        self.assertFalse(mc.is_trading_day(dt.date(2026, 11, 26)))   # Thanksgiving
        self.assertFalse(mc.is_trading_day(dt.date(2026, 10, 3)))    # Saturday
        self.assertTrue(mc.is_trading_day(dt.date(2026, 9, 30)))
        self.assertEqual(mc.close_time_et(dt.date(2026, 11, 27)), dt.time(13, 0))
        fri_close = dt.datetime(2026, 10, 2, 21, 0, tzinfo=UTC)
        self.assertEqual(mc.completed_trading_days_since(fri_close, dt.datetime(2026, 10, 4, 12, tzinfo=UTC)), 0)
        self.assertEqual(mc.next_expected_scan(dt.datetime(2026, 10, 3, 12, tzinfo=UTC)),
                         dt.datetime(2026, 10, 5, 12, 35, tzinfo=UTC))


class ScenarioTests(unittest.TestCase):
    def test_01_everything_healthy(self):
        r = run(baseline())
        self.assertEqual((r["system_status"], r["human_action"]), ("HEALTHY", "NO_ACTION"))
        self.assertEqual(r["incidents"], [])
        self.assertEqual(r["health_score"], 100)
        self.assertEqual(r["subsystems"]["forward_evidence"]["status"], "WAITING")
        self.assertEqual(r["autonomy_readiness"]["state"], "OBSERVABLE")

    def test_02_weekend_no_scans_expected(self):
        sat = dt.datetime(2026, 10, 3, 15, 0, tzinfo=UTC)
        b = baseline(dt.datetime(2026, 10, 2, 23, 50, tzinfo=UTC))
        b["now"] = sat.isoformat()
        r = run(b)
        self.assertEqual(r["system_status"], "HEALTHY", r["incidents"])
        self.assertFalse(r["market"]["is_trading_day"])
        self.assertEqual(r["market"]["next_expected_scan"], "2026-10-05T12:35:00+00:00")

    def test_03_holiday_no_scans_expected(self):
        wed = dt.datetime(2026, 11, 25, 23, 50, tzinfo=UTC)
        b = baseline(wed)
        b["now"] = dt.datetime(2026, 11, 26, 18, 0, tzinfo=UTC).isoformat()  # Thanksgiving
        r = run(b)
        self.assertNotIn("MISSED_SCAN", codes(r))
        self.assertEqual(r["subsystems"]["scanner"]["status"], "HEALTHY")

    def test_04_missed_one_run(self):
        b = baseline()
        b["scan_runs"] = [x for x in b["scan_runs"] if not x["created_at"].startswith("2026-09-30T16:35")]
        r = run(b)
        self.assertIn("MISSED_SCAN", codes(r))
        self.assertEqual(r["system_status"], "DEGRADED")
        self.assertTrue(any(i["automation_candidate"] for i in r["incidents"]))

    def test_05_failed_repeatedly(self):
        b = baseline()
        for x in b["scan_runs"][-4:]:
            x["conclusion"] = "failure"
        r = run(b)
        self.assertEqual(r["subsystems"]["scanner"]["status"], "ACTION_REQUIRED")
        self.assertEqual(r["human_action"], "HUMAN_ACTION_REQUIRED")

    def test_06_zero_candidates_but_healthy_scanner(self):
        b = baseline()
        b["db_runs"] = [{"created_at": WED.isoformat(), "label": "US_MARKET", "row_count": 15, "duration_sec": 90}]
        for p in b["observations_summary"]["per_scan"]:
            p["by_cohort"]["CANDIDATE"] = 0
        r = run(b)
        self.assertEqual(r["subsystems"]["scanner"]["status"], "HEALTHY")

    def test_07_zero_observations_unexpectedly(self):
        b = baseline()
        b["observations_summary"]["per_scan"] = []
        b["observations_summary"]["observations"] = 0
        r = run(b)
        self.assertIn("ZERO_OBSERVATIONS", codes(r))
        self.assertEqual(r["subsystems"]["research_capture"]["status"], "DEGRADED")

    def test_08_stale_universe(self):
        b = baseline()
        b["universe"].update({"source": "cached", "cached_at": "2026-09-20T00:00:00+00:00"})
        r = run(b)
        self.assertEqual(r["subsystems"]["universe"]["detail_state"], "STALE")

    def test_09_suspicious_shrink(self):
        b = baseline()
        b["universe"]["symbol_count"] = 9500
        r = run(b)
        self.assertEqual(r["subsystems"]["universe"]["detail_state"], "SUSPICIOUS")
        b["universe"]["symbol_count"] = 5000
        self.assertEqual(run(b)["subsystems"]["universe"]["detail_state"], "BROKEN")

    def test_10_database_unavailable(self):
        b = baseline()
        b["db_probe"] = {"connected": False, "error": "timeout"}
        r = run(b)
        self.assertEqual(r["system_status"], "ACTION_REQUIRED")
        self.assertEqual(r["incidents"][0]["incident_id"], "database:DATABASE_UNAVAILABLE")

    def test_11_transient_429_recovered(self):
        b = baseline()
        b["maturation_report"]["alpaca_429_count"] = 2
        r = run(b)
        self.assertEqual(r["system_status"], "HEALTHY")
        inc = [i for i in r["incidents"] if i["incident_id"].endswith("RATE_LIMIT_RECOVERED")]
        self.assertEqual((inc[0]["severity"], inc[0]["human_action"]), ("INFO", "NO_ACTION"))

    def test_12_persistent_alpaca_failures(self):
        b = baseline()
        b["maturation_report"].update({"rate_limited_symbols": 300, "alpaca_429_count": 40})
        r = run(b)
        self.assertIn("RATE_LIMIT_PRESSURE", codes(r))
        self.assertIn("ALPACA_PERSISTENT_FAILURES", codes(r))
        self.assertEqual(r["subsystems"]["market_data"]["status"], "DEGRADED")

    def test_13_backlog_growing(self):
        b = baseline()
        b["maturation_report"].update({"symbols_deferred": 800})
        b["maturation_report"]["backlog"].update({"ready_symbols": 2800, "deferred_symbols": 800})
        r = run(b)
        self.assertIn("BACKLOG_GROWING", codes(r))

    def test_14_maturation_stale(self):
        b = baseline()
        b["maturation_runs"] = [x for x in b["maturation_runs"] if x["created_at"] < "2026-09-29"]
        b["maturation_report"]["generated_at"] = b["maturation_runs"][-1]["created_at"]
        r = run(b)
        stale = [i for i in r["incidents"] if i["incident_id"] == "maturation:MATURATION_STALE"]
        self.assertTrue(stale and stale[0]["automation_candidate"])
        self.assertIn(r["automation_candidates"][0]["subsystem"], {"maturation", "workflows"})

    def test_15_historical_parity_critical_but_explained(self):
        r = run(baseline())
        cp = r["subsystems"]["cohort_parity"]
        self.assertEqual(cp["status"], "WAITING")
        self.assertTrue(cp["metrics"]["historical_parity_explained"])
        self.assertEqual(r["system_status"], "HEALTHY")

    def _fwd_parity(self, gap, cls):
        return {h: {"candidate_maturation_pct": 90.0, "near_miss_maturation_pct": 88.0,
                    "control_maturation_pct": 90.0 - gap, "maturation_parity_gap": gap,
                    "measurable": True, "parity_classification": cls} for h in ("+5m", "+15m", "+30m", "+60m")}

    def test_16_forward_parity_healthy(self):
        b = baseline()
        b["readiness"] = _readiness(parity=self._fwd_parity(3.0, "HEALTHY"))
        r = run(b)
        self.assertEqual(r["subsystems"]["cohort_parity"]["detail_state"], "HEALTHY")

    def test_17_forward_parity_warning(self):
        b = baseline()
        b["readiness"] = _readiness(parity=self._fwd_parity(15.0, "WARNING"))
        r = run(b)
        self.assertEqual(r["subsystems"]["cohort_parity"]["status"], "DEGRADED")
        b["readiness"] = _readiness(parity=self._fwd_parity(60.0, "CRITICAL"))
        r = run(b)
        self.assertEqual(r["subsystems"]["cohort_parity"]["status"], "ACTION_REQUIRED")
        self.assertEqual(r["human_action"], "HUMAN_ACTION_REQUIRED")

    def test_18_no_forward_data_yet(self):
        b = baseline()
        b["readiness"] = _readiness(limiting="NO_FORWARD_DATA", days=0, runs=0)
        r = run(b)
        self.assertEqual(r["forward_evidence_status"], "NO_FORWARD_DATA")
        self.assertEqual(r["system_status"], "HEALTHY")

    def test_19_forward_collecting_normally(self):
        r = run(baseline())
        self.assertEqual(r["forward_evidence_status"], "COLLECTING")
        self.assertEqual(r["subsystems"]["forward_evidence"]["metrics"]["trading_days_collected"], 3)

    def test_20_readiness_gates_reached(self):
        b = baseline()
        b["readiness"] = _readiness(state="READY_FOR_RUN55_RERUN", limiting="NONE", days=20, runs=60)
        r = run(b)
        self.assertEqual(r["forward_evidence_status"], "READY_FOR_FORMAL_EVALUATION")
        self.assertEqual((r["system_status"], r["human_action"]), ("HEALTHY", "HUMAN_ACTION_REQUIRED"))
        self.assertIn("FORMAL_EVALUATION_READY", codes(r))

    def test_21_stale_run56_artifact(self):
        b = baseline()
        b["artifacts"]["forward_evidence_readiness"]["generated_at"] = "2026-09-25T23:15:00+00:00"
        r = run(b)
        self.assertIn("STALE_ARTIFACT_FORWARD_EVIDENCE_READINESS", codes(r))

    def test_22_stale_run58_artifact(self):
        b = baseline()
        b["artifacts"]["maturation_parity_audit"]["generated_at"] = "2026-09-20T00:00:00+00:00"
        r = run(b)
        self.assertIn("STALE_ARTIFACT_MATURATION_PARITY_AUDIT", codes(r))

    def test_23_workflow_metadata_unavailable(self):
        b = baseline()
        b["workflows"] = None
        b["scan_runs"] = None
        b["maturation_runs"] = None
        r = run(b)
        self.assertEqual(r["subsystems"]["workflows"]["status"], "UNKNOWN")
        self.assertEqual(r["subsystems"]["scanner"]["status"], "UNKNOWN")
        self.assertEqual(r["subsystems"]["database"]["status"], "HEALTHY")
        self.assertEqual(r["system_status"], "UNKNOWN")
        self.assertEqual(r["autonomy_readiness"]["state"], "NOT_READY")

    def test_24_multiple_incidents_sorted(self):
        b = baseline()
        b["db_probe"]["latency_ms"] = 5000
        b["universe"]["symbol_count"] = 9500
        b["maturation_report"]["alpaca_429_count"] = 1
        b["db_probe"]["duplicate_outcome_keys"] = 2
        r = run(b)
        sev = [i["severity"] for i in r["incidents"]]
        self.assertEqual(sev, sorted(sev, key=sh.SEVERITY_ORDER.get))
        self.assertEqual(r["incidents"][0]["severity"], "CRITICAL")
        self.assertGreaterEqual(len(r["incidents"]), 4)

    def test_25_critical_failure_with_high_score(self):
        b = baseline()
        b["db_probe"] = {"connected": False}
        r = run(b)
        self.assertGreaterEqual(r["health_score"], 75)
        self.assertEqual(r["system_status"], "ACTION_REQUIRED")

    def test_26_expected_waiting_does_not_degrade(self):
        # Friday's complete telemetry, evaluated on Sunday, with no forward data yet.
        b = baseline(dt.datetime(2026, 10, 2, 23, 50, tzinfo=UTC))
        b["now"] = dt.datetime(2026, 10, 4, 12, 0, tzinfo=UTC).isoformat()
        b["readiness"] = _readiness(limiting="NO_FORWARD_DATA", days=0, runs=0)
        r = run(b)
        self.assertEqual((r["system_status"], r["human_action"]), ("HEALTHY", "NO_ACTION"), r["incidents"])
        self.assertEqual(r["incidents"], [])
        self.assertEqual(r["subsystems"]["forward_evidence"]["status"], "WAITING")
        self.assertEqual(r["subsystems"]["cohort_parity"]["status"], "WAITING")

    def test_27_anti_peeking_contract(self):
        b = baseline()
        b["readiness"]["horizons"]["+60m"]["CANDIDATE"]["win_rate"] = 0.9  # must not pass through
        b["readiness"]["mean_return"] = 0.05
        r = run(b)
        text = json.dumps(r)
        for bad in ("win_rate", "mean_return", "spearman", "effect size", "mfe_mean"):
            self.assertNotIn(bad, text)
        self.assertEqual(sh.forbidden_keys(r), [])
        md = sh.render_markdown(r)
        self.assertEqual(sh.forbidden_text(md), [])
        with self.assertRaises(ValueError):
            sh.assert_clean({"subsystems": {"x": {"metrics": {"win_rate": 0.6}}}})
        with self.assertRaises(ValueError):
            sh.assert_clean({"note": "candidate return 2%"})

    def test_evaluator_crash_is_isolated(self):
        b = baseline()
        with mock.patch.object(sh, "eval_database", side_effect=RuntimeError("boom")):
            r = run(b)
        self.assertEqual(r["subsystems"]["database"]["status"], "UNKNOWN")
        self.assertEqual(r["subsystems"]["scanner"]["status"], "HEALTHY")

    def test_markdown_header_and_schema(self):
        r = run(baseline())
        md = sh.render_markdown(r)
        self.assertTrue(md.startswith("# HSF SYSTEM HEALTH"))
        self.assertIn("System Status: HEALTHY", md)
        self.assertIn("Human Action: **NONE**", md)
        self.assertEqual(r["schema"], "hsf-system-health-1.0")
        self.assertEqual(set(r["subsystems"]), set(sh.SUBSYSTEMS))
        for s in r["subsystems"].values():
            for k in ("status", "reason", "observed_value", "expected_value", "last_updated", "recommended_action",
                      "human_action"):
                self.assertIn(k, s)

    def test_inputs_not_mutated(self):
        b = baseline()
        before = copy.deepcopy(b)
        run(b)
        self.assertEqual(b, before)


class FalsePositiveTests(unittest.TestCase):
    def test_small_saved_scan_is_not_partial(self):
        b = baseline()
        b["db_runs"] = [{"created_at": WED.isoformat(), "label": "SP500", "row_count": 3, "duration_sec": 20}]
        r = run(b)
        self.assertNotIn("PARTIAL_SCAN", codes(r))
        self.assertEqual(r["system_status"], "HEALTHY")

    def test_same_hour_second_scan_dedup_is_expected(self):
        b = baseline()
        first = b["observations_summary"]["per_scan"][-1]
        t = dt.datetime.fromisoformat(first["scan_time"]) + dt.timedelta(minutes=19)
        b["observations_summary"]["per_scan"].append(
            {"scan_time": t.isoformat(), "n": 110, "by_cohort": {"NEAR_MISS": 10, "CONTROL": 100}})
        self.assertNotIn("COHORT_MISSING_IN_SCAN", codes(run(b)))
        lone = dict(first, by_cohort={"CANDIDATE": 100, "NEAR_MISS": 50, "CONTROL": 0})
        b["observations_summary"]["per_scan"] = [lone]
        self.assertIn("COHORT_MISSING_IN_SCAN", codes(run(b)))


class StoreAndScriptTests(unittest.TestCase):
    def test_snapshot_store_roundtrip(self):
        import sqlite3

        from db import system_health as store
        conn = sqlite3.connect(":memory:")
        try:
            r = run(baseline())
            self.assertTrue(store.save_snapshot(r, conn=conn))
            later = dict(r, generated_at=(WED + dt.timedelta(hours=1)).isoformat(), system_status="DEGRADED")
            self.assertTrue(store.save_snapshot(later, conn=conn))
            self.assertEqual(store.load_latest(conn=conn)["system_status"], "DEGRADED")
        finally:
            conn.close()

    def test_script_evaluates_saved_inputs(self):
        import tempfile
        from pathlib import Path

        from scripts import system_health as script
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inputs.json"
            p.write_text(json.dumps(baseline()))
            with mock.patch("sys.argv", ["x", "--inputs", str(p), "--out", d]):
                self.assertEqual(script.main(), 0)
            rep = json.loads((Path(d) / "system_health.json").read_text())
            self.assertEqual(rep["system_status"], "HEALTHY")
            self.assertTrue((Path(d) / "system_health.md").read_text().startswith("# HSF SYSTEM HEALTH"))

    def test_collect_survives_every_collector_failing(self):
        from scripts import system_health as script
        boom = mock.Mock(side_effect=RuntimeError("down"))
        with mock.patch.multiple(script, collect_workflows=boom, latest_maturation_report=boom,
                                 recent_observations=boom, forward_outcomes=boom, db_runs=boom,
                                 universe_probe=boom, parity_report=boom), \
                mock.patch("db.system_health.load_latest", side_effect=RuntimeError("down")):
            inputs = script.collect(WED)
        r = sh.evaluate(inputs)
        self.assertIn(r["system_status"], ("UNKNOWN", "ACTION_REQUIRED"))
        self.assertTrue(inputs["collection_errors"])
        self.assertEqual(r["subsystems"]["workflows"]["status"], "UNKNOWN")

    @unittest.skipUnless(_STREAMLIT, "ui.system_health_view imports streamlit")
    def test_streamlit_view_renders_same_model(self):
        from ui import system_health_view as view
        r = run(baseline())
        calls = []
        fake = mock.MagicMock()
        fake.columns.side_effect = lambda n: [mock.MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        fake.markdown.side_effect = lambda s, **k: calls.append(s)
        fake.caption.side_effect = lambda s, **k: calls.append(s)
        with mock.patch.object(view, "st", fake):
            view.render_system_health(r)
        text = " ".join(str(c) for c in calls)
        self.assertIn("System Health", text)
        self.assertEqual(sh.forbidden_text(text), [])


if __name__ == "__main__":
    unittest.main()
