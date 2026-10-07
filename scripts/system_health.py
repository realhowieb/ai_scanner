#!/usr/bin/env python3
"""Run 59 — HSF system health control plane (READ-ONLY, anti-peeking).

One command builds the complete health model:

    python -m scripts.system_health [--out artifacts/health] [--persist]

Collectors gather telemetry that already exists (GitHub workflow runs, the latest
scheduled maturation report, a bounded DB probe, the Run 56 readiness monitor,
the newer of the committed and latest-run Run 58 parity audit, one Alpaca assets
call for the universe).
Each collector is isolated: a failure becomes None, which the model reports as
UNKNOWN; it never becomes HEALTHY. Nothing here writes research data. With
--persist, one snapshot row is appended to the separate operational table
`hsf_system_health`, for the Streamlit view and trend comparisons.
No self-healing: it observes, classifies, prioritizes and recommends.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from analytics import forward_readiness as fr
from analytics import system_health as sh

ROOT = Path(__file__).resolve().parents[1]
GH_API = "https://api.github.com"
RECENT_DAYS = 3


def _timed(errors: Dict[str, str], timings: Dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    t0 = time.perf_counter()
    try:
        return fn()
    except Exception as e:
        errors[name] = f"{type(e).__name__}: {e}"[:300]
        return None
    finally:
        timings[name] = round(time.perf_counter() - t0, 2)


# ---- GitHub ------------------------------------------------------------------------------
def _gh():
    import requests
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("no GitHub token (GH_TOKEN/GITHUB_TOKEN)")
    repo = os.getenv("GITHUB_REPOSITORY") or "realhowieb/ai_scanner"
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    return s, repo


def workflow_runs(session, repo: str, workflow: str, *, event: Optional[str] = None, per_page: int = 60,
                  status: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"per_page": per_page}
    if event:
        params["event"] = event
    if status:
        params["status"] = status
    r = session.get(f"{GH_API}/repos/{repo}/actions/workflows/{workflow}/runs", params=params, timeout=15)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return [{"id": x.get("id"), "created_at": x.get("created_at"), "updated_at": x.get("updated_at"),
             "conclusion": x.get("conclusion"), "status": x.get("status"), "event": x.get("event")}
            for x in (r.json().get("workflow_runs") or [])]


def _never_got_runner(jobs: List[Dict[str, Any]]) -> bool:
    """True when no job of the run was ever assigned a runner (no runner, no steps).

    GitHub cancels such jobs after waiting for a machine; that is a GitHub
    capacity problem, not a scan failure."""
    return bool(jobs) and all(not j.get("runner_id") and not j.get("runner_name") and not j.get("steps")
                              for j in jobs)


def mark_runnerless(session, repo: str, runs: List[Dict[str, Any]], now: _dt.datetime) -> List[Dict[str, Any]]:
    """Flag recent unsuccessful runs whose jobs never got a runner (bounded: last RECENT_DAYS)."""
    since = now - _dt.timedelta(days=RECENT_DAYS)
    for r in runs:
        created = sh._parse(r.get("created_at"))
        if r.get("status") != "completed" or r.get("conclusion") == "success" or created is None or created < since:
            continue
        try:
            resp = session.get(f"{GH_API}/repos/{repo}/actions/runs/{r['id']}/jobs", timeout=15)
            resp.raise_for_status()
            r["no_runner"] = _never_got_runner(resp.json().get("jobs") or [])
        except Exception:
            pass  # unknown stays unflagged: reported as a failed scan, as before
    return runs


def collect_workflows(now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    session, repo = _gh()
    out: Dict[str, Any] = {}
    for name in sh.WORKFLOW_SPECS:
        try:
            out[name] = workflow_runs(session, repo, name)
        except Exception:
            out[name] = None
    if out.get("scheduled-scans.yml"):
        mark_runnerless(session, repo, out["scheduled-scans.yml"], now or _dt.datetime.now(_dt.timezone.utc))
    return out


def _run_artifact_json(session, repo: str, run_id: Any, artifact: str, filename: str) -> Optional[Dict[str, Any]]:
    """One JSON file from a run's uploaded artifact (None if absent or expired)."""
    arts = session.get(f"{GH_API}/repos/{repo}/actions/runs/{run_id}/artifacts", timeout=15).json()
    for a in arts.get("artifacts") or []:
        if a.get("name") != artifact or a.get("expired"):
            continue
        z = session.get(a["archive_download_url"], timeout=30)
        z.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            if filename in zf.namelist():
                return json.loads(zf.read(filename))
    return None


def latest_maturation_report() -> Optional[Dict[str, Any]]:
    """maturation_report.json from the latest successful non-dry-run run.

    Maturation is dispatched by cron-job.org (workflow_dispatch) since 2026-10-02,
    so runs of either event count; dry-run reports are skipped below."""
    session, repo = _gh()
    runs = workflow_runs(session, repo, "mature-observations.yml", status="success", per_page=10)
    for run in runs:
        rep = _run_artifact_json(session, repo, run["id"], "maturation-report", "maturation_report.json")
        if rep is not None and not rep.get("dry_run"):
            return rep
    return None


# ---- Database ----------------------------------------------------------------------------
def _conn():
    from db.engine import get_neon_conn
    c = get_neon_conn()
    if c is None:
        raise RuntimeError("database unavailable (no Neon connection)")
    return c


def _q(c, sql: str, params=()) -> List[Any]:
    cur = c.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall() or []
    cur.close()
    return [list(r.values()) if isinstance(r, dict) else list(r) for r in rows]


def db_probe(recent: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        c = _conn()
    except Exception as e:
        return {"connected": False, "error": str(e)[:200]}
    try:
        _q(c, "SELECT 1")
        latency = round((time.perf_counter() - t0) * 1000.0, 1)
        out: Dict[str, Any] = {"connected": True, "latency_ms": latency}
        for key, table in (("observations", "hsf_observations"), ("outcomes", "hsf_observation_outcomes")):
            try:
                n, latest = _q(c, f"SELECT count(*), max(created_at) FROM {table}")[0]
                out[f"{key}_accessible"] = True
                out[f"{key}_rows"] = int(n)
                out[f"latest_{key[:-1]}_created_at"] = latest.isoformat() if hasattr(latest, "isoformat") else latest
            except Exception:
                out[f"{key}_accessible"] = False
                try:
                    c.rollback()
                except Exception:
                    pass
        try:
            out["duplicate_outcome_keys"] = len(_q(c, "SELECT observation_id, horizon FROM hsf_observation_outcomes "
                                                      "GROUP BY observation_id, horizon HAVING count(*) > 1 LIMIT 10"))
        except Exception:
            out["duplicate_outcome_keys"] = None
        rows = recent or []
        n = len(rows)

        def rate(pred):
            return round(100.0 * sum(1 for o in rows if pred(o)) / n, 3) if n else None
        out["null_rates_pct"] = {
            "symbol": rate(lambda o: not o.get("symbol")),
            "timestamp": rate(lambda o: not (o.get("scan_timestamp") or o.get("timestamp"))),
            "scan_id": rate(lambda o: not (o.get("market_context") or {}).get("scan_id")),
            "market.price": rate(lambda o: (o.get("market") or {}).get("price") is None),
        }
        out["null_rate_sample"] = n
        return out
    finally:
        try:
            c.close()
        except Exception:
            pass


def recent_observations() -> List[Dict[str, Any]]:
    """Recent scanner research observations.

    Stair-stepper observations (``day_trader:stair_stepper``) are a separate
    study captured from the Day Trader page. They carry no research cohort,
    research_metadata or scan_id by design, so counting them made the capture
    checks report untagged rows, incomplete metadata and null scan ids.
    """
    from analytics.stair_step_research import CONTEXT as STAIR_STEP_CONTEXT
    from db.hsf_observations import load_recent_observations
    return [o for o in (load_recent_observations(limit=20000) or [])
            if o.get("context") != STAIR_STEP_CONTEXT]


def forward_outcomes() -> Dict[str, List[Dict[str, Any]]]:
    """Outcome records created since the Run 56 epoch (bounded; presence-only use)."""
    from db.hsf_observations import _loads
    c = _conn()
    try:
        rows = _q(c, "SELECT observation_id, record FROM hsf_observation_outcomes WHERE created_at >= %s",
                  (fr.FORWARD_EPOCH["forward_epoch_start_timestamp"],))
    finally:
        c.close()
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for oid, rec in rows:
        out[str(oid)].append(rec if isinstance(rec, dict) else _loads(rec))
    return out


def db_runs() -> List[Dict[str, Any]]:
    c = _conn()
    try:
        rows = _q(c, "SELECT created_at, label, row_count, duration_sec FROM runs "
                     "WHERE created_at > now() - interval '3 days' AND COALESCE(is_snapshot, false) = false "
                     "ORDER BY created_at")
    finally:
        c.close()
    return [{"created_at": (r[0].replace(tzinfo=_dt.timezone.utc).isoformat() if hasattr(r[0], "isoformat")
                            and r[0].tzinfo is None else str(r[0])), "label": r[1], "row_count": r[2],
             "duration_sec": r[3]} for r in rows]


def observations_summary(obs: List[Dict[str, Any]], now: _dt.datetime,
                         latest_created: Optional[str]) -> Dict[str, Any]:
    """Counts only (no outcomes): recent research capture by scan and cohort."""
    start = now - _dt.timedelta(days=RECENT_DAYS)
    recent = [o for o in obs if (fr.anchor(o) or start) > start]
    ids = Counter(str(o.get("observation_id")) for o in recent)
    per_scan: Dict[str, Counter] = defaultdict(Counter)
    for o in recent:
        per_scan[str(o.get("scan_timestamp") or o.get("timestamp"))][fr.cohort(o) or "UNTAGGED"] += 1
    fwd = [o for o in recent if (fr.anchor(o) or start) >= fr.epoch_start()]
    md = sum(1 for o in fwd if o.get("research_metadata"))
    return {
        "observations": len(recent),
        "by_cohort": dict(Counter(fr.cohort(o) or "UNTAGGED" for o in recent)),
        "untagged": sum(1 for o in recent if not fr.cohort(o)),
        "duplicate_ids": sum(v - 1 for v in ids.values() if v > 1),
        "malformed": sum(1 for o in recent if not o.get("symbol") or not (o.get("scan_timestamp") or o.get("timestamp"))),
        "metadata_scope_n": len(fwd),
        "metadata_block_pct": round(100.0 * md / len(fwd), 2) if fwd else None,
        "per_scan": [{"scan_time": k, "n": sum(v.values()), "by_cohort": dict(v)} for k, v in sorted(per_scan.items())],
        "latest_created_at": latest_created,
    }


def universe_probe() -> Dict[str, Any]:
    from data.us_market_universe import _is_malformed, build_us_market_universe
    u = build_us_market_universe()
    syms = u.get("symbols") or []
    return {"symbol_count": u.get("symbol_count"), "source": u.get("source"), "generated_at": u.get("generated_at"),
            "cached_at": u.get("cached_at"), "exclusions": u.get("exclusions"),
            "provider_assets": u.get("provider_assets"), "duplicates": len(syms) - len(set(syms)),
            "malformed": sum(1 for s in syms if _is_malformed(s))}


def _recovery_status() -> Dict[str, Any]:
    """Run 60 capability for autonomy readiness: kill-switch state + ledger reachability
    + whether a human-unreset circuit breaker is open (ledger only; no health recursion)."""
    from analytics import recovery_policy as rp
    cfg = rp.autonomy_config(os.environ)
    try:
        from db.recovery_ledger import list_recent
        ledger = list_recent(days=7)
        ok = True
    except Exception:
        ledger, ok = [], False
    opened = [e for e in ledger if e.get("result") == "CIRCUIT_OPENED"]
    reset = [e for e in ledger if e.get("result") == "CIRCUIT_RESET"]
    circuit_open = bool(opened) and (not reset or str(reset[-1].get("started_at")) < str(opened[-1].get("started_at")))
    return {"implemented": True, "ledger_available": ok, "circuit_open": circuit_open,
            "production_state": cfg["production_state"], "reason": cfg["reason"]}


def _committed_parity_report() -> Optional[Dict[str, Any]]:
    p = ROOT / "artifacts" / "research" / "maturation_parity_audit.json"
    return json.loads(p.read_text()) if p.exists() else None


def _latest_parity_artifact() -> Optional[Dict[str, Any]]:
    """maturation_parity_audit.json uploaded by the newest successful audit run.

    The audit workflow is read-only and never commits, so re-running it (by hand
    or by recovery's RERUN_PARITY_AUDIT) only produces this artifact."""
    session, repo = _gh()
    for run in workflow_runs(session, repo, "maturation-parity-audit.yml", status="success", per_page=5):
        rep = _run_artifact_json(session, repo, run["id"], "maturation-parity-audit",
                                 "maturation_parity_audit.json")
        if rep is not None:
            return rep
    return None


def _newer_report(*reports: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dated = [(sh._parse(r.get("generated_at")), r) for r in reports if r]
    dated = [(t, r) for t, r in dated if t is not None]
    if dated:
        return max(dated, key=lambda x: x[0])[1]
    return next((r for r in reports if r), None)


def parity_report() -> Optional[Dict[str, Any]]:
    """The newer of the committed Run 58 audit and the latest audit run's artifact."""
    try:
        latest = _latest_parity_artifact()
    except Exception:
        latest = None  # no token / GitHub unavailable: the committed file still counts
    return _newer_report(_committed_parity_report(), latest)


# ---- orchestration -------------------------------------------------------------------------
def collect(now: _dt.datetime) -> Dict[str, Any]:
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    T = lambda n, f: _timed(errors, timings, n, f)  # noqa: E731
    workflows = T("workflows", lambda: collect_workflows(now))
    mat_report = T("maturation_report", latest_maturation_report)
    obs = T("observations", recent_observations)
    probe = T("db_probe", lambda: db_probe(obs))
    outs = T("forward_outcomes", forward_outcomes) if obs is not None else None
    readiness = T("readiness", lambda: fr.monitor(obs, outs or {}, now=now, maturation_report=mat_report)) \
        if obs is not None and probe and probe.get("connected") else None
    summary = observations_summary(obs, now, (probe or {}).get("latest_observation_created_at")) \
        if obs is not None and probe and probe.get("connected") else None
    runs = T("db_runs", db_runs)
    universe = T("universe", universe_probe)
    parity = T("parity", parity_report)
    previous = T("previous_health", lambda: __import__("db.system_health", fromlist=["x"]).load_latest())
    wf = workflows or {}

    def last_success(name):
        rs = [r for r in (wf.get(name) or []) if r.get("conclusion") == "success"]
        return max((r["created_at"] for r in rs), default=None)
    artifacts = {
        "forward_evidence_readiness": {"generated_at": last_success("forward-evidence-readiness.yml")}
        if workflows is not None else None,
        "maturation_parity_audit": {"generated_at": (parity or {}).get("generated_at")},
        "latest_scanner_observation": {"generated_at": (probe or {}).get("latest_observation_created_at")},
        "previous_system_health": {"generated_at": (previous or {}).get("generated_at")},
    }
    recovery = T("recovery", _recovery_status)
    return {
        "now": now.isoformat(),
        "recovery": recovery,
        "universe": universe,
        "previous": previous,
        "scan_runs": wf.get("scheduled-scans.yml") if workflows is not None else None,
        "db_runs": runs,
        "observations_summary": summary,
        "maturation_report": mat_report,
        "maturation_runs": [r for r in (wf.get("mature-observations.yml") or [])
                            if r.get("event") in ("schedule", "workflow_dispatch")]
        if workflows is not None and wf.get("mature-observations.yml") is not None else None,
        "db_probe": probe,
        "workflows": workflows,
        "readiness": readiness,
        "parity": parity,
        "artifacts": artifacts,
        "collection_errors": errors,
        "collection_timings_sec": timings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HSF system health (read-only)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "health"))
    ap.add_argument("--persist", action="store_true", help="append a snapshot to hsf_system_health")
    ap.add_argument("--inputs", default=None, help="evaluate a saved inputs JSON instead of collecting")
    args = ap.parse_args()
    t0 = time.perf_counter()
    now = _dt.datetime.now(_dt.timezone.utc)
    inputs = json.loads(Path(args.inputs).read_text()) if args.inputs else collect(now)
    report = sh.evaluate(inputs)
    report["collection_timings_sec"] = inputs.get("collection_timings_sec")
    report["generation_seconds"] = round(time.perf_counter() - t0, 2)
    if args.persist:
        from db.system_health import save_snapshot
        report["persisted"] = save_snapshot(report)
    sh.assert_clean(report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_health.json").write_text(json.dumps(report, indent=2, default=str))
    md = sh.render_markdown(report)
    if sh.forbidden_text(md):
        raise ValueError("anti-peeking contract violated in markdown")
    (out / "system_health.md").write_text(md)
    print(f"SYSTEM STATUS: {report['system_status']} · score {report['health_score']} · "
          f"human action {report['human_action']} · autonomy {report['autonomy_readiness']['state']}")
    for n in sh.SUBSYSTEMS:
        s = report["subsystems"][n]
        print(f"  {n}: {sh.subsystem_label(s)} — {s['reason']}")
    for i in report["incidents"]:
        print(f"  INCIDENT {i['severity']} {i['incident_id']}: {i['summary']}")
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"system_status={report['system_status']}\nhuman_action={report['human_action']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
