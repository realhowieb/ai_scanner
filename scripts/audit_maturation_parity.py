#!/usr/bin/env python3
"""Run 58 — read-only cohort maturation parity & missingness audit.

Measures whether CANDIDATE / NEAR_MISS / CONTROL have equal OPPORTUNITY to
mature. It never writes to the store, never changes maturation, and never
reports outcome values (anti-peeking guard).

    # Run 55 snapshot section (offline, from the saved Run 55 input snapshot)
    python -m scripts.audit_maturation_parity --input run55_input_snapshot.json \
        --label run55 --run-times run_times.json

    # Current state + forward epoch (CI: DB + Alpaca), with a scheduler trace from a
    # production-faithful DRY-RUN of the maturation worker (nothing is written)
    python -m scripts.audit_maturation_parity --simulate --run-times run_times.json \
        --run55-section artifacts/research/maturation_parity_run55_snapshot.json

Outputs artifacts/research/maturation_parity_audit.{json,md} (or the run55
section file with --label run55).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from analytics import maturation_parity as mp

ROOT = Path(__file__).resolve().parents[1]

FIX_CONDITIONS = (
    "reproducible implementation/pipeline defect",
    "defect affects cohorts unequally",
    "expected correct behavior is unambiguous",
    "fix does not alter outcome formulas",
    "fix does not alter cohort membership",
    "fix does not require looking at outcome effectiveness",
    "deterministic tests can prove the correction",
)


def _parse(v) -> Optional[_dt.datetime]:
    try:
        d = _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def simulate_trace(now: _dt.datetime) -> Dict[str, Any]:
    """Production-faithful DRY RUN of the current maturation worker with a trace:
    same loader call (limit 5000, outcomes attached), current cap, retries and
    retirement. Nothing is written; only per-(observation, horizon) statuses
    are kept."""
    from db.hsf_observations import load_recent_observations
    from scripts.mature_observations import MAX_SYMBOLS, mature_observations
    obs = load_recent_observations(limit=5000, attach_outcomes=True) or []
    trace: List[Dict[str, Any]] = []
    rep = mature_observations(obs, now=now, dry_run=True, trace=trace)
    keep = ("symbols_with_ready_horizons", "symbols_deferred", "alpaca_requests", "alpaca_429_count",
            "rate_limited_symbols", "price_data_unavailable_symbols", "outcomes_matured",
            "fetch_mode", "ineligible")
    return {"trace": trace, "loaded_observations": len(obs), "max_symbols": MAX_SYMBOLS,
            "worker_report": {k: rep.get(k) for k in keep}}


def build(observations, outcomes, *, now, trace=None, run_times=None, run55_section=None,
          sim_meta=None, label=None) -> Dict[str, Any]:
    hist = mp.audit_scope(observations, outcomes, scope="historical", now=now, trace=trace,
                          run_times=run_times)
    fwd = mp.audit_scope(observations, outcomes, scope="forward", now=now, trace=trace)
    if label == "run55":
        return {"schema": mp.SCHEMA, "section": "RUN55_SNAPSHOT", "generated_at": now.isoformat(),
                "as_of": now.isoformat(), **hist}
    rc = hist.get("root_cause") or (run55_section or {}).get("root_cause")
    starvation = _starvation_verdict(run55_section, hist)
    report = {
        "schema": mp.SCHEMA,
        "generated_at": now.isoformat(),
        "run55_snapshot": run55_section,
        "current_historical": hist,
        "forward_epoch": fwd,
        "forward_parity_status": fwd.get("status") if fwd.get("status") != "OK" else (
            (fwd.get("parity") or {}).get("+60m", {}).get("classification")),
        "simulation": sim_meta,
        "historical_scheduler_starvation": starvation,
        "root_cause": rc,
        "fix_applied": False,
        "fix_policy": _fix_policy(hist, starvation),
    }
    mp.assert_clean(report)
    return report


def _starvation_verdict(run55: Optional[Dict[str, Any]], hist: Dict[str, Any]) -> Dict[str, Any]:
    src = run55 or hist
    alloc = (src.get("scheduler_cap_allocation") or {})
    rep400 = src.get("scheduler_replay_400") or {}
    never = ((rep400.get("by_horizon") or {}).get("+60m") or {})
    occurred = bool(never) and (never.get("CONTROL", {}).get("never_attempted_pct") or 0) >= 5.0
    cur = (hist.get("scheduler_cap_allocation") or {}).get("2000") or {}
    return {
        "occurred_historically": occurred,
        "evidence_replay_400_plus60": never or None,
        "cap_400_allocation": alloc.get("400"),
        "current_cap_2000_binding": (cur.get("deferred_symbols") or 0) > 0 if cur else None,
        "current_cap_2000_reach_gap_pp": cur.get("reach_gap_pp") if cur else None,
    }


def _fix_policy(hist: Dict[str, Any], starvation: Dict[str, Any]) -> Dict[str, Any]:
    binding = starvation.get("current_cap_2000_binding")
    same_anchor = ((hist.get("symbol_sharing") or {}).get("mismatch_cases") or {}).get("SAME_ANCHOR_MISMATCH", 0)
    defect_now = bool(binding) or bool(same_anchor)
    return {
        "conditions": list(FIX_CONDITIONS),
        "current_defect_reproducible_in_production": defect_now,
        "decision": "FIX" if defect_now else "AUDIT_ONLY",
        "reason": ("current 2,000-symbol cap does not bind and no same-anchor sharing mismatch exists; "
                   "historical starvation affected pre-epoch data only" if not defect_now else
                   "a cohort-unequal defect is present in the current scheduler"),
    }


def _p(x, suffix="%"):
    return "—" if x is None else f"{x}{suffix}"


def render_markdown(r: Dict[str, Any]) -> str:
    L: List[str] = ["# Maturation Parity & Missingness Audit", "",
                    f"Generated {r['generated_at']} · schema `{r['schema']}` · READ-ONLY · completeness only "
                    "(no outcome values)", ""]
    rc = r.get("root_cause") or {}
    L += [f"## Root cause: **{rc.get('classification')}** (confidence {rc.get('confidence')})", "",
          f"- Material mechanisms at {rc.get('horizon')}: {rc.get('material_mechanisms')}",
          f"- Components (control − candidate missing rate, pp): {rc.get('components_pp')}",
          f"- Control/candidate median capture dollar-volume ratio: "
          f"{rc.get('control_to_candidate_median_dollar_volume_ratio')}",
          f"- Basis: {rc.get('reason_basis')}", "",
          f"Historical scheduler starvation: **{r['historical_scheduler_starvation']['occurred_historically']}** · "
          f"current 2,000 cap binding: **{r['historical_scheduler_starvation']['current_cap_2000_binding']}** · "
          f"fix: **{r['fix_policy']['decision']}** ({r['fix_policy']['reason']}) · "
          f"forward parity status: **{r['forward_parity_status']}**", ""]
    for title, sec in (("Run 55 snapshot", r.get("run55_snapshot")),
                       ("Current historical", r.get("current_historical")),
                       ("Forward epoch", r.get("forward_epoch"))):
        L += [f"## {title}", ""]
        if not sec or sec.get("status") not in ("OK", None) and "coverage" not in sec:
            L += [f"Status: **{(sec or {}).get('status', 'NOT_AVAILABLE')}**", ""]
            continue
        L += render_scope(sec)
    return "\n".join(L)


def render_scope(sec: Dict[str, Any]) -> List[str]:
    L = [f"Observations {sec['observations']} · cohorts {sec['cohort_counts']}", "",
         "Run 55 primary population (regular-session anchors):", "",
         "| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class |", "|---|---|---|---|---|---|"]
    for h in mp.HORIZONS:
        p = sec["run55_primary_population_parity"][h]
        L.append(f"| {h} | {_p(p['candidate_coverage'])} | {_p(p['near_miss_coverage'])} | "
                 f"{_p(p['control_coverage'])} | {_p(p['parity_gap'], '')} | {p['classification']} |")
    L += ["", "All sessions:", "",
         "| Horizon | Cohort | Eligible | Matured | Unmatured | Maturation % | Projected % (after backlog) |",
         "|---|---|---|---|---|---|---|"]
    for h in mp.HORIZONS:
        for c in mp.COHORTS:
            x = sec["coverage"][h][c]
            L.append(f"| {h} | {c} | {x['eligible']} | {x['matured']} | {x['unmatured']} | "
                     f"{_p(x['maturation_pct'])} | {_p(x['projected_maturation_pct_after_backlog'])} |")
    L += ["", "| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class | Projected gap | Projected class |",
          "|---|---|---|---|---|---|---|---|"]
    for h in mp.HORIZONS:
        p = sec["parity"][h]
        q = (sec.get("projected_parity_after_backlog") or {}).get(h) or {}
        L.append(f"| {h} | {_p(p['candidate_coverage'])} | {_p(p['near_miss_coverage'])} | "
                 f"{_p(p['control_coverage'])} | {_p(p['parity_gap'], '')} | {p['classification']} | "
                 f"{_p(q.get('parity_gap'), '')} | {q.get('classification', '—')} |")
    L += ["", "Missingness at +60m (count · % of eligible · % of missing):", "",
          "| Reason | " + " | ".join(mp.COHORTS) + " |", "|---|" + "---|" * 3]
    for rsn in mp.REASONS:
        if rsn == "NOT_YET_ELIGIBLE":
            continue
        cells = []
        for c in mp.COHORTS:
            m = sec["coverage"]["+60m"][c]["missing_reasons"][rsn]
            cells.append(f"{m['count']} · {_p(m['pct_of_cohort_eligible'])} · {_p(m['pct_of_missing'])}")
        L.append(f"| {rsn} | " + " | ".join(cells) + " |")
    t = sec["timing"]["by_cohort"]
    L += ["", "Timing (ET):", "", "| Cohort | P25 | Median | P75 | +60m crosses close | "
          + " | ".join(b[2] for b in mp.TOD_BUCKETS) + " |", "|---|---|---|---|---|" + "---|" * len(mp.TOD_BUCKETS)]
    for c in mp.COHORTS:
        x = t[c]
        L.append(f"| {c} | {x['p25_time_et']} | {x['median_time_et']} | {x['p75_time_et']} | "
                 f"{_p(x['plus60_crosses_close_pct'])} | " + " | ".join(str(x["bucket_counts"][b[2]])
                                                                      for b in mp.TOD_BUCKETS) + " |")
    comp = sec["symbol_composition"]["by_cohort"]
    L += ["", "Symbol composition:", "", "| Cohort | Unique symbols | Obs/symbol | Median capture $-vol (M) | "
          "Median price | Never-matured symbols % | Exclusion | Format |", "|---|---|---|---|---|---|---|---|"]
    for c in mp.COHORTS:
        x = comp[c]
        L.append(f"| {c} | {x['unique_symbols']} | {x['observations_per_symbol']} | "
                 f"{x['capture_dollar_volume_quantiles_musd'][1]} | {x['capture_price_median']} | "
                 f"{_p(x['symbols_never_matured_pct'])} | {x['exclusion_status']} | {x['symbol_format']} |")
    L += ["", f"Symbol overlap: {sec['symbol_composition']['overlap']}", "",
          f"Symbol sharing mismatches: {sec['symbol_sharing']['mismatch_cases']}", "",
          "Scheduler cap allocation (share of each cohort's ready work reached by the worker's own ordering):", ""]
    for cap, a in sec["scheduler_cap_allocation"].items():
        L.append(f"- cap {cap}: ready symbols {a['ready_symbols']}, deferred {a['deferred_symbols']}, reach gap "
                 f"{a['reach_gap_pp']}pp — " + ", ".join(f"{c} {v['reached_pct']}%" for c, v in a["by_cohort"].items()))
    if sec.get("scheduler_replay_400"):
        rp = sec["scheduler_replay_400"]
        L += ["", f"Historical replay (cap {rp['cap']}, {rp['runs']} real maturation runs): never-attempted share "
              "of still-unmatured eligible work:", ""]
        for h in mp.HORIZONS:
            L.append(f"- {h}: " + ", ".join(f"{c} {v['never_attempted']}/{v['unmatured_eligible']} "
                                           f"({_p(v['never_attempted_pct'])})"
                                           for c, v in rp["by_horizon"][h].items()))
    ret = sec["retirement"]
    L += ["", f"Retirement: active={ret['active']} · " + ", ".join(
        f"{c} {ret[c]['retired_observations']} obs ({_p(ret[c]['retirement_rate_pct'])})" for c in mp.COHORTS), ""]
    L += ["Static scheduler audit:", ""] + [f"- **{x['item']}**: {x['finding']} — cohort-neutral: "
                                            f"{x['cohort_neutral']}; risk: {x['risk']}"
                                            for x in sec["scheduler_static_audit"]] + [""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 58 maturation parity audit (read-only)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "research"))
    ap.add_argument("--input", default=None, help="snapshot {observations, outcomes_by_id}")
    ap.add_argument("--label", default=None, choices=[None, "run55"])
    ap.add_argument("--as-of", default=None, help="evaluation time for --input replays")
    ap.add_argument("--simulate", action="store_true", help="dry-run the worker with a trace (CI)")
    ap.add_argument("--run-times", default=None, help="JSON list of maturation run start times")
    ap.add_argument("--run55-section", default=None)
    args = ap.parse_args()

    now = _parse(args.as_of) if args.as_of else _dt.datetime.now(_dt.timezone.utc)
    if args.input:
        data = json.loads(Path(args.input).read_text())
        observations, outcomes = data.get("observations") or [], data.get("outcomes_by_id") or {}
    else:
        from scripts.audit_research_cohorts import _load_live
        observations, outcomes = _load_live()
    run_times = None
    if args.run_times and Path(args.run_times).exists():
        run_times = [t for t in (_parse(x) for x in json.loads(Path(args.run_times).read_text())) if t and t <= now]
    trace = sim = None
    if args.simulate:
        s = simulate_trace(now)
        trace, sim = s.pop("trace"), s
    run55 = None
    if args.run55_section and Path(args.run55_section).exists():
        run55 = json.loads(Path(args.run55_section).read_text())
    report = build(observations, outcomes, now=now, trace=trace, run_times=run_times,
                   run55_section=run55, sim_meta=sim, label=args.label)
    mp.assert_clean(report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.label == "run55":
        (out / "maturation_parity_run55_snapshot.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"RUN55_SNAPSHOT root_cause={report.get('root_cause', {}).get('classification')}")
        return 0
    (out / "maturation_parity_audit.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "maturation_parity_audit.md").write_text(render_markdown(report))
    rc = report["root_cause"] or {}
    print(f"RUN 58 — root_cause={rc.get('classification')} confidence={rc.get('confidence')} "
          f"forward={report['forward_parity_status']} fix={report['fix_policy']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
