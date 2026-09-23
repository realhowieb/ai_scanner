#!/usr/bin/env python3
"""Run 38A/38B — outcome maturation worker (idempotent, safe to rerun).

Finds captured observations whose intraday horizons (+5m/+15m/+30m/+60m) have
matured, fetches the appropriate future minute bars ONCE per symbol, computes
outcomes (no lookahead), and attaches them via the Run 36 outcome schema. Skips
already-matured (observation_id, horizon) pairs (first-write-wins) and classifies
failures. Produces a machine-readable report artifact.

Safe to run repeatedly (Task 7): each horizon matures independently the moment
its wall-clock elapses; nothing is overwritten. It only writes to
hsf_observation_outcomes and never touches scans, observations, or scanners.

    python -m scripts.mature_observations [--limit N] [--slack-min M] [--dry-run] [--out DIR]

Scheduling: `.github/workflows/mature-observations.yml` (every 30 min during/after
US market hours, workflow_dispatch supported).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analytics.observation_capture import (
    HORIZON_BARS,
    compute_matured_outcomes,
    horizon_eligibility,
)

ROOT = Path(__file__).resolve().parents[1]


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse(v: Any):
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _new_report() -> Dict[str, Any]:
    return {
        "schema": "hsf-maturation-1.1",
        "generated_at": _now().isoformat(),
        "observations_scanned": 0,
        "eligible_observations": 0,
        "symbols_with_ready_horizons": 0,
        "symbols_deferred": 0,
        "horizons": {h: {"new": 0, "already": 0, "not_ready": 0, "failed": 0}
                     for h in HORIZON_BARS},
        "failures": {},
        "attached": 0,
        # Run 52 — backlog capacity telemetry (measurement only).
        "backlog": {
            "ready_symbols": 0,          # distinct symbols with >=1 ready horizon
            "ready_observations": 0,     # observations with >=1 ready horizon
            "processed_symbols": 0,      # symbols actually fetched this run (<= cap)
            "deferred_symbols": 0,       # ready symbols not reached (cap overflow)
            "matured_observations": 0,   # NEW outcome rows written this run
            "failed_symbols": 0,         # symbols that failed wholesale (no maturation)
            "oldest_pending_age_min": None,   # age of oldest READY anchor still pending
            "estimated_clearance_runs": None,  # ceil(ready_symbols / cap), snapshot
            "max_symbols": 0,
        },
    }


def _fail(report: Dict[str, Any], horizon: str | None, reason: str) -> None:
    report["failures"][reason] = report["failures"].get(reason, 0) + 1
    if horizon:
        report["horizons"][horizon]["failed"] += 1


def mature_observations(observations, *, now=None, slack_min: int = 15,
                        dry_run: bool = False, fetch_bars=None, save_fn=None,
                        max_symbols: int = 400) -> Dict[str, Any]:
    """Pure-ish orchestration (injectable `fetch_bars`/`save_fn` for tests).

    Bounded per run: at most `max_symbols` distinct symbols are fetched, in
    oldest-anchor-first order, so the worker finishes within its CI timeout and
    the backlog drains deterministically across the every-30-min schedule
    (idempotent — deferred symbols mature on a later run). Set max_symbols<=0 for
    no cap (tests)."""
    now = now or _now()
    if not isinstance(now, _dt.datetime):
        now = _parse(now) or _now()
    report = _new_report()

    # Group by symbol so each symbol's future bars are fetched ONCE per run.
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations or []:
        report["observations_scanned"] += 1
        by_symbol[str(o.get("symbol") or "").upper()].append(o)

    # Pass 1 (no I/O): build per-symbol plans + earliest ready anchor.
    symbol_plans: Dict[str, tuple] = {}
    for symbol, obs_list in by_symbol.items():
        plans = []
        for o in obs_list:
            anchor = o.get("scan_timestamp") or o.get("timestamp")
            elig = horizon_eligibility(anchor, now, (o.get("outcomes") or {}).keys(),
                                       slack_min=slack_min)
            for h, status in elig.items():
                if status in ("already", "not_ready"):
                    report["horizons"][h][status] += 1
            ready = [h for h, s in elig.items() if s == "ready"]
            if ready:
                plans.append((o, anchor, ready))
        if plans:
            anchors = [_parse(p[1]) for p in plans if _parse(p[1])]
            symbol_plans[symbol] = (plans, min(anchors) if anchors else None)

    # Oldest-anchor first so the backlog drains and near-complete symbols finish.
    ordered = sorted(symbol_plans.items(),
                     key=lambda kv: (kv[1][1] is None, kv[1][1] or now))
    report["symbols_with_ready_horizons"] = len(ordered)
    ready_symbols_total = len(ordered)
    ready_observations_total = sum(len(v[0]) for _s, v in ordered)
    deferred_symbols = 0
    if max_symbols and max_symbols > 0 and len(ordered) > max_symbols:
        deferred = ordered[max_symbols:]
        deferred_symbols = len(deferred)
        report["symbols_deferred"] = deferred_symbols
        # Oldest observation still pending AFTER this run = oldest deferred anchor.
        oldest_deferred = min((v[1] for _s, v in deferred if v[1] is not None),
                              default=None)
        report["backlog"]["oldest_pending_age_min"] = (
            round((now - oldest_deferred).total_seconds() / 60.0, 1)
            if oldest_deferred is not None else None)
        ordered = ordered[:max_symbols]

    report["backlog"].update({
        "ready_symbols": ready_symbols_total,
        "ready_observations": ready_observations_total,
        "processed_symbols": len(ordered),
        "deferred_symbols": deferred_symbols,
        "max_symbols": int(max_symbols),
        "estimated_clearance_runs": (
            -(-ready_symbols_total // max_symbols) if max_symbols and max_symbols > 0
            else 1),
    })

    # Pass 2 (I/O): fetch + mature the bounded set.
    for symbol, (plans, earliest) in ordered:
        report["eligible_observations"] += len(plans)
        if earliest is None:
            _fail(report, None, "INVALID_TIMESTAMP")
            report["backlog"]["failed_symbols"] += 1
            continue
        try:
            bars = (fetch_bars or _default_fetch)(symbol, earliest.date().isoformat())
        except Exception:
            _fail(report, None, "PROVIDER_ERROR")
            report["backlog"]["failed_symbols"] += 1
            continue
        if not bars:
            for _o, _a, ready in plans:
                for h in ready:
                    _fail(report, h, "PRICE_DATA_UNAVAILABLE")
            report["backlog"]["failed_symbols"] += 1
            continue

        for o, anchor, ready in plans:
            a = _parse(anchor)
            after = [b for b in bars if (_parse(b.get("t")) or a) >= a]
            closes = [b.get("c") for b in after if b.get("c") is not None]
            if len(closes) < 2:
                for h in ready:
                    _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
                continue
            eval_times = {h: (a + _dt.timedelta(minutes=HORIZON_BARS[h])).isoformat()
                          for h in ready}
            direction = str((o.get("scanners") or [{}])[0].get("direction") or "long")
            outcomes = compute_matured_outcomes(
                o, prices_after=closes,
                horizon_bars={h: HORIZON_BARS[h] for h in ready},
                evaluation_times=eval_times, direction=direction)
            produced = {oc["horizon"] for oc in outcomes}
            for h in ready:
                if h not in produced:
                    _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
            if dry_run:
                for oc in outcomes:
                    report["horizons"][oc["horizon"]]["new"] += 1
                    report["attached"] += 1
                continue
            for oc in outcomes:
                try:
                    wrote = (save_fn or _default_save)(oc)
                except Exception:
                    _fail(report, oc["horizon"], "DATABASE_ERROR")
                    continue
                if wrote:
                    report["horizons"][oc["horizon"]]["new"] += 1
                    report["attached"] += 1
                else:
                    report["horizons"][oc["horizon"]]["already"] += 1
    # matured_observations counts NEW outcome rows written (real drain this run);
    # in dry-run it mirrors would-be writes so clearance estimates stay comparable.
    report["backlog"]["matured_observations"] = report["attached"]
    return report


def _default_fetch(symbol: str, start: str):
    from data.price_alpaca import fetch_minute_bars
    return fetch_minute_bars(symbol, start, None) or []


def _default_save(outcome) -> bool:
    from db.hsf_observations import save_outcome
    return save_outcome(outcome)


def render_report_text(r: Dict[str, Any]) -> str:
    lines = ["HSF Outcome Maturation", "----------------------",
             f"Eligible observations: {r['eligible_observations']} "
             f"(scanned {r['observations_scanned']})",
             f"Symbols ready: {r.get('symbols_with_ready_horizons', 0)} "
             f"(deferred this run: {r.get('symbols_deferred', 0)})"]
    for h in HORIZON_BARS:
        s = r["horizons"][h]
        lines.append(f"{h}: new={s['new']} already={s['already']} "
                     f"not_ready={s['not_ready']} failed={s['failed']}")
    if r["failures"]:
        lines.append("Failures: " + ", ".join(f"{k}={v}" for k, v in sorted(r["failures"].items())))
    b = r.get("backlog") or {}
    if b:
        lines.append(
            f"Backlog: ready_symbols={b.get('ready_symbols')} "
            f"processed={b.get('processed_symbols')} deferred={b.get('deferred_symbols')} "
            f"matured={b.get('matured_observations')} failed_symbols={b.get('failed_symbols')} "
            f"oldest_pending_min={b.get('oldest_pending_age_min')} "
            f"clearance_runs={b.get('estimated_clearance_runs')}")
    return "\n".join(lines)


def main() -> int:
    from db.hsf_observations import load_recent_observations

    ap = argparse.ArgumentParser(description="Mature captured observation outcomes")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--slack-min", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=400,
                    help="max distinct symbols to fetch this run (bounds runtime; "
                         "backlog drains across the schedule). <=0 = no cap.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()

    # attach_outcomes=True (Run 52): matured horizons come back on each record so
    # already-matured work is classified `already` and skipped, not re-fetched.
    observations = load_recent_observations(limit=args.limit,
                                            attach_outcomes=True) or []
    report = mature_observations(observations, slack_min=args.slack_min,
                                 dry_run=args.dry_run, max_symbols=args.max_symbols)
    report["dry_run"] = bool(args.dry_run)

    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "maturation_report.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    print(render_report_text(report))
    print(f"[mature_observations] attached={report['attached']} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
