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
        "schema": "hsf-maturation-1.0",
        "generated_at": _now().isoformat(),
        "observations_scanned": 0,
        "eligible_observations": 0,
        "horizons": {h: {"new": 0, "already": 0, "not_ready": 0, "failed": 0}
                     for h in HORIZON_BARS},
        "failures": {},
        "attached": 0,
    }


def _fail(report: Dict[str, Any], horizon: str | None, reason: str) -> None:
    report["failures"][reason] = report["failures"].get(reason, 0) + 1
    if horizon:
        report["horizons"][horizon]["failed"] += 1


def mature_observations(observations, *, now=None, slack_min: int = 15,
                        dry_run: bool = False, fetch_bars=None, save_fn=None) -> Dict[str, Any]:
    """Pure-ish orchestration (injectable `fetch_bars`/`save_fn` for tests)."""
    now = now or _now()
    report = _new_report()

    # Group by symbol so each symbol's future bars are fetched ONCE per run.
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations or []:
        report["observations_scanned"] += 1
        by_symbol[str(o.get("symbol") or "").upper()].append(o)

    for symbol, obs_list in by_symbol.items():
        # Determine which observations have any ready horizon; fetch once.
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
        if not plans:
            continue
        report["eligible_observations"] += len(plans)

        earliest = min(_parse(p[1]) for p in plans if _parse(p[1]))
        if earliest is None:
            _fail(report, None, "INVALID_TIMESTAMP")
            continue
        try:
            bars = (fetch_bars or _default_fetch)(symbol, earliest.date().isoformat())
        except Exception:
            _fail(report, None, "PROVIDER_ERROR")
            continue
        if not bars:
            for _o, _a, ready in plans:
                for h in ready:
                    _fail(report, h, "PRICE_DATA_UNAVAILABLE")
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
             f"(scanned {r['observations_scanned']})"]
    for h in HORIZON_BARS:
        s = r["horizons"][h]
        lines.append(f"{h}: new={s['new']} already={s['already']} "
                     f"not_ready={s['not_ready']} failed={s['failed']}")
    if r["failures"]:
        lines.append("Failures: " + ", ".join(f"{k}={v}" for k, v in sorted(r["failures"].items())))
    return "\n".join(lines)


def main() -> int:
    from db.hsf_observations import load_recent_observations

    ap = argparse.ArgumentParser(description="Mature captured observation outcomes")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--slack-min", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()

    observations = load_recent_observations(limit=args.limit) or []
    report = mature_observations(observations, slack_min=args.slack_min, dry_run=args.dry_run)
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
