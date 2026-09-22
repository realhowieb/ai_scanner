#!/usr/bin/env python3
"""Run 38B — read-only audit of the canonical HSF observation dataset.

Reports counts, quality distribution, per-scanner/session breakdowns, and outcome
maturation progress. Pure read: never writes, never matures, never scans.

    python -m scripts.observation_audit [--limit N] [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from analytics.observation_capture import HORIZON_BARS

ROOT = Path(__file__).resolve().parents[1]


def audit(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure aggregation of an observation list into an audit report."""
    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    n = len(observations)
    by_source: Counter = Counter()
    by_scanner: Counter = Counter()
    by_session: Counter = Counter()
    quality = {"complete": 0, "partial": 0, "fallback": 0, "stale": 0}
    outcomes = {"none": 0, "partial": 0, "full": 0}
    today_count = 0
    timestamps: List[str] = []
    n_horizons = len(HORIZON_BARS)

    for o in observations:
        ts = str(o.get("timestamp") or "")
        timestamps.append(ts)
        if ts[:10] == today:
            today_count += 1
        ctx = o.get("market_context") or {}
        by_source[str(ctx.get("source") or str(o.get("context") or "").split(":")[0] or "unknown")] += 1
        by_session[str(o.get("session") or "unknown")] += 1
        for s in (o.get("scanners") or []):
            if s.get("triggered", True):
                by_scanner[str(s.get("name") or "unknown")] += 1
        dq = o.get("data_quality") or {}
        if dq.get("fallback_used"):
            quality["fallback"] += 1
        elif (dq.get("feature_completeness") or 0) >= 0.999:
            quality["complete"] += 1
        else:
            quality["partial"] += 1
        if dq.get("stale"):
            quality["stale"] += 1
        got = len(o.get("outcomes") or {})
        if got == 0:
            outcomes["none"] += 1
        elif got >= n_horizons:
            outcomes["full"] += 1
        else:
            outcomes["partial"] += 1

    return {
        "schema": "hsf-observation-audit-1.0",
        "total": n,
        "today": today_count,
        "by_source": dict(by_source),
        "by_scanner": dict(by_scanner.most_common()),
        "by_session": dict(by_session),
        "quality": quality,
        "outcomes": outcomes,
        "oldest": min(timestamps) if timestamps else None,
        "newest": max(timestamps) if timestamps else None,
    }


def render_text(r: Dict[str, Any]) -> str:
    q, oc = r["quality"], r["outcomes"]
    return "\n".join([
        "HSF Production Dataset", "----------------------",
        f"Observations: {r['total']}", f"Today: {r['today']}",
        f"By source: {r['by_source']}",
        f"Quality: complete={q['complete']} partial={q['partial']} "
        f"fallback={q['fallback']} stale={q['stale']}",
        f"Outcomes: none={oc['none']} partial={oc['partial']} full={oc['full']}",
        f"By scanner: {r['by_scanner']}",
        f"Oldest: {r['oldest']}  Newest: {r['newest']}",
    ])


def main() -> int:
    from db.hsf_observations import load_recent_observations

    ap = argparse.ArgumentParser(description="Audit canonical HSF observations")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()

    observations = load_recent_observations(limit=args.limit) or []
    report = audit(observations)
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "observation_audit.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
