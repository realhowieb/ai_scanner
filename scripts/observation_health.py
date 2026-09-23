#!/usr/bin/env python3
"""Run 46 — observation dataset health report + research export (read-only).

Loads persisted canonical observations (bounded) and prints/writes a dataset
health report. Also supports a deterministic, point-in-time-safe research export.
Read-only: never mutates, deletes, dedupes, or backfills historical data.

    python -m scripts.observation_health [--limit N] [--out DIR]
    python -m scripts.observation_health --export export.json [--healthy-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics.observation_integrity import (
    dataset_health_report,
    render_health_text,
    research_export,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(limit: int):
    try:
        from db.hsf_observations import load_recent_observations
        return load_recent_observations(limit=limit) or []
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="HSF observation dataset health")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    ap.add_argument("--export", default=None, help="write research export to this path")
    ap.add_argument("--healthy-only", action="store_true")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    observations = _load(args.limit)
    report = dataset_health_report(observations)
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "observation_health.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    print(render_health_text(report))

    if args.export:
        rows = research_export(observations, healthy_only=args.healthy_only,
                               start=args.start, end=args.end)
        Path(args.export).write_text(json.dumps(rows, indent=2, default=str))
        print(f"[export] {len(rows)} point-in-time rows -> {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
