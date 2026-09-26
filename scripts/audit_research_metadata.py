#!/usr/bin/env python3
"""Run 57 — read-only research metadata completeness audit.

"Are we collecting the context needed for future analysis?", not "which metadata
values perform best?" It reads OBSERVATIONS ONLY (outcomes are never loaded), so
it cannot compute effectiveness. Scope is the Run 56 forward epoch.

    python -m scripts.audit_research_metadata [--out DIR] [--input SNAPSHOT.json]

Writes artifacts/research/research_metadata_audit.{json,md}.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from analytics import forward_readiness as fr
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS
from analytics.research_metadata import RESEARCH_METADATA_SCHEMA, coverage

ROOT = Path(__file__).resolve().parents[1]
COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)
FIELDS = ("with_metadata_block", "tier_at_observation", "market_regime_at_observation",
          "scoring_version", "scanner_commit_sha", "universe_name", "price_provider",
          "scan_mode", "rank_at_observation")
CATEGORICAL = ("tier_source", "market_regime_source", "scoring_version", "universe_name",
               "scan_mode", "price_provider", "price_feed", "scanner_commit_sha", "schema_version")


def audit(observations: Sequence[Dict[str, Any]], *, now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    sel = fr.select_forward(observations)
    fwd = sel["forward"]
    by_cohort = {c: [o for o in fwd if fr.cohort(o) == c] for c in COHORTS}
    by_dir = {"LONG": [o for o in fwd if fr.direction(o) == "LONG"],
              "SHORT": [o for o in fwd if fr.direction(o) == "SHORT"],
              "UNDIRECTED (CONTROL)": [o for o in fwd if fr.direction(o) is None]}
    values = {c: {f: dict(Counter(str((o.get("research_metadata") or {}).get(f)) for o in rows))
                  for f in CATEGORICAL} for c, rows in by_cohort.items()}
    feat_cov = {}
    for c, rows in by_cohort.items():
        n = len(rows)
        names = Counter(k for o in rows for k in ((o.get("research_metadata") or {}).get("row_features") or {}))
        feat_cov[c] = {k: round(100.0 * v / n, 2) for k, v in sorted(names.items())} if n else {}
    report = {
        "schema": "hsf-research-metadata-audit-1.0",
        "generated_at": now.isoformat(),
        "research_metadata_schema": RESEARCH_METADATA_SCHEMA,
        "epoch_start": fr.FORWARD_EPOCH["forward_epoch_start_timestamp"],
        "forward_observations": len(fwd),
        "pre_epoch_excluded": sel["pre_epoch_excluded"],
        "legacy_excluded": sel["legacy_excluded"],
        "coverage_by_cohort": {c: coverage(rows) for c, rows in by_cohort.items()},
        "coverage_by_direction": {d: coverage(rows) for d, rows in by_dir.items()},
        "row_feature_coverage_pct_by_cohort": feat_cov,
        "value_counts_by_cohort": values,
        "notes": {
            "tier": "tier_at_observation is None by design: the scheduled scan emits no tier "
                    "(tier_source=TIER_NOT_EMITTED_BY_SCHEDULED_SCAN).",
            "regime": "market_regime_at_observation is None by design (REGIME_CAPTURE_UNAVAILABLE).",
            "control": "CONTROL rows are unranked and were filtered before scoring, so rank and "
                       "row_features are empty by construction; provenance/provider are captured.",
            "anti_peeking": "Observations only; no outcomes are read and no effectiveness is computed.",
        },
    }
    return report


def render_markdown(r: Dict[str, Any]) -> str:
    L: List[str] = ["# Research Metadata Audit", "",
                    f"Generated {r['generated_at']} · schema `{r['schema']}` · metadata `{r['research_metadata_schema']}` · "
                    "READ-ONLY · observations only (no outcomes)", "",
                    f"Forward epoch start `{r['epoch_start']}` · forward observations **{r['forward_observations']}** "
                    f"(excluded: {r['pre_epoch_excluded']} pre-epoch, {r['legacy_excluded']} legacy)", "",
                    "## Coverage by cohort (% of forward observations with the field present)", "",
                    "| Field | " + " | ".join(COHORTS) + " |", "|---|" + "---|" * len(COHORTS)]
    cov = r["coverage_by_cohort"]
    for f in FIELDS:
        cells = []
        for c in COHORTS:
            x = cov[c]
            if f == "with_metadata_block":
                n = x["observations"]
                cells.append(f"{x['with_metadata_block']}/{n}" if n else "—")
            else:
                cells.append("—" if x[f]["pct"] is None else f"{x[f]['pct']}%")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "## Coverage by direction", "", "| Field | " + " | ".join(r["coverage_by_direction"]) + " |",
          "|---|" + "---|" * len(r["coverage_by_direction"])]
    for f in FIELDS[1:]:
        L.append(f"| {f} | " + " | ".join("—" if x[f]["pct"] is None else f"{x[f]['pct']}%"
                                          for x in r["coverage_by_direction"].values()) + " |")
    L += ["", "## Row-feature coverage (%)", ""]
    for c, fc in r["row_feature_coverage_pct_by_cohort"].items():
        L.append(f"- {c}: {fc or '—'}")
    L += ["", "## Provenance values (counts)", ""]
    for c, vals in r["value_counts_by_cohort"].items():
        L.append(f"- **{c}**: " + "; ".join(f"{k}={v}" for k, v in vals.items()))
    L += ["", "## Notes", ""] + [f"- {k}: {v}" for k, v in r["notes"].items()] + [""]
    return "\n".join(L)


def _load(input_path):
    if input_path:
        return json.loads(Path(input_path).read_text()).get("observations") or []
    from db.hsf_observations import load_recent_observations
    return load_recent_observations(limit=100000) or []


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 57 research metadata audit (read-only)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "research"))
    ap.add_argument("--input", default=None)
    args = ap.parse_args()
    report = audit(_load(args.input))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "research_metadata_audit.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "research_metadata_audit.md").write_text(render_markdown(report))
    print(f"RUN 57 — RESEARCH METADATA AUDIT: forward_observations={report['forward_observations']}")
    for c in COHORTS:
        x = report["coverage_by_cohort"][c]
        print(f"  {c}: n={x['observations']} with_block={x['with_metadata_block']} "
              f"scoring={x['scoring_version']['pct']} provider={x['price_provider']['pct']} "
              f"rank={x['rank_at_observation']['pct']} tier={x['tier_at_observation']['pct']} "
              f"regime={x['market_regime_at_observation']['pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
