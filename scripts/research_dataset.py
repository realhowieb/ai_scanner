"""HSF research dataset CLI: coverage report, export, and version finalization.

Read-only by default. It reads persisted rows only (signal_outcomes
opportunities + scheduled hsf_observations) and makes no provider calls.

    python scripts/research_dataset.py                       # report, all history
    python scripts/research_dataset.py --start 2026-09-01 --end 2026-10-08
    python scripts/research_dataset.py --export artifacts/research/ds.parquet
    python scripts/research_dataset.py --finalize            # WRITES one immutable registry row

The Diagnostics workflow (manual, read-only) runs it with no arguments:
    workflow_dispatch -> script=research_dataset.py
--finalize is the only write and is never the default.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics import research_dataset as rd  # noqa: E402
from analytics import research_schema as rs  # noqa: E402

EARLIEST = dt.date(2025, 1, 1)


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def load(start: dt.date, end: dt.date):
    from api.research import load_window

    return load_window(start, end)


def report(start: dt.date, end: dt.date) -> dict:
    t0 = time.perf_counter()
    w = load(start, end)
    records = w["records"]
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    cov = rd.coverage(records)
    quality = rd.quality_report(w["rows"], records)
    t_cov = time.perf_counter() - t1

    t2 = time.perf_counter()
    built = rd.build_dataset(w["rows"], w["scans"], filters={"start_date": start, "end_date": end},
                             code_revision=os.getenv("GITHUB_SHA"))
    t_build = time.perf_counter() - t2
    again = rd.build_dataset(w["rows"], w["scans"], filters={"start_date": start, "end_date": end},
                             code_revision=os.getenv("GITHUB_SHA"))
    deterministic = again["metadata"]["fingerprint"] == built["metadata"]["fingerprint"]

    page_ms = feat_ms = None
    if records:
        t3 = time.perf_counter()
        rd.apply_filters(records, rd.normalize_filters(start_date=start, end_date=end))[:100]
        page_ms = round((time.perf_counter() - t3) * 1000, 2)
        try:
            from api.research import features

            t4 = time.perf_counter()
            features(records[-1]["observation"]["observation_id"])
            feat_ms = round((time.perf_counter() - t4) * 1000, 1)
        except Exception as e:  # report, don't fail the run
            feat_ms = f"failed: {type(e).__name__}"

    print(f"# HSF research dataset report ({start} .. {end})\n")
    print(f"observations={cov['total_observations']} matured_5d={cov['matured_observations']} "
          f"pending={cov['pending_observations']} unavailable={cov['unavailable_observations']} "
          f"certified={cov['certified_observations']}")
    print(f"earliest={cov['earliest_observation']} latest={cov['latest_observation']}")
    print(f"scan_feature_join_rate={_pct(cov['scan_feature_join_rate'])} "
          f"research_metadata_rate={_pct(cov['research_metadata_rate'])}\n")

    print("## Feature coverage\n")
    specs = {f.name: f for f in rs.feature_schema()}
    print(_table(["FEATURE", "SOURCE", "POINT-IN-TIME", "COVERAGE", "SCHEMA"],
                 [[n, specs[n].source, specs[n].pit, _pct(v), f"v{rs.FEATURE_SCHEMA_VERSION}"]
                  for n, v in cov["features"].items()]
                 + [[u["name"], "-", u["classification"], "0.0%", "not in schema"] for u in cov["unavailable_features"]]))

    print("\n## Labels\n")
    rows = []
    for key in ("horizons", "benchmark", "excess"):
        for h, v in cov[key].items():
            rows.append([f"{key}:{h}", v["present"], v["of_scored"], _pct(v["rate"])])
    for k in ("mfe_5d", "mae_5d"):
        rows.append([k, cov[k]["present"], cov[k]["of_scored"], _pct(cov[k]["rate"])])
    print(_table(["LABEL", "PRESENT", "SCORED ROWS", "RATE"], rows))

    print("\n## Data quality\n")
    print("```json\n" + json.dumps(quality, indent=1, sort_keys=True, default=str) + "\n```")

    print("\n## Join diagnostics\n")
    lags = sorted(r["features"].join.get("lag_seconds") for r in records
                  if r["features"].join.get("status") == rd.JOIN_MATCHED)
    if lags:
        print(f"matched lag seconds: min={lags[0]} median={lags[len(lags) // 2]} max={lags[-1]}")
    views = [rd.scan_record_view(s) for s in w["scans"]]
    st = sorted(v["scan_timestamp"] for v in views if v["scan_timestamp"])
    print(f"scan records fetched: {len(views)}; scan_timestamp range: "
          f"{st[0].isoformat() if st else None} .. {st[-1].isoformat() if st else None}")
    from collections import Counter, defaultdict
    print("contexts:", dict(Counter(v["context"] for v in views)))
    by_day = defaultdict(Counter)
    for r in records:
        by_day[(r["observation"]["observed_at"] or "")[:10]][r["features"].join.get("status")] += 1
    print(_table(["DAY", "MATCHED", "ONLY_LATER_SCANS", "NO_SCAN_RECORD", "NO_SCAN_WITHIN_LAG"],
                 [[d, c.get(rd.JOIN_MATCHED, 0), c.get(rd.JOIN_FUTURE_ONLY, 0), c.get(rd.JOIN_MISSING, 0),
                   c.get(rd.JOIN_STALE, 0)] for d, c in sorted(by_day.items())]))
    # For refused joins: how far after observed_at was the nearest record written?
    idx = rd.index_scan_records(w["scans"])
    gaps = []
    for r in records:
        if r["features"].join.get("status") != rd.JOIN_FUTURE_ONLY:
            continue
        obs = rd.to_dt(r["observation"]["observed_at"])
        c = [v for v in idx.get(r["observation"]["ticker"], []) if v["scan_timestamp"] <= obs]
        if c:
            best = max(c, key=lambda v: v["scan_timestamp"])
            gaps.append(((best["known_at"] - obs).total_seconds() if best["known_at"] else None,
                         (obs - best["scan_timestamp"]).total_seconds()))
    print(f"ONLY_LATER_SCANS with a scan started before observed_at: {len(gaps)}")
    if gaps:
        w_after = sorted(g[0] for g in gaps if g[0] is not None)
        print(f"  written after observed_at by (s): min={w_after[0]} median={w_after[len(w_after)//2]} "
              f"max={w_after[-1]}")
    unverified = sum(1 for r in records if r["features"].join.get("known_at_verified") is False)
    print(f"matched without a write time (known_at unverified): {unverified}")

    print("\n## Performance\n")
    print(json.dumps({"window_load_s": round(t_load, 3), **w["timings"],
                      "coverage_calc_ms": round(t_cov * 1000, 1), "dataset_build_ms": round(t_build * 1000, 1),
                      "page_filter_ms": page_ms, "single_feature_snapshot_ms": feat_ms,
                      "deterministic_rebuild": deterministic,
                      "fingerprint": built["metadata"]["fingerprint"]}, indent=1))
    return {"window": w, "built": built}


def export(built: dict, path: Path) -> None:
    import pandas as pd

    meta = dict(built["metadata"])
    df = pd.DataFrame({"observation_id": built["observation_ids"], "observed_at": built["observed_at"]})
    feats = pd.DataFrame(built["features"], columns=[f"f__{c}" for c in built["feature_columns"]])
    labels = pd.DataFrame(built["labels"], columns=[f"y__{c}" for c in built["label_columns"]])
    for c in feats.columns:  # list features -> stable string for columnar formats
        if feats[c].map(lambda v: isinstance(v, list)).any():
            feats[c] = feats[c].map(lambda v: "|".join(v) if isinstance(v, list) else v)
    out = pd.concat([df, feats, labels, pd.DataFrame({"maturity_5d": built["maturity_5d"]})], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(out, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               b"hsf_research": json.dumps(meta, default=str).encode()})
        pq.write_table(table, path)
    else:
        out.to_csv(path, index=False)
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"exported {len(out)} rows to {path} (feature columns f__*, label columns y__*)")


def finalize(built: dict, start: dt.date, end: dt.date) -> None:
    from db.research_datasets import list_dataset_versions, save_dataset_version

    name = rd.version_name(dt.datetime.now(dt.timezone.utc).date(),
                           [e["dataset_version"] for e in list_dataset_versions()])
    meta = {**built["metadata"], "dataset_version": name,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    ok = save_dataset_version({"dataset_version": name, "feature_schema_version": meta["feature_schema_version"],
                               "label_schema_version": meta["label_schema_version"],
                               "fingerprint": meta["fingerprint"], "observation_count": meta["observation_count"],
                               "observation_ids": built["observation_ids"], "metadata": meta})
    print(f"finalized {name}: {meta['observation_count']} observations, {meta['fingerprint']}" if ok
          else f"{name} already exists; nothing written")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=dt.date.fromisoformat, default=EARLIEST)
    p.add_argument("--end", type=dt.date.fromisoformat, default=None)
    p.add_argument("--export", type=Path, default=None, help=".parquet (preferred) or .csv")
    p.add_argument("--finalize", action="store_true", help="WRITE an immutable dataset version row")
    a = p.parse_args(argv)
    end = a.end or dt.datetime.now(dt.timezone.utc).date()
    out = report(a.start, end)
    if a.export:
        export(out["built"], a.export)
    if a.finalize:
        finalize(out["built"], a.start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
