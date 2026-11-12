"""Freeze one new ML research dataset version, built exactly the way the ML v4
trainer will rebuild it.

    python scripts/ml_freeze_dataset.py                   # preview (read-only)
    python scripts/ml_freeze_dataset.py --freeze FREEZE   # WRITES one registry row

Why a separate script: scripts/research_dataset.py --finalize builds from
un-masked rows, while the trainer (scripts/ml_v4_train.load_audit_rows) rebuilds
a version from rows masked "as of" its created_at. Building and checking here
through the trainer's own path means the frozen fingerprint is the one the
preflight will reproduce.

Checks before any write (all must pass, otherwise nothing is written):
  * the cohort starts 2026-09-14, so the 2026-09-12/13 bootstrap burst is out;
  * two independent builds give the same membership and fingerprint;
  * the readiness report on the frozen population, at the freeze instant, is
    READY (the trainer's preflight repeats this check at created_at);
  * right before the INSERT, a fresh reload still gives the same fingerprint
    (no outcome was written in between).
After the write, the version is reloaded through the trainer's loader and
must pass its fingerprint and membership check.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_START = dt.date(2026, 9, 14)  # first day after the bootstrap burst (#317)
CONFIRM = "FREEZE"


def _log(msg: str) -> None:
    print(f"[freeze {dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load_rows():
    """SELECT only: the same window + benchmark times the trainer reads."""
    from api.research import load_window
    from scripts.ml_v4_train import EARLIEST, _benchmark_computed_at

    window = load_window(EARLIEST, _now().date())
    bca = _benchmark_computed_at([r["id"] for r in window["rows"]])
    rows = [{**r, "benchmark_computed_at": bca.get(int(r["id"]))} for r in window["rows"]]
    return rows, window["scans"]


def build_as_of(rows, scans, at: dt.datetime, filters: dict, members=None):
    from analytics import ml_v3_pipeline as V3
    from analytics import research_dataset as rd

    rows_asof = V3.as_of(rows, at)
    built = rd.build_dataset(rows_asof, scans, filters=filters, members=members,
                             code_revision=os.getenv("GITHUB_SHA"))
    return built, rows_asof


def readiness_at(rows_asof, scans, member_ids, at: dt.datetime) -> dict:
    """Same call the trainer's preflight makes on the frozen population."""
    from analytics import ml_readiness as MR

    keep = set(member_ids)
    return MR.build_report([r for r in rows_asof if int(r["id"]) in keep], scans, now=at)


def _summary(built: dict) -> dict:
    m = built["metadata"]
    return {"observations": m["observation_count"], "matured": m["matured_count"],
            "certified": m["certified_count"], "range": m["observation_range"],
            "fingerprint": m["fingerprint"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=dt.date.fromisoformat, default=DEFAULT_START)
    p.add_argument("--freeze", default="", help=f"type {CONFIRM} to write the registry row")
    a = p.parse_args(argv)
    if a.start < DEFAULT_START:
        raise SystemExit(f"Start {a.start} would include the bootstrap burst; use {DEFAULT_START} or later.")
    freeze = a.freeze == CONFIRM
    if a.freeze and not freeze:
        raise SystemExit(f"--freeze must be exactly {CONFIRM}; nothing written.")

    at = _now()
    filters = {"start_date": a.start, "end_date": at.date()}
    _log(f"mode={'FREEZE' if freeze else 'preview'} start={a.start} end={at.date()} as_of={at.isoformat()}")

    rows, scans = load_rows()
    first, rows_asof = build_as_of(rows, scans, at, filters)
    second, _ = build_as_of(*load_rows(), at, filters)
    blockers = []
    if first["observation_ids"] != second["observation_ids"]:
        blockers.append("Twin builds differ in membership.")
    if first["metadata"]["fingerprint"] != second["metadata"]["fingerprint"]:
        blockers.append("Twin builds differ in fingerprint.")
    rep = readiness_at(rows_asof, scans, first["observation_ids"], at)
    if rep.get("status") != "READY":
        blockers.append(f"Readiness at freeze is {rep.get('status')}: {rep.get('blocking_reasons')}")

    summary = {**_summary(first), "readiness": rep.get("status"),
               "readiness_blockers": rep.get("blocking_reasons"),
               "component_fingerprints": first["metadata"].get("component_fingerprints"),
               "code_revision": os.getenv("GITHUB_SHA"), "blockers": blockers}
    print(json.dumps(summary, indent=1, default=str))
    if blockers:
        _log("BLOCKED: nothing written.")
        return 1
    if not freeze:
        _log("Preview passed. Nothing written.")
        return 0

    # Last look right before the INSERT: a fresh reload, masked to now, must
    # still match, so no outcome landed between the build and the write.
    from analytics import research_dataset as rd
    from db.research_datasets import list_dataset_versions, save_dataset_version

    last, _ = build_as_of(*load_rows(), _now(), filters, members=first["observation_ids"])
    if last["metadata"]["fingerprint"] != first["metadata"]["fingerprint"]:
        _log("BLOCKED: data changed during the freeze (an outcome job likely ran). Nothing written; retry later.")
        return 1
    name = rd.version_name(_now().date(), [e["dataset_version"] for e in list_dataset_versions()])
    meta = {**first["metadata"], "dataset_version": name, "built_as_of": at.isoformat(),
            "freeze_tool": "scripts/ml_freeze_dataset.py"}
    ok = save_dataset_version({"dataset_version": name, "feature_schema_version": meta["feature_schema_version"],
                               "label_schema_version": meta["label_schema_version"],
                               "fingerprint": meta["fingerprint"], "observation_count": meta["observation_count"],
                               "observation_ids": first["observation_ids"], "metadata": meta})
    if not ok:
        _log(f"{name} already exists; nothing written.")
        return 1
    _log(f"WROTE {name}: {meta['observation_count']} observations, {meta['fingerprint']}")

    from scripts.ml_v4_train import load_audit_rows

    try:
        loaded = load_audit_rows(name)
    except Exception as e:  # report loudly; the row is immutable
        _log(f"POST-CHECK FAILED for {name}: {e}. Do not train on it; freeze a new version.")
        return 1
    _log(f"Post-check passed: trainer reload of {name} matches (frozen_at={loaded['frozen_at']}, "
         f"signal-day rows={len(loaded['audit_rows'])}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
