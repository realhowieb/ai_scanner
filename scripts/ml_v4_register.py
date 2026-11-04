"""HSF ML v4: register trained models as registry candidates.

    workflow_dispatch "ML v4 Train & Evaluate" (mode=train, register=true)
    python scripts/ml_v4_register.py --dir <training out folder> --release <tag>

Reads the per-model ``<model_version>.json`` manifests a training run wrote
(``ml_v4_providers.save_artifact``), checks each ``.joblib`` file against the
sha256 in its manifest, and inserts one ``candidate`` row per model into the
ML v4 registry (``db/ml_model_registry.py``). The model FILE is not stored in
the database: the workflow uploads it to a GitHub Release first and the
registry keeps its location (``github-release://<tag>/<file>``) + checksum.

This is a database write (two additive tables), so it only runs when an admin
asks for it. It never changes a model's status beyond ``candidate``, never
touches ``ai_confidence_models`` / ``prebreakout_models``, and is idempotent:
re-registering the same model_version returns the existing row unchanged.
``--dry-run`` checks the files and prints what would be registered.

``--evaluation <ml_v4_evaluation.json>`` (from ``scripts/ml_v4_evaluate.py`` on
the same dataset version) stores each model's walk-forward + holdout metrics
and its serving calibrator on the row, matched by provider + feature set. A
model needs that evaluation before it can be marked "validated".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMMARY_FILE = "ml_v4_training_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifests(folder: Path) -> list:
    """Per-model manifests in ``folder`` whose artifact file exists and matches its checksum."""
    out = []
    for meta in sorted(folder.glob("*.json")):
        if meta.name == SUMMARY_FILE:
            continue
        man = json.loads(meta.read_text(encoding="utf-8"))
        if not (isinstance(man, dict) and man.get("model_version") and man.get("artifact_file")):
            continue
        art = folder / man["artifact_file"]
        if not art.is_file():
            raise SystemExit(f"{art.name} listed in {meta.name} is missing")
        digest = _sha256_file(art)
        if digest != man.get("artifact_sha256"):
            raise SystemExit(f"{art.name}: sha256 {digest[:12]} does not match its manifest; not registering")
        out.append(man)
    return out


def evaluation_for(report: dict, manifest: dict):
    """(metrics, calibration) for one manifest from an evaluation report, or (None, None)."""
    if not report:
        return None, None
    expected = manifest.get("evaluation_binding")
    if not expected or report.get("binding") != expected:
        raise SystemExit("Evaluation binding mismatch: dataset, target, settings, code and library versions must match training")
    target = {k: manifest.get("target", {}).get(k) for k in ("label", "horizon_days")}
    if (expected.get("target") != target or expected.get("git_sha") != manifest.get("git_sha")
            or expected.get("providers", {}).get(manifest.get("provider")) != manifest.get("params")
            or manifest.get("feature_set") not in expected.get("feature_sets", [])
            or expected.get("training_slice_fingerprint") != manifest.get("training_slice_fingerprint")
            or expected.get("train_ids_sha256") != manifest.get("train_ids_sha256")
            or expected.get("library_versions") != manifest.get("library_versions")
            or expected.get("engine_version") != manifest.get("engine_version")):
        raise SystemExit("Training manifest does not match its evaluation binding")
    if not report.get("holdout_claim"):
        raise SystemExit("Registration needs a completed final-test ledger claim, not a development report")
    ds = (report.get("dataset") or {}).get("dataset_version")
    if ds != manifest.get("dataset_version"):
        raise SystemExit("Evaluation dataset version mismatch")
    if (report.get("dataset") or {}).get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise SystemExit("Evaluation dataset fingerprint mismatch")
    actual_target = {k: (report.get("primary") or {}).get("target", {}).get(k) for k in ("label", "horizon_days")}
    if actual_target != target:
        raise SystemExit("Evaluation target mismatch")
    name = f"{manifest.get('provider')}_{manifest.get('feature_set')}"
    res = ((report.get("primary") or {}).get("models") or {}).get(name)
    if not res:
        raise SystemExit("Evaluation is missing the trained provider / feature set")
    if not ((res.get("holdout") or {}).get("raw") or {}).get("n"):
        raise SystemExit("Evaluation has no scored final test")
    h2h = ((report.get("head_to_head_ai_confidence") or {}).get("comparisons") or {}).get(name)
    metrics = {"evaluation_version": report.get("evaluation_version"), "dataset_version": ds,
               "evaluation_binding": expected, "holdout_claim": report["holdout_claim"],
               "dataset_fingerprint": (report.get("dataset") or {}).get("dataset_fingerprint"),
               "evaluated_as": name, "evaluated_at": report.get("generated_at"),
               "matched_by": "exact evaluation binding + provider + feature set",
               **{k: res.get(k) for k in ("fold_auc", "walk_forward_raw", "walk_forward_calibrated", "holdout")
                  if k in res},
               "vs_ai_confidence_as_served": h2h,
               "calibration_note": "serving calibrator fitted on every walk-forward fold's out-of-fold predictions"}
    return metrics, res.get("serving_calibrator")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="folder a training run wrote (models + manifests)")
    ap.add_argument("--release", default=os.getenv("ML_V4_RELEASE_TAG") or None,
                    help="GitHub Release tag the model files were uploaded to")
    ap.add_argument("--actor", default=os.getenv("ML_V4_ACTOR") or os.getenv("GITHUB_ACTOR") or None)
    ap.add_argument("--evaluation", default=None, help="ml_v4_evaluation.json from the same dataset version")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    folder = Path(args.dir)
    manifests = load_manifests(folder)
    if not manifests:
        raise SystemExit(f"no model manifests in {folder}")
    if not args.dry_run and not args.release:
        raise SystemExit("--release is required: the registry stores where each model file lives")
    if not args.dry_run and not args.actor:
        raise SystemExit("--actor is required (who registered these models)")

    report = json.loads(Path(args.evaluation).read_text(encoding="utf-8")) if args.evaluation else None
    # Validate every manifest before the first registration can write anything.
    for man in manifests:
        evaluation_for(report, man)
    if report and not args.dry_run:
        from db.ml_holdout_evaluations import verify_completed

        verify_completed(report["holdout_claim"], report["binding"], report)
    results = []
    for man in manifests:
        metrics, calibration = evaluation_for(report, man)
        uri = f"github-release://{args.release}/{man['artifact_file']}" if args.release else None
        if args.dry_run:
            results.append({"model_version": man["model_version"], "artifact_uri": uri, "would_register": True,
                            "evaluation": bool(metrics)})
            continue
        from db import ml_model_registry as R

        res = R.register_model(man, actor=args.actor, artifact_uri=uri, metrics=metrics, calibration=calibration)
        if metrics and not res["created"] and res["status"] in ("candidate", "validated"):
            R.attach_evaluation(res["id"], metrics=metrics, calibration=calibration, actor=args.actor)
        results.append({"model_version": man["model_version"], "artifact_uri": uri, "evaluation": bool(metrics),
                        **res})
    for r in results:
        state = "dry run" if args.dry_run else ("registered" if r.get("created") else f"already {r.get('status')}")
        print(f"{r['model_version']}: {state} ({r['artifact_uri']})")
    print(json.dumps({"registered": sum(1 for r in results if r.get("created")),
                      "already_present": sum(1 for r in results if r.get("created") is False),
                      "dry_run": bool(args.dry_run), "models": results}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
