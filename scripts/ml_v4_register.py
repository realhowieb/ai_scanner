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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="folder a training run wrote (models + manifests)")
    ap.add_argument("--release", default=os.getenv("ML_V4_RELEASE_TAG") or None,
                    help="GitHub Release tag the model files were uploaded to")
    ap.add_argument("--actor", default=os.getenv("ML_V4_ACTOR") or os.getenv("GITHUB_ACTOR") or None)
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

    results = []
    for man in manifests:
        uri = f"github-release://{args.release}/{man['artifact_file']}" if args.release else None
        if args.dry_run:
            results.append({"model_version": man["model_version"], "artifact_uri": uri, "would_register": True})
            continue
        from db import ml_model_registry as R

        res = R.register_model(man, actor=args.actor, artifact_uri=uri)
        results.append({"model_version": man["model_version"], "artifact_uri": uri, **res})
    for r in results:
        state = "dry run" if args.dry_run else ("registered" if r.get("created") else f"already {r.get('status')}")
        print(f"{r['model_version']}: {state} ({r['artifact_uri']})")
    print(json.dumps({"registered": sum(1 for r in results if r.get("created")),
                      "already_present": sum(1 for r in results if r.get("created") is False),
                      "dry_run": bool(args.dry_run), "models": results}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
