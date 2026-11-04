"""Offline ML rehearsal: fictional data, local artifacts and SQLite ledger only.

    python scripts/ml_v4_rehearsal.py --out /tmp/ml-v4-rehearsal

Exercises preflight, real training for all three providers, development and final
evaluation, durable report reuse, registration validation and shadow predictions.
Never connects to Neon, GitHub or a market-data provider. Synthetic scores are
not evidence about the production models.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Rehearsal output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    import ml_v4_evaluate as E
    import ml_v4_register as R
    import ml_v4_train as T

    from analytics import ml_v4_evaluation as EV
    from analytics import ml_v4_providers as P
    from analytics import ml_v4_shadow as S
    from analytics.ml_v4_synthetic import dataset
    from db import ml_holdout_evaluations as ledger

    loaded = dataset()
    params = {p: {**P.DEFAULT_PARAMS[p], **({"iterations": 8} if p == "catboost" else {"n_estimators": 8})}
              for p in P.PROVIDERS}
    conn = sqlite3.connect(out / "synthetic-final-test.sqlite")
    ledger.set_connection_factory(lambda: conn)
    try:
        with patch("db.engine.get_neon_conn", side_effect=AssertionError("Rehearsal must stay offline")), \
                patch.object(T, "load_audit_rows", return_value=loaded), \
                patch.object(E, "_spy_closes", return_value={}), \
                patch.object(P, "DEFAULT_PARAMS", params), patch.object(EV, "BOOT", 50):
            common = ["--feature-sets", "production", "--git-sha", "synthetic-rehearsal"]
            T.main([*common, "--out", str(out / "preflight"), "--preflight-only"])
            T.main([*common, "--out", str(out / "models")])
            E.main([*common, "--out", str(out / "development")])
            E.main([*common, "--out", str(out / "final"), "--final-holdout"])
            E.main([*common, "--out", str(out / "reused-final"), "--final-holdout"])
            report_path = out / "final" / "ml_v4_evaluation.json"
            report = json.loads(report_path.read_text())
            reused = json.loads((out / "reused-final" / "ml_v4_evaluation.json").read_text())
            assert report == reused, "Final evaluation must be reused byte-for-byte"
            R.main(["--dir", str(out / "models"), "--evaluation", str(report_path), "--dry-run"])
            ledger.verify_completed(report["holdout_claim"], report["binding"], report)
            predictions = 0
            for i, manifest in enumerate(R.load_manifests(out / "models"), 1):
                _, cal = R.evaluation_for(report, manifest)
                estimator = P.load_artifact(out / "models" / manifest["artifact_file"], manifest["artifact_sha256"])
                model = {**manifest, "id": i, "target_label": manifest["target"]["label"],
                         "horizon_days": manifest["target"]["horizon_days"], "calibration": cal}
                predictions += len(S.score(model, estimator, loaded["audit_rows"][-10:], scored_at="synthetic"))
            summary = {"synthetic_only": True, "trained_models": 3, "shadow_predictions": predictions,
                       "final_report_reused": True, "production_writes": 0}
            (out / "rehearsal.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary))
    finally:
        ledger.set_connection_factory(None)
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
