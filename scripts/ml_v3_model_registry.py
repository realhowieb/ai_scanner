"""ML v3 audit: read-only listing of model registry metadata (no model bytes).

Prints every ai_confidence_models / prebreakout_models row (id, version,
active flag, trained_at, created_at, stored AUC and metadata keys) plus the
research dataset registry and SPY benchmark coverage. SELECT only.

    workflow_dispatch Diagnostics -> script=ml_v3_model_registry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.engine import get_neon_conn  # noqa: E402

SMALL_KEYS = ("auc", "mean_auc", "std_auc", "rows", "positive_rows", "training_rows", "validation_rows",
              "validation_method", "target", "target_rule", "purge_days", "trained_at", "model_version",
              "selected_feature_set", "promotion_result", "eligible_rows", "source", "feature_names",
              "calibration_map", "validation_metrics", "fold_metrics", "baseline_comparison", "leakage_audit")


def _q(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:  # report, keep going
        cur.connection.rollback()
        return [{"error": f"{type(e).__name__}: {e}"}]


def _meta(m):
    if isinstance(m, (bytes, memoryview)):
        m = bytes(m).decode("utf-8", "replace")
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except ValueError:
            return {"raw": m[:200]}
    m = m or {}
    out = {k: m.get(k) for k in SMALL_KEYS if k in m}
    if isinstance(out.get("fold_metrics"), list):
        out["fold_metrics"] = [{k: f.get(k) for k in ("fold", "auc", "train_rows", "validation_rows",
                                                       "validation_start", "validation_end", "positive_rate")}
                               for f in out["fold_metrics"] if isinstance(f, dict)]
    if isinstance(out.get("calibration_map"), dict):
        out["calibration_map"] = {k: out["calibration_map"].get(k) for k in ("method", "n")}
    out["all_keys"] = sorted(m.keys())
    return out


def main() -> int:
    conn = get_neon_conn()
    if conn is None:
        print("no database connection")
        return 1
    cur = conn.cursor()
    for table, extra in (("ai_confidence_models", ""), ("prebreakout_models", ", auc")):
        rows = _q(cur, f"SELECT id, model_version, is_active, trained_at, created_at, feature_names, "
                       f"metadata{extra} FROM {table} ORDER BY id")
        print(f"=== {table} ({len(rows)} rows) ===")
        for r in rows:
            if "error" in r:
                print(r)
                continue
            fn = r.get("feature_names")
            if isinstance(fn, (bytes, memoryview)):
                fn = bytes(fn).decode("utf-8", "replace")
            if isinstance(fn, str):
                try:
                    fn = json.loads(fn)
                except ValueError:
                    fn = [x.strip() for x in fn.strip("{}[]").split(",") if x.strip()]
            fn = list(fn or [])
            print(json.dumps({"id": r["id"], "model_version": r["model_version"], "is_active": r["is_active"],
                              "trained_at": r["trained_at"], "created_at": r["created_at"],
                              "auc_column": r.get("auc"), "n_features": len(fn), "features": fn,
                              "metadata": _meta(r.get("metadata"))}, default=str))
    print("=== research_dataset_versions ===")
    for r in _q(cur, "SELECT dataset_version, fingerprint, observation_count, created_at "
                     "FROM research_dataset_versions ORDER BY created_at"):
        print(json.dumps(r, default=str))
    print("=== benchmark coverage (opportunity rows) ===")
    for r in _q(cur, "SELECT COUNT(*) AS n, COUNT(return_5d) AS ret5, MIN(fired_at) AS first, MAX(fired_at) AS last "
                     "FROM signal_outcomes WHERE source = 'opportunity'"):
        print(json.dumps(r, default=str))
    for r in _q(cur, "SELECT column_name FROM information_schema.columns WHERE table_name = 'signal_outcomes' "
                     "AND column_name LIKE 'benchmark%' ORDER BY column_name"):
        print("signal_outcomes column:", json.dumps(r, default=str))
    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
