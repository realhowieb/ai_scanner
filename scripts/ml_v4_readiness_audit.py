"""HSF ML v4 production readiness check (read-only, aggregates only).

    workflow_dispatch "Diagnostics" (script=ml_v4_readiness_audit.py)   or   python scripts/ml_v4_readiness_audit.py

Answers "what is live right now?" for the Milestone 7 readiness audit, using
SELECTs only and printing counts, versions and statuses (Diagnostics logs are
public: no rows, tickers, credentials or paths):
  * finalized research dataset versions (names + sizes);
  * ML v4 readiness status and failing gates (the same view as Admin > Research);
  * registry models by status (ml_models), events, champion per target;
  * shadow predictions: rows, matured rows, last scored, per model;
  * production AI Confidence model version in use (unchanged by ML v4);
  * which ML libraries this runner has.
Missing tables are reported as "not created yet", never created.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _scalar(cur, sql: str, params: tuple = ()):
    cur.execute(sql, params)
    r = cur.fetchone()
    if r is None:
        return None
    return list(r.values())[0] if isinstance(r, dict) else r[0]


def _rows(cur, sql: str, params: tuple = ()) -> list:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in cur.fetchall()]


def collect(conn) -> dict:
    cur = conn.cursor()
    out: dict = {}

    def table(name: str) -> bool:
        return bool(_scalar(cur, "SELECT to_regclass(%s) IS NOT NULL", (name,)))

    out["tables"] = {t: table(t) for t in ("research_dataset_versions", "ml_models", "ml_model_events",
                                           "ml_shadow_predictions", "ai_confidence_models")}
    if out["tables"]["research_dataset_versions"]:
        out["datasets"] = [{"dataset_version": r["dataset_version"], "created_at": str(r["created_at"]),
                            "members": r["members"]} for r in _rows(
            cur, "SELECT dataset_version, created_at, observation_count AS members "
                 "FROM research_dataset_versions ORDER BY created_at")]
    if out["tables"]["ml_models"]:
        out["registry"] = {
            "by_status": {r["status"]: int(r["n"]) for r in _rows(
                cur, "SELECT status, COUNT(*) AS n FROM ml_models GROUP BY status")},
            "champions": [{"target": r["target_label"], "horizon_days": r["horizon_days"],
                           "model_version": r["model_version"]} for r in _rows(
                cur, "SELECT target_label, horizon_days, model_version FROM ml_models WHERE status = 'champion'")],
            "events": int(_scalar(cur, "SELECT COUNT(*) FROM ml_model_events") or 0)
            if out["tables"]["ml_model_events"] else 0,
        }
    if out["tables"]["ml_shadow_predictions"]:
        out["shadow"] = [{"model_version": r["model_version"], "rows": int(r["n"]), "matured": int(r["labeled"]),
                          "last_scored_at": str(r["last"])} for r in _rows(
            cur, "SELECT model_version, COUNT(*) AS n, COUNT(label) AS labeled, MAX(scored_at) AS last "
                 "FROM ml_shadow_predictions GROUP BY model_version ORDER BY model_version")]
    if out["tables"]["ai_confidence_models"]:
        # Never SELECT * here: model_bytes is the whole model.
        out["ai_confidence_active"] = [{"model_version": r["model_version"], "trained_at": str(r["trained_at"])}
                                       for r in _rows(cur, "SELECT model_version, trained_at FROM ai_confidence_models "
                                                           "WHERE is_active ORDER BY created_at DESC LIMIT 1")]
    cur.close()
    return out


def readiness() -> dict:
    try:
        from api import ml_readiness as mlr

        v = mlr.readiness()
        return {"status": v["status"], "recommendation": v["recommendation"],
                "blocking": list(v["blocking_reasons"]),
                "gates": {g["gate"]: g["status"] for g in v["gates"]}}
    except Exception as e:
        return {"error": type(e).__name__}


def main() -> int:
    from db.engine import get_neon_conn

    report = {"libraries": {m: bool(importlib.util.find_spec(m)) for m in
                            ("xgboost", "lightgbm", "catboost", "sklearn")}}
    conn = get_neon_conn()
    if conn is None:
        print(json.dumps({"error": "database unavailable"}))
        return 1
    try:
        report.update(collect(conn))
    finally:
        conn.close()
    report["ml_readiness"] = readiness()
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
