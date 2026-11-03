"""HSF ML v4 shadow scoring: what shadow models would have said, never shown to customers.

    workflow "ML v4 Shadow" (scheduled weekdays + manual)   or   python scripts/ml_v4_shadow.py

For every registry model in ``shadow`` (and the registry ``champion``, which is
also not wired into AI Confidence yet):
  1. downloads its model file from the GitHub Release named in the registry and
     checks the sha256 before loading it;
  2. scores new observations made AFTER it entered shadow (a forward test),
     with the features frozen at observation time;
  3. stores the predictions in ``ml_shadow_predictions`` (an existing
     model/observation pair is never overwritten);
then fills labels for predictions whose outcome has matured and drops rows
older than 180 days.

Writes only ml_shadow_predictions. Never changes AI Confidence, PreBreakout,
alerts, the registry or any model. ``--count`` prints how many models need
scoring (the workflow skips installing the ML libraries when it is 0).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCORED_STATUSES = ("shadow", "champion")
LOOKBACK_DAYS = 21  # covers 5 trading days of maturation plus weekends/holidays


def _log(msg: str) -> None:
    print(f"[ml-v4-shadow] {msg}", flush=True)


def scored_models() -> list:
    from db import ml_model_registry as R

    out = []
    for status in SCORED_STATUSES:
        out.extend(m for m in R.list_models(status=status) if m.get("kind", "model") == "model")
    return out


def shadow_start(model_id: int):
    """When the model first entered shadow (forward test starts there)."""
    from analytics import research_dataset as rd
    from db import ml_model_registry as R

    starts = [e["created_at"] for e in R.model_events(model_id) if e.get("to_status") == "shadow"]
    return rd.to_dt(min(starts, key=str)) if starts else None


def fetch_release_asset(uri: str, dest: Path) -> Path:
    """github-release://<tag>/<file> -> local file, via the GitHub API (GH_TOKEN, GITHUB_REPOSITORY)."""
    if not uri or not uri.startswith("github-release://"):
        raise ValueError(f"unsupported artifact location {uri!r}")
    tag, _, name = uri[len("github-release://"):].partition("/")
    repo, token = os.getenv("GITHUB_REPOSITORY"), os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not (repo and token and tag and name):
        raise RuntimeError("GITHUB_REPOSITORY and GH_TOKEN are needed to download model files")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "hsf-ml-v4-shadow"}
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 - fixed https host
        release = json.load(resp)
    asset = next((a for a in release.get("assets", []) if a.get("name") == name), None)
    if asset is None:
        raise FileNotFoundError(f"{name} is not attached to release {tag}")
    req = urllib.request.Request(asset["url"], headers={**headers, "Accept": "application/octet-stream"})
    out = dest / name
    with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as fh:  # nosec B310
        fh.write(resp.read())
    return out


def load_estimator(model: dict, workdir: Path, fetch=fetch_release_asset):
    from analytics import ml_v4_providers as P

    path = fetch(model.get("artifact_uri"), workdir)
    return P.load_artifact(path, model.get("artifact_sha256"))


def recent_rows(days: int = LOOKBACK_DAYS) -> list:
    """SELECT only: recent observations as signal-day audit rows (features as frozen)."""
    from analytics import ml_v3_audit as A
    from api.research import load_window

    today = dt.datetime.now(dt.timezone.utc).date()
    window = load_window(today - dt.timedelta(days=days), today)
    return A.build_audit_rows(window["records"], window["rows"], unit="signal_day")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", action="store_true", help="print the number of models to score and exit")
    ap.add_argument("--lookback-days", type=int, default=int(os.getenv("ML_V4_SHADOW_LOOKBACK_DAYS") or LOOKBACK_DAYS))
    args = ap.parse_args(argv)

    models = scored_models()
    if args.count:
        print(len(models))
        return 0
    if not models:
        _log("no shadow or champion models in the registry; nothing to score")
        return 0

    from analytics import ml_v3_audit as A
    from analytics import ml_v4_monitoring as M
    from analytics import ml_v4_providers as P
    from analytics import ml_v4_shadow as S
    from db import ml_shadow_predictions as D

    schemas = {name: P.feature_schema_version(fs) for name, fs in P.FEATURE_SETS.items()}

    rows = recent_rows(args.lookback_days)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _log(f"models={len(models)} recent signal-days={len(rows)}")
    inserted, failures, targets = 0, [], set()
    with tempfile.TemporaryDirectory() as tmp:
        for m in models:
            targets.add((m["target_label"], int(m["horizon_days"])))
            if m.get("feature_schema_version") != schemas.get(str(m.get("feature_set"))):
                # Features changed since training: scores would be silently wrong, so refuse.
                failures.append({"model_version": m["model_version"], "error": "feature schema changed since training"})
                _log(f"SKIPPED {m['model_version']}: feature schema {m.get('feature_schema_version')} != "
                     f"{schemas.get(str(m.get('feature_set')))}")
                continue
            start = shadow_start(int(m["id"]))
            todo = S.forward_rows(rows, start)
            if not todo:
                _log(f"{m['model_version']}: no observations since shadow start {start}")
                continue
            try:
                est = load_estimator(m, Path(tmp))
                preds = S.score(m, est, todo, scored_at=now)
            except Exception as e:  # one broken model must not stop the others
                failures.append({"model_version": m["model_version"], "error": f"{type(e).__name__}: {e}"})
                _log(f"FAILED {m['model_version']}: {type(e).__name__}: {e}")
                continue
            n = D.insert_predictions(preds)
            inserted += n
            _log(f"{m['model_version']}: {n} new predictions ({len(preds)} forward signal-days in window)")

    filled = 0
    by_id = {r.observation_id: r for r in rows}
    for label, h in sorted(targets):
        waiting = set(D.unlabeled(label, h))
        matured = [(r.observation_id, y) for r, y in A.eligible([by_id[i] for i in waiting if i in by_id], h, label)]
        filled += D.fill_labels(label, h, matured)
    pruned = D.prune()
    now_dt = dt.datetime.now(dt.timezone.utc)
    preds = D.predictions([int(m["id"]) for m in models], since=now_dt - dt.timedelta(days=D.RETENTION_DAYS))
    by_model: dict = {}
    for p in preds:
        by_model.setdefault(int(p["model_id"]), []).append(p)
    health = M.monitor(models, by_model, now=now_dt, feature_schemas=schemas)
    summary = {"models": len(models), "inserted": inserted, "labels_filled": filled, "pruned": pruned,
               "failures": failures, "monitoring_status": health["status"]}
    _log(json.dumps(summary))
    write_step_summary(summary, health)
    return 1 if failures else 0


def write_step_summary(summary: dict, health: dict) -> None:
    """Markdown for the GitHub run page (no-op outside Actions)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["# ML v4 shadow scoring", "",
             f"Models {summary['models']} · new predictions {summary['inserted']} · labels filled "
             f"{summary['labels_filled']} · pruned {summary['pruned']} · monitoring **{health['status']}**", "",
             "| Model | Status | Check | Value | Detail |", "|---|---|---|---|---|"]
    for m in health["models"]:
        for c in m["checks"]:
            lines.append(f"| {m['model_version']} | {m['status']} | {c['check']} | {c['value']} | {c['detail']} |")
    for f in summary["failures"]:
        lines.append(f"\n**Failed:** {f['model_version']}: {f['error']}")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
