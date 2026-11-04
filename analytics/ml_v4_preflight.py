"""Read-only preparation checks and reproducible evaluation bindings for ML v4."""
from __future__ import annotations

import hashlib
import json

from analytics import ml_v3_audit as A
from analytics import ml_v4_providers as P
from analytics import research_dataset as rd


class PreparationError(ValueError):
    pass


def verify_dataset(entry, built):
    expected = entry.get("fingerprint")
    if not expected or expected != built["metadata"].get("fingerprint"):
        raise PreparationError("Frozen dataset fingerprint mismatch. Finalize and review a fresh version; do not reuse this name.")
    members = entry.get("observation_ids") or []
    if isinstance(members, str):
        members = json.loads(members)
    if set(map(int, members)) != set(map(int, built["observation_ids"])):
        raise PreparationError("Frozen dataset membership changed or observations are missing.")


def binding(loaded, split, providers, feature_sets, git_sha):
    from analytics import ml_v4_evaluation as E

    return {
        "dataset_version": loaded["entry"]["dataset_version"],
        "dataset_fingerprint": loaded["built"]["metadata"]["fingerprint"],
        "git_sha": git_sha,
        "engine_version": P.ENGINE_VERSION,
        "evaluation_version": E.EVALUATION_VERSION,
        "fold_spec": dict(E.FOLD_KW), "holdout_spec": dict(E.HOLDOUT_KW),
        "target": {"label": P.PRIMARY_TARGET.label, "horizon_days": P.PRIMARY_TARGET.horizon},
        "providers": {p: P.DEFAULT_PARAMS[p] for p in sorted(providers)},
        "feature_sets": sorted(feature_sets),
        "library_versions": P.library_versions(),
        "training_slice_fingerprint": A.slice_fingerprint([r for r, _ in split["train"]]),
        "train_ids_sha256": hashlib.sha256(json.dumps(sorted(r.observation_id for r, _ in split["train"])).encode()).hexdigest(),
        "served_ai_sha256": rd.fingerprint(sorted((int(r["id"]), r.get("ai_confidence")) for r in loaded["raw_rows"])),
    }


def prepare(loaded, providers, feature_sets, *, git_sha=None):
    from analytics import ml_readiness as MR

    if not providers or len(set(providers)) != len(providers) or any(p not in P.PROVIDERS for p in providers):
        raise PreparationError("Choose at least one known provider, without duplicates")
    if not feature_sets or len(set(feature_sets)) != len(feature_sets) or any(f not in P.FEATURE_SETS for f in feature_sets):
        raise PreparationError("Choose at least one known feature set, without duplicates")
    verify_dataset(loaded["entry"], loaded["built"])
    # Audit the selected frozen population, not today's newer observations.
    readiness = MR.build_report(loaded["raw_rows"], loaded["scan_rows"], now=loaded["frozen_at"])
    split = P.training_split(loaded["audit_rows"], P.PRIMARY_TARGET)
    eligible = A.eligible(loaded["audit_rows"], P.PRIMARY_TARGET.horizon, P.PRIMARY_TARGET.label)
    positive = sum(y for _, y in eligible)
    minority = min(positive, len(eligible) - positive) / len(eligible) if eligible else 0
    hold = split["holdout"]
    before_holdout = split["train"] if hold["valid"] else []
    folds = A.walk_forward_folds(before_holdout, P.PRIMARY_TARGET.horizon, **MR.FOLD_SPEC)
    blockers = list(readiness["blocking_reasons"])
    if minority < .20:
        blockers.append(f"Actual v4 target minority class is {minority:.1%}; needs at least 20%.")
    if not hold["valid"]:
        blockers.append(f"Actual v4 target has no valid final holdout: {hold['reason']}")
    if len(folds) < 3:
        blockers.append(f"Actual v4 target has {len(folds)} pre-holdout walk-forward folds; needs at least 3.")
    if not split["train_rows"] or split["train_positive"] in (0, split["train_rows"]):
        blockers.append("Training partition must contain both outcome classes.")
    if not git_sha:
        blockers.append("A code commit is required (GITHUB_SHA or --git-sha).")
    report = {
        "ready": not blockers, "blockers": blockers, "readiness_status": readiness["status"],
        "dataset_version": loaded["entry"]["dataset_version"], "fingerprint": loaded["built"]["metadata"]["fingerprint"],
        "frozen_at": str(loaded["frozen_at"]), "target": P.PRIMARY_TARGET.description,
        "eligible": len(eligible), "positive": positive, "negative": len(eligible) - positive,
        "minority_share": minority, "training_rows": split["train_rows"],
        "training_start": split["train_start"], "training_end": split["train_end"],
        "walk_forward_folds": len(folds), "holdout": {k: v for k, v in hold.items() if k != "ids"},
        "providers": list(providers), "feature_sets": list(feature_sets),
    }
    return report, split, binding(loaded, split, providers, feature_sets, git_sha)


def require_ready(report):
    if not report["ready"]:
        raise PreparationError("Training blocked:\n- " + "\n- ".join(report["blockers"]))


def write_load_failure(out, reason, dataset_version):
    """Keep a reviewable report even when the snapshot fails identity verification."""
    out.mkdir(parents=True, exist_ok=True)
    report = {"ready": False, "dataset_version": dataset_version, "blockers": [str(reason)]}
    (out / "ml_v4_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "ml_v4_preflight.md").write_text(
        f"# ML v4: ready to train\n\n**BLOCKED**\n\nDataset: {dataset_version or 'newest requested'}.\n\n- {reason}\n")


def markdown(report):
    lines = ["# ML v4: ready to train", "", f"**{'READY' if report['ready'] else 'BLOCKED'}**", "",
             f"Dataset: `{report['dataset_version']}`. Frozen: {report['frozen_at']}.", "",
             f"Target: {report['target']}", "",
             f"Usable examples: {report['eligible']} ({report['positive']} positive / {report['negative']} negative).",
             f"Training: {report['training_rows']} examples, {report['training_start']} to {report['training_end']}.",
             f"Walk-forward periods before final test: {report['walk_forward_folds']}.",
             f"Reserved final test: {report['holdout']['rows']} examples, starting {report['holdout']['start']}.", ""]
    lines += [f"- {b}" for b in report["blockers"]] or ["All preparation checks passed. This does not establish model quality."]
    return "\n".join(lines) + "\n"
