"""ML v4 model providers: XGBoost, LightGBM and CatBoost on one shared pipeline.

Research only. Nothing here touches production scoring: AI Confidence
(``scan/ai_confidence.py``), PreBreakout (``ml_prebreakout.py``) and their
model tables are never imported, read or written. No database access at all.

What is shared, so the three providers are compared like for like:
  * rows, labels and maturity come from the ML v3 audit layer
    (``analytics.ml_v3_audit``), built from a finalized research dataset;
  * the feature matrix comes from the same ``Encoder``, fitted on the training
    rows only (vocab never sees later rows). Missing values stay NaN: all three
    libraries handle them natively, so nothing is filled with 0 or a median;
  * the target is fixed before training (Howard, 2026-11-03): the production
    AI Confidence event, "+4% before -2% within 5 trading days", measured with
    the canonical order-free proxy ``F_clean_path_5d`` (MFE >= +4% and MAE > -2%
    over the 5-day window). Secondary targets are reported, never trained on
    silently;
  * training stops before a reserved final holdout, and training rows whose
    outcome window reaches into the holdout are purged (plus an embargo), so the
    holdout stays untouched for Milestone 3's evaluation.

Every model gets a deterministic seed, single-thread training and a recorded
configuration (params, library versions, feature and label schema versions,
dataset fingerprint, duration). Artifacts are versioned files with a sha256
and are never overwritten.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analytics import ml_v3_audit as A
from analytics import research_schema as rs

ENGINE_VERSION = "hsf-ml-v4-1"
SEED = 20261103
PROVIDERS = ("xgboost", "lightgbm", "catboost")
ARTIFACT_FORMAT = "hsf-ml-v4-artifact-1"


# ----------------------------------------------------------------------------- targets
@dataclass(frozen=True)
class Target:
    label: str
    horizon: int
    description: str


# Pre-declared 2026-11-03 (decision card in the ML v4 thread): the event AI
# Confidence shows customers as "5D outcome probability".
PRIMARY_TARGET = Target(
    label="F_clean_path_5d", horizon=5,
    description="Reaches +4% and never touches -2% over five trading days (MFE_5d >= +4%, MAE_5d > -2%). "
                "This is an order-free proxy for '+4% before -2%', not a measurement of which price was reached first.")
# Reported alongside the primary target; never used to pick a winner after the fact.
SECONDARY_TARGETS = tuple(
    Target(label=label, horizon=h, description=A.LABELS[label].description)
    for label in ("C_excess_gt_0", "A_return_gt_0") for h in A.HORIZONS)


def target_by_key(label: str, horizon: int) -> Target:
    for t in (PRIMARY_TARGET, *SECONDARY_TARGETS):
        if t.label == label and t.horizon == horizon:
            return t
    if label not in A.LABELS or horizon not in A.LABELS[label].horizons:
        raise ValueError(f"no label {label!r} at horizon {horizon}")
    return Target(label=label, horizon=horizon, description=A.LABELS[label].description)


# ----------------------------------------------------------------------------- feature sets
# Removed from the leakage-safe set per the ML v3 audit (ml_v3_recommendations.md #32):
# PreBreakout % and hsf_model_component carry the unrecorded PreBreakout version gap;
# same-row IsBreakout is excluded too.
LEAKAGE_REVIEW_FEATURES = ("prebreakout_prob", "hsf_model_component", "is_breakout")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]
    signals: bool
    description: str
    research_schema: int = 1   # minimum research feature schema version the set needs


FEATURE_SETS: Dict[str, FeatureSet] = {
    # The 6 production AI Confidence inputs, mapped to research columns: the
    # like-for-like comparison with the champion recipe.
    "production": FeatureSet(
        "production", tuple(A.AI_CONFIDENCE_FEATURE_MAP.values()), (), False,
        "the 6 production AI Confidence features (Trend10D%, Trend20D%, VolRel20, DollarVol20, "
        "BreakoutScore, GapPct)"),
    # Every point-in-time feature the ML v3 leakage audit did not flag.
    "leakage_safe": FeatureSet(
        "leakage_safe", tuple(c for c in A.NUMERIC_FEATURES if c not in LEAKAGE_REVIEW_FEATURES),
        tuple(A.CATEGORICAL_FEATURES), True,
        "all SAFE_STORED / SAFE_ASOF_JOIN research features minus " + ", ".join(LEAKAGE_REVIEW_FEATURES)),
}
# Schema v2 ablation arms (2026-11-13 plan): the same six core inputs, read
# with the prediction-record repair, then + RSI, + regime, + both. Only usable
# on datasets built with research feature schema v2.
_CORE = tuple(A.AI_CONFIDENCE_FEATURE_MAP.values())
_REGIME = ("spy_vs_sma20_pct", "spy_above_sma20")
FEATURE_SETS.update({
    "core_v2": FeatureSet("core_v2", _CORE, (), False,
                          "6 core inputs, empty ones filled from the pick's prediction record (schema v2)", 2),
    "core_rsi_v2": FeatureSet("core_rsi_v2", _CORE + ("rsi_14",), (), False, "core_v2 + RSI-14", 2),
    "core_regime_v2": FeatureSet("core_regime_v2", _CORE + _REGIME, (), False,
                                 "core_v2 + SPY vs 20-day average (% and above/below)", 2),
    "core_rsi_regime_v2": FeatureSet("core_rsi_regime_v2", _CORE + ("rsi_14",) + _REGIME, (), False,
                                     "core_v2 + RSI-14 + SPY regime", 2),
})
ABLATION_ARMS_V2 = ("core_v2", "core_rsi_v2", "core_regime_v2", "core_rsi_regime_v2")
DEFAULT_FEATURE_SETS = ("production", "leakage_safe")


def feature_schema_version(fs: FeatureSet) -> str:
    """Names the exact inputs: research schema version + this engine's feature set."""
    digest = hashlib.sha256(json.dumps([fs.numeric, fs.categorical, fs.signals]).encode()).hexdigest()[:8]
    return f"research-v{max(rs.FEATURE_SCHEMA_VERSION, fs.research_schema)}/{ENGINE_VERSION}/{fs.name}-{digest}"


# ----------------------------------------------------------------------------- providers
# Fixed, pre-registered hyperparameters (no search on this data size). Shallow
# trees and the same learning rate / tree count across providers keep the
# comparison about the library, not about tuning effort. Single thread +
# fixed seed = bit-for-bit reproducible.
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "xgboost": {**A.XGB_PRODUCTION_PARAMS, "random_state": SEED, "n_jobs": 1},
    "lightgbm": dict(n_estimators=400, learning_rate=0.05, max_depth=5, num_leaves=15, min_child_samples=20,
                     subsample=0.9, subsample_freq=1, colsample_bytree=0.9, objective="binary",
                     random_state=SEED, n_jobs=1, deterministic=True, force_col_wise=True, verbose=-1),
    "catboost": dict(iterations=400, learning_rate=0.05, depth=5, loss_function="Logloss",
                     random_seed=SEED, thread_count=1, verbose=False, allow_writing_files=False),
}


def _library(provider: str):
    if provider == "xgboost":
        import xgboost as lib
    elif provider == "lightgbm":
        import lightgbm as lib
    elif provider == "catboost":
        import catboost as lib
    else:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    return lib


def available_providers() -> List[str]:
    out = []
    for p in PROVIDERS:
        try:
            _library(p)
        except ImportError:
            continue
        out.append(p)
    return out


def library_versions() -> Dict[str, Optional[str]]:
    import sklearn

    out: Dict[str, Optional[str]] = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}
    for p in PROVIDERS:
        try:
            out[p] = _library(p).__version__
        except ImportError:
            out[p] = None
    return out


class ProviderModel(A.Model):
    """One gradient-boosted classifier behind the ML v3 ``Model`` interface.

    ``fit`` takes ML v3 ``AuditRow`` objects and 0/1 labels; ``predict`` returns
    the raw (uncalibrated) probability of the positive class. Calibration is a
    separate step (Milestone 3) and is never folded into these numbers."""

    kind = "probability"

    def __init__(self, provider: str, feature_set: str = "production", params: Optional[Mapping[str, Any]] = None):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}; expected one of {tuple(FEATURE_SETS)}")
        self.provider = provider
        self.feature_set = FEATURE_SETS[feature_set]
        self.name = f"{provider}_{feature_set}"
        self.params = dict(DEFAULT_PARAMS[provider] if params is None else params)
        self.hyperparameters = dict(self.params)
        self.enc: Optional[A.Encoder] = None
        self.m: Any = None

    # -- matrix
    def _encoder(self) -> A.Encoder:
        fs = self.feature_set
        return A.Encoder(fs.numeric, fs.categorical, fs.signals)

    def matrix(self, rows: Sequence[A.AuditRow]) -> np.ndarray:
        if self.enc is None:
            raise RuntimeError("model is not fitted")
        return self.enc.transform(rows, impute=False)

    @property
    def columns(self) -> List[str]:
        return list(self.enc.columns) if self.enc is not None else []

    # -- train / predict
    def fit(self, train: Sequence[A.AuditRow], y) -> "ProviderModel":
        y = np.asarray(y, dtype=int)
        if len(train) == 0:
            raise ValueError("no training rows")
        if len(train) != len(y):
            raise ValueError(f"{len(train)} rows but {len(y)} labels")
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("labels must be 0/1")
        if len(np.unique(y)) < 2:
            raise ValueError("training labels need both classes")
        if self.feature_set.research_schema >= 2 and not any("core_inputs_source" in r.features for r in train):
            raise ValueError(f"feature set {self.feature_set.name!r} needs a research feature schema v2 dataset")
        self.enc = self._encoder().fit(train)
        X = self.enc.transform(train, impute=False)
        lib = _library(self.provider)
        if self.provider == "xgboost":
            self.m = lib.XGBClassifier(**self.params).fit(X, y)
        elif self.provider == "lightgbm":
            self.m = lib.LGBMClassifier(**self.params).fit(X, y)
        else:
            self.m = lib.CatBoostClassifier(**self.params).fit(X, y)
        return self

    def predict(self, rows: Sequence[A.AuditRow]) -> np.ndarray:
        if self.m is None:
            raise RuntimeError("model is not fitted")
        if len(rows) == 0:
            return np.zeros(0)
        p = np.asarray(self.m.predict_proba(self.matrix(rows))[:, 1], dtype=float)
        return np.clip(p, 0.0, 1.0)

    # -- serialization (exact bytes; pickle of this object incl. its training-fold encoder)
    def to_bytes(self) -> bytes:
        import joblib

        if self.m is None:
            raise RuntimeError("model is not fitted")
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(blob: bytes, *, expected_sha256: Optional[str] = None) -> "ProviderModel":
        """Load only verified bytes: the checksum is checked before unpickling."""
        import joblib

        if expected_sha256 is not None and sha256(blob) != expected_sha256:
            raise ValueError("artifact checksum mismatch; refusing to load")
        obj = joblib.load(io.BytesIO(blob))
        if not isinstance(obj, ProviderModel):
            raise TypeError("artifact is not an ML v4 provider model")
        return obj


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------- training data
def training_split(rows: Sequence[A.AuditRow], target: Target = PRIMARY_TARGET, *, embargo: int = 1,
                   holdout_kw: Optional[Mapping[str, Any]] = None,
                   fold_kw: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Eligible labelled rows split into (training, reserved holdout).

    The holdout is the ML v3 ``final_holdout`` (most recent block, only when big
    enough). Training keeps rows that enter before the holdout AND whose outcome
    window closes before it, minus an embargo (``purge_train``), so no training
    label reads a holdout-period bar. With no valid holdout every eligible row
    trains and the result says so."""
    data = A.eligible(rows, target.horizon, target.label)
    hk = dict(min_rows=100, min_days=5, min_remaining_folds=3) if holdout_kw is None else dict(holdout_kw)
    fk = dict(min_train=60, min_val=30, min_val_class=5, embargo=embargo) if fold_kw is None else dict(fold_kw)
    hold = A.final_holdout(data, target.horizon, **hk, **fk) if data else {"valid": False, "reason": "no rows"}
    label_of = {r.observation_id: y for r, y in data}
    if hold.get("valid"):
        start = _dt.date.fromisoformat(hold["start"])
        kept, purged, embargoed = A.purge_train([r for r, _ in data], start, target.horizon, embargo)
        train = [(r, label_of[r.observation_id]) for r in kept]
        holdout_ids = list(hold["ids"])
    else:
        start, purged, embargoed = None, 0, 0
        train = list(data)
        holdout_ids = []
    ys = [y for _, y in train]
    return {
        "target": target, "eligible_rows": len(data), "train": train,
        "holdout": {"valid": bool(hold.get("valid")), "start": start.isoformat() if start else None,
                    "rows": len(holdout_ids), "ids": holdout_ids, "reason": hold.get("reason")},
        "purged": purged, "embargoed": embargoed, "embargo_days": embargo,
        "train_rows": len(train), "train_positive": int(sum(ys)),
        "train_start": min(r.entry_day for r, _ in train).isoformat() if train else None,
        "train_end": max(r.entry_day for r, _ in train).isoformat() if train else None,
    }


# ----------------------------------------------------------------------------- train + record
def _config_hash(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(A.to_jsonable(cfg), sort_keys=True).encode()).hexdigest()


def train_model(train: Sequence[Tuple[A.AuditRow, int]], provider: str, feature_set: str, *,
                target: Target = PRIMARY_TARGET, dataset: Optional[Mapping[str, Any]] = None,
                params: Optional[Mapping[str, Any]] = None, git_sha: Optional[str] = None) -> Dict[str, Any]:
    """Fit one provider on ``train`` and return (model, full training record)."""
    rows = [r for r, _ in train]
    y = np.array([v for _, v in train], dtype=int)
    model = ProviderModel(provider, feature_set, params)
    t0 = time.perf_counter()
    model.fit(rows, y)
    duration = time.perf_counter() - t0
    fs = model.feature_set
    dataset = dict(dataset or {})
    config = {
        "engine_version": ENGINE_VERSION, "provider": provider, "params": model.params, "seed": SEED,
        "feature_set": fs.name, "feature_schema_version": feature_schema_version(fs),
        "research_feature_schema_version": rs.FEATURE_SCHEMA_VERSION,
        "label_schema_version": rs.LABEL_SCHEMA_VERSION,
        "target": {"label": target.label, "horizon_days": target.horizon, "description": target.description},
        "dataset_version": dataset.get("dataset_version"),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "training_slice_fingerprint": A.slice_fingerprint(rows),
        "train_ids_sha256": hashlib.sha256(json.dumps(sorted(r.observation_id for r in rows)).encode()).hexdigest(),
        "unit": dataset.get("unit", "signal_day"),
    }
    chash = _config_hash(config)
    version = f"ml-v4-{provider}-{fs.name}-{target.label.split('_')[0].lower()}{target.horizon}d-{chash[:12]}"
    record = {
        **config,
        "model_version": version, "config_sha256": chash,
        "columns": model.columns, "missing_values": "native (NaN passed through; no fill)",
        "categoricals": "one-hot, vocabulary fitted on training rows (min 3 occurrences)",
        "train_rows": int(len(rows)), "train_positive": int(y.sum()),
        "train_start": min(r.entry_day for r in rows).isoformat(),
        "train_end": max(r.entry_day for r in rows).isoformat(),
        "training_seconds": round(duration, 3),
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "library_versions": library_versions(), "git_sha": git_sha,
        "probability": "raw model output, uncalibrated; not a probability of profit",
    }
    return {"model": model, "record": record}


def train_all(train: Sequence[Tuple[A.AuditRow, int]], *, providers: Sequence[str] = PROVIDERS,
              feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS, target: Target = PRIMARY_TARGET,
              dataset: Optional[Mapping[str, Any]] = None, git_sha: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every provider x feature set on the identical training rows. A provider
    that fails is reported with its error; the others still train."""
    out = []
    for fs in feature_sets:
        for p in providers:
            try:
                out.append(train_model(train, p, fs, target=target, dataset=dataset, git_sha=git_sha))
            except Exception as exc:  # report, don't hide: one library failing must not stop the others
                out.append({"model": None, "record": {"provider": p, "feature_set": fs,
                                                      "error": f"{type(exc).__name__}: {exc}"}})
    return out


# ----------------------------------------------------------------------------- artifacts
def save_artifact(model: ProviderModel, record: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Write ``<model_version>.joblib`` + ``.json`` under ``out_dir``. Never
    overwrites: an existing file with the same name is an error (versions are
    content-addressed by config, so a clash means a duplicate run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = str(record["model_version"])
    blob = model.to_bytes()
    art, meta = out_dir / f"{version}.joblib", out_dir / f"{version}.json"
    for p in (art, meta):
        if p.exists():
            raise FileExistsError(f"{p} already exists; ML v4 artifacts are never overwritten")
    digest = sha256(blob)
    with open(art, "xb") as fh:  # "x": fail if the file appeared meanwhile
        fh.write(blob)
    manifest = {**A.to_jsonable(dict(record)), "artifact_format": ARTIFACT_FORMAT, "artifact_file": art.name,
                "artifact_sha256": digest, "artifact_bytes": len(blob)}
    with open(meta, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def load_artifact(path: Path, expected_sha256: str) -> ProviderModel:
    """Load an artifact file after verifying its checksum (required)."""
    if not expected_sha256:
        raise ValueError("expected_sha256 is required to load an ML v4 artifact")
    return ProviderModel.from_bytes(Path(path).read_bytes(), expected_sha256=expected_sha256)
