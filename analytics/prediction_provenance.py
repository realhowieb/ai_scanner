"""Prediction-time evidence; never reconstruct served identity from registry state."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

COLUMN = "ModelProvenance"


def clean(value):
    """Return strict JSON values without inventing missing numbers."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return None


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact_identity(blob, *, registry_id=None, version=None, artifact=None):
    return {"registry_id": registry_id, "version": version, "artifact": artifact,
            "sha256": hashlib.sha256(blob).hexdigest()}


def local_identity(path):
    """Hash once at cached resource load, not once per candidate."""
    try:
        return artifact_identity(Path(path).read_bytes(), artifact=Path(path).name)
    except OSError:
        return {"registry_id": None, "version": None, "artifact": Path(path).name,
                "sha256": None, "unavailable_reason": "artifact_bytes_unavailable"}


def load_local(joblib_module, path):
    """Deserialize and identify the same bytes, avoiding file replacement races."""
    from io import BytesIO
    try:
        blob = Path(path).read_bytes()
    except OSError:
        return joblib_module.load(path), local_identity(path)
    return joblib_module.load(BytesIO(blob)), artifact_identity(blob, artifact=Path(path).name)


def models_from_row(row):
    value = row.get(COLUMN)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return clean(value) if isinstance(value, dict) else {}


def unavailable(frame, role, reason):
    records = []
    for _, row in frame.iterrows():
        models = models_from_row(row)
        models[role] = {"schema_version": "prediction-provenance-1.0", "status": "unavailable",
                        "reason": reason, "raw_probability": None, "calibrated_probability": None}
        records.append(models)
    frame[COLUMN] = [json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) for record in records]
    return frame


def attach(frame, role, inputs, default_mask, raw, calibrated, metadata, *, availability=None):
    """Capture the actual final model matrix in row order, before any ranking."""
    if len(frame) != len(inputs):
        raise ValueError("Prediction provenance row alignment mismatch")
    names = list(inputs.columns)
    calibration = clean(metadata.get("calibration_map"))
    common = {
        "schema_version": "prediction-provenance-1.0",
        "status": "captured",
        "inferred_at": datetime.now(timezone.utc).isoformat(),
        "build_sha": os.environ.get("GITHUB_SHA") or os.environ.get("HSF_COMMIT_SHA"),
        "loaded_artifact": clean(metadata.get("loaded_artifact")),
        "feature_names": names, "feature_schema_hash": digest(names),
        "preprocessing": clean(metadata.get("preprocessing")),
        "preprocessing_hash": digest(metadata.get("preprocessing")),
        "calibration_snapshot": calibration,
        "calibration_hash": digest(calibration) if calibration is not None else None,
        "probability_units": "0-1",
        "target": clean(metadata.get("target") or metadata.get("target_rule")),
        "target_version": metadata.get("target_version"),
        "trained_at": clean(metadata.get("trained_at")),
        "training_data_end": clean(metadata.get("training_data_end")),
        "calibration_data_end": clean(metadata.get("calibration_data_end")),
        "enrichment_availability": clean(availability),
        "preprocessing_identity": "prebreakout_bundle_plan_then_fillna_zero" if role == "prebreakout" else "numeric_coercion_then_fillna_zero",
        "target_evidence": {"status": "unavailable", "reason": "original_target_not_matured; top_n_absence_is_not_negative"},
    }
    records = []
    for i, (_, row) in enumerate(frame.iterrows()):
        models = models_from_row(row)
        models[role] = {**common, "source_scan_id": clean(row.get("SourceScanId")),
                        "input_timestamp": clean(row.get("Timestamp")),
                        "input_timestamp_status": "supplied" if clean(row.get("Timestamp")) is not None else "unavailable",
                        "feature_values": clean(inputs.iloc[i].tolist()),
                        "default_mask": clean(default_mask.iloc[i].tolist()),
                        "raw_probability": clean(float(raw[i])),
                        "calibrated_probability": clean(float(calibrated[i]))}
        records.append(models)
    # pandas.to_json rounds nested floats; a string preserves exact inputs and
    # calibration snapshots until models_from_row decodes at storage boundaries.
    frame[COLUMN] = [json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) for record in records]
    return frame


def customer_frame(frame):
    """Remove internal prediction evidence from the presentation copy only."""
    return frame.drop(columns=[COLUMN, "SourceScanId"], errors="ignore")
