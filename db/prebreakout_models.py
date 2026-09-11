"""Persistence helpers for trained pre-breakout models."""
from __future__ import annotations

import io
import json
from typing import Any

from db.engine import get_neon_conn


def ensure_prebreakout_models_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prebreakout_models (
            id BIGSERIAL PRIMARY KEY,
            model_version TEXT NOT NULL,
            model_bytes BYTEA NOT NULL,
            feature_names JSONB NOT NULL,
            metadata JSONB,
            auc DOUBLE PRECISION,
            trained_at TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prebreakout_models_active_created "
        "ON prebreakout_models (is_active, created_at DESC)"
    )
    cur.execute("ALTER TABLE prebreakout_models ADD COLUMN IF NOT EXISTS metadata JSONB")
    conn.commit()
    cur.close()


def _row_get(row: Any, key: str, idx: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[idx]


def _coerce_model_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def _coerce_feature_names(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in list(value or []) if str(item).strip()]


def serialize_model_to_bytes(model: Any, joblib_module: Any) -> bytes:
    buffer = io.BytesIO()
    joblib_module.dump(model, buffer)
    return buffer.getvalue()


def deserialize_model_from_bytes(model_bytes: bytes, joblib_module: Any) -> Any:
    return joblib_module.load(io.BytesIO(model_bytes))


def save_prebreakout_model(
    *,
    model_bytes: bytes,
    feature_names: list[str],
    auc: float,
    trained_at: str,
    model_version: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save a new active model version to Neon/Postgres."""
    conn = get_neon_conn()
    if conn is None:
        return False
    ensure_prebreakout_models_schema(conn)
    cur = conn.cursor()
    cur.execute("UPDATE prebreakout_models SET is_active = FALSE WHERE is_active = TRUE")
    cur.execute(
        """
        INSERT INTO prebreakout_models (
            model_version, model_bytes, feature_names, metadata, auc, trained_at, is_active
        )
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, TRUE)
        """,
        (
            model_version,
            model_bytes,
            json.dumps(feature_names),
            json.dumps(metadata or {}),
            float(auc),
            trained_at,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def restore_previous_model_if_active_run16_incomplete(*, min_valid_folds: int = 5) -> dict[str, Any]:
    """Reactivate the prior model if an incomplete-fold Run #16 model is active."""
    conn = get_neon_conn()
    if conn is None:
        return {"restored": False, "reason": "database unavailable"}
    ensure_prebreakout_models_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, model_version, metadata, auc, trained_at
        FROM prebreakout_models
        WHERE is_active = TRUE
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """
    )
    active = cur.fetchone()
    if not active:
        cur.close()
        conn.close()
        return {"restored": False, "reason": "no active prebreakout model"}
    active_id = _row_get(active, "id", 0)
    metadata = _row_get(active, "metadata", 2) or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    audit = metadata.get("run16_dataset_audit") or metadata.get("run6_reproduction_audit") or {}
    valid_fold_count = int(audit.get("valid_fold_count") or 0)
    is_incomplete_run16 = (
        metadata.get("run_number") == 16
        and metadata.get("run16_result") == "PROMOTED"
        and valid_fold_count < int(min_valid_folds)
    )
    if not is_incomplete_run16:
        cur.close()
        conn.close()
        return {
            "restored": False,
            "reason": "active model does not match incomplete Run #16 rollback criteria",
            "active_id": active_id,
            "valid_fold_count": valid_fold_count,
        }
    cur.execute(
        """
        SELECT id, model_version, auc, trained_at
        FROM prebreakout_models
        WHERE id <> %s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (active_id,),
    )
    previous = cur.fetchone()
    if not previous:
        cur.close()
        conn.close()
        return {"restored": False, "reason": "no previous model row available", "active_id": active_id}
    previous_id = _row_get(previous, "id", 0)
    cur.execute("UPDATE prebreakout_models SET is_active = FALSE WHERE is_active = TRUE")
    cur.execute("UPDATE prebreakout_models SET is_active = TRUE WHERE id = %s", (previous_id,))
    conn.commit()
    restored = {
        "restored": True,
        "deactivated_id": active_id,
        "restored_id": previous_id,
        "restored_model_version": _row_get(previous, "model_version", 1),
        "restored_auc": _row_get(previous, "auc", 2),
        "restored_trained_at": str(_row_get(previous, "trained_at", 3)),
    }
    cur.close()
    conn.close()
    return restored


def update_active_prebreakout_model_metadata(patch: dict[str, Any]) -> bool:
    """Merge ``patch`` into the active model's metadata JSONB, in place.

    Used to attach a calibration map to the live champion without retraining or
    swapping the model row (no change to model_bytes or which row is_active).
    Returns False when there is no active row or the database is unavailable.
    """
    if not isinstance(patch, dict) or not patch:
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    ensure_prebreakout_models_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, metadata FROM prebreakout_models
        WHERE is_active = TRUE
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return False
    active_id = _row_get(row, "id", 0)
    metadata = _row_get(row, "metadata", 1) or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(patch)
    cur.execute(
        "UPDATE prebreakout_models SET metadata = %s::jsonb WHERE id = %s",
        (json.dumps(metadata), active_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def load_latest_prebreakout_model_bundle(joblib_module: Any) -> dict[str, Any] | None:
    """Load the latest active model bundle from Neon/Postgres."""
    conn = get_neon_conn()
    if conn is None:
        return None
    ensure_prebreakout_models_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, model_version, model_bytes, feature_names, metadata, auc, trained_at
        FROM prebreakout_models
        WHERE is_active = TRUE
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None

    feature_names = _coerce_feature_names(_row_get(row, "feature_names", 3))
    model = deserialize_model_from_bytes(_coerce_model_bytes(_row_get(row, "model_bytes", 2)), joblib_module)
    metadata = _row_get(row, "metadata", 4) or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    trained_at = _row_get(row, "trained_at", 6)
    if trained_at is not None:
        trained_at = str(trained_at)
    bundle = {
        "model": model,
        "features": feature_names,
        "feature_names": feature_names,
        "auc": _row_get(row, "auc", 5),
        "trained_at": trained_at,
        "model_version": _row_get(row, "model_version", 1),
        "source": "database",
    }
    if isinstance(metadata, dict):
        bundle.update(metadata)
        bundle["model"] = model
        bundle["features"] = feature_names
        bundle["feature_names"] = feature_names
        bundle["source"] = "database"
    return bundle
