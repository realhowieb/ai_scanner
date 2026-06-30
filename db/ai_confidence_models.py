"""Persistence helpers for trained AI Confidence models."""
from __future__ import annotations

import io
import json
from typing import Any

from db.engine import get_neon_conn


def ensure_ai_confidence_models_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_confidence_models (
            id BIGSERIAL PRIMARY KEY,
            model_version TEXT NOT NULL,
            model_bytes BYTEA NOT NULL,
            feature_names JSONB NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            trained_at TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_confidence_models_active_created "
        "ON ai_confidence_models (is_active, created_at DESC)"
    )
    conn.commit()
    cur.close()


def _row_get(row: Any, key: str, idx: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[idx]


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _coerce_feature_names(value: Any) -> list[str]:
    value = _coerce_json(value)
    return [str(item) for item in list(value or []) if str(item).strip()]


def serialize_model_to_bytes(model: Any, joblib_module: Any) -> bytes:
    buffer = io.BytesIO()
    joblib_module.dump(model, buffer)
    return buffer.getvalue()


def deserialize_model_from_bytes(model_bytes: bytes, joblib_module: Any) -> Any:
    return joblib_module.load(io.BytesIO(model_bytes))


def save_ai_confidence_model(
    *,
    model_bytes: bytes,
    feature_names: list[str],
    trained_at: str | None,
    model_version: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save a new active AI Confidence model version to Neon/Postgres."""
    conn = get_neon_conn()
    if conn is None:
        return False
    ensure_ai_confidence_models_schema(conn)
    cur = conn.cursor()
    cur.execute("UPDATE ai_confidence_models SET is_active = FALSE WHERE is_active = TRUE")
    cur.execute(
        """
        INSERT INTO ai_confidence_models (
            model_version, model_bytes, feature_names, metadata, trained_at, is_active
        )
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, TRUE)
        """,
        (
            model_version,
            model_bytes,
            json.dumps(feature_names),
            json.dumps(metadata or {}),
            trained_at,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def load_latest_ai_confidence_model_bundle(joblib_module: Any) -> dict[str, Any] | None:
    """Load the latest active AI Confidence model from Neon/Postgres."""
    conn = get_neon_conn()
    if conn is None:
        return None
    ensure_ai_confidence_models_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, model_version, model_bytes, feature_names, metadata, trained_at
        FROM ai_confidence_models
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

    metadata = _coerce_json(_row_get(row, "metadata", 4)) or {}
    feature_names = _coerce_feature_names(_row_get(row, "feature_names", 3))
    model = deserialize_model_from_bytes(_coerce_bytes(_row_get(row, "model_bytes", 2)), joblib_module)
    trained_at = _row_get(row, "trained_at", 5) or metadata.get("trained_at")
    if trained_at is not None:
        trained_at = str(trained_at)
    metadata = dict(metadata)
    metadata["feature_names"] = feature_names
    metadata["trained_at"] = trained_at
    metadata["model_version"] = _row_get(row, "model_version", 1)
    metadata["source"] = "database"
    return {
        "model": model,
        "metadata": metadata,
        "feature_names": feature_names,
        "trained_at": trained_at,
        "model_version": metadata["model_version"],
        "source": "database",
    }
