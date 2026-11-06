"""Read-only, slim operational evidence for the admin model summary."""
from db.engine import get_neon_conn
from db.ml_model_registry import RegistryUnavailable


def _values(row):
    return list(row.values()) if isinstance(row, dict) else row


def production_confidence():
    try:
        conn = get_neon_conn()
    except Exception as exc:
        raise RegistryUnavailable("database unavailable") from exc
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('ai_confidence_models') AS models, to_regclass('signal_outcomes') AS signals")
        tables = _values(cur.fetchone())
        configured, captured = None, None
        if tables[0]:
            cur.execute("SELECT id, model_version, trained_at, metadata->>'dataset_version' AS dataset_version "
                        "FROM ai_confidence_models WHERE is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                configured = dict(zip(("registry_id", "model_version", "trained_at", "dataset_version"), row))
        if tables[1]:
            cur.execute("SELECT fired_at, raw_signal->'models'->'ai_confidence'->'loaded_artifact'->>'sha256' AS artifact_sha256, "
                        "raw_signal->'models'->'ai_confidence'->>'inferred_at' AS inferred_at, "
                        "raw_signal->'models'->'ai_confidence'->'loaded_artifact'->>'version' AS inferred_model_version, "
                        "raw_signal->>'ai_confidence_model_version' AS legacy_version FROM signal_outcomes "
                        "WHERE source='opportunity' AND ai_confidence IS NOT NULL ORDER BY fired_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                captured = {"signal_at": row[0], "artifact_sha256": row[1], "inferred_at": row[2],
                            "model_version": row[3] or row[4]}
        return {"configured_model": configured, "latest_capture": captured,
                "note": "Configured database model and latest recorded inference are separate evidence. "
                        "The scorer can fall back to a local artifact; missing evidence is unknown."}
    except Exception as exc:
        raise RegistryUnavailable("production confidence evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()


def production_prebreakout():
    """Configured model metadata and recorded inference identity, without loading artifacts."""
    try:
        conn = get_neon_conn()
    except Exception as exc:
        raise RegistryUnavailable("database unavailable") from exc
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('prebreakout_models') AS models, to_regclass('signal_outcomes') AS signals")
        tables = _values(cur.fetchone())
        configured, captured = None, None
        if tables[0]:
            cur.execute("SELECT id, model_version, trained_at, metadata->>'dataset_version' AS dataset_version "
                        "FROM prebreakout_models WHERE is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                configured = dict(zip(("registry_id", "model_version", "trained_at", "dataset_version"), row))
        if tables[1]:
            cur.execute("SELECT fired_at, raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'sha256' AS artifact_sha256, "
                        "raw_signal->'models'->'prebreakout'->>'inferred_at' AS inferred_at, "
                        "raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'version' AS inferred_model_version, "
                        "raw_signal->>'prebreakout_model_version' AS legacy_version FROM signal_outcomes "
                        "WHERE source='opportunity' AND prebreakout_prob IS NOT NULL "
                        "AND (raw_signal->'models'->'prebreakout'->>'status' = 'captured' OR "
                        "(raw_signal->'models'->'prebreakout'->>'status' IS NULL "
                        "AND raw_signal->>'prebreakout_model_version' IS NOT NULL AND prebreakout_prob > 0)) "
                        "ORDER BY fired_at DESC, id DESC LIMIT 1")
            row = _values(cur.fetchone())
            if row:
                captured = {"signal_at": row[0], "artifact_sha256": row[1], "inferred_at": row[2],
                            "model_version": row[3] or row[4]}
        return {"configured_model": configured, "latest_capture": captured,
                "note": "Configured database model and latest recorded inference are separate evidence. "
                        "The scorer can fall back to a local artifact; missing evidence is unknown."}
    except Exception as exc:
        raise RegistryUnavailable("production PreBreakout evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()
