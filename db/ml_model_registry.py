"""ML v4 model registry: metadata, lifecycle and promotion history in Neon.

Two additive tables, beside (never replacing) ``ai_confidence_models`` and
``prebreakout_models``, which production scoring keeps using unchanged:

  ml_models        one row per trained model or ensemble version: provider,
                   target + horizon, feature/label schema versions, dataset
                   version + fingerprint, training config, metrics,
                   calibration, artifact URI + sha256 + size, status.
                   Model FILES are not stored here (GitHub Release assets);
                   only their location and checksum.
  ml_model_events  every registration and status change: from, to, admin,
                   reason, evidence, time. Append-only; this is the promotion
                   history and what rollback reads.

Rules live in ``analytics/ml_v4_lifecycle.py``. At most one champion per target
+ horizon, enforced by a partial unique index as well as in code. Writes create
the tables if missing (CREATE ... IF NOT EXISTS); reads never create anything,
so read-only callers (the API) stay read-only. Size: tens of rows per training
run, a handful of events per model, so no retention policy is needed here.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from analytics import ml_v4_lifecycle as L
from db.engine import get_neon_conn, schema_once

_conn_factory: Optional[Callable[[], Any]] = None

MODEL_COLUMNS = ("id", "model_version", "provider", "kind", "feature_set", "feature_schema_version",
                 "label_schema_version", "target_label", "horizon_days", "dataset_version", "dataset_fingerprint",
                 "training_config", "metrics", "calibration", "members", "artifact_uri", "artifact_sha256",
                 "artifact_bytes", "git_sha", "trained_at", "status", "status_changed_at", "created_at")


class RegistryUnavailable(RuntimeError):
    pass


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Tests point the registry at a throwaway database; None restores Neon."""
    global _conn_factory
    _conn_factory = factory


def _connect():
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    return conn


def _rows(cur) -> List[Dict[str, Any]]:
    rows = cur.fetchall() or []
    cols = [d[0] for d in cur.description] if cur.description else []
    out = []
    for r in rows:
        d = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
        for k in ("training_config", "metrics", "calibration", "members", "evidence"):
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        out.append(d)
    return out


@schema_once
def ensure_schema(conn) -> None:
    cur = conn.cursor()
    statuses = ", ".join(f"'{s}'" for s in L.STATUSES)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS ml_models (
            id BIGSERIAL PRIMARY KEY,
            model_version TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'model',
            feature_set TEXT,
            feature_schema_version TEXT,
            label_schema_version INTEGER,
            target_label TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            dataset_version TEXT,
            dataset_fingerprint TEXT,
            training_config JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            calibration JSONB,
            members JSONB,
            artifact_uri TEXT,
            artifact_sha256 TEXT,
            artifact_bytes BIGINT,
            git_sha TEXT,
            trained_at TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ({statuses})),
            status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""")  # nosec B608 - fixed status list
    # The registry's queries: models for one target by status (admin list, shadow scorer).
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_models_target_status "
                "ON ml_models (target_label, horizon_days, status)")
    # One champion per target + horizon, enforced by the database too.
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_models_one_champion "
                "ON ml_models (target_label, horizon_days) WHERE status = 'champion'")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_model_events (
            id BIGSERIAL PRIMARY KEY,
            model_id BIGINT NOT NULL REFERENCES ml_models (id),
            from_status TEXT,
            to_status TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            evidence JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""")
    # History of one model (detail view) and rollback's scan of former champions.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_model_events_model_created "
                "ON ml_model_events (model_id, created_at)")
    conn.commit()
    cur.close()


@contextmanager
def _tx() -> Iterator[Any]:
    conn = _connect()
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        yield cur
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _exists(cur) -> bool:
    cur.execute("SELECT to_regclass('ml_models') IS NOT NULL")
    r = cur.fetchone()
    return bool(list(r.values())[0] if isinstance(r, dict) else r[0])


def _event(cur, model_id: int, from_status: Optional[str], to_status: str, actor: str, reason: Optional[str],
           evidence: Optional[Mapping[str, Any]] = None) -> None:
    cur.execute("INSERT INTO ml_model_events (model_id, from_status, to_status, actor, reason, evidence) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                (model_id, from_status, to_status, actor, reason,
                 json.dumps(evidence, default=str) if evidence is not None else None))


def _j(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, default=str)


# ----------------------------------------------------------------------------- writes
def register_model(manifest: Mapping[str, Any], *, actor: str, artifact_uri: Optional[str] = None,
                   metrics: Optional[Mapping[str, Any]] = None, calibration: Optional[Mapping[str, Any]] = None,
                   members: Optional[Any] = None, kind: str = "model") -> Dict[str, Any]:
    """Insert a candidate from a training manifest (``ml_v4_providers.save_artifact``).
    Idempotent: an existing model_version is returned unchanged, never overwritten."""
    if not (actor and str(actor).strip()):
        raise L.LifecycleError("registration needs an actor")
    target = manifest.get("target") or {}
    version = str(manifest["model_version"])
    with _tx() as cur:
        cur.execute(
            "INSERT INTO ml_models (model_version, provider, kind, feature_set, feature_schema_version, "
            "label_schema_version, target_label, horizon_days, dataset_version, dataset_fingerprint, "
            "training_config, metrics, calibration, members, artifact_uri, artifact_sha256, artifact_bytes, "
            "git_sha, trained_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, "
            "%s::jsonb, %s::jsonb, %s, %s, %s, %s, %s) ON CONFLICT (model_version) DO NOTHING RETURNING id",
            (version, manifest.get("provider") or kind, kind, manifest.get("feature_set"),
             manifest.get("feature_schema_version"), manifest.get("label_schema_version"),
             target.get("label"), target.get("horizon_days"), manifest.get("dataset_version"),
             manifest.get("dataset_fingerprint"), _j({k: manifest.get(k) for k in (
                 "engine_version", "params", "seed", "columns", "train_rows", "train_positive", "train_start",
                 "train_end", "training_seconds", "library_versions", "training_slice_fingerprint",
                 "train_ids_sha256", "config_sha256", "unit", "evaluation_binding") if k in manifest}),
             _j(dict(metrics or {})), _j(calibration), _j(members), artifact_uri,
             manifest.get("artifact_sha256"), manifest.get("artifact_bytes"), manifest.get("git_sha"),
             manifest.get("trained_at")))
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT id, status FROM ml_models WHERE model_version = %s", (version,))
            existing = _rows(cur)[0]
            return {"id": int(existing["id"]), "created": False, "status": existing["status"]}
        model_id = int(row["id"] if isinstance(row, dict) else row[0])
        _event(cur, model_id, None, "candidate", actor, "registered after training",
               {"artifact_uri": artifact_uri, "artifact_sha256": manifest.get("artifact_sha256")})
        return {"id": model_id, "created": True, "status": "candidate"}


def attach_evaluation(model_id: int, *, metrics: Mapping[str, Any], calibration: Optional[Mapping[str, Any]],
                      actor: str) -> None:
    """Record evaluation results on a candidate/validated model (history keeps the event)."""
    with _tx() as cur:
        cur.execute("SELECT status FROM ml_models WHERE id = %s FOR UPDATE", (int(model_id),))
        rows = _rows(cur)
        if not rows:
            raise L.LifecycleError(f"no model {model_id}")
        status = rows[0]["status"]
        if status not in ("candidate", "validated"):
            raise L.LifecycleError(f"evaluation can't be replaced on a {status} model")
        cur.execute("UPDATE ml_models SET metrics = %s::jsonb, calibration = %s::jsonb WHERE id = %s",
                    (_j(dict(metrics)), _j(calibration), int(model_id)))
        _event(cur, int(model_id), status, status, actor, "evaluation attached",
               {"evaluation_version": (metrics or {}).get("evaluation_version")})


def _was_champion(cur, model_id: int) -> bool:
    cur.execute("SELECT 1 FROM ml_model_events WHERE model_id = %s AND to_status = 'champion' LIMIT 1", (model_id,))
    return cur.fetchone() is not None


def transition(model_id: int, to_status: str, *, actor: str, reason: str,
               evidence: Optional[Mapping[str, Any]] = None, approved: bool = False) -> Dict[str, Any]:
    """Move one model through the lifecycle. Promoting a champion retires the
    current champion of the same target + horizon in the same transaction."""
    with _tx() as cur:
        cur.execute("SELECT id, status, target_label, horizon_days FROM ml_models WHERE id = %s FOR UPDATE",
                    (int(model_id),))
        rows = _rows(cur)
        if not rows:
            raise L.LifecycleError(f"no model {model_id}")
        m = rows[0]
        L.check_transition(m["status"], to_status, actor=actor, reason=reason, evidence=evidence, approved=approved,
                           was_champion_before=_was_champion(cur, int(model_id)))
        superseded = None
        if to_status == "champion":
            cur.execute("SELECT id FROM ml_models WHERE target_label = %s AND horizon_days = %s "
                        "AND status = 'champion' AND id <> %s FOR UPDATE",
                        (m["target_label"], m["horizon_days"], int(model_id)))
            cur_champ = _rows(cur)
            if cur_champ:
                superseded = int(cur_champ[0]["id"])
                cur.execute("UPDATE ml_models SET status = 'retired', status_changed_at = NOW() WHERE id = %s",
                            (superseded,))
                _event(cur, superseded, "champion", "retired", actor, f"superseded by model {model_id}: {reason}")
        cur.execute("UPDATE ml_models SET status = %s, status_changed_at = NOW() WHERE id = %s",
                    (to_status, int(model_id)))
        _event(cur, int(model_id), m["status"], to_status, actor, reason, evidence)
        return {"id": int(model_id), "from": m["status"], "to": to_status, "superseded": superseded}


def rollback(target_label: str, horizon_days: int, *, actor: str, reason: str) -> Dict[str, Any]:
    """Retire the current champion and restore the most recent former champion."""
    with _tx() as cur:
        cur.execute("SELECT id FROM ml_models WHERE target_label = %s AND horizon_days = %s AND status = 'champion' "
                    "FOR UPDATE", (target_label, int(horizon_days)))
        cur_rows = _rows(cur)
        current = int(cur_rows[0]["id"]) if cur_rows else None
        cur.execute("SELECT e.model_id, e.to_status, e.created_at, m.status FROM ml_model_events e "
                    "JOIN ml_models m ON m.id = e.model_id WHERE m.target_label = %s AND m.horizon_days = %s "
                    "AND e.to_status = 'champion'", (target_label, int(horizon_days)))
        previous = L.rollback_target(_rows(cur), current)
        if previous is None:
            raise L.LifecycleError("no former champion to roll back to")
        L.check_transition("retired", "champion", actor=actor, reason=reason, approved=True, was_champion_before=True)
        if current is not None:
            L.check_transition("champion", "retired", actor=actor, reason=reason)
            cur.execute("UPDATE ml_models SET status = 'retired', status_changed_at = NOW() WHERE id = %s", (current,))
            _event(cur, current, "champion", "retired", actor, f"rollback: {reason}")
        cur.execute("UPDATE ml_models SET status = 'champion', status_changed_at = NOW() WHERE id = %s", (previous,))
        _event(cur, previous, "retired", "champion", actor, f"rollback: {reason}", {"rolled_back_from": current})
        return {"restored": previous, "retired": current}


# ----------------------------------------------------------------------------- reads (never create tables)
def _read(sql: str, params: tuple) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        if not _exists(cur):
            cur.close()
            return []
        cur.execute(sql, params)
        out = _rows(cur)
        cur.close()
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_models(*, status: Optional[str] = None, target_label: Optional[str] = None,
                horizon_days: Optional[int] = None) -> List[Dict[str, Any]]:
    where, params = [], []
    if status:
        where.append("status = %s")
        params.append(status)
    if target_label:
        where.append("target_label = %s")
        params.append(target_label)
    if horizon_days is not None:
        where.append("horizon_days = %s")
        params.append(int(horizon_days))
    sql = f"SELECT {', '.join(MODEL_COLUMNS)} FROM ml_models"  # nosec B608 - fixed columns
    if where:
        sql += " WHERE " + " AND ".join(where)
    return _read(sql + " ORDER BY created_at DESC, id DESC", tuple(params))


def get_model(model_id: int) -> Optional[Dict[str, Any]]:
    rows = _read(f"SELECT {', '.join(MODEL_COLUMNS)} FROM ml_models WHERE id = %s", (int(model_id),))  # nosec B608
    return rows[0] if rows else None


def model_events(model_id: int) -> List[Dict[str, Any]]:
    return _read("SELECT id, model_id, from_status, to_status, actor, reason, evidence, created_at "
                 "FROM ml_model_events WHERE model_id = %s ORDER BY created_at, id", (int(model_id),))
