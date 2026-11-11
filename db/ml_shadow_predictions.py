"""ML v4 shadow predictions: what registry models WOULD have said, stored beside production.

One additive table, written only by the scheduled shadow workflow
(``scripts/ml_v4_shadow.py``); production scoring never reads it.

  ml_shadow_predictions  one row per (model, observation): raw + calibrated
                         probability, target + horizon, feature schema version,
                         when it was scored and how long scoring took per
                         prediction, and the label once the outcome
                         matured (filled later by the same workflow).

Size: about one row per shadow model per signal-day (~30-160/day), kept 180
days (``prune``). Reads never create the table, so the API stays read-only.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from db.engine import get_neon_conn, schema_once
from utils.swallowed import swallowed

RETENTION_DAYS = 180
_conn_factory: Optional[Callable[[], Any]] = None
COLUMNS = ("model_id", "model_version", "observation_id", "ticker", "entry_day", "observed_at", "target_label",
           "horizon_days", "raw_probability", "calibrated_probability", "feature_schema_version", "scored_at",
           "scoring_ms")


class ShadowUnavailable(RuntimeError):
    pass


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory


def _connect():
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise ShadowUnavailable("database unavailable")
    return conn


def _rows(cur) -> List[Dict[str, Any]]:
    rows = cur.fetchall() or []
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


@schema_once
def ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_shadow_predictions (
            id BIGSERIAL PRIMARY KEY,
            model_id BIGINT NOT NULL,
            model_version TEXT NOT NULL,
            observation_id BIGINT NOT NULL,
            ticker TEXT,
            entry_day DATE NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            target_label TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            raw_probability DOUBLE PRECISION,
            calibrated_probability DOUBLE PRECISION,
            feature_schema_version TEXT,
            scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            scoring_ms DOUBLE PRECISION,
            label SMALLINT,
            label_filled_at TIMESTAMPTZ,
            UNIQUE (model_id, observation_id)
        )""")
    # Summary reads: one model's recent predictions; label fill: unmatured rows per target.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_shadow_model_scored ON ml_shadow_predictions (model_id, scored_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_shadow_unlabeled ON ml_shadow_predictions "
                "(target_label, horizon_days, observation_id) WHERE label IS NULL")
    conn.commit()
    cur.close()


def insert_predictions(preds: Sequence[Mapping[str, Any]]) -> int:
    """Insert new predictions; an existing (model, observation) pair is left untouched."""
    if not preds:
        return 0
    conn = _connect()
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        n = 0
        sql = (f"INSERT INTO ml_shadow_predictions ({', '.join(COLUMNS)}) VALUES "  # nosec B608 - fixed columns
               f"({', '.join(['%s'] * len(COLUMNS))}) ON CONFLICT (model_id, observation_id) DO NOTHING")
        for p in preds:
            cur.execute(sql, tuple(p.get(c) for c in COLUMNS))
            n += max(cur.rowcount or 0, 0)
        conn.commit()
        cur.close()
        return n
    except Exception:
        conn.rollback()
        raise
    finally:
        _close(conn)


def unlabeled(target_label: str, horizon_days: int) -> List[int]:
    """Observation ids that still wait for their outcome (any model)."""
    return [int(r["observation_id"]) for r in _read(
        "SELECT DISTINCT observation_id FROM ml_shadow_predictions WHERE target_label = %s AND horizon_days = %s "
        "AND label IS NULL", (target_label, int(horizon_days)))]


def fill_labels(target_label: str, horizon_days: int, labels: Iterable[Tuple[int, int]]) -> int:
    """Set the matured label on every model's prediction for those observations (once)."""
    labels = list(labels)
    if not labels:
        return 0
    conn = _connect()
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        n = 0
        for oid, y in labels:
            cur.execute("UPDATE ml_shadow_predictions SET label = %s, label_filled_at = NOW() "
                        "WHERE observation_id = %s AND target_label = %s AND horizon_days = %s AND label IS NULL",
                        (int(y), int(oid), target_label, int(horizon_days)))
            n += max(cur.rowcount or 0, 0)
        conn.commit()
        cur.close()
        return n
    except Exception:
        conn.rollback()
        raise
    finally:
        _close(conn)


def prune(days: int = RETENTION_DAYS) -> int:
    conn = _connect()
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM ml_shadow_predictions WHERE scored_at < NOW() - make_interval(days => %s)",
                    (int(days),))
        n = max(cur.rowcount or 0, 0)
        conn.commit()
        cur.close()
        return n
    except Exception:
        conn.rollback()
        raise
    finally:
        _close(conn)


# ----------------------------------------------------------------------------- reads (never create tables)
def _read(sql: str, params: tuple) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('ml_shadow_predictions') IS NOT NULL")
        r = cur.fetchone()
        if not (list(r.values())[0] if isinstance(r, dict) else r[0]):
            cur.close()
            return []
        cur.execute(sql, params)
        out = _rows(cur)
        cur.close()
        return out
    finally:
        _close(conn)


def predictions(model_ids: Sequence[int], *, since: dt.datetime, with_served: bool = True) -> List[Dict[str, Any]]:
    """Predictions for these models scored since ``since``, with AI Confidence as served
    (frozen on the observation) when ``with_served`` and the column exists."""
    if not model_ids:
        return []
    base = ("SELECT p.model_id, p.model_version, p.observation_id, p.ticker, p.entry_day, p.observed_at, "
            "p.target_label, p.horizon_days, p.raw_probability, p.calibrated_probability, p.scored_at, p.label, "
            "p.scoring_ms")
    where = " WHERE p.model_id = ANY(%s) AND p.scored_at >= %s ORDER BY p.scored_at, p.id"
    params = ([int(m) for m in model_ids], since)
    if with_served:
        try:
            return _read(base + ", s.ai_confidence FROM ml_shadow_predictions p "
                         "LEFT JOIN signal_outcomes s ON s.id = p.observation_id" + where, params)
        except Exception as _ignored:  # signal_outcomes missing (tests, fresh DB): fall back without it
            swallowed("db.ml_shadow_predictions.predictions", _ignored)
    return _read(base + ", NULL AS ai_confidence FROM ml_shadow_predictions p" + where, params)
