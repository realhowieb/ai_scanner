"""Run 36 — opt-in persistence for canonical HSF observations + outcomes.

The smallest production-safe foundation for a durable market-intelligence
dataset. Reuses the existing dual-backend engine (Neon Postgres in production,
SQLite locally/CI). Observations are IMMUTABLE and idempotent by
``observation_id``; outcomes are a separate table keyed by
(observation_id, horizon), first-write-wins — so a matured label never rewrites
the observation's features/predictions.

Not wired into any live scan path: nothing here changes production behavior. A
future run (Run 37) connects the scheduled pipeline to ``save_observation``.
Every function is non-fatal and returns a safe default when the DB is
unavailable, matching db.opportunity_snapshots' contract.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn, get_sqlite_conn


def _resolve_conn(conn):
    """Return (conn, opened_here, is_sqlite). Prefer a caller-supplied conn (tests),
    else Neon, else local SQLite."""
    if conn is not None:
        return conn, False, isinstance(conn, sqlite3.Connection)
    neon = get_neon_conn()
    if neon is not None:
        return neon, True, False
    try:
        return get_sqlite_conn(), True, True
    except Exception:
        return None, False, True


def _ph(is_sqlite: bool) -> str:
    return "?" if is_sqlite else "%s"


def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    if is_sqlite:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hsf_observations (
                observation_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                record TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hsf_observation_outcomes (
                observation_id TEXT NOT NULL,
                horizon TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                record TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (observation_id, horizon)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_obs_symbol_ts "
                    "ON hsf_observations (symbol, timestamp)")
    else:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hsf_observations (
                observation_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                context TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                record JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hsf_observation_outcomes (
                observation_id TEXT NOT NULL,
                horizon TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                record JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (observation_id, horizon)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_obs_symbol_ts "
                    "ON hsf_observations (symbol, timestamp)")
    conn.commit()
    cur.close()


def save_observation(observation: Dict[str, Any], *, conn=None) -> bool:
    """Idempotently persist one canonical observation. First write wins
    (immutable); a duplicate observation_id is a no-op. Returns True only when a
    NEW row was written. Non-fatal."""
    oid = observation.get("observation_id")
    if not oid:
        return False
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return False
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cast = "" if is_sqlite else "::jsonb"
        cur.execute(
            f"INSERT INTO hsf_observations "
            f"(observation_id, symbol, timestamp, context, schema_version, record) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}{cast}) "
            f"ON CONFLICT (observation_id) DO NOTHING",
            (str(oid), str(observation.get("symbol") or "").upper(),
             str(observation.get("timestamp")), str(observation.get("context") or "default"),
             str(observation.get("schema_version") or ""), json.dumps(observation)),
        )
        wrote = cur.rowcount == 1
        c.commit()
        cur.close()
        return wrote
    except Exception:
        return False
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def save_observations_batch(observations, *, conn=None) -> Dict[str, int]:
    """Persist many observations over ONE connection/transaction (Run 38A).

    First-write-wins per observation_id (idempotent). Returns
    {"attempted","written","duplicates","failed"}. Non-fatal: a per-row error is
    counted, never raised, and never aborts the scan. Batching one connection +
    one commit keeps overhead bounded vs a connection-per-symbol.
    """
    obs = [o for o in (observations or []) if o and o.get("observation_id")]
    stats = {"attempted": len(observations or []), "written": 0,
             "duplicates": 0, "failed": 0}
    if not obs:
        return stats
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        stats["failed"] = stats["attempted"]
        return stats
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cast = "" if is_sqlite else "::jsonb"
        sql = (f"INSERT INTO hsf_observations "
               f"(observation_id, symbol, timestamp, context, schema_version, record) "
               f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}{cast}) "
               f"ON CONFLICT (observation_id) DO NOTHING")
        for o in obs:
            try:
                cur.execute(sql, (
                    str(o.get("observation_id")),
                    str(o.get("symbol") or "").upper(),
                    str(o.get("timestamp")),
                    str(o.get("context") or "default"),
                    str(o.get("schema_version") or ""),
                    json.dumps(o),
                ))
                if cur.rowcount == 1:
                    stats["written"] += 1
                else:
                    stats["duplicates"] += 1
            except Exception:
                stats["failed"] += 1
                try:
                    c.rollback()
                except Exception:
                    pass
        c.commit()
        cur.close()
        return stats
    except Exception:
        stats["failed"] = stats["attempted"] - stats["written"] - stats["duplicates"]
        return stats
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def save_outcome(outcome: Dict[str, Any], *, conn=None) -> bool:
    """Idempotently persist one outcome for (observation_id, horizon). First write
    wins — never rewrites the observation. Returns True only on a NEW row."""
    oid, horizon = outcome.get("observation_id"), outcome.get("horizon")
    if not oid or not horizon:
        return False
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return False
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cast = "" if is_sqlite else "::jsonb"
        cur.execute(
            f"INSERT INTO hsf_observation_outcomes "
            f"(observation_id, horizon, schema_version, record) "
            f"VALUES ({ph},{ph},{ph},{ph}{cast}) "
            f"ON CONFLICT (observation_id, horizon) DO NOTHING",
            (str(oid), str(horizon), str(outcome.get("schema_version") or ""),
             json.dumps(outcome)),
        )
        wrote = cur.rowcount == 1
        c.commit()
        cur.close()
        return wrote
    except Exception:
        return False
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def _loads(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def load_observation(observation_id: str, *, conn=None) -> Optional[Dict[str, Any]]:
    """Return the stored observation record, with any attached outcomes, or None."""
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return None
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cur.execute(f"SELECT record FROM hsf_observations WHERE observation_id = {ph}",
                    (str(observation_id),))
        row = cur.fetchone()
        if not row:
            cur.close()
            return None
        rec = _loads(row[0] if not isinstance(row, dict) else list(row.values())[0])
        cur.execute(
            f"SELECT horizon, record FROM hsf_observation_outcomes WHERE observation_id = {ph}",
            (str(observation_id),))
        outs = cur.fetchall() or []
        cur.close()
        if outs:
            rec["outcomes"] = {}
            for r in outs:
                h, payload = (r[0], r[1]) if not isinstance(r, dict) else tuple(r.values())
                rec["outcomes"][h] = _loads(payload)
        return rec
    except Exception:
        return None
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def load_observations_for_symbol(symbol: str, *, limit: int = 500,
                                 conn=None) -> List[Dict[str, Any]]:
    """Bounded query for one symbol's observations (Run 43 replay, Task 25).
    Newest first; non-fatal; [] if DB unavailable. Rehydrates attached outcomes
    per observation so replay can display them separately."""
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return []
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        cur.execute(f"SELECT observation_id, record FROM hsf_observations "
                    f"WHERE symbol = {ph} ORDER BY timestamp DESC LIMIT {ph}",
                    (str(symbol).upper(), int(limit)))
        rows = cur.fetchall() or []
        cur.close()
        out = []
        for r in rows:
            oid = r[0] if not isinstance(r, dict) else list(r.values())[0]
            rec = _loads(r[1] if not isinstance(r, dict) else list(r.values())[1])
            if rec:
                rec["_observation_id"] = oid
                out.append(rec)
        return out
    except Exception:
        return []
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass


def load_recent_observations(*, limit: int = 100, context: Optional[str] = None,
                             conn=None) -> List[Dict[str, Any]]:
    """Most recent observations (newest first). Non-fatal; [] if DB unavailable."""
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        return []
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        ph = _ph(is_sqlite)
        if context is None:
            cur.execute(f"SELECT record FROM hsf_observations "
                        f"ORDER BY timestamp DESC LIMIT {ph}", (int(limit),))
        else:
            cur.execute(f"SELECT record FROM hsf_observations WHERE context = {ph} "
                        f"ORDER BY timestamp DESC LIMIT {ph}", (str(context), int(limit)))
        rows = cur.fetchall() or []
        cur.close()
        return [_loads(r[0] if not isinstance(r, dict) else list(r.values())[0]) for r in rows]
    except Exception:
        return []
    finally:
        if opened:
            try:
                c.close()
            except Exception:
                pass
