"""Read side of the HSF research dataset, plus the finalized-version registry.

Reads are bulk and bounded: one query for the opportunity rows of a date range
(``signal_outcomes``, source='opportunity') and one for the scheduled scan
records of those tickers (``hsf_observations``). Never one query per
observation, never a market-data provider call.

The registry table ``research_dataset_versions`` is insert-once: a finalized
version (its filters, member ids, fingerprint and metadata) is never updated.
Changed data means a new version. Only the CLI (scripts/research_dataset.py
--finalize) writes it; the API is read-only.

Unlike most db.* readers, these raise ``ResearchDataUnavailable`` when the
database can't answer, so an outage is never mistaken for an empty dataset.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from db.engine import get_neon_conn, schema_once

OPTIONAL_OUTCOME_COLUMNS = ("benchmark_return_1d", "benchmark_return_3d", "benchmark_return_5d")
_BASE_COLUMNS = ("id", "ticker", "fired_at", "setup_score", "prebreakout_prob", "indicators", "raw_signal",
                 "return_1d", "return_3d", "return_5d", "mfe_5d", "mae_5d", "outcome_computed_at", "created_at")
# Scan-record fields the research snapshot reads; everything else stays in the DB.
_SCAN_JSON_PG = (
    "jsonb_build_object("
    "'market', record->'market', 'indicators', record->'indicators', 'scanners', record->'scanners', "
    "'versions', record->'versions', 'scan_timestamp', record->'scan_timestamp', "
    "'schema_version', record->'schema_version', 'universe_version', record->'universe_version', "
    "'market_context', record->'market_context', "
    "'research_metadata', jsonb_build_object("
    "'row_features', record->'research_metadata'->'row_features', "
    "'rank_at_observation', record->'research_metadata'->'rank_at_observation', "
    "'scan_id', record->'research_metadata'->'scan_id', "
    "'universe_name', record->'research_metadata'->'universe_name', "
    "'scoring_version', record->'research_metadata'->'scoring_version', "
    "'scanner_commit_sha', record->'research_metadata'->'scanner_commit_sha'))"
)


class ResearchDataUnavailable(RuntimeError):
    """The database could not answer; never treat as 'no data'."""


def _rows(cur) -> List[Dict[str, Any]]:
    rows = cur.fetchall() or []
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _neon():
    try:
        conn = get_neon_conn()
    except Exception as e:
        raise ResearchDataUnavailable("database unavailable") from e
    if conn is None:
        raise ResearchDataUnavailable("database unavailable")
    return conn


def _present_optional_columns(cur) -> List[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'signal_outcomes' AND column_name = ANY(%s)",
        (list(OPTIONAL_OUTCOME_COLUMNS),))
    have = {(r["column_name"] if isinstance(r, dict) else r[0]) for r in (cur.fetchall() or [])}
    return [c for c in OPTIONAL_OUTCOME_COLUMNS if c in have]


def fetch_opportunity_rows(start: _dt.datetime, end: _dt.datetime, *,
                           ids: Optional[Sequence[int]] = None, conn=None) -> List[Dict[str, Any]]:
    """Frozen HSF opportunity rows with fired_at in [start, end), oldest first.
    Benchmark columns are included when Outcome Intelligence has added them."""
    c = conn or _neon()
    try:
        cur = c.cursor()
        cols = list(_BASE_COLUMNS) + _present_optional_columns(cur)
        where = "source = 'opportunity' AND fired_at >= %s AND fired_at < %s"
        params: List[Any] = [start, end]
        if ids is not None:
            where += " AND id = ANY(%s)"
            params.append([int(i) for i in ids])
        cur.execute(f"SELECT {', '.join(cols)} FROM signal_outcomes WHERE {where} "  # nosec B608
                    "ORDER BY fired_at ASC, id ASC", tuple(params))
        out = _rows(cur)
        cur.close()
        return out
    except ResearchDataUnavailable:
        raise
    except Exception as e:
        raise ResearchDataUnavailable(f"opportunity read failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)


def fetch_opportunity_row(observation_id: int, *, conn=None) -> Optional[Dict[str, Any]]:
    """One opportunity row plus every row frozen at the same instant (needed for
    its snapshot rank). Returns {"row": ..., "snapshot": [...]} or None."""
    c = conn or _neon()
    try:
        cur = c.cursor()
        cols = list(_BASE_COLUMNS) + _present_optional_columns(cur)
        sel = ", ".join(cols)
        cur.execute(f"SELECT {sel} FROM signal_outcomes WHERE source = 'opportunity' AND id = %s",  # nosec B608
                    (int(observation_id),))
        found = _rows(cur)
        if not found:
            cur.close()
            return None
        cur.execute(f"SELECT {sel} FROM signal_outcomes WHERE source = 'opportunity' AND fired_at = %s "  # nosec B608
                    "ORDER BY id ASC", (found[0]["fired_at"],))
        snap = _rows(cur)
        cur.close()
        return {"row": found[0], "snapshot": snap}
    except Exception as e:
        raise ResearchDataUnavailable(f"opportunity read failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)


def fetch_scan_records(tickers: Iterable[str], start: _dt.datetime, end: _dt.datetime, *,
                       conn=None) -> List[Dict[str, Any]]:
    """Scheduled-scan records for `tickers` with timestamp in [start, end].

    Callers widen `start` by the join's max lag so the backward join can see
    the scan just before the first observation. Works on Neon (slim JSON
    projection) and on the local SQLite fallback used by tests."""
    from db.hsf_observations import _ensure_schema, _resolve_conn

    syms = sorted({str(t).upper() for t in tickers if t})
    if not syms:
        return []
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        raise ResearchDataUnavailable("database unavailable")
    try:
        _ensure_schema(c, is_sqlite)
        cur = c.cursor()
        out: List[Dict[str, Any]] = []
        if is_sqlite:
            s, e = start.isoformat(), end.isoformat()
            for i in range(0, len(syms), 500):
                chunk = syms[i:i + 500]
                marks = ",".join("?" for _ in chunk)
                cur.execute("SELECT observation_id, symbol, context, timestamp, created_at, record "  # nosec B608
                            f"FROM hsf_observations WHERE context LIKE 'scheduled:%' AND symbol IN ({marks}) "
                            "AND timestamp >= ? AND timestamp <= ?", (*chunk, s, e))
                out.extend(_rows(cur))
        else:
            cur.execute("SELECT observation_id, symbol, context, timestamp, created_at, "  # nosec B608
                        f"{_SCAN_JSON_PG} AS record FROM hsf_observations "
                        "WHERE context LIKE 'scheduled:%%' AND symbol = ANY(%s) "
                        "AND timestamp >= %s AND timestamp <= %s", (syms, start, end))
            out = _rows(cur)
        cur.close()
        for r in out:
            if isinstance(r.get("record"), str):
                try:
                    r["record"] = json.loads(r["record"])
                except json.JSONDecodeError:
                    r["record"] = {}
        return out
    except ResearchDataUnavailable:
        raise
    except Exception as e:
        raise ResearchDataUnavailable(f"scan record read failed: {type(e).__name__}") from e
    finally:
        if opened:
            _close(c)


# --------------------------------------------------------------------------- ML readiness (slim reads)
# ML readiness needs identity, timing, the HSF status fields and the outcome
# columns, not the full frozen payload. These projections keep the scheduled
# readiness checks light on Neon egress (about a fifth of the full rows).
_READINESS_RAW_PG = (
    "jsonb_build_object('hsf_score', raw_signal->'hsf_score', 'score_version', raw_signal->'score_version', "
    "'primary_setup', raw_signal->'primary_setup', 'status', raw_signal->'status', "
    "'prebreakout_model_version', raw_signal->'prebreakout_model_version', "
    "'model_version', raw_signal->'model_version')"
)
_READINESS_IND_PG = (
    "jsonb_build_object('signals', indicators->'signals', 'status', indicators->'status', "
    "'primary_setup', indicators->'primary_setup')"
)
_READINESS_COLUMNS = ("id", "ticker", "fired_at", "created_at", "ai_confidence", "return_1d", "return_3d",
                      "return_5d", "outcome_computed_at")


def fetch_readiness_rows(start: _dt.datetime, end: _dt.datetime, *, conn=None) -> List[Dict[str, Any]]:
    """Slim opportunity rows with fired_at in [start, end) for ML readiness:
    identity, timing, HSF status fields, AI Confidence, 1/3/5-day returns and
    the 5-day benchmark. Raises ResearchDataUnavailable on any failure."""
    c = conn or _neon()
    try:
        cur = c.cursor()
        bench = [x for x in _present_optional_columns(cur) if x.endswith("_5d")]
        cols = list(_READINESS_COLUMNS) + bench + [f"{_READINESS_RAW_PG} AS raw_signal",
                                                   f"{_READINESS_IND_PG} AS indicators"]
        cur.execute(f"SELECT {', '.join(cols)} FROM signal_outcomes "  # nosec B608
                    "WHERE source = 'opportunity' AND fired_at >= %s AND fired_at < %s "
                    "ORDER BY fired_at ASC, id ASC", (start, end))
        out = _rows(cur)
        cur.close()
        for r in out:
            for k in ("raw_signal", "indicators"):
                if isinstance(r.get(k), str):
                    try:
                        r[k] = json.loads(r[k])
                    except json.JSONDecodeError:
                        r[k] = {}
        return out
    except ResearchDataUnavailable:
        raise
    except Exception as e:
        raise ResearchDataUnavailable(f"readiness read failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)


def fetch_scan_index(tickers: Iterable[str], start: _dt.datetime, end: _dt.datetime, *,
                     conn=None) -> List[Dict[str, Any]]:
    """Only what the backward join needs (symbol, context, scan time, write
    time) for scheduled-scan records of `tickers` in [start, end]. The record
    payload is reduced to its scan_timestamp, so a readiness check never pulls
    feature JSON. SQLite (tests, local) falls back to fetch_scan_records."""
    from db.hsf_observations import _resolve_conn

    syms = sorted({str(t).upper() for t in tickers if t})
    if not syms:
        return []
    c, opened, is_sqlite = _resolve_conn(conn)
    if c is None:
        raise ResearchDataUnavailable("database unavailable")
    if is_sqlite:
        if opened:
            _close(c)
        return fetch_scan_records(syms, start, end, conn=conn)
    try:
        cur = c.cursor()
        cur.execute("SELECT observation_id, symbol, context, timestamp, created_at, "
                    "jsonb_build_object('scan_timestamp', record->'scan_timestamp') AS record "
                    "FROM hsf_observations WHERE context LIKE 'scheduled:%%' AND symbol = ANY(%s) "
                    "AND timestamp >= %s AND timestamp <= %s", (syms, start, end))
        out = _rows(cur)
        cur.close()
        for r in out:
            if isinstance(r.get("record"), str):
                try:
                    r["record"] = json.loads(r["record"])
                except json.JSONDecodeError:
                    r["record"] = {}
        return out
    except Exception as e:
        raise ResearchDataUnavailable(f"scan index read failed: {type(e).__name__}") from e
    finally:
        if opened:
            _close(c)


# --------------------------------------------------------------------------- registry
@schema_once
def _ensure_registry(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS research_dataset_versions (
            dataset_version TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            feature_schema_version INTEGER NOT NULL,
            label_schema_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            observation_count INTEGER NOT NULL,
            observation_ids JSONB NOT NULL,
            metadata JSONB NOT NULL
        )
        """
    )
    conn.commit()
    cur.close()


def save_dataset_version(entry: Dict[str, Any], *, conn=None) -> bool:
    """Insert a finalized version once. Returns False when the name already
    exists: a finalized version is never overwritten."""
    c = conn or _neon()
    try:
        _ensure_registry(c)
        cur = c.cursor()
        cur.execute(
            "INSERT INTO research_dataset_versions (dataset_version, feature_schema_version, "
            "label_schema_version, fingerprint, observation_count, observation_ids, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) ON CONFLICT (dataset_version) DO NOTHING",
            (entry["dataset_version"], int(entry["feature_schema_version"]), int(entry["label_schema_version"]),
             entry["fingerprint"], int(entry["observation_count"]),
             json.dumps([int(i) for i in entry["observation_ids"]]),
             json.dumps(entry["metadata"], default=str)))
        written = cur.rowcount == 1
        c.commit()
        cur.close()
        return written
    except Exception as e:
        raise ResearchDataUnavailable(f"registry write failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)


def _registry_exists(cur) -> bool:
    """Reads never create the table (the API stays read-only)."""
    cur.execute("SELECT to_regclass('research_dataset_versions') IS NOT NULL")
    r = cur.fetchone()
    return bool(list(r.values())[0] if isinstance(r, dict) else r[0])


def list_dataset_versions(*, conn=None) -> List[Dict[str, Any]]:
    c = conn or _neon()
    try:
        cur = c.cursor()
        if not _registry_exists(cur):
            cur.close()
            return []
        cur.execute("SELECT dataset_version, created_at, feature_schema_version, label_schema_version, "
                    "fingerprint, observation_count, metadata FROM research_dataset_versions "
                    "ORDER BY created_at DESC, dataset_version DESC")
        out = _rows(cur)
        cur.close()
        return out
    except Exception as e:
        raise ResearchDataUnavailable(f"registry read failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)


def get_dataset_version(name: str, *, conn=None) -> Optional[Dict[str, Any]]:
    c = conn or _neon()
    try:
        cur = c.cursor()
        if not _registry_exists(cur):
            cur.close()
            return None
        cur.execute("SELECT dataset_version, created_at, feature_schema_version, label_schema_version, "
                    "fingerprint, observation_count, observation_ids, metadata "
                    "FROM research_dataset_versions WHERE dataset_version = %s", (str(name),))
        rows = _rows(cur)
        cur.close()
        return rows[0] if rows else None
    except Exception as e:
        raise ResearchDataUnavailable(f"registry read failed: {type(e).__name__}") from e
    finally:
        if conn is None:
            _close(c)
