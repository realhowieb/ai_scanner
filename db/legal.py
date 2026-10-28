"""Policy versions and who accepted which (domain/legal.py reads the texts).

Tables:

* legal_policy_versions: one row per (policy_type, version) with the exact text shown,
  its SHA-256 (content_hash) and status (proposed / approved). Rows are never deleted.
  A version's text may change only while nobody has accepted it (drafts, and filling in
  business details before launch); once accepted it is frozen, and a changed file for
  it is refused (the policy then reports unavailable, so checkout stops) until it is
  published under a new version number.
* user_policy_acceptances: insert-only. One row per accepted policy per checkout: the
  account, the version row, the tier and checkout kind, the time and the Stripe
  checkout session. Admins match a purchase to the words its buyer accepted through
  checkout_session_id.

Uses db/founding.py's connections (and its test connection factory).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from db import founding as _db
from db.engine import schema_once
from domain import legal as policies

log = logging.getLogger("hsf.legal")

LOCK_KEY = 0x4853464C  # "HSFL"
_sqlite_lock = threading.RLock()


@contextmanager
def _transaction(*, write: bool = True) -> Iterator["_db._Tx"]:
    conn, sq = _db._raw_connect()
    guard = _sqlite_lock if sq else None
    if guard:
        guard.acquire()
    try:
        _ensure_schema(conn, sq)
        tx = _db._Tx(conn, sq)
        try:
            if write and not sq:
                tx.run("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            yield tx
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        _db._close(conn, sq)
        if guard:
            guard.release()


@schema_once
def _ensure_schema(conn, sq: bool) -> None:
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if sq else "BIGSERIAL PRIMARY KEY"
    types = ", ".join(f"'{t}'" for t in policies.POLICY_TYPES)
    statuses = ", ".join(f"'{s}'" for s in policies.STATUSES)
    cur = conn.cursor()
    if not sq:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS legal_policy_versions (
            id {pk},
            policy_type TEXT NOT NULL CHECK (policy_type IN ({types})),
            version INTEGER NOT NULL CHECK (version >= 1),
            title TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({statuses})),
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (policy_type, version)
        )""")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS user_policy_acceptances (
            id {pk},
            user_id TEXT NOT NULL,
            policy_version_id INTEGER NOT NULL REFERENCES legal_policy_versions (id),
            tier TEXT,
            kind TEXT,
            accepted_at TEXT NOT NULL,
            checkout_session_id TEXT NOT NULL,
            reservation_id INTEGER,
            UNIQUE (checkout_session_id, policy_version_id)
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_policy_acceptances_user ON user_policy_acceptances (user_id)")
    conn.commit()
    cur.close()


def _now(now: Optional[dt.datetime]) -> str:
    return _db._iso(_db._utc(now))


def sync(loaded: Dict[str, List[Dict[str, Any]]], *, now: Optional[dt.datetime] = None) -> Dict[str, Dict[str, Any]]:
    """Record every version from domain.legal.load(). Returns the newest version of each
    type: {type: {id, version, content_hash, status, available, problem}}. A version whose
    text changed after someone accepted it is left as recorded and marked unavailable."""
    stamp = _now(now)
    out: Dict[str, Dict[str, Any]] = {}
    with _transaction() as tx:
        for ptype, versions in loaded.items():
            for p in versions:
                row = tx.one("SELECT * FROM legal_policy_versions WHERE policy_type = %s AND version = %s",
                             (ptype, p["version"]))
                problem = None
                if not row:
                    tx.run("INSERT INTO legal_policy_versions (policy_type, version, title, effective_date, status, "
                           "content, content_hash, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (ptype, p["version"], p["title"], p["effective_date"], p["status"], p["text"],
                            p["content_hash"], stamp, stamp))
                elif row["content_hash"] != p["content_hash"]:
                    accepted = tx.one("SELECT COUNT(*) AS n FROM user_policy_acceptances WHERE policy_version_id = %s",
                                      (row["id"],))
                    if int((accepted or {}).get("n") or 0):
                        problem = "accepted_version_changed"
                        log.error("legal policy %s v%s text changed after acceptance; publish a new version",
                                  ptype, p["version"])
                    else:
                        tx.run("UPDATE legal_policy_versions SET title = %s, effective_date = %s, status = %s, "
                               "content = %s, content_hash = %s, updated_at = %s WHERE id = %s",
                               (p["title"], p["effective_date"], p["status"], p["text"], p["content_hash"], stamp,
                                row["id"]))
                elif row["status"] != p["status"] or row["effective_date"] != p["effective_date"]:
                    tx.run("UPDATE legal_policy_versions SET status = %s, effective_date = %s, updated_at = %s "
                           "WHERE id = %s", (p["status"], p["effective_date"], stamp, row["id"]))
                row = tx.one("SELECT id, version, content_hash, status FROM legal_policy_versions "
                             "WHERE policy_type = %s AND version = %s", (ptype, p["version"]))
                out[ptype] = {**row, "available": problem is None, "problem": problem}
    return out


def version(policy_type: str, number: int) -> Optional[Dict[str, Any]]:
    with _transaction(write=False) as tx:
        return tx.one("SELECT * FROM legal_policy_versions WHERE policy_type = %s AND version = %s",
                      (policy_type, int(number)))


def history(policy_type: str) -> List[Dict[str, Any]]:
    """Every recorded version of a policy, oldest first (texts included)."""
    with _transaction(write=False) as tx:
        return tx.rows("SELECT * FROM legal_policy_versions WHERE policy_type = %s ORDER BY version", (policy_type,))


def record_acceptance(user_id: str, version_ids: List[int], *, checkout_session_id: str, tier: str, kind: str,
                      reservation_id: Optional[int] = None, now: Optional[dt.datetime] = None) -> int:
    """Insert one acceptance per policy version for this checkout. A retry of the same
    checkout adds nothing. Returns rows added."""
    if not checkout_session_id or not version_ids:
        raise ValueError("a checkout session and at least one policy version are required")
    stamp, user_id, added = _now(now), user_id.strip().lower(), 0
    with _transaction() as tx:
        for vid in version_ids:
            cur = tx.run("INSERT INTO user_policy_acceptances (user_id, policy_version_id, tier, kind, accepted_at, "
                         "checkout_session_id, reservation_id) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                         "ON CONFLICT (checkout_session_id, policy_version_id) DO NOTHING",
                         (user_id, int(vid), tier, kind, stamp, checkout_session_id, reservation_id))
            added += int(cur.rowcount or 0)
    return added


def acceptances(*, user_id: Optional[str] = None, checkout_session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """What an account, or one checkout, accepted: version, hash and time, oldest first."""
    if not user_id and not checkout_session_id:
        raise ValueError("user_id or checkout_session_id is required")
    where, params = [], []
    if user_id:
        where.append("a.user_id = %s")
        params.append(user_id.strip().lower())
    if checkout_session_id:
        where.append("a.checkout_session_id = %s")
        params.append(checkout_session_id)
    with _transaction(write=False) as tx:
        return tx.rows("SELECT a.id, a.user_id, a.tier, a.kind, a.accepted_at, a.checkout_session_id, a.reservation_id, "
                       "v.policy_type, v.version, v.content_hash, v.status FROM user_policy_acceptances a "
                       "JOIN legal_policy_versions v ON v.id = a.policy_version_id "
                       f"WHERE {' AND '.join(where)} ORDER BY a.id", tuple(params))
