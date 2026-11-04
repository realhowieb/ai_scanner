"""Durable final-test ledger. Only explicit final evaluation writes here.

Claims are committed BEFORE any holdout predictions are made. Failed/interrupted
runs keep their claim; they never silently make a final test available again.
Completed reports can be reused only with exactly the same evaluation binding.
"""
from __future__ import annotations

import hashlib
import json

from db.engine import get_neon_conn

_factory = None


class HoldoutAlreadyUsed(ValueError):
    pass


def set_connection_factory(factory):
    global _factory
    _factory = factory


def _connection():
    c = _factory() if _factory else get_neon_conn()
    if c is None:
        raise RuntimeError("Final-test ledger unavailable; holdout scoring refused")
    return c, c.__class__.__module__.startswith("sqlite3")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reserve(binding, observation_ids, *, run_id):
    """Return (claim_key, cached_report). Refuse overlapping or interrupted tests."""
    ids = sorted(set(map(int, observation_ids)))
    if not ids or not run_id:
        raise ValueError("Final evaluation needs holdout observations and a run identifier")
    target = _canonical(binding["target"])
    key = hashlib.sha256(_canonical({"target": target, "ids": ids}).encode()).hexdigest()
    encoded_binding = _canonical(binding)
    c, sqlite = _connection()
    cur = c.cursor()
    query = lambda sql: sql.replace("%s", "?") if sqlite else sql
    try:
        cur.execute("CREATE TABLE IF NOT EXISTS ml_holdout_evaluations (claim_key TEXT PRIMARY KEY, "
                    "target_key TEXT NOT NULL, binding TEXT NOT NULL, observation_ids TEXT NOT NULL, "
                    "run_id TEXT NOT NULL, report TEXT, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        c.commit()
        if sqlite:
            cur.execute("BEGIN IMMEDIATE")
        else:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("ml-holdout:" + target,))
        cur.execute(query("SELECT claim_key, binding, observation_ids, report FROM ml_holdout_evaluations WHERE target_key = %s"), (target,))
        for row in cur.fetchall():
            old = dict(row) if isinstance(row, dict) else dict(zip(("claim_key", "binding", "observation_ids", "report"), row))
            if set(json.loads(old["observation_ids"])) & set(ids):
                if old["claim_key"] == key and old["binding"] == encoded_binding and old["report"]:
                    c.commit()
                    return key, json.loads(old["report"])
                raise HoldoutAlreadyUsed("Final-test observations already claimed or evaluated. Reuse the exact completed report or reserve genuinely unseen observations; no automatic retry.")
        cur.execute(query("INSERT INTO ml_holdout_evaluations (claim_key,target_key,binding,observation_ids,run_id) VALUES (%s,%s,%s,%s,%s)"),
                    (key, target, encoded_binding, _canonical(ids), str(run_id)))
        c.commit()
        return key, None
    except BaseException:
        c.rollback()
        raise
    finally:
        cur.close()
        if not sqlite:
            c.close()


def complete(key, binding, report):
    c, sqlite = _connection()
    cur = c.cursor()
    try:
        sql = "UPDATE ml_holdout_evaluations SET report = %s WHERE claim_key = %s AND binding = %s AND report IS NULL"
        cur.execute(sql.replace("%s", "?") if sqlite else sql, (_canonical(report), key, _canonical(binding)))
        if cur.rowcount != 1:
            raise HoldoutAlreadyUsed("Final-test claim changed or is already complete")
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        cur.close()
        if not sqlite:
            c.close()


def verify_completed(key, binding, report):
    """Read-only proof that a supplied report is the ledger's completed report."""
    c, sqlite = _connection()
    cur = c.cursor()
    try:
        sql = "SELECT binding, report FROM ml_holdout_evaluations WHERE claim_key = %s"
        cur.execute(sql.replace("%s", "?") if sqlite else sql, (key,))
        row = cur.fetchone()
        saved = dict(row) if isinstance(row, dict) else dict(zip(("binding", "report"), row)) if row else {}
        if saved.get("binding") != _canonical(binding) or saved.get("report") != _canonical(report):
            raise HoldoutAlreadyUsed("Supplied final evaluation is not the ledger's completed report")
    finally:
        cur.close()
        if not sqlite:
            c.close()
