"""Test helper: an in-memory SQLite stand-in for the Neon connection used by the
Run 83 token store. It translates the few Postgres-isms in that SQL so the real
queries (INSERT / DELETE … RETURNING) run unchanged, and it supports both the
psycopg-style `conn.cursor()` calls in ui/auth_tokens.py and the psycopg2
`with conn: with conn.cursor() as cur:` style in billing_service/main.py.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime


def _sql(q: str) -> str:
    return (q.replace("%s", "?")
             .replace("timestamptz", "text")
             .replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP"))


def _arg(v):
    return v.isoformat() if isinstance(v, datetime) else v


class _Cursor:
    def __init__(self, raw: sqlite3.Cursor):
        self._raw = raw

    def execute(self, q, params=()):
        self._raw.execute(_sql(q), tuple(_arg(p) for p in params))
        return self

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TokenDB:
    """One shared in-memory database; `conn()` hands out connection views."""

    def __init__(self):
        self.raw = sqlite3.connect(":memory:", check_same_thread=False)  # TestClient uses a worker thread

    def cursor(self):
        return _Cursor(self.raw.cursor())

    def commit(self):
        self.raw.commit()

    def close(self):          # views never close the shared database
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.raw.commit()
        return False

    def rows(self):
        return self.raw.execute("SELECT username, purpose, token_hash FROM hsf_auth_tokens").fetchall()

    def shutdown(self):
        self.raw.close()
