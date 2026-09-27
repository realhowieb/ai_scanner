"""Run 83 — short-lived, single-use, purpose-bound account tokens.

Two uses, both of which previously relied on a reusable 14-day login session id:

* ``restore`` — the ``?rt=`` value on the Stripe checkout-success / portal-return
  links, which signs the user back in after the Stripe round trip.
* ``billing`` — proof, sent to the billing service, that the request comes from
  the signed-in HSF account (the service resolves the Stripe customer from this
  identity; it never trusts a client-supplied email or customer id).

Properties:
* unpredictable: ``secrets.token_urlsafe(32)`` (256 bits);
* only a SHA-256 hash is stored, so a database read does not reveal usable tokens;
* bound to one purpose and a short expiry;
* consumed by one atomic ``DELETE … RETURNING``: a token works exactly once,
  and an unknown, expired or wrong-purpose token is rejected without consuming
  anything.

The billing service (a separate FastAPI deploy) implements the same consume
query against the same table; see ``billing_service/main.py``.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

TABLE = "hsf_auth_tokens"
TTL = {
    "restore": timedelta(hours=2),    # long enough to fill in Stripe checkout
    "billing": timedelta(minutes=10),  # one billing-service call
}

SCHEMA_SQL = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} ("
    " token_hash text PRIMARY KEY,"
    " username text NOT NULL,"
    " purpose text NOT NULL,"
    " expires_at timestamptz NOT NULL,"
    " created_at timestamptz NOT NULL DEFAULT now())"
)
CONSUME_SQL = (
    f"DELETE FROM {TABLE} WHERE token_hash = %s AND purpose = %s AND expires_at > %s "
    "RETURNING username"
)


def token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _conn():
    from db.engine import get_neon_conn

    return get_neon_conn()


def _close(conn: Any) -> None:
    try:
        conn.close()
    except Exception:
        pass


def issue_token(username: str, purpose: str, *, conn: Any = None,
                now: Optional[datetime] = None) -> Optional[str]:
    """Create a token for ``username``; returns the raw token (store nowhere) or
    None when the database is unavailable (callers must fail closed)."""
    user = str(username or "").strip().lower()
    if not user or purpose not in TTL:
        return None
    own = conn is None
    try:
        c = _conn() if own else conn
        if c is None:
            return None
        now = now or datetime.now(timezone.utc)
        token = secrets.token_urlsafe(32)
        cur = c.cursor()
        cur.execute(SCHEMA_SQL)
        cur.execute(f"DELETE FROM {TABLE} WHERE expires_at <= %s", (now,))   # housekeeping
        cur.execute(
            f"INSERT INTO {TABLE} (token_hash, username, purpose, expires_at) VALUES (%s, %s, %s, %s)",
            (token_hash(token), user, purpose, now + TTL[purpose]),
        )
        c.commit()
        cur.close()
        return token
    except Exception:
        return None
    finally:
        if own and "c" in locals() and c is not None:
            _close(c)


def consume_token(token: str, purpose: str, *, conn: Any = None,
                  now: Optional[datetime] = None) -> Optional[str]:
    """Return the username for a valid token and invalidate it (single use).
    Unknown, expired, already-used or wrong-purpose tokens return None."""
    raw = str(token or "").strip()
    if not raw or purpose not in TTL or len(raw) > 256:
        return None
    own = conn is None
    try:
        c = _conn() if own else conn
        if c is None:
            return None
        cur = c.cursor()
        cur.execute(SCHEMA_SQL)
        cur.execute(CONSUME_SQL, (token_hash(raw), purpose, now or datetime.now(timezone.utc)))
        row = cur.fetchone()
        c.commit()
        cur.close()
        if not row:
            return None
        user = row[0] if isinstance(row, (tuple, list)) else row.get("username")
        return str(user).strip().lower() or None
    except Exception:
        return None
    finally:
        if own and "c" in locals() and c is not None:
            _close(c)
