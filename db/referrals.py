"""Invite a friend: referral codes, who signed up with whose link, and Pro trial grants.

Three tables, keyed by the account's user_id (its lowercase email username):

* hsf_referral_codes: one share code per account.
* hsf_referrals: one row per account that signed up with someone's code (UNIQUE
  referred, so an account is credited to one inviter, once). `referred` is a SHA-256
  of the new account's email, so the row holds nothing personal about them and stays
  after they delete their account (a re-created account can't earn a second trial). verified_at is set when
  the new account confirms its email; rewarded_at when both people got their trial.
* hsf_plan_grants: a free plan until a time (today only 'pro' from referrals).
  Billing never reads or writes it; the API treats a Starter account with a live
  grant as Pro until `until` passes, then it is Starter again with nothing to undo.

Postgres (Neon) in production; tests pass a SQLite connection through
set_connection_factory. Timestamps are ISO strings written from Python so both
backends run the same statements.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import secrets
import sqlite3
from typing import Any, Callable, Dict, Optional, Tuple

from db.engine import get_neon_conn, schema_once

_conn_factory: Optional[Callable[[], Any]] = None
CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o/1/l/i
CODE_LEN = 8


def referred_key(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Tests: route every call to `factory()` (e.g. one shared SQLite connection)."""
    global _conn_factory
    _conn_factory = factory


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def parse_ts(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _connect() -> Tuple[Any, bool]:
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")
    is_sqlite = isinstance(conn, sqlite3.Connection)
    _ensure_schema(conn, is_sqlite)
    return conn, is_sqlite


def _sql(query: str, is_sqlite: bool) -> str:
    return query.replace("%s", "?") if is_sqlite else query


def _close(conn, is_sqlite: bool) -> None:
    if is_sqlite and _conn_factory is not None:
        return  # the test's shared connection
    try:
        conn.close()
    except Exception:
        pass


def _one(cur) -> Optional[Tuple[Any, ...]]:
    row = cur.fetchone()
    if row is None:
        return None
    return tuple(row.values()) if isinstance(row, dict) else tuple(row)


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "BIGSERIAL PRIMARY KEY"
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_referral_codes (
            user_id TEXT PRIMARY KEY,
            code TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )""")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS hsf_referrals (
            id {pk},
            referrer TEXT NOT NULL,
            referred TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            verified_at TEXT,
            rewarded_at TEXT
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_referrals_referrer ON hsf_referrals (referrer)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_plan_grants (
            user_id TEXT PRIMARY KEY,
            tier TEXT NOT NULL,
            until_at TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        )""")
    conn.commit()
    cur.close()


def _new_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def get_or_create_code(user_id: str) -> str:
    user_id = user_id.strip().lower()
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT code FROM hsf_referral_codes WHERE user_id = %s", sq), (user_id,))
        row = _one(cur)
        if row:
            return str(row[0])
        for _ in range(5):
            code = _new_code()
            try:
                cur.execute(_sql("INSERT INTO hsf_referral_codes (user_id, code, created_at) VALUES (%s, %s, %s)", sq),
                            (user_id, code, _iso(_now())))
                conn.commit()
                return code
            except Exception:
                conn.rollback()
                cur = conn.cursor()
                cur.execute(_sql("SELECT code FROM hsf_referral_codes WHERE user_id = %s", sq), (user_id,))
                row = _one(cur)
                if row:  # created by another request in between
                    return str(row[0])
        raise RuntimeError("could not allocate a referral code")
    finally:
        _close(conn, sq)


def owner_of(code: str) -> Optional[str]:
    code = (code or "").strip().lower()
    if not code or len(code) > 32:
        return None
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT user_id FROM hsf_referral_codes WHERE code = %s", sq), (code,))
        row = _one(cur)
        return str(row[0]) if row else None
    finally:
        _close(conn, sq)


def record_signup(code: str, referred: str) -> bool:
    """Credit a new account to the code's owner. False for an unknown code, a
    self-referral, or an account that was already credited."""
    referred = referred.strip().lower()
    referrer = owner_of(code)
    if not referrer or referrer == referred:
        return False
    key = referred_key(referred)
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT 1 FROM hsf_referrals WHERE referred = %s", sq), (key,))
        if _one(cur):
            return False
        try:
            cur.execute(_sql("INSERT INTO hsf_referrals (referrer, referred, code, created_at) VALUES (%s, %s, %s, %s)", sq),
                        (referrer, key, code.strip().lower(), _iso(_now())))
            conn.commit()
        except Exception:
            conn.rollback()
            return False
        return True
    finally:
        _close(conn, sq)


def _extend(cur, sq: bool, user_id: str, days: int, cap_days: int, reason: str, now: dt.datetime) -> str:
    cur.execute(_sql("SELECT until_at FROM hsf_plan_grants WHERE user_id = %s", sq), (user_id,))
    row = _one(cur)
    current = parse_ts(row[0]) if row else None
    start = current if current and current > now else now
    until = min(start + dt.timedelta(days=days), now + dt.timedelta(days=cap_days))
    if row:
        cur.execute(_sql("UPDATE hsf_plan_grants SET tier = 'pro', until_at = %s, reason = %s, updated_at = %s "
                         "WHERE user_id = %s", sq), (_iso(until), reason, _iso(now), user_id))
    else:
        cur.execute(_sql("INSERT INTO hsf_plan_grants (user_id, tier, until_at, reason, updated_at) "
                         "VALUES (%s, 'pro', %s, %s, %s)", sq), (user_id, _iso(until), reason, _iso(now)))
    return _iso(until)


def reward_verified(referred: str, *, days: int, cap_days: int) -> Optional[str]:
    """When a credited account verifies its email: mark it verified and give both people
    `days` of Pro (added to any trial they still have, at most `cap_days` from now).
    Returns the referrer, or None when there is nothing to reward (never twice)."""
    referred = referred.strip().lower()
    key = referred_key(referred)
    now = _now()
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        # Claim the row first; the WHERE makes a second verification (or a second server) a no-op.
        cur.execute(_sql("UPDATE hsf_referrals SET verified_at = %s, rewarded_at = %s "
                         "WHERE referred = %s AND rewarded_at IS NULL", sq), (_iso(now), _iso(now), key))
        if not cur.rowcount:
            conn.rollback()
            return None
        cur.execute(_sql("SELECT referrer FROM hsf_referrals WHERE referred = %s", sq), (key,))
        row = _one(cur)
        referrer = str(row[0]) if row else None
        if days > 0:
            _extend(cur, sq, referred, days, cap_days, "referral:invited", now)
            if referrer:
                _extend(cur, sq, referrer, days, cap_days, "referral:inviter", now)
        conn.commit()
        return referrer
    except Exception:
        conn.rollback()
        raise
    finally:
        _close(conn, sq)


def grant_until(user_id: str) -> Optional[dt.datetime]:
    """When the account's free Pro ends, if it hasn't yet."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT until_at FROM hsf_plan_grants WHERE user_id = %s AND tier = 'pro'", sq),
                    (user_id.strip().lower(),))
        row = _one(cur)
    finally:
        _close(conn, sq)
    until = parse_ts(row[0]) if row else None
    return until if until and until > _now() else None


def stats(user_id: str) -> Dict[str, int]:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT COUNT(*) AS signed_up, COUNT(rewarded_at) AS verified FROM hsf_referrals WHERE referrer = %s", sq),
                    (user_id.strip().lower(),))
        row = _one(cur) or (0, 0)
        return {"signed_up": int(row[0] or 0), "verified": int(row[1] or 0)}
    finally:
        _close(conn, sq)


# ---- lifecycle email log (services/lifecycle_emails.py) ------------------------------------------
# One row per email sent, so each one goes out once: ("welcome", "") at the first verify,
# ("trial_ending", <until_at>) per trial end (a trial extended to a new end can get a new one).
@schema_once
def _ensure_lifecycle_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_lifecycle_emails (
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            PRIMARY KEY (user_id, kind, ref)
        )""")
    conn.commit()
    cur.close()


def _lifecycle_connect() -> Tuple[Any, bool]:
    conn, sq = _connect()
    _ensure_lifecycle_schema(conn, sq)
    return conn, sq


def claim_email(user_id: str, kind: str, ref: str = "") -> bool:
    """Reserve an email before sending it. False when it was already sent (or claimed by
    another run at the same moment), so two jobs never send the same email twice."""
    conn, sq = _lifecycle_connect()
    try:
        cur = conn.cursor()
        try:
            cur.execute(_sql("INSERT INTO hsf_lifecycle_emails (user_id, kind, ref, sent_at) "
                             "VALUES (%s, %s, %s, %s)", sq), (user_id.strip().lower(), kind, ref, _iso(_now())))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
    finally:
        _close(conn, sq)


def release_email(user_id: str, kind: str, ref: str = "") -> None:
    """Undo a claim when the send failed, so the next run tries again."""
    conn, sq = _lifecycle_connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_lifecycle_emails WHERE user_id = %s AND kind = %s AND ref = %s", sq),
                    (user_id.strip().lower(), kind, ref))
        conn.commit()
    finally:
        _close(conn, sq)


def grants_ending(within: dt.timedelta) -> list:
    """[(user_id, until_at text)] for free Pro that is still running and ends within `within`."""
    now = _now()
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT user_id, until_at FROM hsf_plan_grants WHERE tier = 'pro' "
                         "AND until_at > %s AND until_at <= %s ORDER BY until_at", sq),
                    (_iso(now), _iso(now + within)))
        rows = cur.fetchall() or []
    finally:
        _close(conn, sq)
    out = []
    for row in rows:
        vals = tuple(row.values()) if isinstance(row, dict) else tuple(row)
        out.append((str(vals[0]), str(vals[1])))
    return out
