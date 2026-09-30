"""P2-52 — delete sign-in data that can no longer be used (run once a day).

Only rows past their expiry (plus a grace period) are removed, so nothing a
user could still use is touched:

- auth_sessions: expired more than 1 day ago (a session past expires_at is
  already rejected at sign-in restore).
- hsf_auth_tokens (billing / restore tokens): expired more than 1 day ago.
- password_reset_tokens: used or expired, and created more than 1 day ago (the
  reset rate limit only counts the last hour).
- email_verifications: never verified and expired more than 7 days ago.
  Verification status lives in users.email_verified, so this never
  un-verifies anyone; verified rows are kept as history.

A missing table is skipped. Returns {table: rows_deleted}.
"""
from __future__ import annotations

from typing import Dict

PURGES = (
    ("auth_sessions",
     "DELETE FROM auth_sessions WHERE expires_at < NOW() - INTERVAL '1 day'"),
    ("hsf_auth_tokens",
     "DELETE FROM hsf_auth_tokens WHERE expires_at < NOW() - INTERVAL '1 day'"),
    ("password_reset_tokens",
     "DELETE FROM password_reset_tokens "
     "WHERE (used OR expires_at < NOW()) AND created_at < NOW() - INTERVAL '1 day'"),
    ("email_verifications",
     "DELETE FROM email_verifications "
     "WHERE verified_at IS NULL AND expires_at < NOW() - INTERVAL '7 days'"),
)


def purge_expired_credentials(conn) -> Dict[str, int]:
    """Run each purge in its own transaction; one failure never blocks the rest."""
    out: Dict[str, int] = {}
    for table, sql in PURGES:
        cur = conn.cursor()
        try:
            cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            row = cur.fetchone()
            exists = (row[0] if isinstance(row, (tuple, list)) else next(iter(row.values()))) if row else None
            if not exists:
                continue
            cur.execute(sql)
            out[table] = max(int(cur.rowcount or 0), 0)
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[maintenance] credential purge skipped {table}: {type(e).__name__}")
        finally:
            cur.close()
    return out
