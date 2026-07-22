"""Per-user Alpaca paper-trading account credentials (encrypted at rest).

Each user connects their OWN Alpaca paper account so positions/orders are theirs
alone. The API key/secret are encrypted via db.secret_box before storage — the
plaintext never lands in the DB. Same plain-cursor pattern as the rest of db/.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from db.engine import get_neon_conn
from db.secret_box import decrypt_secret, encrypt_secret, encryption_available


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alpaca_paper_accounts (
            user_id TEXT PRIMARY KEY,
            api_key_enc TEXT NOT NULL,
            api_secret_enc TEXT NOT NULL,
            connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.commit()
    cur.close()


def _get_conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available.")
    _ensure_schema(conn)
    return conn


def save_paper_account(user_id: str, api_key: str, api_secret: str) -> bool:
    """Encrypt and upsert a user's Alpaca paper keys. False if unstorable."""
    user = (user_id or "").strip().lower()
    if not user or not api_key or not api_secret:
        return False
    if not encryption_available():
        return False  # refuse to store secrets we can't encrypt
    key_enc = encrypt_secret(api_key)
    sec_enc = encrypt_secret(api_secret)
    if not key_enc or not sec_enc:
        return False
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alpaca_paper_accounts (user_id, api_key_enc, api_secret_enc, connected_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE
                SET api_key_enc = EXCLUDED.api_key_enc,
                    api_secret_enc = EXCLUDED.api_secret_enc,
                    connected_at = NOW()
            """,
            (user, key_enc, sec_enc),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False


def get_paper_account(user_id: str) -> Optional[Dict[str, str]]:
    """Return {'api_key', 'api_secret'} (decrypted) for the user, or None."""
    user = (user_id or "").strip().lower()
    if not user:
        return None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT api_key_enc, api_secret_enc FROM alpaca_paper_accounts WHERE user_id = %s",
            (user,),
        )
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    if not row:
        return None
    key_enc = row["api_key_enc"] if isinstance(row, dict) else row[0]
    sec_enc = row["api_secret_enc"] if isinstance(row, dict) else row[1]
    api_key = decrypt_secret(key_enc)
    api_secret = decrypt_secret(sec_enc)
    if not api_key or not api_secret:
        return None
    return {"api_key": api_key, "api_secret": api_secret}


def has_paper_account(user_id: str) -> bool:
    user = (user_id or "").strip().lower()
    if not user:
        return False
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM alpaca_paper_accounts WHERE user_id = %s", (user,))
        found = cur.fetchone() is not None
        cur.close()
        return found
    except Exception:
        return False


def delete_paper_account(user_id: str) -> None:
    user = (user_id or "").strip().lower()
    if not user:
        return
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM alpaca_paper_accounts WHERE user_id = %s", (user,))
        conn.commit()
        cur.close()
    except Exception:
        pass


def account_meta(user_id: str) -> Optional[Dict[str, Any]]:
    """Non-secret connection metadata (connected_at) for display."""
    user = (user_id or "").strip().lower()
    if not user:
        return None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT connected_at FROM alpaca_paper_accounts WHERE user_id = %s", (user,)
        )
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    if not row:
        return None
    return {"connected_at": row["connected_at"] if isinstance(row, dict) else row[0]}
