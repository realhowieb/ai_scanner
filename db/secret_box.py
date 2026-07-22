"""Symmetric encryption for at-rest secrets (per-user Alpaca paper keys).

Fernet (AES-128-CBC + HMAC) with a key derived from APP_ENCRYPTION_KEY, falling
back to COOKIE_PASSWORD. A domain separator keeps this key distinct from the
cookie-session key even when they share the same source secret. Plaintext never
touches the database — only opaque tokens are stored.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

_DOMAIN = "alpaca_paper_v1:"  # key separation from other uses of the app secret


def _secret_source() -> Optional[str]:
    for name in ("APP_ENCRYPTION_KEY", "COOKIE_PASSWORD"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    # Streamlit secrets fallback (cloud), best-effort.
    try:
        import streamlit as st  # type: ignore

        for name in ("APP_ENCRYPTION_KEY", "COOKIE_PASSWORD"):
            val = str(st.secrets.get(name, "") or "").strip()  # type: ignore[attr-defined]
            if val:
                return val
    except Exception:
        pass
    return None


def _fernet():
    secret = _secret_source()
    if not secret:
        return None
    try:
        from cryptography.fernet import Fernet

        digest = hashlib.sha256((_DOMAIN + secret).encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))
    except Exception:
        return None


def encryption_available() -> bool:
    return _fernet() is not None


def encrypt_secret(plaintext: str) -> Optional[str]:
    """Encrypt a secret to an opaque token, or None if encryption is unavailable."""
    f = _fernet()
    if f is None or plaintext is None:
        return None
    return f.encrypt(str(plaintext).encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> Optional[str]:
    """Decrypt a token back to plaintext, or None on failure / bad token."""
    f = _fernet()
    if f is None or not token:
        return None
    try:
        return f.decrypt(str(token).encode("ascii")).decode("utf-8")
    except Exception:
        return None
