"""P2 — small per-browser preferences (tour done, saved screens).

Stored in the encrypted cookie jar sign-in already uses (ui.auth_sessions), so
no database schema is needed. Values are short strings; JSON helpers keep
structured values small. Preferences follow the browser, not the account.
Every call is best-effort and never raises.
"""
from __future__ import annotations

import json
from typing import Any, Optional

MAX_VALUE_CHARS = 2000   # keep cookies well under browser size limits


def _jar():
    from ui.auth_sessions import cookies_ready_or_stop

    return cookies_ready_or_stop()


def get(key: str) -> Optional[str]:
    try:
        jar = _jar()
        v = jar.get(key) if jar is not None else None
        return None if v in (None, "") else str(v)
    except Exception:
        return None


def put(key: str, value: str) -> bool:
    """Store a value; False if it is too large or cookies are unavailable."""
    value = str(value)
    if len(value) > MAX_VALUE_CHARS:
        return False
    try:
        from ui.auth_sessions import save_cookies

        jar = _jar()
        if jar is None:
            return False
        jar[key] = value
        save_cookies(jar)
        return True
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("browser preference", exc)
        return False


def get_json(key: str, default: Any = None) -> Any:
    raw = get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def put_json(key: str, value: Any) -> bool:
    return put(key, json.dumps(value, separators=(",", ":")))
