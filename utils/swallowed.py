"""Leave a trace when best-effort code ignores an error.

Many optional steps (an audit row, a cache warm, a follow-up email) are wrapped in
`try/except` so a failure never breaks the main action. Ignoring the error silently
also hides real bugs, so those handlers call `swallowed(where, exc)`: one WARNING
per call site every few minutes, with the error type and a short redacted message.
Behaviour is otherwise unchanged, and this function never raises.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, Optional, Tuple

log = logging.getLogger("hsf.swallowed")

INTERVAL_S = 300.0
_MAX_SITES = 2000
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_CREDS_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_lock = threading.Lock()
_seen: Dict[str, Tuple[float, int]] = {}   # where -> (last logged at, ignored since then)


def _clean(exc: Optional[BaseException]) -> str:
    text = " ".join(str(exc).split())[:200] if exc is not None else ""
    text = _URL_CREDS_RE.sub("://***@", text)
    return _EMAIL_RE.sub("<email>", text)


def swallowed(where: str, exc: Optional[BaseException] = None) -> None:
    """Note that `where` ignored `exc`. Logs at most once per INTERVAL_S per call site."""
    try:
        now = time.monotonic()
        with _lock:
            last, skipped = _seen.get(where, (0.0, 0))
            if last and now - last < INTERVAL_S:
                _seen[where] = (last, skipped + 1)
                return
            if len(_seen) >= _MAX_SITES:
                _seen.clear()
            _seen[where] = (now, 0)
        more = f" (+{skipped} more since last note)" if skipped else ""
        log.warning("ignored error in %s: %s: %s%s", where, type(exc).__name__ if exc is not None else "error",
                    _clean(exc), more)
    except Exception:  # noqa: BLE001,S110 - the note itself must never fail the caller
        pass


def reset() -> None:
    """For tests."""
    with _lock:
        _seen.clear()
