# ai_scanner/telemetry.py
from __future__ import annotations

import time
import traceback
from functools import wraps
from typing import Any, Callable


def timed(save_cb: Callable[[float], None]):
    """Decorator to measure and save execution time of a function."""
    def deco(fn: Callable[..., Any]):
        @wraps(fn)
        def wrapper(*a, **kw) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed = time.perf_counter() - t0
                print(f"Execution time of {fn.__name__}: {elapsed:.6f} seconds")
                save_cb(elapsed)
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Structured scan error logging
# ---------------------------------------------------------------------------

def log_scan_error(
    error: BaseException,
    *,
    context: str = "",
    username: str | None = None,
    universe: str | None = None,
    ticker_count: int | None = None,
) -> None:
    """Persist a scan failure to the DB and emit a console log.

    Best-effort: never raises so it cannot block the scan path.
    """
    tb = traceback.format_exc()
    error_type = type(error).__name__
    message = str(error)[:500]

    # Always emit to stdout so cloud log aggregators capture it.
    print(
        f"[SCAN ERROR] context={context!r} user={username!r} universe={universe!r} "
        f"tickers={ticker_count} error={error_type}: {message}"
    )

    try:
        from db.engine import get_neon_conn
        from db.schema import ensure_neon_scan_errors_schema
        conn = get_neon_conn()
        if conn is None:
            return
        ensure_neon_scan_errors_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scan_errors (context, username, universe, ticker_count, error_type, message, traceback)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (context or "", username, universe, ticker_count, error_type, message, tb[:4000]),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def log_provider_warning(provider: str, message: str, *, username: str | None = None) -> None:
    """Log a non-fatal provider issue (rate limit, timeout, partial data)."""
    print(f"[PROVIDER WARNING] provider={provider!r} user={username!r} msg={message!r}")
    try:
        from db.engine import get_neon_conn
        from db.schema import ensure_neon_scan_errors_schema
        conn = get_neon_conn()
        if conn is None:
            return
        ensure_neon_scan_errors_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scan_errors (context, username, error_type, message)
            VALUES (%s, %s, %s, %s)
            """,
            (f"provider:{provider}", username, "ProviderWarning", message[:500]),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass