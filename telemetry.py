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


def send_slack_alert(message: str) -> bool:
    """POST a message to SLACK_WEBHOOK_URL. Returns True on success."""
    try:
        from config import SLACK_WEBHOOK_URL
        if not SLACK_WEBHOOK_URL:
            return False
        import json
        import urllib.request
        payload = json.dumps({"text": message}).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_and_alert_scan_errors() -> None:
    """Count recent scan_errors and fire a Slack/email alert if above threshold.

    Called by the nightly scheduler. Best-effort; never raises.
    """
    try:
        from config import (
            ALERT_EMAIL,
            SCAN_ERROR_ALERT_THRESHOLD,
            SCAN_ERROR_ALERT_WINDOW_MINUTES,
        )
        from db.engine import get_neon_conn
        from db.schema import ensure_neon_scan_errors_schema

        conn = get_neon_conn()
        if conn is None:
            return
        ensure_neon_scan_errors_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*), MAX(occurred_at)
            FROM scan_errors
            WHERE occurred_at > NOW() - INTERVAL '%s minutes'
              AND error_type != 'ProviderWarning'
            """,
            (SCAN_ERROR_ALERT_WINDOW_MINUTES,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        count = int(row[0]) if row else 0
        last_at = row[1] if row else None

        if count < SCAN_ERROR_ALERT_THRESHOLD:
            return

        msg = (
            f":rotating_light: *AI Scanner alert*: {count} scan errors in the last "
            f"{SCAN_ERROR_ALERT_WINDOW_MINUTES} minutes (threshold: {SCAN_ERROR_ALERT_THRESHOLD}). "
            f"Last error at {last_at}. Check the Admin > Diagnostics panel."
        )
        print(f"[telemetry] {msg}")
        sent = send_slack_alert(msg)

        if not sent and ALERT_EMAIL:
            try:
                from ui.email_utils import send_alert_email
                send_alert_email(
                    to_address=ALERT_EMAIL,
                    subject=f"AI Scanner: {count} scan errors in {SCAN_ERROR_ALERT_WINDOW_MINUTES}m",
                    body=msg.replace(":rotating_light: ", "").replace("*", ""),
                )
            except Exception:
                pass
    except Exception as e:
        print(f"[telemetry] check_and_alert_scan_errors failed: {e}")


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