"""Admin console for the web app: accounts, health and maintenance tools.

The same data and actions as the Streamlit Admin page (ui/admin_results_tab.py,
ui/admin_users.py, ui/provider_health.py, ui/email_health_view.py), without
Streamlit. Account changes go through db.user_admin, which writes the admin
audit log. Every function here is called only after require_admin.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PLANS = ("basic", "pro", "premium")
PROBE_TTL_S = 120  # FMP/Finnhub free tiers are rate-limited; reuse a probe for 2 minutes


class AdminError(ValueError):
    """A refused admin action; the message is shown to the admin as is."""


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return None if value is None else str(value)


def _forget(username: str) -> None:
    from api.deps import forget_account

    forget_account(username)


def _done(result: Tuple[bool, str], username: str = "") -> Dict[str, Any]:
    ok, message = result
    if not ok:
        raise AdminError(message)
    if username:
        _forget(username)
    return {"ok": True, "message": message}


# ---- accounts ------------------------------------------------------------------------------------
def users() -> List[Dict[str, Any]]:
    from db.user_admin import fetch_users_for_admin

    return [{"username": str(r.get("username") or ""), "full_name": r.get("full_name"),
             "tier": str(r.get("tier") or "basic").strip().lower(), "is_active": bool(r.get("is_active")),
             "is_admin": bool(r.get("is_admin")), "email_verified": bool(r.get("email_verified")),
             "created_at": _iso(r.get("created_at"))} for r in fetch_users_for_admin()]


def create_user(actor: str, email: str, full_name: str, password: str, tier: str, active: bool) -> Dict[str, Any]:
    from db.user_admin import create_user as create

    return _done(create(email, full_name, password, tier=tier, active=active, actor=actor))


def update_user(actor: str, username: str, tier: Optional[str], is_active: Optional[bool]) -> Dict[str, Any]:
    """Change the plan and/or the active flag, like the Streamlit "Update User" button."""
    from db.admin_audit import record_admin_event
    from db.user_admin import _update, normalize_email

    u = normalize_email(username)
    current = next((r for r in users() if r["username"].strip().lower() == u), None)
    if current is None:
        raise AdminError("No such account.")
    if tier is not None and tier not in PLANS:
        raise AdminError("Choose a plan: basic, pro or premium.")
    if is_active is False and u == normalize_email(actor):
        raise AdminError("You can't deactivate your own account.")
    changes: Dict[str, Any] = {}
    if tier is not None and tier != current["tier"]:
        changes["tier"] = [current["tier"], tier]
    if is_active is not None and is_active != current["is_active"]:
        changes["active"] = [current["is_active"], is_active]
    if not changes:
        return {"ok": True, "message": "Nothing to change."}
    new_tier = tier if "tier" in changes else current["tier"]
    new_active = is_active if "active" in changes else current["is_active"]
    if not _update("UPDATE users SET tier = %s, is_active = %s WHERE lower(username) = %s", (new_tier, new_active, u)):
        raise AdminError("Couldn't update the account.")
    record_admin_event(actor, "update_user", u, changes)
    _forget(u)
    return {"ok": True, "message": f"Updated {u}."}


def mark_verified(actor: str, username: str) -> Dict[str, Any]:
    from db.user_admin import mark_email_verified

    return _done(mark_email_verified(username, actor=actor), username)


def resend_verification(username: str) -> Dict[str, Any]:
    from api.account import send_verification
    from db.user_admin import is_valid_email, normalize_email

    u = normalize_email(username)
    if not is_valid_email(u):
        raise AdminError("This username isn't an email address, so the account can't receive email.")
    if not send_verification(u):
        raise AdminError("Couldn't send the email. Check the email setup (Resend test mode only "
                         "delivers to the Resend account owner).")
    return {"ok": True, "message": f"Verification email sent to {u}."}


def set_admin(actor: str, username: str, make_admin: bool) -> Dict[str, Any]:
    from db.user_admin import set_admin as set_role

    return _done(set_role(username, make_admin, acting_user=actor), username)


def audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    from db.admin_audit import describe, recent_admin_events

    return [{"at": _iso(e["at"]), "actor": e["actor"], "action": str(e["action"]).replace("_", " "),
             "target": e["target"], "details": describe(e["detail"])} for e in recent_admin_events(limit=limit)]


# ---- health --------------------------------------------------------------------------------------
def system_health() -> Dict[str, Any]:
    """The latest System Health snapshot, trimmed to what the Streamlit panel shows."""
    from analytics import system_health as sh
    from api.operations import get_summary, load_snapshot

    report = load_snapshot()["report"] or {}
    subsystems = []
    for name in getattr(sh, "SUBSYSTEMS", ()):
        s = (report.get("subsystems") or {}).get(name)
        if isinstance(s, dict):
            try:
                label = sh.subsystem_label(s)
            except (KeyError, TypeError):
                label = s.get("status")
            subsystems.append({"name": sh._TITLES.get(name, name), "status": s.get("status"),
                               "label": label, "reason": s.get("reason")})
    incidents = [{"severity": i.get("severity"), "id": i.get("incident_id"), "age_hours": i.get("age_hours"),
                  "summary": i.get("summary"), "recommended_action": i.get("recommended_action")}
                 for i in report.get("incidents") or [] if isinstance(i, dict)]
    return {"operations": get_summary(), "health_score": report.get("health_score"),
            "human_action": report.get("human_action"),
            "next_expected_scan": (report.get("market") or {}).get("next_expected_scan"),
            "subsystems": subsystems, "incidents": incidents}


def email_health() -> Dict[str, Any]:
    from analytics.email_health import evaluate
    from db.email_job_runs import recent_email_runs

    result = evaluate(recent_email_runs(days=5))
    return {"status": result["status"], "day": _iso(result.get("day")),
            "jobs": [{"job": job, "label": j.get("label"), "status": j.get("status"), "detail": j.get("detail")}
                     for job, j in (result.get("jobs") or {}).items()]}


_probe_cache: Dict[str, Tuple[float, Any]] = {}


def _cached_probe(key: str, fn: Callable[[], Any]) -> Any:
    hit = _probe_cache.get(key)
    if hit and time.monotonic() - hit[0] < PROBE_TTL_S:
        return hit[1]
    value = fn()
    _probe_cache[key] = (time.monotonic(), value)
    return value


def _timed(fn: Callable[[], Any]) -> Tuple[Any, Optional[float]]:
    t0 = time.perf_counter()
    value = fn()
    return value, round((time.perf_counter() - t0) * 1000.0, 1)


def _probe_database() -> Dict[str, Any]:
    try:
        from db.engine import get_db_status

        status, ms = _timed(get_db_status)
        status = str(status)
    except Exception:
        status, ms = "error", None
    return {"name": "Database", "status": "ok" if status == "neon" else "error",
            "detail": status, "ms": ms}


def _probe_alpaca() -> Dict[str, Any]:
    try:
        from data.price_alpaca import get_alpaca_config

        configured = get_alpaca_config() is not None
    except Exception:
        configured = False
    if not configured:
        return {"name": "Alpaca", "status": "not_configured", "detail": "No API keys", "ms": None}
    try:
        from market_data import get_latest_quotes

        quotes, ms = _timed(lambda: get_latest_quotes(["AAPL"]) or {})
        info = quotes.get("AAPL")
        ok = isinstance(info, dict) and info.get("last") is not None
        return {"name": "Alpaca", "status": "ok" if ok else "error",
                "detail": "AAPL quote" if ok else "No AAPL quote", "ms": ms}
    except Exception as exc:
        return {"name": "Alpaca", "status": "error", "detail": type(exc).__name__, "ms": None}


def _probe_earnings(which: str) -> Dict[str, Any]:
    label = "FMP (earnings)" if which == "fmp" else "Finnhub (earnings)"
    try:
        from data.earnings_sources import _secret, fetch_earnings_window_finnhub, fetch_earnings_window_fmp
    except Exception:
        return {"name": label, "status": "error", "detail": "Unavailable", "ms": None}
    if not _secret("FMP_API_KEY" if which == "fmp" else "FINNHUB_API_KEY"):
        return {"name": label, "status": "not_configured", "detail": "No API key", "ms": None}
    fn = fetch_earnings_window_fmp if which == "fmp" else fetch_earnings_window_finnhub
    start = dt.date.today()
    try:
        rows, ms = _timed(lambda: fn(start.isoformat(), (start + dt.timedelta(days=30)).isoformat()))
    except Exception as exc:
        return {"name": label, "status": "error", "detail": type(exc).__name__, "ms": None}
    n = len(rows or {})
    return {"name": label, "status": "ok" if n else "warn", "detail": f"{n} rows in the next 30 days", "ms": ms}


def _probe_billing() -> Dict[str, Any]:
    import requests

    base = (os.getenv("BILLING_API_BASE") or "https://ai-scanner-h2c8.onrender.com").strip().rstrip("/")
    try:
        r, ms = _timed(lambda: requests.get(f"{base}/health", timeout=float(os.getenv("BILLING_HEALTH_TIMEOUT", "5"))))
        ok = r.status_code == 200
        return {"name": "Billing service", "status": "ok" if ok else "error",
                "detail": base if ok else f"{base}/health returned {r.status_code}", "ms": ms}
    except Exception:
        return {"name": "Billing service", "status": "error", "detail": f"{base}/health did not respond", "ms": None}


def _earnings_cache() -> Dict[str, Any]:
    out: Dict[str, Any] = {"last_refresh": None, "dated_rows": None}
    try:
        from db.engine import get_neon_conn

        conn = get_neon_conn()
    except Exception:
        conn = None
    if conn is None:
        return out
    try:
        cur = conn.cursor()
        for sql, params, key in (
            ("SELECT refreshed_at FROM earnings_refresh_log WHERE refresh_key = %s", ("cron_earnings",), "last_refresh"),
            ("SELECT COUNT(*) FROM earnings_calendar WHERE earnings_date IS NOT NULL", (), "dated_rows"),
        ):
            try:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row:
                    out[key] = list(row.values())[0] if isinstance(row, dict) else row[0]
            except Exception:
                conn.rollback()
        cur.close()
    finally:
        conn.close()
    return {"last_refresh": _iso(out["last_refresh"]),
            "dated_rows": int(out["dated_rows"]) if out["dated_rows"] is not None else None}


def providers() -> Dict[str, Any]:
    """Live checks of the services HSF depends on. The earnings APIs are cached 2 minutes."""
    checks = [_probe_database(), _probe_alpaca(),
              _cached_probe("fmp", lambda: _probe_earnings("fmp")),
              _cached_probe("finnhub", lambda: _probe_earnings("finnhub")),
              _probe_billing()]
    return {"checked_at": _iso(dt.datetime.now(dt.timezone.utc)), "checks": checks,
            "earnings_cache": _earnings_cache()}


# ---- maintenance tools ---------------------------------------------------------------------------
def _query(sql: str, params: tuple = (), ensure: Optional[Callable[[Any], None]] = None) -> List[tuple]:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        raise AdminError("Database unavailable.")
    try:
        if ensure is not None:
            ensure(conn)
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        cur.close()
        return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]
    finally:
        conn.close()


def scan_errors(limit: int = 50) -> List[Dict[str, Any]]:
    from db.schema import ensure_neon_scan_errors_schema

    rows = _query("SELECT occurred_at, context, username, ticker_count, error_type, message FROM scan_errors "
                  "ORDER BY occurred_at DESC LIMIT %s", (int(limit),), ensure_neon_scan_errors_schema)
    return [{"at": _iso(r[0]), "context": r[1], "username": r[2], "tickers": r[3], "error_type": r[4],
             "message": str(r[5] or "")[:500]} for r in rows]


def login_attempts(limit: int = 100) -> List[Dict[str, Any]]:
    """Recent sign-in attempts (the cron keeps only the last 24 hours)."""
    from db.schema import ensure_neon_login_attempts_schema

    rows = _query("SELECT attempted_at, username, success, ip_address FROM login_attempts "
                  "ORDER BY attempted_at DESC LIMIT %s", (int(limit),), ensure_neon_login_attempts_schema)
    return [{"at": _iso(r[0]), "username": r[1], "success": bool(r[2]),
             "ip_address": None if r[3] is None else str(r[3])} for r in rows]


def ai_usage(days: int = 30) -> List[Dict[str, Any]]:
    from db.ai_usage import feature_usage_counts

    return [{"feature": str(f), "calls": int(n)} for f, n in feature_usage_counts(days)]


def db_check() -> Dict[str, Any]:
    """Database reachability and earnings-table hygiene, like the Streamlit "DB integrity checks"."""
    from db.engine import get_neon_conn

    issues: List[str] = []
    out: Dict[str, Any] = {"connected": False, "earnings_rows": None, "bad_symbols": None, "issues": issues}
    conn = get_neon_conn()
    if conn is None:
        issues.append("Database unavailable.")
        return out
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        out["connected"] = True
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'earnings_calendar'")
        row = cur.fetchone()
        if not (list(row.values())[0] if isinstance(row, dict) else row[0]):
            issues.append("The earnings_calendar table is missing.")
        else:
            def one(sql: str) -> int:
                cur.execute(sql)
                r = cur.fetchone()
                return int(list(r.values())[0] if isinstance(r, dict) else r[0])

            out["earnings_rows"] = one("SELECT COUNT(*) FROM earnings_calendar")
            out["bad_symbols"] = one("SELECT COUNT(*) FROM earnings_calendar WHERE symbol IS NULL "
                                     "OR symbol <> UPPER(TRIM(symbol)) OR symbol = ''")
        cur.close()
    except Exception as exc:
        issues.append(f"Check failed: {type(exc).__name__}")
    finally:
        conn.close()
    if out["bad_symbols"]:
        try:
            from db.earnings import ensure_earnings_table

            ensure_earnings_table()
            issues.append(f"Found {out['bad_symbols']} non-normalized symbols; ran the normalization update.")
        except Exception:
            issues.append(f"Found {out['bad_symbols']} non-normalized symbols; normalization unavailable.")
    return out


_earnings_lock = threading.Lock()
_earnings_job: Dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None,
                                 "started_by": None, "symbols": None, "dated": None, "error": None}


def earnings_refresh_status() -> Dict[str, Any]:
    with _earnings_lock:
        return dict(_earnings_job)


def _universe() -> List[str]:
    seen: Dict[str, None] = {}
    for name in ("sp500.txt", "nasdaq.txt"):
        try:
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
                sym = line.strip().upper()
                if sym:
                    seen.setdefault(sym, None)
        except (OSError, UnicodeError):
            continue
    return list(seen)


def _run_earnings_refresh(symbols: List[str]) -> None:
    dated, error = None, None
    try:
        from db.earnings import mark_earnings_refreshed_today, populate_earnings_calendar

        result = populate_earnings_calendar(symbols, use_yf_fallback=False, sleep_s=0)
        dated = sum(1 for info in (result or {}).values() if getattr(info, "earnings_date", None) is not None)
        if dated:
            mark_earnings_refreshed_today("cron_earnings")
    except Exception as exc:
        error = type(exc).__name__
    with _earnings_lock:
        _earnings_job.update(state="failed" if error else "done", dated=dated, error=error,
                             finished_at=_iso(dt.datetime.now(dt.timezone.utc)))


def start_earnings_refresh(actor: str) -> Dict[str, Any]:
    """Refresh the earnings calendar now (same FMP -> Finnhub bulk path as the daily cron), in the background."""
    with _earnings_lock:
        if _earnings_job["state"] == "running":
            return dict(_earnings_job)
        symbols = _universe()
        if not symbols:
            raise AdminError("No symbols available (sp500.txt and nasdaq.txt are missing).")
        _earnings_job.update(state="running", started_at=_iso(dt.datetime.now(dt.timezone.utc)), finished_at=None,
                             started_by=actor, symbols=len(symbols), dated=None, error=None)
        snapshot = dict(_earnings_job)
    threading.Thread(target=_run_earnings_refresh, args=(symbols,), name="admin-earnings-refresh", daemon=True).start()
    try:
        from db.admin_audit import record_admin_event

        record_admin_event(actor, "refresh_earnings", "earnings_calendar", {"symbols": len(symbols)})
    except Exception:
        pass
    return snapshot
