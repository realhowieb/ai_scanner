"""Admin-only provider/health diagnostics dashboard.

Shows live status for the data + infra the app depends on: database, Alpaca,
earnings refresh, the last scheduled scan, and measured provider latency. Each
section is independently guarded so one failing probe never breaks the page (or
the app). Admin-gated by the caller.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st


def _neon_conn():
    try:
        from db.engine import get_neon_conn

        return get_neon_conn()
    except Exception:
        return None


def _fmt_ago(ts) -> str:
    if ts is None:
        return "never"
    try:
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        t = ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
        secs = (now - t).total_seconds()
        if secs < 90:
            return f"{int(secs)}s ago"
        if secs < 5400:
            return f"{int(secs // 60)}m ago"
        if secs < 172800:
            return f"{int(secs // 3600)}h ago"
        return f"{int(secs // 86400)}d ago"
    except Exception:
        return str(ts)


def _probe_database() -> tuple[str, Optional[float]]:
    """Return (status, latency_ms). status is 'neon'/'sqlite'/'none'/'error'."""
    try:
        from db.engine import get_db_status

        t0 = time.perf_counter()
        status = get_db_status()
        return str(status), (time.perf_counter() - t0) * 1000.0
    except Exception:
        return "error", None


def _probe_alpaca() -> tuple[bool, bool, Optional[float]]:
    """Return (configured, working, latency_ms) via a live AAPL quote."""
    configured = False
    try:
        from data.price_alpaca import get_alpaca_config

        configured = get_alpaca_config() is not None
    except Exception:
        configured = False
    if not configured:
        return False, False, None
    try:
        from market_data import get_latest_quotes

        t0 = time.perf_counter()
        quotes = get_latest_quotes(["AAPL"]) or {}
        ms = (time.perf_counter() - t0) * 1000.0
        info = quotes.get("AAPL")
        working = isinstance(info, dict) and info.get("last") is not None
        return True, bool(working), ms
    except Exception:
        return True, False, None


def _earnings_status() -> dict:
    out = {"last_refresh": None, "dated_rows": None}
    conn = _neon_conn()
    if conn is None:
        return out
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT refreshed_at FROM earnings_refresh_log WHERE refresh_key = %s",
                    ("cron_earnings",),
                )
                row = cur.fetchone()
                if row:
                    out["last_refresh"] = row[0] if not isinstance(row, dict) else row.get("refreshed_at")
            except Exception:
                pass
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM earnings_calendar WHERE earnings_date IS NOT NULL"
                )
                row = cur.fetchone()
                if row:
                    out["dated_rows"] = row[0] if not isinstance(row, dict) else list(row.values())[0]
            except Exception:
                pass
    except Exception:
        pass
    return out


def _last_scan() -> Optional[dict]:
    try:
        from db.runs import list_runs

        runs = list_runs(limit=10) or []
        return runs[0] if runs else None
    except Exception:
        return None


def render_provider_health() -> None:
    """Render the admin provider-health dashboard. Never raises."""
    try:
        st.markdown("## 🩺 Provider Health")
        st.caption("Live status of the data sources and infrastructure the app depends on.")

        # --- Database + Alpaca latency row ---
        db_status, db_ms = _probe_database()
        alp_conf, alp_ok, alp_ms = _probe_alpaca()

        c1, c2, c3 = st.columns(3)
        with c1:
            ok = db_status in ("neon", "sqlite")
            st.metric(
                "Database",
                f"{'🟢' if ok else '🔴'} {db_status}",
                f"{db_ms:.0f} ms" if db_ms is not None else "—",
            )
        with c2:
            if not alp_conf:
                st.metric("Alpaca", "⚪ not configured", "—")
            else:
                st.metric(
                    "Alpaca",
                    f"{'🟢 ok' if alp_ok else '🔴 error'}",
                    f"{alp_ms:.0f} ms" if alp_ms is not None else "—",
                )
        with c3:
            # Provider latency summary (live-measured above).
            lat = alp_ms if alp_ms is not None else db_ms
            st.metric("Provider latency (live)", f"{lat:.0f} ms" if lat is not None else "—")

        # --- Earnings + last scan row ---
        earn = _earnings_status()
        last = _last_scan()

        c4, c5 = st.columns(2)
        with c4:
            st.metric(
                "Earnings refresh",
                _fmt_ago(earn.get("last_refresh")),
                f"{earn.get('dated_rows')} dated symbols" if earn.get("dated_rows") is not None else "—",
            )
        with c5:
            if last:
                st.metric(
                    "Last scheduled scan",
                    _fmt_ago(last.get("created_at")),
                    f"{last.get('label') or last.get('name') or ''} · {last.get('row_count', '?')} rows",
                )
            else:
                st.metric("Last scheduled scan", "none found", "—")

        # --- Not-yet-instrumented ---
        st.caption(
            "ℹ️ **Cache hit rate** is not instrumented yet — adding lightweight "
            "counters to the price cache is a post-launch item (kept out of the "
            "hot path for now)."
        )
    except Exception as e:  # the whole panel must never break the app
        try:
            st.caption(f"Provider health unavailable: {type(e).__name__}: {e}")
        except Exception:
            pass
