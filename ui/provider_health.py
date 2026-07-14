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


@st.cache_data(ttl=120, show_spinner=False)
def _probe_earnings_source(which: str) -> tuple[bool, Optional[int], Optional[float]]:
    """Probe FMP/Finnhub with a small live window. Returns (configured, rows, ms).

    Cached 2 min so opening the panel repeatedly doesn't burn the (rate-limited)
    free-tier API quotas.
    """
    import datetime as dt

    try:
        from data.earnings_sources import (
            _secret,
            fetch_earnings_window_finnhub,
            fetch_earnings_window_fmp,
        )
    except Exception:
        return (False, None, None)

    keyname = "FMP_API_KEY" if which == "fmp" else "FINNHUB_API_KEY"
    if not _secret(keyname):
        return (False, None, None)

    start = dt.date.today().isoformat()
    end = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    fn = fetch_earnings_window_fmp if which == "fmp" else fetch_earnings_window_finnhub
    t0 = time.perf_counter()
    try:
        m = fn(start, end)
        return (True, len(m), (time.perf_counter() - t0) * 1000.0)
    except Exception:
        return (True, None, None)


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


def _render_ranking_ab() -> None:
    """Admin view: breakout vs prebreakout excess-vs-SPY per horizon (grouped bars).

    The most decision-relevant internal number — previously visible only in
    cron logs. Fixed hue per ranking (identity), zero line for polarity.
    """
    try:
        import plotly.graph_objects as go

        from db.track_record import load_latest_track_record

        horizons = (1, 5, 20)
        series = {"breakout": [], "prebreakout": []}
        labels = []
        for h in horizons:
            row_b = load_latest_track_record(h, ranking="breakout")
            row_p = load_latest_track_record(h, ranking="prebreakout")
            if not row_b and not row_p:
                continue
            labels.append(f"{h}d")
            series["breakout"].append(
                (row_b or {}).get("avg_return") if row_b else None
            )
            series["prebreakout"].append(
                (row_p or {}).get("avg_return") if row_p else None
            )
        if not labels:
            return
        st.markdown("#### 🥊 Ranking A/B — excess vs SPY (top-5)")
        fig = go.Figure()
        for name, color in (("breakout", "#94a3b8"), ("prebreakout", "#60a5fa")):
            vals = [v * 100 if v is not None else None for v in series[name]]
            fig.add_trace(
                go.Bar(
                    x=labels, y=vals, name=name, marker_color=color,
                    hovertemplate=name + " %{x}: %{y:+.2f}%<extra></extra>",
                )
            )
        fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
        fig.update_layout(
            barmode="group", height=220, margin=dict(l=0, r=0, t=8, b=0),
            legend=dict(orientation="h", y=1.1),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)", ticksuffix="%"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
        st.caption("Small samples early on — direction matters more than magnitude.")
    except Exception:
        pass


def _render_earnings_debug() -> None:
    """One-click earnings-calendar write test (moved here from the scan flow)."""
    with st.expander("🧪 Earnings Calendar Debug", expanded=False):
        st.caption("Runs a small earnings refresh and shows the returned dates (best-effort).")
        if st.button(
            "Fetch earnings for AAPL / MSFT / TSLA",
            key="btn_earnings_debug",
            width="stretch",
        ):
            try:
                from db.earnings import populate_earnings_calendar

                with st.spinner("Fetching earnings..."):
                    result = populate_earnings_calendar(["AAPL", "MSFT", "TSLA"])
                st.success("Earnings fetch attempted.")
                st.write(result)
                st.caption("If all dates are None, this is usually a network/API-key issue.")
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                st.error(f"Earnings debug failed: {e}")


def render_provider_health() -> None:
    """Render the admin provider-health dashboard. Never raises."""
    try:
        st.markdown("## 🩺 Provider Health")
        st.caption(
            "Live status of the data sources and infrastructure the app depends "
            "on. 🛠️ Admin override active: universe caps are disabled."
        )
        _render_ranking_ab()
        _render_earnings_debug()

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

        # --- Earnings sources row (FMP + Finnhub, 30-day probe) ---
        fmp_conf, fmp_rows, fmp_ms = _probe_earnings_source("fmp")
        fh_conf, fh_rows, fh_ms = _probe_earnings_source("finnhub")

        def _src_metric(label, conf, rows, ms):
            if not conf:
                st.metric(label, "⚪ not configured", "—")
                return
            icon = "🟢" if (rows or 0) > 0 else "🟡"
            st.metric(
                label,
                f"{icon} {rows if rows is not None else 'error'} rows",
                f"{ms:.0f} ms (30d window)" if ms is not None else "—",
            )

        c6, c7 = st.columns(2)
        with c6:
            _src_metric("FMP (earnings)", fmp_conf, fmp_rows, fmp_ms)
        with c7:
            _src_metric("Finnhub (earnings)", fh_conf, fh_rows, fh_ms)

        # --- Earnings cache + last scan row ---
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
