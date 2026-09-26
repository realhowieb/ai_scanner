"""Run 63 — the latest scheduled full-market scan as the Scanner's default view.

Until a user runs their own scan in a session, the Scanner shows the newest
scheduled full-market run (username "cron", label "US_MARKET") instead of an
empty table. Read-only: it loads what production already saved through the
existing `db.runs.list_runs` / `load_run_results`; nothing about what the
scheduler saves, scores or ranks changes. A session scan always takes
precedence and is never overwritten.

`st.session_state[MARKET_VIEW_KEY]` holds the shown run's metadata so the
results tab can label the view and show the scan's own time (not "now").
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

MARKET_VIEW_KEY = "hsf_market_view"
REPLACE_HINT = "Run your own scan above to replace this view."


def pick_market_run(runs: Optional[Sequence[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Newest scheduled full-market run (same rule as the trust banner)."""
    from ui.trust_banner import latest_market_scan

    return latest_market_scan(runs)


def market_view_caption(meta: Mapping[str, Any], now: Optional[_dt.datetime] = None) -> str:
    """'Latest full-market scan · Fri Sep 25, 3:35 PM ET (2 h ago) · 100 ranked setups. …'"""
    from ui.trust_banner import _fmt_age, _fmt_et, _parse_ts

    now = now or _dt.datetime.now(_dt.timezone.utc)
    parts = ["Latest full-market scan"]
    ts = _parse_ts(meta.get("created_at"))
    if ts is not None:
        parts.append(f"{_fmt_et(ts, now)} ({_fmt_age(ts, now)})")
    rows = meta.get("rows")
    if isinstance(rows, int) and rows > 0:
        parts.append(f"{rows} ranked setups")
    return " · ".join(parts) + ". " + REPLACE_HINT


def _load_latest_market_results() -> Optional[Tuple[Any, Dict[str, Any]]]:
    from db.runs import list_runs, load_run_results
    from ui.app_runtime import normalize_results_to_df

    run = pick_market_run(list_runs(limit=25, include_snapshots=True, username="cron") or [])
    if run is None or run.get("id") is None:
        return None
    df = normalize_results_to_df(load_run_results(int(run["id"])))
    if df is None or df.empty:
        return None
    ts = run.get("created_at")
    meta = {"run_id": int(run["id"]), "created_at": ts.isoformat() if hasattr(ts, "isoformat") else ts,
            "rows": int(len(df))}
    return df, meta


if st is not None:
    _load_cached = st.cache_data(ttl=120, show_spinner=False)(_load_latest_market_results)
else:  # pragma: no cover
    _load_cached = _load_latest_market_results


def default_results(session_df: Any) -> Any:
    """The user's session scan if there is one, else the latest market scan.

    Returns None when neither exists (the tab then shows its empty state).
    Never raises.
    """
    if st is None:
        return session_df
    if session_df is not None:
        st.session_state.pop(MARKET_VIEW_KEY, None)
        return session_df
    try:
        loaded = _load_cached()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("latest market scan", exc)
        loaded = None
    if not loaded:
        st.session_state.pop(MARKET_VIEW_KEY, None)
        return None
    df, meta = loaded
    st.session_state[MARKET_VIEW_KEY] = dict(meta)
    return df.copy()   # downstream enrichment mutates in place; keep the cache clean
