"""P1 shared layer — read-only access to recent scheduled full-market scans.

Used by the lens bar, "new since your last visit", the end-of-day recap and
the Today page. Reads what production already saved (`runs` rows written by
the scheduler: username "cron", label "US_MARKET") through the existing
`db.runs` helpers. Never writes, never re-scores: HSF Scores come from the
canonical `consolidate_scanner_results` path.

Pure helpers (`market_runs`, `runs_on_day`, `tickers_of`, `diff_tickers`,
`top_setups`) take plain data so they are testable without a database.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from analytics import market_calendar as mc

MARKET_USER = "cron"
MARKET_LABEL = "US_MARKET"


def _ts(v: Any) -> Optional[_dt.datetime]:
    from ui.trust_banner import _parse_ts

    return _parse_ts(v)


def market_runs(runs: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Scheduled full-market runs, newest first, with parsed timestamps."""
    out = []
    for r in runs or []:
        if str(r.get("username") or "").lower() != MARKET_USER:
            continue
        if str(r.get("label") or "").upper() != MARKET_LABEL or r.get("id") is None:
            continue
        ts = _ts(r.get("created_at"))
        if ts is not None:
            out.append({**r, "created_at": ts})
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def runs_on_day(runs: Sequence[Dict[str, Any]], day: _dt.date) -> List[Dict[str, Any]]:
    """Runs whose ET date is `day`, newest first."""
    return [r for r in runs if r["created_at"].astimezone(mc.ET).date() == day]


def tickers_of(df: Any) -> List[str]:
    """Upper-cased tickers of a results frame, in row order (deduplicated)."""
    if df is None or getattr(df, "empty", True):
        return []
    col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
    if col is None:
        return []
    seen, out = set(), []
    for t in df[col]:
        k = str(t or "").strip().upper()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def diff_tickers(before: Iterable[str], after: Iterable[str]) -> Dict[str, List[str]]:
    """Which names entered / left the ranked list between two scans (after-order kept)."""
    b, a = list(before), list(after)
    bs, as_ = set(b), set(a)
    return {"entered": [t for t in a if t not in bs], "left": [t for t in b if t not in as_]}


def top_setups(df: Any, n: int = 5) -> List[Dict[str, Any]]:
    """Top-n HSF opportunities in a scan via the canonical HSF Score ranking."""
    if df is None or getattr(df, "empty", True):
        return []
    from ui.results_intelligence import consolidate_scanner_results

    return consolidate_scanner_results(df.to_dict(orient="records"), top_n=n)


# ---- cached loaders (read-only) ------------------------------------------------------------------
def _list_runs_uncached() -> List[Dict[str, Any]]:
    from db.runs import list_runs

    return market_runs(list_runs(limit=60, include_snapshots=True, username=MARKET_USER) or [])


def _run_df_uncached(run_id: int) -> Any:
    from db.runs import load_run_results
    from ui.app_runtime import normalize_results_to_df

    return normalize_results_to_df(load_run_results(int(run_id)))


if st is not None:
    recent_market_runs = st.cache_data(ttl=120, show_spinner=False)(_list_runs_uncached)
    # A saved run never changes, so its results can be cached longer.
    run_df = st.cache_data(ttl=1800, show_spinner=False, max_entries=64)(_run_df_uncached)
else:  # pragma: no cover
    recent_market_runs = _list_runs_uncached
    run_df = _run_df_uncached


def safe_recent_runs() -> List[Dict[str, Any]]:
    try:
        return recent_market_runs()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("recent market scans", exc)
        return []


def safe_run_df(run_id: Optional[int]) -> Any:
    if run_id is None:
        return None
    try:
        df = run_df(int(run_id))
        return None if df is None else df.copy()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("a saved market scan", exc)
        return None
