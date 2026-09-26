"""P1-9 — "new since your last visit", remembered in this browser.

The first time a session shows the market view, HSF reads (from an encrypted
browser cookie, the same cookie manager sign-in uses) which full-market scan
this browser last saw, keeps that as the session's baseline, and records the
current latest scan for next time. Names in the current market scan that were
not in the baseline scan are "new since your last visit".

No database schema or server-side storage: the marker follows the browser.
First visit (no baseline), or no newer scan since, means nothing is marked.
"""
from __future__ import annotations

from typing import Any, Optional, Set

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

COOKIE_KEY = "hsf_last_market_run"
BASELINE_KEY = "hsf_visit_baseline_run"   # session: the run this visit compares against
SYNCED_KEY = "hsf_visit_cookie_synced"


def new_tickers(current: Any, baseline: Any) -> Set[str]:
    """Tickers in `current` that are absent from `baseline` (empty if no baseline)."""
    from ui.market_scans import tickers_of

    if baseline is None or getattr(baseline, "empty", True):
        return set()
    base = set(tickers_of(baseline))
    return {t for t in tickers_of(current) if t not in base}


def _latest_run_id() -> Optional[int]:
    from ui.market_scans import safe_recent_runs

    runs = safe_recent_runs()
    return int(runs[0]["id"]) if runs else None


def baseline_run_id() -> Optional[int]:
    """The run this browser last saw before this session (read once per session)."""
    if st is None:
        return None
    if BASELINE_KEY in st.session_state:
        return st.session_state[BASELINE_KEY]
    base: Optional[int] = None
    try:
        from ui.auth_sessions import cookies_ready_or_stop, save_cookies

        cookies = cookies_ready_or_stop()
        if cookies is not None:
            raw = cookies.get(COOKIE_KEY)
            base = int(raw) if raw and str(raw).isdigit() else None
            latest = _latest_run_id()
            if latest is not None and str(latest) != str(raw or ""):
                cookies[COOKIE_KEY] = str(latest)
                save_cookies(cookies)
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("last-visit marker", exc)
    st.session_state[BASELINE_KEY] = base
    return base


def new_since_last_visit(current_df: Any, shown_run_id: Optional[int] = None) -> Set[str]:
    """Names new since this browser's last visit. `shown_run_id` is the market
    run being displayed (defaults to the Scanner's market view). Never raises."""
    try:
        from ui.market_default import MARKET_VIEW_KEY
        from ui.market_scans import safe_run_df

        base_id = baseline_run_id()
        shown = shown_run_id
        if shown is None and st is not None:
            shown = (st.session_state.get(MARKET_VIEW_KEY) or {}).get("run_id")
        if base_id is None or (shown is not None and int(shown) == int(base_id)):
            return set()
        return new_tickers(current_df, safe_run_df(base_id))
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("new since last visit", exc)
        return set()
