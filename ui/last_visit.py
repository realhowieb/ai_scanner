"""P1-9 — "new since your last visit", remembered in this browser.

The cookie (encrypted, the same cookie manager sign-in uses) remembers two
full-market scans: the one this browser last saw and the baseline it was
compared against. When a session starts:
  - a newer scan exists → the baseline becomes the last-seen scan, and the
    newest scan is recorded as seen;
  - no newer scan → the baseline is kept, so a page refresh or a fresh sign-in
    (each a new Streamlit session) still shows the same "new" names instead of
    comparing the latest scan with itself.
Names in the current market scan that were not in the baseline scan are "new
since your last visit".

No database schema or server-side storage: the marker follows the browser.
First visit (no baseline) means nothing is marked.
"""
from __future__ import annotations

from typing import Any, Optional, Set, Tuple

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


def _parse_marker(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    """Cookie value → (seen, baseline). Accepts "seen:baseline", "seen:" and the
    older single-value "seen" format."""
    seen_s, _, base_s = str(raw or "").partition(":")
    seen = int(seen_s) if seen_s.isdigit() else None
    base = int(base_s) if base_s.isdigit() else None
    return seen, base


def next_marker(raw: Any, latest: Optional[int]) -> Tuple[Optional[int], str]:
    """(baseline for this session, cookie value to store) given the stored
    marker and the newest full-market run id."""
    seen, base = _parse_marker(raw)
    if latest is None:                     # runs unavailable: change nothing
        return base, str(raw or "")
    if seen is None:                       # first visit on this browser
        return None, f"{latest}:"
    if latest != seen:                     # a newer scan since the last visit
        return seen, f"{latest}:{seen}"
    return base, f"{seen}:{base if base is not None else ''}"   # same scan: keep the baseline


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
            base, value = next_marker(raw, _latest_run_id())
            if value and value != str(raw or ""):
                cookies[COOKIE_KEY] = value
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
