"""Daily snapshot job.

This module now also refreshes the earnings calendar (slow-moving metadata) once per day.
Earnings refresh is best-effort and will never fail the snapshot job.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Earnings calendar refresh
try:
    from db.earnings import populate_earnings_calendar  # type: ignore
except Exception:  # pragma: no cover
    populate_earnings_calendar = None  # type: ignore


def _refresh_earnings_calendar_best_effort(conn, symbols: list[str]) -> None:
    """Populate earnings_calendar for the given symbols.

    - Best-effort: errors are swallowed so daily snapshot never fails.
    - Runs at most once per UTC day per Streamlit/worker process (via session_state-like module cache).
    """
    if not symbols:
        return
    if populate_earnings_calendar is None:
        return

    # Simple per-process daily guard
    today = datetime.now(timezone.utc).date().isoformat()
    cache_key = "_earnings_refreshed_utc_date"
    last = globals().get(cache_key)
    if last == today:
        return

    try:
        populate_earnings_calendar(symbols, conn=conn, sleep_s=0.02, commit_every=200)
        globals()[cache_key] = today
    except Exception:
        # Never break snapshot on earnings fetch
        return


def run_daily_snapshot(conn, symbols: list[str]) -> None:
    """Wrapper for callers that run daily snapshot logic elsewhere.

    Call this after your normal snapshot pipeline finishes, to refresh earnings metadata.
    """
    _refresh_earnings_calendar_best_effort(conn, [str(s).strip().upper() for s in symbols if str(s).strip()])
