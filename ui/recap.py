"""P1-8 — scanner recap for the latest trading session.

What the scheduled full-market scans saw during one session: how many scans
ran, which names entered and left the ranked list between the first and last
scan, and the standout setups in the last scan (canonical HSF Score ranking).
Descriptive only — no returns, hit rates or other performance numbers.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from analytics import market_calendar as mc

MAX_NAMES = 8


def recap_day(runs: Sequence[Dict[str, Any]], now: _dt.datetime) -> Optional[_dt.date]:
    """Today (ET) if a scan ran today, else the ET date of the newest scan."""
    if not runs:
        return None
    today = now.astimezone(mc.ET).date()
    days = [r["created_at"].astimezone(mc.ET).date() for r in runs]
    return today if today in days else max(days)


def build_recap(day_runs: Sequence[Dict[str, Any]], first_df: Any, last_df: Any, *,
                day: _dt.date, now: _dt.datetime) -> Dict[str, Any]:
    """Plain-data recap for one session (newest-first runs of that day)."""
    from ui.market_scans import diff_tickers, tickers_of, top_setups

    first, last = tickers_of(first_df), tickers_of(last_df)
    d = diff_tickers(first, last) if len(day_runs) > 1 else {"entered": [], "left": []}
    today = now.astimezone(mc.ET).date()
    closed = day < today or (mc.is_trading_day(day) and now >= mc.session_bounds_utc(day)[1])
    return {
        "day": day,
        "title": ("End-of-day recap" if closed else "Today so far") if day == today
        else f"Last session recap · {day.strftime('%a %b')} {day.day}",
        "scans": len(day_runs),
        "entered": d["entered"],
        "left": d["left"],
        "standouts": [{"ticker": o["ticker"], "score": o["score"], "setup": o["primary_setup"]}
                      for o in top_setups(last_df, n=5)],
    }


def recap_lines(r: Dict[str, Any]) -> List[str]:
    """Markdown bullets for the recap (plain, descriptive)."""
    def names(xs: List[str]) -> str:
        more = f" and {len(xs) - MAX_NAMES} more" if len(xs) > MAX_NAMES else ""
        return ", ".join(xs[:MAX_NAMES]) + more

    lines = [f"- **{r['scans']}** full-market scan{'s' if r['scans'] != 1 else ''} ran."]
    if r["scans"] > 1:
        lines.append(f"- Entered the ranked list: {names(r['entered'])}." if r["entered"]
                     else "- No new names entered the ranked list.")
        lines.append(f"- Left the ranked list: {names(r['left'])}." if r["left"]
                     else "- No names left the ranked list.")
    if r["standouts"]:
        lines.append("- Standouts by HSF Score: " + ", ".join(
            f"{s['ticker']} ({s['score']}{', ' + s['setup'] if s.get('setup') not in (None, '', 'Signal') else ''})"
            for s in r["standouts"]) + ".")
    return lines


def render_recap(now: Optional[_dt.datetime] = None) -> None:
    """Render the latest session's recap. Never raises; silent with no data."""
    if st is None:
        return
    try:
        from ui.market_scans import runs_on_day, safe_recent_runs, safe_run_df

        now = now or _dt.datetime.now(_dt.timezone.utc)
        runs = safe_recent_runs()
        day = recap_day(runs, now)
        if day is None:
            return
        day_runs = runs_on_day(runs, day)
        first_df = safe_run_df(day_runs[-1]["id"])
        last_df = safe_run_df(day_runs[0]["id"])
        r = build_recap(day_runs, first_df, last_df, day=day, now=now)
        st.markdown(f"### {r['title']}")
        st.markdown("\n".join(recap_lines(r)))
        st.caption("What HSF's scheduled scans saw. Descriptive only: not a performance record.")
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("scanner recap", exc)
