"""P2-74 — Today's "Before the open" card (Pro+).

Shows the biggest pre-market movers from this morning's premarket scan
(PMLast / PMPctChange: today's latest pre-market trade vs the previous close)
until 10:30 ET: "Before the open" until 9:30, then "This morning's pre-market
movers" for the first hour of trading (owner, 2026-10-05: the card was too easy
to miss when it disappeared at the open). It's a snapshot as of that scan; Day Trader has live
pre-market prices. Display only: nothing here changes scores, ranking or
research data.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

PREMARKET_LABEL = "premarket"
SHOW_UNTIL_ET = _dt.time(10, 30)  # keep this morning's movers up through the first hour of trading


def _window_open(now: _dt.datetime) -> bool:
    """A trading day, before SHOW_UNTIL_ET New York time."""
    from analytics import market_calendar as mc

    et = now.astimezone(mc.ET)
    return mc.is_trading_day(et.date()) and et.time() < SHOW_UNTIL_ET


def after_open(now: _dt.datetime) -> bool:
    from analytics import market_calendar as mc

    return now >= mc.session_bounds_utc(now.astimezone(mc.ET).date())[0]


def current_premarket_run(runs: List[Dict[str, Any]], now: _dt.datetime) -> Optional[Dict[str, Any]]:
    """The latest premarket run from today (ET), until SHOW_UNTIL_ET."""
    from analytics import market_calendar as mc
    from ui.market_scans import _ts

    today = now.astimezone(mc.ET).date()
    if not _window_open(now):
        return None
    best = None
    for r in runs or []:
        if str(r.get("label") or "").strip().lower() != PREMARKET_LABEL or r.get("id") is None:
            continue
        ts = _ts(r.get("created_at"))
        if ts is None or ts.astimezone(mc.ET).date() != today or ts > now:
            continue
        if best is None or ts > best[0]:
            best = (ts, r)
    return {**best[1], "created_at": best[0]} if best else None


def premarket_movers(df: Any, n: int = 5) -> List[Dict[str, Any]]:
    """Biggest absolute pre-market moves: [{ticker, pct, last, score}]."""
    from ui.after_close import session_movers

    return session_movers(df, "PMPctChange", "PMLast", n=n)


def render_before_open(now: Optional[_dt.datetime] = None) -> None:
    """Today card. Renders nothing outside 8:35-10:30 ET or with no premarket scan today."""
    if st is None:
        return
    now = now or _dt.datetime.now(_dt.timezone.utc)
    from analytics import market_calendar as mc

    if not _window_open(now):
        return
    title = "This morning's pre-market movers" if after_open(now) else "Before the open"
    from ui.after_close import MIN_ABS_MOVE_PCT, _can_see

    try:
        from db.runs import list_runs

        run = current_premarket_run(list_runs(limit=60, include_snapshots=False, username="scheduler") or [], now)
    except Exception:
        run = None
    if run is None:  # nothing to show (or to upsell) until this morning's scan exists
        return
    if not _can_see():
        st.markdown(f"### {title}")
        st.caption("Pro shows this morning's biggest pre-market movers here, plus live pre-market prices in Day Trader.")
        return
    from ui.market_scans import safe_run_df

    movers = premarket_movers(safe_run_df(int(run["id"])))
    st.markdown(f"### {title}")
    as_of = run["created_at"].astimezone(mc.ET).strftime("%-I:%M %p ET")
    if not movers:
        st.caption(f"No pre-market moves of {MIN_ABS_MOVE_PCT:g}% or more in the {as_of} pre-market scan.")
    for m in movers:
        price = f" (${m['last']:,.2f})" if m["last"] is not None else ""
        score = f" · HSF Score **{m['score']}**" if m["score"] is not None else ""
        st.markdown(f"**{m['ticker']}** {m['pct']:+.2f}% pre-market{price}{score}")
    st.caption(f"As of the {as_of} pre-market scan, vs the previous close — pre-market prices keep moving.")
    st.page_link("pages/day_trader.py", label="Live pre-market prices in Day Trader", icon="⚡")
