"""P2-45 — Today's "After the close" card (Pro+).

Shows the biggest after-hours movers from the latest postmarket scan
(AHLast / AHPctChange, added by e0b58e0) between the close and the next open.
It's a snapshot as of that scan; Day Trader has the live after-hours prices.
Display only: nothing here changes scores, ranking or research data.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

POSTMARKET_LABEL = "postmarket"
# Smaller moves are noise (late evening most names show 0.00%: no after-hours trade).
MIN_ABS_MOVE_PCT = 0.5


def last_close(now: _dt.datetime) -> _dt.datetime:
    """The most recent regular-session close at or before `now` (ET-aware)."""
    from analytics import market_calendar as mc

    now_et = now.astimezone(mc.ET)
    d = now_et.date()
    if mc.is_trading_day(d):
        close = _dt.datetime.combine(d, mc.close_time_et(d), tzinfo=mc.ET)
        if now_et >= close:
            return close
    prev = mc.previous_trading_day(d)
    return _dt.datetime.combine(prev, mc.close_time_et(prev), tzinfo=mc.ET)


def current_postmarket_run(runs: List[Dict[str, Any]], now: _dt.datetime) -> Optional[Dict[str, Any]]:
    """The latest postmarket run if it belongs to the current after-hours window:
    outside regular trading hours and created after the most recent close."""
    from analytics import market_calendar as mc
    from ui.market_scans import _ts

    if mc.is_market_open(now):
        return None
    cutoff = last_close(now)
    best = None
    for r in runs or []:
        if str(r.get("label") or "").strip().lower() != POSTMARKET_LABEL or r.get("id") is None:
            continue
        ts = _ts(r.get("created_at"))
        if ts is None or ts < cutoff:
            continue
        if best is None or ts > best[0]:
            best = (ts, r)
    return {**best[1], "created_at": best[0]} if best else None


def after_hours_movers(df: Any, n: int = 5) -> List[Dict[str, Any]]:
    """Biggest absolute after-hours moves: [{ticker, ah_pct, ah_last, score}]."""
    return [{"ticker": m["ticker"], "ah_pct": m["pct"], "ah_last": m["last"], "score": m["score"]}
            for m in session_movers(df, "AHPctChange", "AHLast", n=n)]


def session_movers(df: Any, pct_col: str, last_col: str, n: int = 5) -> List[Dict[str, Any]]:
    """Biggest absolute extended-hours moves in `pct_col` (at least MIN_ABS_MOVE_PCT):
    [{ticker, pct, last, score}]. Shared by After the close and Before the open."""
    import pandas as pd

    if df is None or getattr(df, "empty", True) or pct_col not in df.columns or "Ticker" not in df.columns:
        return []
    rows = df.copy()
    rows[pct_col] = pd.to_numeric(rows[pct_col], errors="coerce")
    rows = rows[rows[pct_col].notna() & (rows[pct_col].abs() >= MIN_ABS_MOVE_PCT)]
    if rows.empty:
        return []
    try:
        from ui.headline_score import hsf_scores_by_ticker

        scores = hsf_scores_by_ticker(df.to_dict(orient="records"))
    except Exception:
        scores = {}
    rows = rows.reindex(rows[pct_col].abs().sort_values(ascending=False).index).head(n)
    out = []
    for _, r in rows.iterrows():
        t = str(r["Ticker"]).upper()
        last = pd.to_numeric(r.get(last_col), errors="coerce")
        out.append({"ticker": t, "pct": float(r[pct_col]),
                    "last": None if pd.isna(last) else float(last), "score": scores.get(t)})
    return out


def _can_see() -> bool:
    ss = st.session_state
    if ss.get("is_admin"):
        return True
    ent = ss.get("entitlements") or {}
    if "can_day_trader" in ent:
        return bool(ent["can_day_trader"])
    try:
        from auth.tiering import has_min_tier

        return bool(has_min_tier(ss.get("tier_key") or "basic", "pro"))
    except Exception:
        return False


def render_after_close(now: Optional[_dt.datetime] = None) -> None:
    """Today card. Renders nothing during market hours or with no fresh postmarket scan."""
    if st is None:
        return
    now = now or _dt.datetime.now(_dt.timezone.utc)
    from analytics import market_calendar as mc

    if mc.is_market_open(now):
        return
    if not _can_see():
        st.markdown("### After the close")
        st.caption("Pro shows tonight's biggest after-hours movers here, plus live after-hours prices in Day Trader.")
        return
    try:
        from db.runs import list_runs

        run = current_postmarket_run(list_runs(limit=60, include_snapshots=False, username="scheduler") or [], now)
    except Exception:
        run = None
    if run is None:
        return
    from ui.market_scans import safe_run_df

    movers = after_hours_movers(safe_run_df(int(run["id"])))
    st.markdown("### After the close")
    as_of = run["created_at"].astimezone(mc.ET).strftime("%-I:%M %p ET")
    if not movers:
        st.caption(f"No after-hours moves of {MIN_ABS_MOVE_PCT:g}% or more in the {as_of} postmarket scan.")
    for m in movers:
        price = f" (${m['ah_last']:,.2f})" if m["ah_last"] is not None else ""
        score = f" · HSF Score **{m['score']}**" if m["score"] is not None else ""
        st.markdown(f"**{m['ticker']}** {m['ah_pct']:+.2f}% after hours{price}{score}")
    st.caption(f"As of the {as_of} postmarket scan — after-hours prices keep moving.")
    st.page_link("pages/day_trader.py", label="Live after-hours prices in Day Trader", icon="⚡")
