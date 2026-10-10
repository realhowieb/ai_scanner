"""Day Trader web rows: the bounded DT score, its explanation, session-aware
change columns and quote sanity flags (pure, no I/O).

The web table used to rank by ``ui.day_trader.day_trade_score``, which has no
ceiling (about 2x the % change), so one bad print or split-adjusted quote at
+500% buried every other row. This module ranks by the validated 0-100 setup
score from ``analytics.day_trade_intel`` instead and flags quotes that look
wrong, so the API can serve rows the page can sort and explain.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from analytics.day_trade_intel import day_trade_intelligence

EXTREME_MOVE_PCT = 100.0     # |chg| at or above this: likely split, bad print or stale prior close
FAR_FROM_VWAP_PCT = 40.0     # last this far from VWAP: one print far off the day's trading
STALE_QUOTE_DAYS = 4         # covers a weekend plus a holiday
_OFF_HOURS = ("afterhours", "closed")


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or not b:
        return None
    return round((a - b) / b * 100.0, 2)


def _parse_ts(v: Any) -> Optional[dt.datetime]:
    if not v:
        return None
    if isinstance(v, dt.datetime):
        d = v
    else:
        s = str(v).replace("Z", "+00:00")
        # Alpaca sends nanoseconds; fromisoformat takes at most microseconds.
        if "." in s:
            head, _, rest = s.partition(".")
            digits = "".join(ch for ch in rest if ch.isdigit())
            tz = rest[len(digits):]
            s = f"{head}.{digits[:6]}{tz}"
        try:
            d = dt.datetime.fromisoformat(s)
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def session_changes(row: Dict[str, Any], state: str) -> Dict[str, Optional[float]]:
    """Regular-session change and the extended-hours move on top of it.

    Outside the regular session ``chg_pct`` (last vs previous close) mixes the
    day's move with thin after-hours prints, so the page shows them apart.
    Both are None during the regular session and pre-market."""
    out: Dict[str, Optional[float]] = {"session_chg_pct": None, "ext_chg_pct": None}
    if state not in _OFF_HOURS:
        return out
    close, prev, last = _num(row.get("close_today")), _num(row.get("previous_close")), _num(row.get("last"))
    out["session_chg_pct"] = _pct(close, prev)
    if close and last and last != close:
        out["ext_chg_pct"] = _pct(last, close)
    return out


def scoring_view(row: Dict[str, Any], state: str) -> Dict[str, Any]:
    """The row the DT score reads. Outside the regular session it scores the
    completed session (close vs previous close and VWAP) so an after-hours
    print can't re-rank the day."""
    if state not in _OFF_HOURS:
        return row
    close, prev, vwap = _num(row.get("close_today")), _num(row.get("previous_close")), _num(row.get("vwap"))
    if close is None or not prev:
        return row
    view = dict(row)
    view["chg_pct"] = _pct(close, prev)
    if vwap:
        view["vs_vwap_pct"] = _pct(close, vwap)
    return view


def quote_flags(row: Dict[str, Any], now: Optional[dt.datetime] = None) -> List[str]:
    """Reasons a quote may be wrong (split, bad print, stale prior close)."""
    out: List[str] = []
    chg = _num(row.get("chg_pct"))
    vsvwap = _num(row.get("vs_vwap_pct"))
    if chg is not None and abs(chg) >= EXTREME_MOVE_PCT:
        out.append("Extreme move")
    if vsvwap is not None and abs(vsvwap) >= FAR_FROM_VWAP_PCT:
        out.append("Far from VWAP")
    ts = _parse_ts(row.get("trade_ts"))
    if ts is not None:
        now = now or dt.datetime.now(dt.timezone.utc)
        if (now - ts).days > STALE_QUOTE_DAYS:
            out.append("Stale quote")
    return out


def enrich_row(row: Dict[str, Any], state: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """One web row: live fields plus score, tier, direction, reasons, conflicts,
    session columns and quote flags."""
    intel = day_trade_intelligence(scoring_view(row, state))
    flags = quote_flags(row, now)
    return {
        **row,
        **session_changes(row, state),
        "day_trade_score": intel["score"],
        "dt_quality": intel["quality"],
        "dt_direction": intel["direction"],
        "dt_reasons": intel["reasons"],
        "dt_conflicts": intel["conflicts"],
        "quote_flags": flags,
    }


def rank_key(row: Dict[str, Any]) -> tuple:
    """Clean quotes first, then by score (unscored last)."""
    score = row.get("day_trade_score")
    return (bool(row.get("quote_flags")), score is None, -(score or 0.0))


def sparkline(bars: List[Dict[str, Any]], points: int = 48) -> List[float]:
    """Closes from the latest session's 1-minute bars, thinned to ``points``."""
    from analytics.stair_step import latest_session_bars

    closes = [float(b["c"]) for b in latest_session_bars(bars)]
    if len(closes) <= points:
        return [round(c, 4) for c in closes]
    step = (len(closes) - 1) / (points - 1)
    return [round(closes[round(i * step)], 4) for i in range(points)]
