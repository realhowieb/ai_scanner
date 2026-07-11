"""Score fired alerts against what the stock actually did afterward.

For each fired alert event whose forward window is complete: entry = close on
the fire date, "hit" = intraday high reached +HIT_TARGET_PCT% within
HORIZON_DAYS trading days, plus the plain horizon return. Powers the per-alert
scorecard ("4 of your last 7 fires hit +5% within 3 days").

Heavy (bar downloads) — run once/day from the cron. Best-effort; never raises.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from db.alert_outcomes import HIT_TARGET_PCT, HORIZON_DAYS


def _entry_position(bars, fire_date) -> Optional[int]:
    """Index of the first bar dated on/after the fire date."""
    try:
        for pos, ts in enumerate(bars.index):
            d = ts.date() if hasattr(ts, "date") else ts
            if d >= fire_date:
                return pos
    except Exception:
        pass
    return None


def score_event(bars, fired_at) -> Optional[Dict[str, Any]]:
    """Outcome for one event, or None when the window is incomplete/unusable."""
    try:
        fire_date = fired_at.date() if hasattr(fired_at, "date") else fired_at
        entry_pos = _entry_position(bars, fire_date)
        if entry_pos is None:
            return None
        exit_pos = entry_pos + HORIZON_DAYS
        if exit_pos >= len(bars):
            return None
        entry = float(bars["Close"].iloc[entry_pos])
        if entry <= 0:
            return None
        window = bars.iloc[entry_pos + 1 : exit_pos + 1]
        highs = window["High"] if "High" in window.columns else window["Close"]
        max_gain_pct = (float(highs.max()) - entry) / entry * 100.0
        horizon_return_pct = (float(bars["Close"].iloc[exit_pos]) - entry) / entry * 100.0
        return {
            "entry_price": round(entry, 4),
            "max_gain_pct": round(max_gain_pct, 2),
            "horizon_return_pct": round(horizon_return_pct, 2),
            "hit": max_gain_pct >= HIT_TARGET_PCT,
        }
    except Exception:
        return None


def score_pending_outcomes(max_events: int = 500) -> int:
    """Score all unscored, window-complete alert events. Returns rows saved."""
    try:
        from db.alert_outcomes import list_unscored_events, save_outcome
    except Exception:
        return 0

    events = list_unscored_events(limit=max_events)
    if not events:
        return 0

    symbols = sorted({str(e["ticker"]).upper() for e in events if e.get("ticker")})
    if not symbols:
        return 0

    try:
        from data.price_alpaca import download_multi_alpaca

        bars_by_symbol = download_multi_alpaca(
            symbols,
            period="45d",
            interval="1d",
            prepost=False,
            timeout_s=20.0,
        )
    except Exception:
        return 0
    if not bars_by_symbol:
        return 0

    saved = 0
    for event in events:
        sym = str(event.get("ticker") or "").upper()
        bars = bars_by_symbol.get(sym)
        if bars is None:
            bars = bars_by_symbol.get(sym.replace("-", "."))
        outcome = score_event(bars, event["fired_at"]) if bars is not None else None
        if outcome is None:
            # Unscoreable (delisted, no bars, window gap): record a null row so
            # we don't re-download it every day; hit=NULL excludes it from
            # scorecard aggregates.
            outcome = {
                "entry_price": None,
                "max_gain_pct": None,
                "horizon_return_pct": None,
                "hit": None,
            }
        try:
            if save_outcome(
                event_id=event["id"],
                alert_id=event.get("alert_id"),
                user_id=event["user_id"],
                ticker=sym,
                fired_at=event["fired_at"],
                **outcome,
            ):
                saved += 1
        except Exception:
            continue
    return saved
