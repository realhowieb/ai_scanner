"""Watchlist-centered HSF intelligence composition for Run 27.

This module is intentionally read-only with respect to HSF intelligence. It
loads existing user watchlists, persisted opportunity snapshots, and persisted
intelligence alerts, then composes a personalized view. It never scans, scores,
freezes, matures outcomes, delivers alerts, or calls an LLM.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from analytics.opportunity_events import (
    DROPPED,
    FADING,
    FALLING,
    NEW_OPPORTUNITY,
    RISING,
    SIGNAL_ADDED,
    SIGNAL_REMOVED,
    STATUS_DOWNGRADE,
    STATUS_UPGRADE,
    collapse_events,
    derive_opportunity_events,
)

ATTENTION_PRIORITY = [
    STATUS_DOWNGRADE,
    FADING,
    DROPPED,
    STATUS_UPGRADE,
    NEW_OPPORTUNITY,
    FALLING,
    RISING,
    SIGNAL_REMOVED,
    SIGNAL_ADDED,
]

IMPROVING_EVENTS = {STATUS_UPGRADE, NEW_OPPORTUNITY, RISING, SIGNAL_ADDED}
FADING_EVENTS = {FADING, STATUS_DOWNGRADE, FALLING, DROPPED}


def normalize_ticker(value: object) -> str:
    try:
        from db.watchlists import normalize_watchlist_ticker

        return normalize_watchlist_ticker(value)
    except Exception:
        ticker = str(value or "").strip().upper()
        return ticker if ticker else ""


def normalize_tickers(values: List[object]) -> List[str]:
    cleaned = sorted({t for t in (normalize_ticker(v) for v in (values or [])) if t})
    return cleaned


def _ticker(row: Dict[str, Any]) -> str:
    return normalize_ticker(row.get("ticker") or row.get("Ticker") or row.get("Symbol"))


def _rows_by_ticker(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = _ticker(row)
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def _latest_alert_by_ticker(alerts: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for alert in alerts or []:
        ticker = normalize_ticker(alert.get("ticker"))
        if ticker and ticker not in out:
            out[ticker] = alert
    return out


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _age_minutes(ts: Any, now: Optional[datetime]) -> Optional[int]:
    if ts is None or now is None:
        return None
    try:
        if isinstance(ts, str):
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            parsed = ts
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return max(0, int((ref - parsed).total_seconds() // 60))
    except Exception:
        return None


def _event_reason(event: Optional[Dict[str, Any]], current: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> str:
    if not event:
        if current:
            return "No meaningful HSF change detected recently."
        if previous:
            return "Left the current HSF opportunity ranking."
        return "Not currently ranked as an HSF opportunity."
    event_type = event.get("event_type")
    if event_type == STATUS_DOWNGRADE:
        return f"HSF status downgraded {event.get('previous_status')} to {event.get('current_status')}."
    if event_type == STATUS_UPGRADE:
        return f"HSF status upgraded {event.get('previous_status')} to {event.get('current_status')}."
    if event_type == FADING:
        return "Fading behavior detected in the current HSF state."
    if event_type == DROPPED:
        return "Left the current HSF opportunity ranking."
    if event_type == NEW_OPPORTUNITY:
        return "Entered the current HSF opportunity ranking."
    if event_type == FALLING:
        return f"HSF Score weakened by {abs(int(event.get('score_delta') or 0))}."
    if event_type == RISING:
        return f"HSF Score strengthened by {int(event.get('score_delta') or 0)}."
    if event_type == SIGNAL_ADDED:
        return f"Confirming signal added: {event.get('signal') or 'HSF signal'}."
    if event_type == SIGNAL_REMOVED:
        return f"Confirming signal removed: {event.get('signal') or 'HSF signal'}."
    return str(event_type or "Recent HSF change")


def _group_for(event: Optional[Dict[str, Any]], current: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> str:
    if event and event.get("event_type") in FADING_EVENTS:
        return "needs_attention"
    if event and event.get("event_type") in IMPROVING_EVENTS:
        return "improving"
    if current:
        return "stable"
    if previous:
        return "quiet"
    return "quiet"


def _priority_for(row: Dict[str, Any]) -> tuple:
    event_type = (row.get("attention_event") or {}).get("event_type")
    event_rank = ATTENTION_PRIORITY.index(event_type) if event_type in ATTENTION_PRIORITY else 99
    score = row.get("hsf_score")
    score_sort = -float(score) if isinstance(score, (int, float)) else 0.0
    group_rank = {"needs_attention": 0, "improving": 1, "stable": 2, "quiet": 3}.get(row.get("group"), 9)
    return (group_rank, event_rank, score_sort, row.get("ticker") or "")


def match_watchlist_opportunities(watchlist: List[object], opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic watchlist ∩ current HSF opportunities."""
    watch = set(normalize_tickers(watchlist))
    rows = [row for row in opportunities or [] if _ticker(row) in watch]
    return sorted(rows, key=lambda row: (-float(row.get("score") or 0), _ticker(row)))


def build_watchlist_intelligence(
    user_id: str,
    *,
    watchlist: Optional[List[object]] = None,
    current_snapshot: Optional[Dict[str, Any]] = None,
    previous_snapshot: Optional[Dict[str, Any]] = None,
    recent_alerts: Optional[List[Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
    watchlist_loader: Optional[Callable[[str], List[str]]] = None,
    snapshots_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    alerts_loader: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Compose one personalized watchlist intelligence object.

    All loaders are batch-style. Passing injected data makes the function pure
    for tests; default loaders read only existing persisted HSF intelligence.
    """
    user = str(user_id or "").strip().lower()
    now = now or datetime.now(timezone.utc)

    if watchlist is None:
        if watchlist_loader is not None:
            watchlist = watchlist_loader(user)
        else:
            try:
                from db.watchlists import get_user_watchlist

                watchlist = get_user_watchlist(user)
            except Exception:
                watchlist = []
    tickers = normalize_tickers(list(watchlist or []))

    snapshots: List[Dict[str, Any]] = []
    if current_snapshot is None:
        if snapshots_loader is not None:
            snapshots = snapshots_loader()
        else:
            try:
                from db.opportunity_snapshots import load_recent_snapshots

                snapshots = load_recent_snapshots(context="market_brief", limit=2)
            except Exception:
                snapshots = []
        current_snapshot = snapshots[0] if snapshots else None
        previous_snapshot = snapshots[1] if previous_snapshot is None and len(snapshots) > 1 else previous_snapshot

    current_rows = (current_snapshot or {}).get("opportunities") or []
    previous_rows = (previous_snapshot or {}).get("opportunities") or []
    current_by = _rows_by_ticker(current_rows)
    previous_by = _rows_by_ticker(previous_rows)

    watched_current = [current_by[t] for t in tickers if t in current_by]
    watched_previous = [previous_by[t] for t in tickers if t in previous_by]
    events = collapse_events(
        derive_opportunity_events(
            watched_previous,
            watched_current,
            previous_snapshot_time=(previous_snapshot or {}).get("snapshot_time"),
            current_snapshot_time=(current_snapshot or {}).get("snapshot_time"),
            source_context="market_brief",
        )
    )
    events_by = _rows_by_ticker(events)

    if recent_alerts is None:
        if alerts_loader is not None:
            recent_alerts = alerts_loader(user)
        else:
            try:
                from db.intelligence_alerts import list_recent_intelligence_alerts

                recent_alerts = list_recent_intelligence_alerts(user, limit=50)
            except Exception:
                recent_alerts = []
    watch_set = set(tickers)
    recent_alerts = [a for a in (recent_alerts or []) if normalize_ticker(a.get("ticker")) in watch_set]
    alerts_by = _latest_alert_by_ticker(recent_alerts)

    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        current = current_by.get(ticker)
        previous = previous_by.get(ticker)
        event = events_by.get(ticker)
        group = _group_for(event, current, previous)
        row = {
            "ticker": ticker,
            "is_watched": True,
            "in_current_opportunities": bool(current),
            "hsf_score": current.get("score") if current else None,
            "hsf_status": current.get("status") if current else None,
            "confirming_signals": list(current.get("signals") or []) if current else [],
            "fading": bool(current.get("fading")) if current else bool(event and event.get("event_type") == FADING),
            "previous_hsf_score": previous.get("score") if previous else None,
            "previous_hsf_status": previous.get("status") if previous else None,
            "score_delta": event.get("score_delta") if event else None,
            "attention_event": event,
            "attention_reason": _event_reason(event, current, previous),
            "recent_alert": alerts_by.get(ticker),
            "group": group,
            "last_updated": _iso_or_none((current_snapshot or {}).get("snapshot_time")),
            "last_updated_minutes": _age_minutes((current_snapshot or {}).get("snapshot_time"), now),
            "has_historical_context": bool(previous),
        }
        rows.append(row)

    rows.sort(key=_priority_for)
    summary = {
        "tracked": len(tickers),
        "needs_attention": sum(1 for row in rows if row["group"] == "needs_attention"),
        "strengthening": sum(1 for row in rows if row["group"] == "improving"),
        "fading": sum(1 for row in rows if row.get("fading")),
        "active_opportunities": sum(1 for row in rows if row["in_current_opportunities"]),
        "stable": sum(1 for row in rows if row["group"] in {"stable", "quiet"}),
        "recent_alerts": len(recent_alerts),
    }
    return {
        "user_id": user,
        "watchlist": tickers,
        "summary": summary,
        "rows": rows,
        "groups": {
            "needs_attention": [row for row in rows if row["group"] == "needs_attention"],
            "improving": [row for row in rows if row["group"] == "improving"],
            "stable": [row for row in rows if row["group"] == "stable"],
            "quiet": [row for row in rows if row["group"] == "quiet"],
        },
        "recent_changes": [event for event in events if _ticker(event) in watch_set],
        "recent_alerts": recent_alerts,
        "current_snapshot_time": _iso_or_none((current_snapshot or {}).get("snapshot_time")),
        "previous_snapshot_time": _iso_or_none((previous_snapshot or {}).get("snapshot_time")),
    }
