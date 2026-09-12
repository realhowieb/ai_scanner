"""Background HSF Intelligence Alert evaluation (two-stage, batch).

STAGE A — market event detection (ONCE): the two latest comparable opportunity
snapshots -> canonical events -> collapsed per-ticker notifications. HSF state is
computed once for the market, never per user.

STAGE B — user matching: for each subscribed user, keep notifications for tickers
they follow, apply their preferences + state-aware dedupe, persist, and deliver
(best-effort email). Owned by the scheduled pipeline (never a page render);
fails independently and never fabricates a delivered state.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set


def _watchers() -> Dict[str, Set[str]]:
    """user_id -> set of watched tickers, for users who explicitly opted into HSF
    intelligence alerts (conservative V1 scope). Each user's own watchlist only —
    strict per-user isolation. {} when none/unavailable."""
    out: Dict[str, Set[str]] = {}
    try:
        from db.intelligence_alerts import list_alert_pref_user_ids
        from db.watchlists import get_watchlist_tickers, list_watchlists
    except Exception:
        return {}
    try:
        user_ids = list_alert_pref_user_ids()
    except Exception:
        return {}
    for user_id in user_ids:
        tickers: Set[str] = set()
        try:
            for wl in (list_watchlists(user_id) or []):
                for t in (get_watchlist_tickers(wl.get("id"), user_id) or []):
                    if str(t).strip():
                        tickers.add(str(t).upper())
        except Exception:
            tickers = set()
        if tickers:
            out[user_id] = tickers
    return out


def run_intelligence_alert_evaluation(
    *,
    deliver: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """Detect + match + dedupe + persist + deliver. Returns metrics. Never raises.

    `deliver(user_id, copy, note) -> bool` is injectable for tests; defaults to
    best-effort email. Delivery failure is recorded as FAILED, event retained.
    """
    metrics = {"events_detected": 0, "notifications_matched": 0, "deduped": 0,
               "delivered": 0, "failed": 0, "persisted": 0, "reason": None}
    try:
        from analytics.opportunity_events import (
            collapse_events,
            derive_opportunity_events,
            event_fingerprint,
            notification_copy,
            should_notify,
        )
        from db.intelligence_alerts import (
            get_hsf_alert_prefs,
            recent_fingerprints,
            record_intelligence_alert,
            update_delivery_status,
        )
        from db.opportunity_snapshots import load_recent_snapshots
    except Exception as e:  # pragma: no cover
        metrics["reason"] = f"import failed: {type(e).__name__}"
        return metrics

    # STAGE A — detect once.
    try:
        snaps = load_recent_snapshots(context="market_brief", limit=2)
    except Exception:
        snaps = []
    if len(snaps) < 2:
        metrics["reason"] = "need two comparable snapshots"
        return metrics
    current, previous = snaps[0], snaps[1]
    try:
        events = derive_opportunity_events(
            previous.get("opportunities"), current.get("opportunities"),
            previous_snapshot_time=previous.get("snapshot_time"),
            current_snapshot_time=current.get("snapshot_time"),
            source_context="market_brief")
        notes = collapse_events(events)
    except Exception as e:
        metrics["reason"] = f"detection failed: {type(e).__name__}"
        return metrics
    metrics["events_detected"] = len(notes)
    if not notes:
        return metrics
    notes_by_ticker: Dict[str, Dict[str, Any]] = {n["ticker"]: n for n in notes}

    # STAGE B — match users (batch; HSF state already computed once above).
    watchers = _watchers()
    for user_id, tickers in watchers.items():
        try:
            prefs = get_hsf_alert_prefs(user_id)
            recent = recent_fingerprints(user_id)
            for ticker in (tickers & set(notes_by_ticker.keys())):
                note = notes_by_ticker[ticker]
                metrics["notifications_matched"] += 1
                decision = should_notify(note, prefs, recent, user_id=user_id)
                if not decision["notify"]:
                    if "dedup" in decision["reason"]:
                        metrics["deduped"] += 1
                    continue
                fp = decision["fingerprint"]
                copy = notification_copy(note)
                aid = record_intelligence_alert(
                    user_id=user_id, note=note, copy=copy, fingerprint=fp,
                    delivery_status="QUEUED")
                if aid:
                    metrics["persisted"] += 1
                recent.add(fp)  # prevent duplicate within this run
                ok = False
                try:
                    ok = bool(deliver(user_id, copy, note)) if deliver else _deliver_email(user_id, copy)
                except Exception:
                    ok = False
                if aid:
                    update_delivery_status(aid, "DELIVERED" if ok else "FAILED")
                metrics["delivered" if ok else "failed"] += 1
        except Exception:
            # One user's failure must not abort the batch.
            continue
    return metrics


def _deliver_email(user_id: str, copy: str) -> bool:
    """Best-effort email via the existing alert email path. False on any failure
    (never a fabricated delivered state)."""
    try:
        from ui.email_utils import send_alert_email

        subject = "HSF Intelligence Alert"
        return bool(send_alert_email(user_id, subject, copy))
    except Exception:
        return False
