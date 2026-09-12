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


def _empty_metrics() -> Dict[str, Any]:
    return {
        "snapshots_loaded": 0, "events_detected": 0, "users_evaluated": 0,
        "watchlist_tickers_evaluated": 0, "notifications_matched": 0,
        "filtered_by_preferences": 0, "deduped": 0, "persisted": 0,
        "delivered": 0, "failed": 0,
    }


def _log(name: str, **fields) -> None:
    """Restrained structured log (no secrets/PII/payloads)."""
    try:
        parts = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[hsf_intelligence] {name} {parts}".rstrip())
    except Exception:
        pass


def run_intelligence_alert_evaluation(
    *,
    deliver: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """Detect + match + dedupe + persist + deliver, fully instrumented.

    Never raises (operational contract). Returns a result carrying status
    (SUCCESS/PARTIAL/SKIPPED/FAILED), pipeline counts, snapshot times, timing,
    and a sanitized error_stage/reason — and persists one evaluation-run record
    for operational truth. `deliver(user_id, copy, note) -> bool` is injectable.
    """
    import time as _time

    started = _time.time()
    m = _empty_metrics()
    result: Dict[str, Any] = {
        "status": "FAILED", "reason": None, "error_stage": None, "error_type": None,
        "previous_snapshot_time": None, "current_snapshot_time": None,
        "duration_ms": 0, **m,
    }
    _log("hsf_intelligence_eval_started")

    def finish(status: str, *, error_stage=None, error_type=None, reason=None) -> Dict[str, Any]:
        result.update({k: m[k] for k in m})
        result["status"] = status
        result["error_stage"] = error_stage
        result["error_type"] = error_type
        result["reason"] = reason
        result["duration_ms"] = int((_time.time() - started) * 1000)
        try:
            from db.intelligence_alerts import record_evaluation_run
            record_evaluation_run(result)
        except Exception:
            pass
        ev = {"SUCCESS": "hsf_intelligence_eval_completed", "PARTIAL": "hsf_intelligence_eval_completed",
              "SKIPPED": "hsf_intelligence_eval_skipped", "FAILED": "hsf_intelligence_eval_failed"}[status]
        _log(ev, status=status, events=m["events_detected"], matched=m["notifications_matched"],
             delivered=m["delivered"], deduped=m["deduped"], filtered=m["filtered_by_preferences"],
             failed=m["failed"], stage=error_stage or "-", duration_ms=result["duration_ms"])
        return result

    # --- Dependencies ---
    try:
        from analytics.opportunity_events import (
            collapse_events,
            derive_opportunity_events,
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
        return finish("FAILED", error_stage="IMPORT", error_type=type(e).__name__)

    # --- STAGE A: detect market events ONCE ---
    try:  # LOAD_SNAPSHOTS
        snaps = load_recent_snapshots(context="market_brief", limit=2)
    except Exception as e:
        return finish("FAILED", error_stage="LOAD_SNAPSHOTS", error_type=type(e).__name__)
    m["snapshots_loaded"] = len(snaps)
    if len(snaps) < 2:  # VALIDATE_SNAPSHOTS
        return finish("SKIPPED", reason="fewer than two comparable snapshots")
    current, previous = snaps[0], snaps[1]
    result["previous_snapshot_time"] = str(previous.get("snapshot_time"))
    result["current_snapshot_time"] = str(current.get("snapshot_time"))
    try:  # DETECT_EVENTS + COLLAPSE_EVENTS
        events = derive_opportunity_events(
            previous.get("opportunities"), current.get("opportunities"),
            previous_snapshot_time=previous.get("snapshot_time"),
            current_snapshot_time=current.get("snapshot_time"),
            source_context="market_brief")
        notes = collapse_events(events)
    except Exception as e:
        return finish("FAILED", error_stage="DETECT_EVENTS", error_type=type(e).__name__)
    m["events_detected"] = len(notes)
    if not notes:
        return finish("SUCCESS", reason="no meaningful state changes")  # zero events is SUCCESS
    notes_by_ticker: Dict[str, Dict[str, Any]] = {n["ticker"]: n for n in notes}

    # --- STAGE B: match users (state already computed once) ---
    user_failures = 0
    for user_id, tickers in _watchers().items():
        try:
            m["users_evaluated"] += 1
            prefs = get_hsf_alert_prefs(user_id)
            recent = recent_fingerprints(user_id)
            matched_tickers = tickers & set(notes_by_ticker.keys())
            m["watchlist_tickers_evaluated"] += len(tickers)
            for ticker in matched_tickers:
                note = notes_by_ticker[ticker]
                m["notifications_matched"] += 1
                decision = should_notify(note, prefs, recent, user_id=user_id)
                if not decision["notify"]:
                    if "dedup" in decision["reason"]:
                        m["deduped"] += 1
                    elif "preference" in decision["reason"]:
                        m["filtered_by_preferences"] += 1
                    continue
                fp = decision["fingerprint"]
                copy = notification_copy(note)
                aid = record_intelligence_alert(
                    user_id=user_id, note=note, copy=copy, fingerprint=fp,
                    delivery_status="QUEUED")
                if aid:
                    m["persisted"] += 1
                recent.add(fp)
                ok = False
                try:
                    ok = bool(deliver(user_id, copy, note)) if deliver else _deliver_email(user_id, copy)
                except Exception:
                    ok = False
                if aid:
                    update_delivery_status(aid, "DELIVERED" if ok else "FAILED")
                if not ok:
                    _log("hsf_intelligence_delivery_failed", ticker=ticker, event=note.get("event_type"))
                m["delivered" if ok else "failed"] += 1
        except Exception as e:
            user_failures += 1
            _log("hsf_intelligence_user_failed", error_type=type(e).__name__)
            continue  # one user's failure never aborts the batch

    status = "PARTIAL" if (m["failed"] > 0 or user_failures > 0) else "SUCCESS"
    return finish(status, reason=("downstream delivery/user failures" if status == "PARTIAL" else None))


def _deliver_email(user_id: str, copy: str) -> bool:
    """Best-effort email via the existing alert email path. False on any failure
    (never a fabricated delivered state)."""
    try:
        from ui.email_utils import send_alert_email

        subject = "HSF Intelligence Alert"
        return bool(send_alert_email(user_id, subject, copy))
    except Exception:
        return False
