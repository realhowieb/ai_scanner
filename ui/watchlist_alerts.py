"""Scheduled per-user watchlist email digest.

Opt-in (WATCHLIST_ALERTS_ENABLED). Runs headless from the scheduler: for each
user with a watchlist, filter the latest saved snapshot to their tickers, have
Claude write a short digest, and email it. Best-effort; never raises.
"""
from __future__ import annotations


def _latest_snapshot_df():
    """Load the most recent saved snapshot as a DataFrame, or None."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df
        runs = list_runs(limit=10) or []
        snap = next((r for r in runs if r.get("is_snapshot")), None) or (runs[0] if runs else None)
        if not snap:
            return None
        raw = load_run_results(snap["id"])
        return normalize_results_to_df(raw) if raw else None
    except Exception:
        return None


def run_watchlist_alerts() -> None:
    """Email each user a digest of their watchlist vs. the latest snapshot."""
    try:
        from config import (
            AI_ENABLED,
            ANTHROPIC_API_KEY,
            WATCHLIST_ALERTS_ENABLED,
            WATCHLIST_ALERTS_MAX_USERS,
        )
    except Exception:
        return

    if not (WATCHLIST_ALERTS_ENABLED and AI_ENABLED and ANTHROPIC_API_KEY):
        return

    df = _latest_snapshot_df()
    if df is None or len(df) == 0:
        print("[watchlist_alerts] no snapshot to summarize")
        return

    try:
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from ui.ai_insights import generate_watchlist_digest
        from ui.email_utils import send_alert_email
    except Exception as e:
        print(f"[watchlist_alerts] import failed: {e}")
        return

    try:
        users = load_users() or {}
    except Exception:
        users = {}

    sent = 0
    for username in list(users.keys())[:WATCHLIST_ALERTS_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            continue
        try:
            wls = list_watchlists(email) or []
            tickers: list[str] = []
            for wl in wls:
                tickers.extend(get_watchlist_tickers(wl.get("id"), email) or [])
            tickers = sorted({str(t).strip().upper() for t in tickers if t})
            if not tickers:
                continue

            digest, err = generate_watchlist_digest(tickers, df)
            if not digest:
                continue

            send_alert_email(
                to_address=email,
                subject="Your AI watchlist digest",
                body=digest,
            )
            sent += 1
        except Exception as e:
            print(f"[watchlist_alerts] {email}: {e}")
            continue

    print(f"[watchlist_alerts] sent {sent} digest(s)")
