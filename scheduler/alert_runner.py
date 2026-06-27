"""Headless evaluation of per-user alerts against the latest scan snapshot.

Runs from the scheduler right after the scans save a fresh snapshot. For each
enabled alert it checks the snapshot, and on a match it (a) records an in-app
event and (b) emails the user (verified addresses only). Repeat sends are
throttled by last_fired_at so a standing condition doesn't email every run.

Best-effort: a failure on one alert never aborts the rest, and the whole run is
a no-op when ALERTS_ENABLED is off.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


def _latest_snapshot_df():
    """Load the most recent saved scan as a DataFrame, or None."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_runs(limit=10) or []
        snap = next((r for r in runs if r.get("is_snapshot")), None) or (
            runs[0] if runs else None
        )
        if not snap:
            return None
        raw = load_run_results(snap["id"])
        return normalize_results_to_df(raw) if raw else None
    except Exception as e:  # pragma: no cover - best effort loader
        print(f"[alert_runner] snapshot load failed: {e}")
        return None


def _col(df, *candidates: str) -> Optional[str]:
    """Return the first matching column name (case-insensitive), or None."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _throttled(last_fired_at, throttle_hours: float) -> bool:
    """True if the alert fired within the throttle window."""
    if last_fired_at is None:
        return False
    try:
        now = _dt.datetime.now(_dt.timezone.utc)
        ts = last_fired_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.timezone.utc)
        return (now - ts).total_seconds() < throttle_hours * 3600.0
    except (TypeError, ValueError, AttributeError):
        return False


def _evaluate(alert: Dict[str, Any], df, watch_tickers: set) -> List[str]:
    """Return human-readable match lines for an alert, or [] when none."""
    ticker_col = _col(df, "Ticker", "Symbol")
    if ticker_col is None:
        return []
    score_col = _col(df, "BreakoutScore")
    last_col = _col(df, "Last", "Close", "Price")

    atype = alert.get("alert_type")
    lines: List[str] = []

    if atype == "breakout":
        if score_col is None:
            return []
        threshold = float(alert.get("threshold") or 0)
        sub = df
        if alert.get("watchlist_only"):
            if not watch_tickers:
                return []
            sub = df[df[ticker_col].astype(str).str.upper().isin(watch_tickers)]
        for _, row in sub.iterrows():
            try:
                score = float(row[score_col])
            except (TypeError, ValueError):
                continue
            if score >= threshold:
                lines.append(f"{str(row[ticker_col]).upper()}: BreakoutScore {score:.1f} (≥ {threshold:g})")

    elif atype == "watchlist":
        if not watch_tickers:
            return []
        sub = df[df[ticker_col].astype(str).str.upper().isin(watch_tickers)]
        for _, row in sub.iterrows():
            tk = str(row[ticker_col]).upper()
            if score_col is not None:
                try:
                    lines.append(f"{tk}: in scan results (BreakoutScore {float(row[score_col]):.1f})")
                    continue
                except (TypeError, ValueError):
                    pass
            lines.append(f"{tk}: in scan results")

    elif atype == "price":
        if last_col is None:
            return []
        tk = str(alert.get("ticker") or "").upper()
        direction = (alert.get("direction") or "above").lower()
        target = float(alert.get("threshold") or 0)
        match = df[df[ticker_col].astype(str).str.upper() == tk]
        if match.empty:
            return []
        try:
            last = float(match.iloc[0][last_col])
        except (TypeError, ValueError):
            return []
        crossed = (direction == "above" and last >= target) or (
            direction == "below" and last <= target
        )
        if crossed:
            lines.append(f"{tk}: Last {last:.2f} is {direction} {target:.2f}")

    return lines


def _is_verified(email: str) -> bool:
    try:
        from db.email_verification import is_email_verified

        return bool(is_email_verified(email))
    except Exception:
        return False


def run_alerts() -> None:
    """Evaluate all enabled alerts against the latest snapshot and notify."""
    try:
        from config import ALERT_THROTTLE_HOURS, ALERTS_ENABLED
    except Exception:
        return
    if not ALERTS_ENABLED:
        return

    try:
        from db.alerts import (
            list_all_enabled_alerts,
            mark_alert_fired,
            record_alert_event,
        )
    except Exception as e:
        print(f"[alert_runner] import failed: {e}")
        return

    df = _latest_snapshot_df()
    if df is None or len(df) == 0:
        print("[alert_runner] no snapshot to evaluate")
        return

    try:
        alerts = list_all_enabled_alerts() or []
    except Exception as e:
        print(f"[alert_runner] could not load alerts: {e}")
        return

    if not alerts:
        print("[alert_runner] no enabled alerts")
        return

    # Resolve each user's watchlist tickers once (lazily, cached per user).
    watch_cache: Dict[str, set] = {}

    def _watch_for(user_id: str) -> set:
        if user_id in watch_cache:
            return watch_cache[user_id]
        tickers: set = set()
        try:
            from db.watchlists import get_watchlist_tickers, list_watchlists

            for wl in list_watchlists(user_id) or []:
                tickers.update(
                    str(t).upper()
                    for t in (get_watchlist_tickers(wl.get("id"), user_id) or [])
                )
        except Exception:
            pass
        watch_cache[user_id] = tickers
        return tickers

    fired = 0
    emailed = 0
    for alert in alerts:
        try:
            if _throttled(alert.get("last_fired_at"), float(ALERT_THROTTLE_HOURS)):
                continue
            user_id = str(alert.get("user_id") or "")
            needs_watch = alert.get("alert_type") == "watchlist" or alert.get(
                "watchlist_only"
            )
            watch = _watch_for(user_id) if needs_watch else set()

            lines = _evaluate(alert, df, watch)
            if not lines:
                continue

            # Cap the body so a broad breakout alert doesn't email 100 lines.
            shown = lines[:25]
            extra = len(lines) - len(shown)
            body = "\n".join(shown)
            if extra > 0:
                body += f"\n…and {extra} more."

            label = {
                "breakout": "Breakout alert",
                "watchlist": "Watchlist alert",
                "price": "Price alert",
            }.get(alert.get("alert_type"), "Alert")

            first_ticker = alert.get("ticker") or (
                lines[0].split(":", 1)[0] if lines else None
            )
            record_alert_event(user_id, alert.get("id"), first_ticker, f"{label}: {body}")
            mark_alert_fired(alert.get("id"))
            fired += 1

            if "@" in user_id and _is_verified(user_id):
                try:
                    from ui.email_utils import send_alert_email

                    if send_alert_email(
                        to_address=user_id,
                        subject=f"📈 {label} triggered",
                        body=body,
                    ):
                        emailed += 1
                except Exception as e:
                    print(f"[alert_runner] email to {user_id} failed: {e}")
        except Exception as e:  # never let one alert kill the run
            print(f"[alert_runner] alert {alert.get('id')} failed: {e}")
            continue

    print(f"[alert_runner] fired {fired} alert(s), emailed {emailed}")
