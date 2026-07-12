"""Evening wrap email (Pro+): how today went, sent from the postmarket slot.

Per verified Pro+ user with a watchlist: today's watchlist closing moves, the
alerts that fired for them today, and tomorrow's earnings on their names.
Gated by the same MORNING_DIGEST_ENABLED flag (one switch for daily emails)
and throttled once per day. Deterministic, best-effort, never raises.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

try:
    from ui.monitoring import capture as _capture
except Exception:  # pragma: no cover
    def _capture(exc: BaseException) -> None:
        pass

_WRAP_KEY = "evening_wrap"


def _todays_events(user_id: str) -> List[str]:
    """Messages of alerts that fired for this user today (UTC)."""
    try:
        from db.alerts import list_recent_events

        today = datetime.now(timezone.utc).date()
        events = list_recent_events(user_id, limit=25) or []
        seen: List[str] = []
        for ev in events:
            fired = ev.get("fired_at")
            if fired is None or not hasattr(fired, "date") or fired.date() != today:
                continue
            msg = str(ev.get("message") or "")
            if msg and msg not in seen:
                seen.append(msg)
        return seen[:8]
    except Exception:
        return []


def _tomorrows_earnings(tickers: List[str]) -> List[str]:
    try:
        from db.earnings import load_earnings_map

        tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
        emap = load_earnings_map(tickers) or {}
        return sorted(
            {str(s).upper() for s, d in emap.items() if d == tomorrow and str(s).upper() in tickers}
        )
    except Exception:
        return []


def _compose_wrap(
    watch_rows: List[Dict[str, Any]],
    fired_today: List[str],
    earnings_tomorrow: List[str],
) -> tuple:
    from scheduler.morning_digest import _movers_table, _movers_text

    date_s = datetime.now(timezone.utc).strftime("%A, %b %d")
    html = [f"<p style='color:#666;margin:0 0 12px'>Evening wrap · {date_s}</p>"]
    text = [f"Evening wrap · {date_s}", ""]

    html.append("<h3 style='margin:16px 0 6px'>📋 Your watchlist today</h3>")
    html.append(_movers_table(watch_rows, show_gap=False))
    text += ["Your watchlist today:", _movers_text(watch_rows), ""]

    if fired_today:
        items = "".join(f"<li>{m}</li>" for m in fired_today)
        html.append(
            f"<h3 style='margin:16px 0 6px'>🔔 Alerts that fired today</h3><ul>{items}</ul>"
        )
        text += ["Alerts that fired today:"] + [f"  - {m}" for m in fired_today] + [""]

    if earnings_tomorrow:
        names = ", ".join(earnings_tomorrow)
        html.append(
            "<h3 style='margin:16px 0 6px'>📅 Earnings tomorrow (your watchlist)</h3>"
            f"<p>{names} ⚠️</p>"
        )
        text += ["Earnings tomorrow (your watchlist):", f"  {names}", ""]

    return "".join(html), "\n".join(text)


def run_evening_wrap(force: bool = False) -> None:
    """Send the evening wrap to eligible Pro+ users (once/day)."""
    try:
        from config import MORNING_DIGEST_ENABLED, MORNING_DIGEST_MAX_USERS
    except Exception:
        return
    if not MORNING_DIGEST_ENABLED:
        return

    if not force:
        try:
            from db.earnings import should_refresh_earnings_today

            if not should_refresh_earnings_today(_WRAP_KEY):
                print("[evening_wrap] already sent today; skipping")
                return
        except Exception:
            pass

    try:
        from auth.tiering import get_user_tier, has_min_tier
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from market_data import build_day_trader_metrics
        from ui.email_utils import send_digest_email
    except Exception as e:
        print(f"[evening_wrap] import failed: {e}")
        return

    try:
        users = load_users() or {}
    except Exception:
        users = {}

    sent = 0
    for username in list(users.keys())[:MORNING_DIGEST_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            continue
        try:
            if not has_min_tier(get_user_tier(email, users), "pro"):
                continue
            from db.email_verification import is_email_verified

            if not is_email_verified(email):
                continue
        except Exception:
            continue

        try:
            wls = list_watchlists(email) or []
            tickers: List[str] = []
            for wl in wls:
                tickers.extend(get_watchlist_tickers(wl.get("id"), email) or [])
            tickers = sorted({str(t).strip().upper() for t in tickers if t})
            if not tickers:
                continue

            watch_rows = build_day_trader_metrics(tickers, with_rvol=False)
            html_inner, text_inner = _compose_wrap(
                watch_rows, _todays_events(email), _tomorrows_earnings(tickers)
            )
            send_digest_email(
                to_address=email,
                subject="Your evening market wrap",
                html_inner=html_inner,
                text_inner=text_inner,
            )
            sent += 1
        except Exception as e:
            print(f"[evening_wrap] {email}: {e}")
            _capture(e)
            continue

    if sent > 0:
        try:
            from db.earnings import mark_earnings_refreshed_today

            mark_earnings_refreshed_today(_WRAP_KEY)
        except Exception:
            pass
    print(f"[evening_wrap] sent {sent} wrap(s)")
