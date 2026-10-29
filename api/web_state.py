"""Getting started checklist, unread alert count and in-app feedback for the web app.

Kill switches (unset = on):
  HSF_GETTING_STARTED=0  the checklist is never shown
  HSF_FEEDBACK_NOTIFY=0  feedback is stored but the admins aren't messaged
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

from api import user_data
from api.deps import _user, entitlements_for

log = logging.getLogger("hsf_api")

UNREAD_CAP = 99
FIRST_VISIT_DAYS = 7  # with no Alerts visit on record, count what fired in the last week


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip() != "0"


def _state(user: str) -> Dict[str, Optional[str]]:
    from db import web_state

    try:
        return web_state.get_state(user)
    except Exception as e:
        log.warning(json.dumps({"event": "web_state_read_failed", "error": type(e).__name__}))
        return {f: None for f in web_state.FIELDS}


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:
        log.warning(json.dumps({"event": "getting_started_check_failed", "error": type(e).__name__}))
        return default


def _has_stocks(user: str) -> bool:
    return any(int(w.get("symbol_count") or 0) > 0 for w in user_data.list_watchlists(user))


def _has_alert(user: str) -> bool:
    from api import alert_rules

    return bool(user_data.list_alerts(user)) or bool(alert_rules.list_rules(user))


def _has_phone_delivery(user: str) -> bool:
    from api import chat_delivery, devices

    chats = chat_delivery.summary(user)
    if any((chats.get(k) or {}).get("connected") for k in ("telegram", "discord")):
        return True
    return bool(devices.list_devices(user))


def _verified(user: str) -> bool:
    from api import account as acct

    return bool(acct.is_verified(user))


def getting_started(account: Dict[str, Any]) -> Dict[str, Any]:
    """Four first steps for a new account, each ticked from what the account has done."""
    from api import webpush

    user = _user(account)
    state = _state(user)
    paid = webpush.plan_allows(account)
    if paid:
        delivery = {"id": "delivery", "title": "Get alerts on your phone",
                    "detail": "Connect Telegram or Discord, or turn on notifications in this browser.",
                    "href": "/alerts#delivery", "done": _safe(lambda: _has_phone_delivery(user), False)}
    else:
        delivery = {"id": "delivery", "title": "Confirm your email",
                    "detail": "Alerts and the morning email go to your inbox once it's confirmed.",
                    "href": "/account", "done": _safe(lambda: _verified(user), True)}
    steps: List[Dict[str, Any]] = [
        {"id": "watchlist", "title": "Save a few stocks to a watchlist",
         "detail": "Today and the morning email then show how your stocks are doing.",
         "href": "/watchlists", "done": _safe(lambda: _has_stocks(user), False)},
        {"id": "alert", "title": "Set your first alert",
         "detail": "Get told when a stock you follow scores high or moves.",
         "href": "/alerts?new=1", "done": _safe(lambda: _has_alert(user), False)},
        delivery,
        {"id": "guide", "title": "See how the HSF Score works",
         "detail": "Two minutes on what the score means, and what it doesn't.",
         "href": "/how-hsf-works", "done": bool(state.get("guide_seen_at"))},
    ]
    done = sum(1 for s in steps if s["done"])
    dismissed = bool(state.get("onboarding_dismissed_at"))
    show = _enabled("HSF_GETTING_STARTED") and not dismissed and done < len(steps)
    return {"show": show, "dismissed": dismissed, "done": done, "total": len(steps), "steps": steps}


def mark(user: str, field: str) -> str:
    from db import web_state

    return web_state.mark(user, field)


def _parse(stamp: Optional[str]) -> Optional[dt.datetime]:
    if not stamp:
        return None
    try:
        t = dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def unread_alerts(user: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Alerts (ticker alerts and alert rules) that fired since the Alerts page was last opened."""
    from api import alert_rules

    seen = _parse(_state(user).get("alerts_seen_at"))
    since = seen or (now or dt.datetime.now(dt.timezone.utc)) - dt.timedelta(days=FIRST_VISIT_DAYS)
    try:
        events = alert_rules.list_events(user, limit=UNREAD_CAP, triggered_after=since)
    except Exception as e:
        log.warning(json.dumps({"event": "unread_alerts_failed", "error": type(e).__name__}))
        events = []
    return {"count": min(len(events), UNREAD_CAP), "seen_at": seen.isoformat() if seen else None}


FEEDBACK_KINDS = {"problem": "Problem", "idea": "Idea", "praise": "Praise", "other": "Other"}


def send_feedback(account: Dict[str, Any], kind: str, message: str, path: Optional[str],
                  build: Optional[str]) -> None:
    """Stored first (never lost), then sent to the admins' chats, or emailed when none is linked."""
    from db import web_state

    user = _user(account)
    plan = entitlements_for(account)["tier"]
    if account.get("founder_tier"):  # founders' feedback is marked so it stands out
        from domain import founding as founding_rules

        plan = f"{plan}, {founding_rules.TIER_LABEL.get(account['founder_tier'], 'founder')}"
    clean_path = (path or "").split("?")[0].split("#")[0][:200] or None
    web_state.add_feedback(user, kind, message.strip(), clean_path, plan, (build or "")[:64] or None)
    if not _enabled("HSF_FEEDBACK_NOTIFY"):
        return
    try:
        from services.ops_notify import notify_ops

        label = FEEDBACK_KINDS.get(kind, "Feedback")
        text = (f"From: {user} ({plan})\nPage: {clean_path or 'unknown'}\n"
                f"Build: {build or 'unknown'}\n\n{message.strip()}")
        notify_ops(f"HSF feedback{' (founder)' if account.get('founder_tier') else ''}: {label}", text)
    except Exception as e:
        log.warning(json.dumps({"event": "feedback_notify_failed", "error": type(e).__name__}))
