"""First-week tips and the come-back email (api/ops_jobs.py runs this once a day).

    tip_watchlist   2-6 days after sign-up, verified, and no watchlist stock saved yet
    tip_alert       5-9 days after sign-up, verified, and no alert or alert rule yet
    comeback        last seen in the app 14-30 days ago (db/user_activity.py): today's
                    top Scanner names and a link back. Once per absence.

Each email is claimed in the lifecycle email log first (db.referrals.claim_email), so it
goes out once. All three respect the "tips" email preference and carry its unsubscribe
link. Admin and inactive accounts are skipped.

Env: HSF_ONBOARDING_EMAILS_ENABLED=0 turns them off (checked by api/ops_jobs.py too).
Best effort: never raises.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("hsf.onboarding_emails")

TIP_WATCHLIST = "tip_watchlist"
TIP_ALERT = "tip_alert"
COMEBACK = "comeback"
MAX_PER_RUN = 200
UTC = dt.timezone.utc


def enabled() -> bool:
    return os.environ.get("HSF_ONBOARDING_EMAILS_ENABLED", "1").strip() != "0"


def _parse(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


# ---- content ------------------------------------------------------------------------------------
def tip_watchlist_content() -> tuple:
    from services.lifecycle_emails import _base_url, _button

    url = f"{_base_url()}/watchlists"
    subject = "Tip: save the stocks you follow"
    text = ("A quick tip for your first week on HSFinest.AI.\n\n"
            "Save the stocks you follow to a watchlist. Today then shows how they're doing, which ones "
            "the Scanner ranks, and who reports earnings this week.\n\n"
            "One tap saves today's top 5 setups as a starter list, or add your own tickers: " + url + "\n")
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Save the stocks you follow</h2>"
             "<p>Save them to a watchlist and Today shows how they're doing, which ones the Scanner ranks, and "
             "who reports earnings this week.</p>"
             "<p>One tap saves today's top 5 setups as a starter list, or add your own tickers.</p>"
             + _button(url, "Make a watchlist"))
    return subject, text, inner


def tip_alert_content() -> tuple:
    from services.lifecycle_emails import _base_url, _button

    url = f"{_base_url()}/alerts"
    subject = "Tip: let HSF watch a stock for you"
    text = ("Set an alert and HSF tells you when a stock crosses a price, moves a lot in a day, or shows up "
            "as a ranked setup in the Scanner, so you don't have to keep checking.\n\n"
            "The Quick start buttons set one up in a tap: " + url + "\n")
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Let HSF watch a stock for you</h2>"
             "<p>Set an alert and HSF tells you when a stock crosses a price, moves a lot in a day, or shows up as "
             "a ranked setup in the Scanner, so you don't have to keep checking.</p>"
             "<p>The Quick start buttons set one up in a tap.</p>" + _button(url, "Set an alert"))
    return subject, text, inner


def comeback_content(top: List[Dict[str, Any]]) -> tuple:
    from services.lifecycle_emails import _base_url, _button

    url = f"{_base_url()}/today"
    subject = "Here's what the Scanner likes today"
    if top:
        rows_t = "".join(f"- {r['ticker']}: HSF Score {r['score']:.0f}\n" for r in top)
        rows_h = "".join(f"<li><strong>{html.escape(r['ticker'])}</strong>: HSF Score {r['score']:.0f}</li>"
                         for r in top)
        lead_t = f"It's been a couple of weeks. The top-ranked stocks in today's scan:\n{rows_t}\n"
        lead_h = ("<p>It's been a couple of weeks. The top-ranked stocks in today's scan:</p>"
                  f"<ul style='padding-left:20px'>{rows_h}</ul>")
    else:
        lead_t = "It's been a couple of weeks. The Scanner keeps ranking the whole market every trading day.\n\n"
        lead_h = "<p>It's been a couple of weeks. The Scanner keeps ranking the whole market every trading day.</p>"
    text = lead_t + f"See the full list and how your watchlist did: {url}\n"
    inner = ("<h2 style='font-size:20px;margin:16px 0 8px'>Here's what the Scanner likes today</h2>"
             + lead_h + _button(url, "Open Today"))
    return subject, text, inner


# ---- selection (pure) ---------------------------------------------------------------------------
def pick(accounts: Iterable[Dict[str, Any]], *, engaged_watchlist: set, engaged_alerts: set,
         last_active: Dict[str, str], now: dt.datetime) -> List[Tuple[str, str, str]]:
    """[(email, kind, ref)] due now. Accounts: {user, created_at, verified, is_admin, is_active}."""
    out = []
    for a in accounts:
        user = str(a.get("user") or "").strip().lower()
        if "@" not in user or a.get("is_admin") or a.get("is_active") is False or not a.get("verified"):
            continue
        created = _parse(a.get("created_at"))
        age = (now - created).days if created else None
        if age is not None and 2 <= age <= 6 and user not in engaged_watchlist:
            out.append((user, TIP_WATCHLIST, ""))
        if age is not None and 5 <= age <= 9 and user not in engaged_alerts:
            out.append((user, TIP_ALERT, ""))
        seen = _parse(last_active.get(user))
        if seen is not None and 14 <= (now - seen).days <= 30 and (age is None or age >= 14):
            out.append((user, COMEBACK, seen.date().isoformat()))
    return out


# ---- data ---------------------------------------------------------------------------------------
def _rows(conn, sql: str, args: tuple = ()) -> List[tuple]:
    cur = conn.cursor()
    try:
        cur.execute(sql, args)
        rows = cur.fetchall() or []
    finally:
        cur.close()
    return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]


def load_inputs(now: dt.datetime) -> Dict[str, Any]:
    from db.engine import get_neon_conn
    from db.user_activity import last_active_map

    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("database unavailable")
    try:
        since = now - dt.timedelta(days=10)
        accounts = [{"user": r[0], "created_at": r[1], "verified": r[2], "is_admin": r[3], "is_active": r[4]}
                    for r in _rows(conn, "SELECT lower(username), created_at, email_verified, is_admin, is_active "
                                         "FROM users WHERE created_at >= %s", (since,))]
        names = [a["user"] for a in accounts]
        watch = {r[0] for r in _rows(conn, "SELECT DISTINCT lower(w.user_id) FROM watchlists w JOIN watchlist_items i "
                                           "ON i.watchlist_id = w.id WHERE lower(w.user_id) = ANY(%s)", (names,))} if names else set()
        alerts = set()
        if names:
            alerts |= {r[0] for r in _rows(conn, "SELECT DISTINCT lower(user_id) FROM user_alerts "
                                                 "WHERE lower(user_id) = ANY(%s)", (names,))}
            try:
                alerts |= {r[0] for r in _rows(conn, "SELECT DISTINCT lower(user_id) FROM hsf_alert_rules "
                                                     "WHERE lower(user_id) = ANY(%s)", (names,))}
            except Exception:
                conn.rollback()
        last_active = last_active_map()
        returning = [u for u, t in last_active.items()
                     if (p := _parse(t)) is not None and 14 <= (now - p).days <= 30]
        if returning:
            known = {a["user"] for a in accounts}
            for r in _rows(conn, "SELECT lower(username), created_at, email_verified, is_admin, is_active "
                                 "FROM users WHERE lower(username) = ANY(%s)", ([u for u in returning if u not in known],)):
                accounts.append({"user": r[0], "created_at": r[1], "verified": r[2], "is_admin": r[3], "is_active": r[4]})
        return {"accounts": accounts, "watch": watch, "alerts": alerts, "last_active": last_active}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def top_scanner_names(limit: int = 3) -> List[Dict[str, Any]]:
    """The latest saved Scanner run's top names by HSF score (frozen hsf_scanner rows)."""
    try:
        from db.engine import get_neon_conn

        conn = get_neon_conn()
        if conn is None:
            return []
        try:
            rows = _rows(conn, "SELECT ticker, setup_score FROM signal_outcomes WHERE source = 'opportunity' "
                               "AND signal_type = 'hsf_scanner' AND fired_at = (SELECT MAX(fired_at) FROM signal_outcomes "
                               "WHERE source = 'opportunity' AND signal_type = 'hsf_scanner') "
                               "AND setup_score IS NOT NULL ORDER BY setup_score DESC LIMIT %s", (int(limit),))
        finally:
            conn.close()
        return [{"ticker": str(r[0]).upper(), "score": float(r[1])} for r in rows]
    except Exception:
        return []


# ---- run ----------------------------------------------------------------------------------------
def run(now: Optional[dt.datetime] = None, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now(UTC)
    counts: Dict[str, Any] = {"sent": 0, "skipped": 0, "failed": 0}
    if not enabled():
        counts["disabled"] = True
        return counts
    try:
        from db import referrals as store
        from db.email_prefs import unsubscribe_url, wants_email
        from ui.email_utils import send_digest_email

        data = inputs or load_inputs(now)
    except Exception as e:
        log.warning(json.dumps({"event": "onboarding_emails_failed", "error": type(e).__name__}))
        counts["error"] = type(e).__name__
        return counts
    due = pick(data["accounts"], engaged_watchlist=data["watch"], engaged_alerts=data["alerts"],
               last_active=data["last_active"], now=now)
    top = None
    for email, kind, ref in due[:MAX_PER_RUN]:
        try:
            if not wants_email(email, "tips"):
                counts["skipped"] += 1
                continue
            if not store.claim_email(email, kind, ref):
                continue
            if kind == TIP_WATCHLIST:
                subject, text, inner = tip_watchlist_content()
            elif kind == TIP_ALERT:
                subject, text, inner = tip_alert_content()
            else:
                top = top if top is not None else top_scanner_names()
                subject, text, inner = comeback_content(top)
            if send_digest_email(email, subject, inner, text, unsubscribe_url=unsubscribe_url(email, "tips")):
                counts["sent"] += 1
                counts[kind] = counts.get(kind, 0) + 1
            else:
                store.release_email(email, kind, ref)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "onboarding_email_failed", "kind": kind, "error": type(e).__name__}))
    return counts
