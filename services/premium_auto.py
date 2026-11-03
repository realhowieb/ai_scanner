"""Premium automations (2026-11-03): things Premium members used to press a button for.

api/ops_jobs.py calls these on its 5-minute pass. Each one is claimed once per key
(services.ops_notify.once or the lifecycle email log), so a restart never repeats it.

    brief summary     trading days 08:00-18:00 ET: after each market scan, the AI Brief
                      summary is written once for everyone (api.ai.brief_narrative) and
                      saved, so the Brief shows it on load. One AI call per scan.
    stock notes       after each market scan: stocks on Premium members' watchlists that
                      newly became ranked setups get their AI note written and saved
                      (api.ai.ticker_note, shared by everyone), so the stock page shows it on
                      load. At most HSF_AUTO_NOTES_PER_SCAN (default 15) per scan.
    watchlist email   Sunday from 21:00 UTC: each Premium member gets one email with an AI
                      rundown of every watchlist (up to 5), which is also saved for the
                      Watchlists page. Respects the "weekly" email preference.
    journal email     Friday from 21:30 UTC: Premium members with 5+ closed trades whose
                      journal changed since the last note get the AI journal coach by email
                      (saved for the Journal page). Respects the "weekly" email preference.

The AI summary in scheduled-scan emails lives in services/scheduled_scans.py, and the
stop and target alerts for logged trades in api/trade_alerts.py.

Env (default on; 0 stops one): HSF_AUTO_BRIEF, HSF_AUTO_NOTES, HSF_AUTO_WATCHLIST_EMAIL,
HSF_AUTO_JOURNAL_EMAIL. HSF_AUTO_NOTES_PER_SCAN, HSF_AUTO_MAX_USERS (default 300).
Every AI call goes through ui.ai (kill switch AI_ENABLED, per-account daily cap). Never raises.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("hsf.premium_auto")

UTC = dt.timezone.utc
FLAG = "can_auto_research"
MAX_WATCHLISTS_PER_EMAIL = 5
NOTE_KEEP_DAYS = 3


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip() != "0"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _et(now: dt.datetime) -> dt.datetime:
    from analytics import market_calendar as mc

    return now.astimezone(mc.ET)


def iso_week(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def _ai_ready() -> bool:
    try:
        from ui.ai import is_configured

        return bool(is_configured())
    except Exception:
        return False


# ---- who gets them --------------------------------------------------------------------------------
def premium_accounts(limit: Optional[int] = None) -> List[str]:
    """Active Premium (and admin) accounts with an email username."""
    from auth.tiering import get_user_tier, has_min_tier
    from db.users import load_users
    from scheduler.morning_digest import _email_tier_key

    users = load_users() or {}
    out: List[str] = []
    for username in users:
        email = str(username or "").strip().lower()
        record = users.get(username) or {}
        if "@" not in email or record.get("is_active") is False:
            continue
        try:
            if has_min_tier(_email_tier_key(email, record, users, get_user_tier), "premium"):
                out.append(email)
        except Exception:
            continue
        if len(out) >= (limit or _int_env("HSF_AUTO_MAX_USERS", 300)):
            break
    return out


def _can_email(email: str, kind: str = "weekly") -> bool:
    from db.email_prefs import wants_email
    from db.email_verification import is_email_verified

    return is_email_verified(email) and wants_email(email, kind)


def _watchlists(email: str) -> List[Dict[str, Any]]:
    """The account's non-empty watchlists, default first, each with its items."""
    from api import user_data

    lists = user_data.list_watchlists(email) or []
    lists = sorted(lists, key=lambda w: (not w.get("is_default"), int(w["id"])))
    out = []
    for w in lists:
        wl = user_data.get_watchlist(email, int(w["id"]))
        if wl.get("items"):
            out.append(wl)
    return out


# ---- 1. brief summary -----------------------------------------------------------------------------
def brief_due(now: dt.datetime) -> bool:
    from analytics import market_calendar as mc

    et = _et(now)
    return mc.is_trading_day(et.date()) and 8 <= et.hour < 18


def run_brief(now: dt.datetime, *, claim=None, release=None) -> Dict[str, Any]:
    """Write the Brief's AI summary once for the latest snapshot."""
    if not brief_due(now) or not _ai_ready():
        return {"skipped": "not_due"}
    from api import ai
    from api.market import _brief_core
    from services import ops_notify

    claim = claim or ops_notify.once
    release = release or ops_notify.release
    ts = ((_brief_core() or {}).get("data") or {}).get("snapshot_time")
    if not ts:
        return {"skipped": "no_snapshot"}
    if ai.stored(ai.SHARED_USER, "brief_narrative", ts) or not claim("auto_brief", str(ts)):
        return {"skipped": "done"}
    try:
        out = ai.brief_narrative(None)
    except Exception as e:
        release("auto_brief", str(ts))
        return {"error": type(e).__name__}
    return {"written": bool(out.get("text")), "snapshot_time": str(ts)}


# ---- 3. stock notes for newly ranked watchlist stocks ---------------------------------------------
def newly_ranked(obs: Dict[str, Any]) -> List[str]:
    """Tickers ranked in the latest scan that weren't ranked in the previous one, best rank first.
    Empty without a previous scan (everything would look new)."""
    if not obs or obs.get("previous_run_id") is None:
        return []
    prev = obs.get("previous_ranked") or {}
    new = [(t, v) for t, v in (obs.get("ranked") or {}).items() if t not in prev]
    return [t for t, _ in sorted(new, key=lambda kv: int(kv[1].get("rank") or 10 ** 6))]


def pick_note_tickers(new: List[str], watched: Dict[str, int], cap: int) -> List[str]:
    """The newly ranked stocks that are on Premium watchlists, the most-watched first, then by
    rank, at most `cap`."""
    order = {t: i for i, t in enumerate(new)}
    hits = [t for t in new if t in watched]
    hits.sort(key=lambda t: (-watched[t], order[t]))
    return hits[:cap]


def run_notes(now: dt.datetime, *, claim=None, accounts: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _ai_ready():
        return {"skipped": "ai_off"}
    from api import ai, watchlist_intel
    from services import ops_notify

    claim = claim or ops_notify.once
    obs = watchlist_intel.market_observation()
    if not obs:
        return {"skipped": "no_scan"}
    rid = str(obs["run_id"])
    new = newly_ranked(obs)
    if not new or not claim("auto_notes", rid):
        return {"skipped": "nothing_new" if not new else "done"}
    watched: Dict[str, int] = {}
    for email in accounts if accounts is not None else premium_accounts():
        try:
            for wl in _watchlists(email):
                for t in {str(i["ticker"]).upper() for i in wl.get("items") or []}:
                    watched[t] = watched.get(t, 0) + 1
        except Exception:
            continue
    picks = pick_note_tickers(new, watched, _int_env("HSF_AUTO_NOTES_PER_SCAN", 15))
    written = 0
    for t in picks:
        try:
            if (ai.ticker_note(None, t) or {}).get("text"):
                written += 1
        except Exception as e:
            log.info(json.dumps({"event": "auto_note_failed", "ticker": t, "error": type(e).__name__}))
    try:
        from db import premium_auto

        premium_auto.prune("ticker_note", now - dt.timedelta(days=NOTE_KEEP_DAYS))
    except Exception:
        pass
    return {"run_id": rid, "newly_ranked": len(new), "on_watchlists": len(picks), "written": written}


# ---- email pieces ---------------------------------------------------------------------------------
def paragraphs(text: str) -> str:
    """AI text -> email HTML: '- ' lines become a list, other lines paragraphs."""
    from ui import email_kit as k

    out: List[str] = []
    bullets: List[str] = []

    def flush() -> None:
        if bullets:
            out.append("<ul style='margin:4px 0 0 18px;padding:0'>" + "".join(
                f"<li style='margin:0 0 6px;font-size:14px;line-height:1.5;color:{k.INK}'>{k.esc(b)}</li>"
                for b in bullets) + "</ul>")
            bullets.clear()

    for line in str(text or "").splitlines():
        s = line.strip()
        if not s:
            flush()
            continue
        if s.startswith(("- ", "* ", "• ")):
            bullets.append(s[2:].strip())
        else:
            flush()
            out.append(f"<p style='margin:0 0 8px;font-size:14px;line-height:1.5;color:{k.INK}'>{k.esc(s)}</p>")
    flush()
    return "".join(out)


DISCLAIMER = "Written by AI from HSF scan facts only. It's research, not advice or a prediction."


def compose_watchlists(rundowns: List[Tuple[Dict[str, Any], str]], *, base_url: str) -> Tuple[str, str, str, str]:
    """(subject, text, html_inner, glance) for the Sunday watchlist email. Pure."""
    from ui import email_kit as k

    n = len(rundowns)
    subject = (f"Your watchlist rundown: {rundowns[0][0].get('name')}" if n == 1
               else f"Your {n} watchlists, rundown for the week ahead")
    html = ""
    text = ["Your watchlists, rundown for the week ahead", ""]
    for wl, body in rundowns:
        syms = len(wl.get("items") or [])
        html += k.card(f"⭐ {k.esc(wl.get('name'))}", paragraphs(body),
                       sub=f"{syms} stock{'s' if syms != 1 else ''} · from the latest scan")
        text += [str(wl.get("name")), body, ""]
    html += k.card("See every stock", k.note(DISCLAIMER) + k.button(f"{base_url}/watchlists", "Open Watchlists →"))
    text += [DISCLAIMER, f"Open Watchlists: {base_url}/watchlists"]
    glance = k.glance_text([f"<b style='color:#fff'>{n} watchlist{'s' if n != 1 else ''}</b> summed up by AI.",
                            "Earnings due this week are called out."])
    return subject, "\n".join(text), html, glance


def compose_journal(coach: str, closed: int, *, base_url: str) -> Tuple[str, str, str, str]:
    from ui import email_kit as k

    subject = "Your weekly journal coaching"
    html = k.card("📓 Patterns in your trades", paragraphs(coach),
                  sub=f"From your {closed} closed trades")
    html += k.card("Your journal", k.note("Written by AI from your own journal. It looks back at past results "
                                          "and isn't advice.") + k.button(f"{base_url}/journal", "Open Journal →"))
    text = ["Your weekly journal coaching", f"From your {closed} closed trades", "", coach, "",
            f"Open Journal: {base_url}/journal"]
    glance = k.glance_text([f"AI read your <b style='color:#fff'>{closed} closed trades</b> and found what stands out."])
    return subject, "\n".join(text), html, glance


def _send(email: str, kind_pref: str, subject: str, text: str, html: str, glance: str, title: str) -> bool:
    from db.email_prefs import unsubscribe_url
    from ui.email_utils import send_digest_email

    return bool(send_digest_email(email, subject, html, text, unsubscribe_url=unsubscribe_url(email, kind_pref),
                                  title_line=title, glance_html=glance))


def _per_account(job: str, key: str, accounts: Optional[List[str]], each: Callable[[str], str]) -> Dict[str, Any]:
    """Run `each(email)` once per Premium account (claimed in the lifecycle email log);
    it returns 'sent' or a skip reason. A failed or skipped send releases the claim."""
    from db import referrals as store

    counts: Dict[str, Any] = {"sent": 0, "skipped": {}}
    for email in accounts if accounts is not None else premium_accounts():
        try:
            if not _can_email(email):
                why = "no_email"
            elif not store.claim_email(email, job, key):
                why = "already_sent"
            else:
                why = each(email)
                if why != "sent":
                    store.release_email(email, job, key)
        except Exception as e:
            log.info(json.dumps({"event": f"{job}_failed", "error": type(e).__name__}))
            why = "error"
            try:
                store.release_email(email, job, key)
            except Exception:
                pass
        if why == "sent":
            counts["sent"] += 1
        else:
            counts["skipped"][why] = counts["skipped"].get(why, 0) + 1
    return counts


# ---- 2. Sunday watchlist email --------------------------------------------------------------------
def watchlist_email_due(now: dt.datetime) -> Optional[str]:
    """The ISO week of the coming Monday, Sunday from 21:00 UTC."""
    if now.weekday() == 6 and now.hour >= 21:
        return iso_week(now.date() + dt.timedelta(days=1))
    return None


def run_watchlist_email(now: dt.datetime, accounts: Optional[List[str]] = None) -> Dict[str, Any]:
    week = watchlist_email_due(now)
    if not week or not _ai_ready():
        return {"skipped": "not_due"}
    from api import ai
    from domain.plans import compute_entitlements, has_min_tier
    from services.lifecycle_emails import _base_url

    ent = compute_entitlements(tier_obj="premium", is_admin=False, has_min_tier_fn=has_min_tier)
    url = _base_url()

    def each(email: str) -> str:
        rundowns = []
        for wl in _watchlists(email)[:MAX_WATCHLISTS_PER_EMAIL]:
            try:
                text = (ai.watchlist_note(email, wl, ent) or {}).get("text")
            except Exception:
                text = None
            if text:
                rundowns.append((wl, text))
        if not rundowns:
            return "no_rundown"
        subject, text, html, glance = compose_watchlists(rundowns, base_url=url)
        return "sent" if _send(email, "weekly", subject, text, html, glance, "Watchlist rundown · week ahead") else "email_failed"

    return {"week": week, **_per_account("auto_watchlist_email", week, accounts, each)}


# ---- 4. Friday journal coach email ----------------------------------------------------------------
def journal_email_due(now: dt.datetime) -> Optional[str]:
    if now.weekday() == 4 and (now.hour, now.minute) >= (21, 30):
        return iso_week(now.date())
    return None


def run_journal_email(now: dt.datetime, accounts: Optional[List[str]] = None) -> Dict[str, Any]:
    week = journal_email_due(now)
    if not week or not _ai_ready():
        return {"skipped": "not_due"}
    from api import ai, ai_premium
    from services.lifecycle_emails import _base_url

    url = _base_url()

    def each(email: str) -> str:
        saved = ai.stored(email, "journal_coach", "latest") or {}
        if saved.get("trades_hash") and saved["trades_hash"] == ai_premium.journal_trades_hash(email):
            return "no_new_trades"
        out = ai_premium.journal_coach(email)
        if not out.get("text"):
            return "too_few_trades"
        subject, text, html, glance = compose_journal(out["text"], int(out.get("closed") or 0), base_url=url)
        return "sent" if _send(email, "weekly", subject, text, html, glance, "Journal coaching · this week") else "email_failed"

    return {"week": week, **_per_account("auto_journal_email", week, accounts, each)}


# ---- the pass -------------------------------------------------------------------------------------
def run_pass(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Called by api/ops_jobs.py every 5 minutes. Never raises."""
    now = now or dt.datetime.now(UTC)
    out: Dict[str, Any] = {}
    for name, env, fn in (("brief", "HSF_AUTO_BRIEF", run_brief), ("notes", "HSF_AUTO_NOTES", run_notes),
                          ("watchlist_email", "HSF_AUTO_WATCHLIST_EMAIL", run_watchlist_email),
                          ("journal_email", "HSF_AUTO_JOURNAL_EMAIL", run_journal_email)):
        if not _on(env):
            continue
        try:
            out[name] = fn(now)
        except Exception as e:
            log.warning(json.dumps({"event": "premium_auto_failed", "job": name, "error": type(e).__name__}))
            out[name] = {"error": type(e).__name__}
    return out
