"""Sunday weekly earnings email: the week ahead at a glance (api/ops_jobs.py, Sunday from 22:00 UTC).

One box per trading day, Monday to Friday. Each box shows that day's biggest reporters
by dollar volume (api.earnings_week) as logo tiles with the ticker, before-open or
after-close timing and the HSF Score, plus any of the reader's watchlist names, then
"+N more" and a link to the Earnings calendar in the app.

Pro and up only (the week-ahead calendar is a Pro feature), verified, active accounts
that haven't switched the "earnings" email off. Each account is claimed in the
lifecycle email log (db.referrals.claim_email, kind "weekly_earnings", ref the ISO week
of the Monday) before sending, so a restart never sends it twice.

Logos come from FMP's public image URL, checked by api.logo_check (cached a week): a
missing logo or a photo shows the ticker's first letter, and a white logo sits on a dark
chip. The ticker is always printed as text, so emails with images off still read fine.

Env: HSF_WEEKLY_EARNINGS_ENABLED=0 stops it (checked in api/ops_jobs.py),
HSF_WEEKLY_EARNINGS_MAX_USERS (default 500) caps one run. Best effort: never raises.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

from utils.swallowed import swallowed

log = logging.getLogger("hsf.weekly_earnings")

KIND = "weekly_earnings"
UTC = dt.timezone.utc
PER_DAY = 12
TILES_PER_ROW = 4


def _max_users() -> int:
    try:
        return max(1, int(os.environ.get("HSF_WEEKLY_EARNINGS_MAX_USERS", "500")))
    except ValueError:
        return 500


def iso_week(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


_TIME = {"bmo": ("☀️", "Before open"), "amc": ("🌙", "After close")}


def tile(name: Dict[str, Any], *, base_url: str) -> str:
    """One logo tile. name["logo"]: standard (light chip), light (white logo: dark chip),
    anything else (missing, a photo, unchecked): the ticker's first letter."""
    from ui import email_kit as k

    t = str(name["ticker"])
    icon, label = _TIME.get(name.get("time") or "", ("", "Time TBA"))
    kind = name.get("logo")
    if kind in ("standard", "light"):
        chip = "#1e293b" if kind == "light" else "#ffffff"
        img = (f"<div style='width:40px;height:40px;margin:0 auto 4px;border-radius:9px;background:{chip};"
               f"border:1px solid {k.LINE}'><img src='{k.esc(name['logo_url'])}' width='32' height='32' alt='' "
               "style='display:block;margin:4px;width:32px;height:32px;object-fit:contain'></div>")
    else:
        img = (f"<div style='width:40px;height:40px;line-height:40px;margin:0 auto 4px;border-radius:9px;"
               f"background:#e2e8f0;color:#334155;font-weight:700;font-size:16px'>{k.esc(t[:1])}</div>")
    star = "⭐ " if name.get("on_watchlist") else ""
    score = name.get("score")
    badge = k.score_badge(score) if score is not None else ""
    when = f"<span title='{label}'>{icon}</span>" if icon else f"<span style='color:{k.FAINT}'>TBA</span>"
    border = "#a5b4fc" if name.get("on_watchlist") else k.LINE
    return (f"<a href='{k.esc(base_url)}/stocks/{k.esc(t)}' style='text-decoration:none;color:{k.INK}'>"
            f"<div style='border:1px solid {border};border-radius:10px;padding:8px 4px;text-align:center;"
            f"background:#f8fafc'>{img}<div style='font-weight:700;font-size:13px'>{star}{k.esc(t)}</div>"
            f"<div style='font-size:12px;margin-top:2px'>{when} {badge}</div></div></a>")


def day_box(day: Dict[str, Any], *, base_url: str) -> str:
    from ui import email_kit as k

    d = dt.date.fromisoformat(day["date"])
    names = day["names"]
    cells = [tile(n, base_url=base_url) for n in names]
    rows = []
    for i in range(0, len(cells), TILES_PER_ROW):
        chunk = cells[i:i + TILES_PER_ROW]
        chunk += [""] * (TILES_PER_ROW - len(chunk))
        rows.append("<tr>" + "".join(f"<td width='25%' valign='top' style='padding:3px'>{c}</td>" for c in chunk) + "</tr>")
    grid = ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
            f"style='table-layout:fixed;width:100%'>{''.join(rows)}</table>") if rows else \
        k.empty("No companies we track report this day.")
    more = day["total"] - len(names)
    sub = f"{day['total']} compan{'y' if day['total'] == 1 else 'ies'} report"
    body = grid + (k.note(f"+ {more} more on the Earnings calendar") if more > 0 else "")
    return k.card(f"{d:%A}, {d:%b} {d.day}", body, sub=sub)


def compose(week: Dict[str, Any], *, base_url: str, watched: List[str]) -> tuple:
    """(subject, text, html_inner, glance_html). Pure."""
    from ui import email_kit as k

    start = dt.date.fromisoformat(week["start"])
    days = week["days"]
    total = sum(d["total"] for d in days)
    biggest = max(days, key=lambda d: d["total"]) if days else None
    subject = f"Earnings this week: {total} companies report (week of {start:%b} {start.day})"
    link = f"{base_url}/scanner?signal=earnings&eview=calendar"
    html = [day_box(d, base_url=base_url) for d in days]
    html.append(k.card("📅 See the whole week", k.note(
        "Biggest names first, by trading volume. ☀️ reports before the open, 🌙 after the close. "
        "⭐ marks your watchlist. Scores come from the latest HSF scan. Dates come from the earnings "
        "calendar and can move. Educational only, not financial advice.")
        + k.button(link, "Open the Earnings calendar →")))
    text = [f"Earnings this week (week of {start:%b} {start.day})", ""]
    for d in days:
        day = dt.date.fromisoformat(d["date"])
        names = ", ".join(("*" if n.get("on_watchlist") else "") + n["ticker"]
                          + {"bmo": " (before open)", "amc": " (after close)"}.get(n.get("time") or "", "")
                          for n in d["names"])
        more = d["total"] - len(d["names"])
        text.append(f"{day:%A %b} {day.day}: {names or 'none'}" + (f" + {more} more" if more > 0 else ""))
    text += ["", f"Earnings calendar: {link}"]
    keys = {"".join(c for c in t.upper() if c.isalnum()) for t in watched}
    mine = [(n["ticker"], d["date"]) for d in days for n in d["names"] if n.get("on_watchlist")]
    parts = [f"<b style='color:#fff'>{total} companies</b> report this week."]
    if biggest and biggest["total"]:
        bd = dt.date.fromisoformat(biggest["date"])
        top = ", ".join(n["ticker"] for n in biggest["names"][:3])
        parts.append(f"Busiest day: {bd:%A} ({biggest['total']}{', including ' + k.esc(top) if top else ''}).")
    if mine:
        parts.append("Your watchlist: " + ", ".join(
            f"<b style='color:#fff'>{k.esc(t)}</b> {dt.date.fromisoformat(dd):%a}" for t, dd in mine[:5]) + ".")
    elif keys:
        parts.append("None of your watchlist names report this week.")
    return subject, "\n".join(text), "".join(html), k.glance_text(parts)


def run(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now(UTC)
    counts: Dict[str, Any] = {"sent": 0, "skipped": {}, "failed": 0}

    def skip(why: str) -> None:
        counts["skipped"][why] = counts["skipped"].get(why, 0) + 1

    try:
        from api import earnings_week as ew
        from api.logo_check import logo_info
        from api.scans import _et_date, earnings_autoscore
        from auth.tiering import get_user_tier, has_min_tier
        from db import referrals as store
        from db.email_prefs import unsubscribe_url, wants_email
        from db.email_verification import is_email_verified
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from scheduler.morning_digest import _email_tier_key, default_watchlist_tickers, record_email_job
        from services.lifecycle_emails import _base_url
        from ui.email_utils import send_digest_email

        users = load_users() or {}
        base = ew.build_base(_et_date(now))
    except Exception as e:
        log.warning(json.dumps({"event": "weekly_earnings_failed", "error": type(e).__name__}))
        return {"error": type(e).__name__}
    if not any(d["rows"] for d in base["days"]):
        log.warning(json.dumps({"event": "weekly_earnings_empty_calendar"}))
        return {"sent": 0, "skipped": {"empty_calendar": 1}}
    week_key = iso_week(base["start"])
    try:
        auto = earnings_autoscore({"can_early_breakout": False}).get("rows") or {}
    except Exception:
        auto = {}
    url = _base_url()

    for username in list(users)[:_max_users()]:
        email = str(username or "").strip().lower()
        record = users.get(username) or {}
        if "@" not in email or record.get("is_active") is False:
            skip("not_eligible")
            continue
        try:
            tier = _email_tier_key(email, record, users, get_user_tier)
            if not (has_min_tier(tier, "pro") or store.grant_until(email)):
                skip("plan_below_pro")
                continue
            if not is_email_verified(email):
                skip("unverified")
                continue
        except Exception:
            skip("plan_lookup_failed")
            continue
        if not wants_email(email, "earnings"):
            skip("unsubscribed")
            continue
        try:
            watch = default_watchlist_tickers(email, list_watchlists, get_watchlist_tickers) or []
        except Exception:
            watch = []
        try:
            if not store.claim_email(email, KIND, week_key):
                skip("already_sent")
                continue
            week = ew.for_plan(base, per_day=PER_DAY, auto=auto, watched=set(watch), all_scores=True)
            for d in week["days"]:   # watchlist names beyond the checked top names
                for n in d["names"]:
                    if n.get("logo") is None:
                        n["logo"] = ew.logo_kind(logo_info(n["ticker"]))
            subject, text, inner, glance = compose(week, base_url=url, watched=watch)
            start = base["start"]
            if send_digest_email(email, subject, inner, text, unsubscribe_url=unsubscribe_url(email, "earnings"),
                                 title_line=f"Earnings this week · {start:%b} {start.day} to "
                                            f"{base['end']:%b} {base['end'].day}",
                                 glance_html=glance):
                counts["sent"] += 1
            else:
                store.release_email(email, KIND, week_key)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "weekly_earnings_send_failed", "error": type(e).__name__}))
    try:
        record_email_job("weekly_earnings", {"sent": counts["sent"], "skipped": counts["skipped"],
                                             "email_failed": counts["failed"]})
    except Exception as _ignored:
        swallowed("services.weekly_earnings.run", _ignored)
    return counts
