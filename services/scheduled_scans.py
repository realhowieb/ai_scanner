"""Scheduled custom scans (Premium, 2026-11-02).

A Premium account saves up to three custom scans (the same request as POST /v1/scans) to
run on every trading day at one of three times (US Eastern):

    open     08:05 to 09:25, before the open (a pre-market scan works here)
    midday   12:00 to 13:30
    close    16:20 to 18:00, after the close

api/ops_jobs.py calls run_pass() every 5 minutes. Inside a window each due schedule is
claimed for the day (db.scheduled_scans.claim), its owner's plan is checked again
(a lapsed Premium account's schedules pause and say so), the request is re-checked
against the plan exactly like POST /v1/scans, and the scan is queued on the API's own
scan pool (api.scan_jobs), so it shares the one-scan-at-a-time memory bound and shows
in the owner's recent scans. When it finishes, the results go out by email.

A schedule whose owner already has a scan running, or that finds the scanner busy, is
released and tried again on the next pass. At most PER_PASS schedules are queued per
pass so customers' own scans keep their place in line.

Emails respect the "alerts" email preference and need a verified address. The email
lists the top results ranked by HSF Score, then a short AI read of the leading setups
(2026-11-03; HSF_SCHEDULED_SCAN_AI=0 leaves it out). It's research, not a recommendation.

Env: HSF_SCHEDULED_SCANS_ENABLED=0 stops it (checked in api/ops_jobs.py),
HSF_SCHEDULED_SCANS_PER_PASS (default 3). Never raises.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("hsf.scheduled_scans")

UTC = dt.timezone.utc
WINDOWS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "open": ((8, 5), (9, 25)),
    "midday": ((12, 0), (13, 30)),
    "close": ((16, 20), (18, 0)),
}
LABELS = {"open": "before the open", "midday": "at midday", "close": "after the close"}
EMAIL_ROWS = 10
_SETUPS = {"golden_cross": "Golden Cross", "breakout": "Breakout", "prebreakout": "PreBreakout",
           "gapper": "Gapper", "gainer": "Momentum", "fading": "Gap Fading"}


def per_pass() -> int:
    try:
        return max(1, int(os.environ.get("HSF_SCHEDULED_SCANS_PER_PASS", "3")))
    except ValueError:
        return 3


def _et(now: dt.datetime) -> dt.datetime:
    from analytics import market_calendar as mc

    return now.astimezone(mc.ET)


def slot_for(now: dt.datetime) -> Optional[Tuple[str, str]]:
    """(run_at, claim key) when `now` is inside a window on a trading day, else None."""
    from analytics import market_calendar as mc

    et = _et(now)
    if not mc.is_trading_day(et.date()):
        return None
    minute = et.hour * 60 + et.minute
    for run_at, ((h1, m1), (h2, m2)) in WINDOWS.items():
        if h1 * 60 + m1 <= minute < h2 * 60 + m2:
            return run_at, f"{et.date().isoformat()}:{run_at}"
    return None


def setup_label(v: Any) -> str:
    s = str(v or "").strip()
    return _SETUPS.get(s.lower(), s.replace("_", " ").title() if s else "")


# ---- the AI summary (Premium automation, 2026-11-03) ---------------------------------------------
AI_ROWS = 10


def ai_enabled() -> bool:
    return os.environ.get("HSF_SCHEDULED_SCAN_AI", "1").strip() != "0"


def summary_table(rows: List[Dict[str, Any]]) -> str:
    """The top results as CSV for the AI prompt (scan numbers only). Pure."""
    cols = ("ticker", "score", "primary_setup", "status", "chg_pct", "gap_pct", "rvol", "breakout_score", "last")
    lines = [",".join(cols)]
    for r in rows[:AI_ROWS]:
        vals = []
        for c in cols:
            v = r.get(c)
            if c == "primary_setup":
                v = setup_label(v)
            vals.append("" if v is None else str(v).replace(",", " "))
        lines.append(",".join(vals))
    return "\n".join(lines)


def ai_summary(username: str, result: Dict[str, Any]) -> Optional[str]:
    """A short AI read of the top results, or None (AI off, no rows, daily limit). Counts
    toward the owner's daily AI requests. Never raises."""
    rows = list(result.get("setups") or [])
    if not rows or not ai_enabled():
        return None
    try:
        from ui.ai import ask_claude
        from ui.ai_summary import _SYSTEM_PROMPT

        text, _err = ask_claude(system=_SYSTEM_PROMPT,
                                user=(f"Here are the top {min(len(rows), AI_ROWS)} results of the trader's own scheduled "
                                      f"scan in HSF Score order (CSV):\n\n{summary_table(rows)}\n\n"
                                      "Explain the leading setups in this order. Keep it under 150 words."),
                                max_tokens=500, username=username, feature="scheduled_scan_summary")
    except Exception as e:
        log.info(json.dumps({"event": "scheduled_scan_ai_failed", "error": type(e).__name__}))
        return None
    if not text or text.startswith("_Some wording was removed"):
        return None
    return text.strip()


# ---- the email ------------------------------------------------------------------------------------
def compose(name: str, run_at: str, result: Dict[str, Any], *, base_url: str,
            ai_text: Optional[str] = None) -> Tuple[str, str, str, str]:
    """(subject, text, html_inner, glance_html). Pure."""
    from ui import email_kit as k

    rows: List[Dict[str, Any]] = list(result.get("setups") or [])
    top = rows[:EMAIL_ROWS]
    total = int(result.get("total") or len(rows))
    scanned = int(result.get("symbols_scanned") or 0)
    label = str(result.get("label") or "")
    n = len(rows)
    subject = (f"Your scan \"{name}\": {n} setup{'s' if n != 1 else ''} {LABELS.get(run_at, '')}".strip()
               if n else f"Your scan \"{name}\": no setups matched {LABELS.get(run_at, '')}".strip())
    link = f"{base_url}/scanner/custom"
    if top:
        table = k.table(["Ticker", "HSF Score", "Setup", "Change"], [
            [f"<a href='{k.esc(base_url)}/stocks/{k.esc(r.get('ticker'))}' style='color:inherit;text-decoration:none'>"
             f"{k.ticker(r.get('ticker'))}</a>",
             k.score_badge(r.get("score")), k.esc(setup_label(r.get("primary_setup"))), k.pct(r.get("chg_pct"))]
            for r in top])
        more = f" Showing the top {len(top)} of {n}." if n > len(top) else ""
        body = table + k.note(f"Ranked by HSF Score.{more} Prices and changes are from the scan, not live quotes.")
    else:
        body = k.empty("Nothing passed your filters this time. Looser filters or a larger list find more.")
    sub = f"{label} · {scanned:,} stocks scanned" if label else f"{scanned:,} stocks scanned"
    html = k.card(f"🔎 {k.esc(name)}", body, sub=k.esc(sub))
    if ai_text and top:
        from services.premium_auto import paragraphs

        html += k.card("🤖 AI read of these results", paragraphs(ai_text.replace("**", "")),
                       sub="Written by AI from the scan numbers only")
    html += k.card("Change or stop this scan", k.note(
        "Open Custom scan to see every result, edit the filters or delete this schedule. "
        "HSF Scores rank setups for research. They aren't predictions or advice.")
        + k.button(link, "Open Custom scan →"))
    text = [f"Your scheduled scan \"{name}\" ({sub})", ""]
    text += [f"{i}. {r.get('ticker')}  HSF {r.get('score')}  {setup_label(r.get('primary_setup'))}"
             for i, r in enumerate(top, 1)] or ["No setups matched your filters this time."]
    if ai_text and top:
        text += ["", "AI read of these results:", ai_text]
    text += ["", f"See every result or change the schedule: {link}"]
    parts = [f"<b style='color:#fff'>{n} setup{'s' if n != 1 else ''}</b> from {scanned:,} stocks."]
    if top:
        parts.append("Top: " + ", ".join(f"<b style='color:#fff'>{k.esc(r.get('ticker'))}</b>" for r in top[:3]) + ".")
    if total > n:
        parts.append(f"{total} scored in all.")
    return subject, "\n".join(text), html, k.glance_text(parts)


def _email(username: str, name: str, run_at: str, result: Dict[str, Any]) -> str:
    """'sent', 'unsubscribed', 'unverified' or 'email_failed'."""
    from db.email_prefs import unsubscribe_url, wants_email
    from db.email_verification import is_email_verified
    from services.lifecycle_emails import _base_url
    from ui.email_utils import send_digest_email

    if "@" not in username:
        return "no_email"
    if not is_email_verified(username):
        return "unverified"
    if not wants_email(username, "alerts"):
        return "unsubscribed"
    subject, text, inner, glance = compose(name, run_at, result, base_url=_base_url(),
                                           ai_text=ai_summary(username, result))
    ok = send_digest_email(username, subject, inner, text, unsubscribe_url=unsubscribe_url(username, "alerts"),
                           title_line=f"Scheduled scan · {name}", glance_html=glance)
    return "sent" if ok else "email_failed"


# ---- the pass -------------------------------------------------------------------------------------
def _account_entitlements(username: str) -> Optional[Dict[str, Any]]:
    from api import plan_access, store
    from api.deps import entitlements_for

    account = store.get_account(username)
    if not account or account.get("is_active") is False:
        return None
    return entitlements_for(plan_access.apply_to_account({k: v for k, v in account.items() if k != "password"}))


def _work(schedule: Dict[str, Any], params: Dict[str, Any], username: str):
    from api import custom_scans
    from db import scheduled_scans as store

    def work(report):
        try:
            result = custom_scans.run_scan(params, username, report)
        except Exception as e:
            from api.scan_jobs import ScanCancelled, ScanFailed

            if not isinstance(e, ScanCancelled):
                store.record(schedule["id"], "failed", str(e) if isinstance(e, ScanFailed) else "The scan failed.")
            raise
        try:
            status = _email(username, schedule["name"], schedule["run_at"], result)
        except Exception as e:
            log.warning(json.dumps({"event": "scheduled_scan_email_failed", "error": type(e).__name__}))
            status = "email_failed"
        store.record(schedule["id"], status, f"{len(result.get('setups') or [])} setups")
        return result

    return work


def run_pass(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now(UTC)
    slot = slot_for(now)
    if not slot:
        return {"slot": None}
    run_at, key = slot
    counts: Dict[str, Any] = {"slot": key, "queued": 0, "skipped": {}}

    def skip(why: str) -> None:
        counts["skipped"][why] = counts["skipped"].get(why, 0) + 1

    try:
        from api import custom_scans, scan_jobs
        from db import scheduled_scans as store

        due = store.list_due(run_at, key)
    except Exception as e:
        log.warning(json.dumps({"event": "scheduled_scans_failed", "error": type(e).__name__}))
        return {"error": type(e).__name__}

    for s in due:
        if counts["queued"] >= per_pass():
            break
        prev = store.claim(s["id"], key)
        if prev is None:
            skip("claimed")
            continue
        user = str(s["username"])
        try:
            ent = _account_entitlements(user)
            if ent is None:
                store.record(s["id"], "paused", "This account isn't active.")
                skip("inactive")
                continue
            if not ent["entitlements"].get("can_scheduled_scans"):
                store.record(s["id"], "paused", "Scheduled scans are part of Premium.")
                skip("not_premium")
                continue
            try:
                params = custom_scans.plan_or_raise(dict(s["request"]), ent)
            except (custom_scans.PlanError, ValueError) as e:
                store.record(s["id"], "failed", str(e))
                skip("plan")
                continue
            try:
                job = scan_jobs.create_job(user, params["universe"], params)
            except (scan_jobs.ScanInProgress, scan_jobs.ScanBusy):
                store.release(s["id"], prev)
                skip("busy")
                continue
            scan_jobs.submit(job["id"], _work(s, params, user))
            store.record(s["id"], "running", "", scan_id=job["id"])
            counts["queued"] += 1
        except Exception as e:
            log.warning(json.dumps({"event": "scheduled_scan_failed", "error": type(e).__name__}))
            try:
                store.release(s["id"], prev)
            except Exception:
                pass
            skip("error")
    return counts
