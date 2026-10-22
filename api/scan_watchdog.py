"""Backup scan check (P2-80): about 20 minutes after each scan slot, confirm the scan ran.

Scans are started by cron-job.org dispatching the scheduled-scans workflow on GitHub. If
that dispatch never arrives (cron-job.org down, an expired token) or GitHub drops the run,
nothing noticed until the next day's health report. This check runs inside the always-on
API, so it doesn't depend on either of those:

    at slot + 20 min   read the latest scheduled-scans runs from the GitHub API and
                       classify the slot: ok / running / failed / missed
    running            look again every 2 minutes until the run finishes (or slot + 60 min)
    failed or missed   flag it once (hsf_scan_watchdog row, so a restart or a second
                       instance doesn't flag it again), log an error (which Sentry
                       picks up), email ops, and re-dispatch the scan when a token is set

A re-dispatch at slot + 20 still counts for that slot in the health report (it matches
runs up to 45 minutes after a slot), so a missed slot becomes a late one instead of a gap.

Env:
    HSF_SCAN_WATCHDOG_ENABLED=0   turns the check off (default on; it only runs on Render)
    HSF_GITHUB_DISPATCH_TOKEN     GitHub token with Actions write on the repo. Set it and a
                                  missed or failed slot is re-run (force=false, session=auto).
                                  Without it the check only flags. It is also used to read runs.
    HSF_OPS_ALERT_EMAIL           where the email goes (comma-separated). Default: the active
                                  admin accounts.
    HSF_SCAN_WATCHDOG_REDISPATCH=0  flag only, never re-dispatch, even with a token.

Everything here is best effort and never raises into the API.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from analytics import market_calendar as mc

log = logging.getLogger("hsf.scan_watchdog")

UTC = dt.timezone.utc
CHECK_AFTER = dt.timedelta(minutes=20)
GIVE_UP_AFTER = dt.timedelta(minutes=60)
SLOT_EARLY = dt.timedelta(minutes=10)   # a run this early still belongs to the slot (as in system_health)
SLOT_LATE = dt.timedelta(minutes=45)
POLL_SECONDS = 120  # GitHub's unauthenticated limit is 60 reads an hour per IP
WORKFLOW = "scheduled-scans.yml"
GH_API = "https://api.github.com"

OK, RUNNING, FAILED, MISSED, UNKNOWN = "ok", "running", "failed", "missed", "unknown"


def enabled() -> bool:
    return os.environ.get("HSF_SCAN_WATCHDOG_ENABLED", "1").strip() != "0"


def _repo() -> str:
    return (os.environ.get("HSF_GITHUB_REPO") or "realhowieb/ai_scanner").strip()


def _token() -> str:
    return (os.environ.get("HSF_GITHUB_DISPATCH_TOKEN") or "").strip()


def _parse(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


# ---- pure parts (tested directly) ---------------------------------------------------------------
def due_slot(now: dt.datetime) -> Optional[dt.datetime]:
    """The scan slot to check at `now`: one that is at least CHECK_AFTER old and less
    than GIVE_UP_AFTER old (so an API restart doesn't flag slots from hours ago)."""
    day = now.astimezone(mc.ET).date()
    for slot in reversed(mc.expected_scan_slots(day)):
        if slot + CHECK_AFTER <= now < slot + GIVE_UP_AFTER:
            return slot
    return None


def classify(slot: dt.datetime, runs: Sequence[Mapping[str, Any]]) -> str:
    """ok / running / failed / missed for one slot, from scheduled-scans workflow runs
    (GitHub API rows: created_at, status, conclusion)."""
    hit = []
    for r in runs or []:
        created = _parse(r.get("created_at"))
        if created is not None and slot - SLOT_EARLY <= created <= slot + SLOT_LATE:
            hit.append(r)
    if not hit:
        return MISSED
    if any(r.get("conclusion") == "success" for r in hit):
        return OK
    if any((r.get("status") or "") != "completed" for r in hit):
        return RUNNING
    return FAILED


def alert_content(slot: dt.datetime, status: str, redispatched: Optional[bool]) -> tuple:
    """(subject, text, html inner) for the ops email."""
    et = slot.astimezone(mc.ET)
    when = f"{et:%a %b} {et.day}, {et:%I:%M %p} ET".replace(" 0", " ")
    what = "never started" if status == MISSED else "started but did not finish successfully"
    if redispatched:
        next_step = "The backup check started the scan again, so this slot should still fill in."
    elif redispatched is False:
        next_step = "The backup check tried to start the scan again, but GitHub refused it. Start it by hand."
    else:
        next_step = ("Re-run is off (no HSF_GITHUB_DISPATCH_TOKEN on hsf-api-v1), so nothing was retried. "
                     "Start scheduled-scans by hand if you want this slot filled.")
    if status == MISSED:
        cause = "Check that cron-job.org fired and that its GitHub token still works."
    else:
        cause = "Open the latest scheduled-scans run on GitHub for the error."
    runs_url = f"https://github.com/{_repo()}/actions/workflows/{WORKFLOW}"
    subject = f"HSF scan {status}: {when}"
    text = (f"The {when} scan {what}.\n\n{next_step}\n\n{cause}\n\nRuns: {runs_url}\n")
    inner = (f"<h2 style='font-size:18px;margin:16px 0 8px'>The {when} scan {what}</h2>"
             f"<p>{next_step}</p><p>{cause}</p>"
             f"<p><a href='{runs_url}'>Open the scheduled-scans runs</a></p>")
    return subject, text, inner


# ---- side effects -------------------------------------------------------------------------------
def fetch_runs(session=None) -> Optional[List[Dict[str, Any]]]:
    """The latest scheduled-scans runs, or None when GitHub can't be read. The repo is
    public, so this works without a token (60 requests an hour; we use at most ~40 a day)."""
    try:
        import requests

        s = session or requests
        headers = {"Accept": "application/vnd.github+json"}
        if _token():
            headers["Authorization"] = f"Bearer {_token()}"
        r = s.get(f"{GH_API}/repos/{_repo()}/actions/workflows/{WORKFLOW}/runs",
                  params={"per_page": 20}, headers=headers, timeout=15)
        if r.status_code != 200:
            log.warning(json.dumps({"event": "scan_watchdog_fetch_failed", "http_status": r.status_code}))
            return None
        return [{"created_at": x.get("created_at"), "status": x.get("status"), "conclusion": x.get("conclusion")}
                for x in (r.json().get("workflow_runs") or [])]
    except Exception as e:
        log.warning(json.dumps({"event": "scan_watchdog_fetch_failed", "error": type(e).__name__}))
        return None


def redispatch(session=None) -> Optional[bool]:
    """Start scheduled-scans on main the way cron-job.org does. None when re-run is off."""
    if not _token() or os.environ.get("HSF_SCAN_WATCHDOG_REDISPATCH", "1").strip() == "0":
        return None
    try:
        import requests

        s = session or requests
        r = s.post(f"{GH_API}/repos/{_repo()}/actions/workflows/{WORKFLOW}/dispatches",
                   json={"ref": "main", "inputs": {"force": "false", "session": "auto"}},
                   headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {_token()}"},
                   timeout=15)
        return r.status_code == 204
    except Exception as e:
        log.warning(json.dumps({"event": "scan_watchdog_dispatch_failed", "error": type(e).__name__}))
        return False


def _recipients() -> List[str]:
    raw = os.environ.get("HSF_OPS_ALERT_EMAIL", "")
    listed = sorted({p.strip().lower() for p in raw.split(",") if "@" in p})
    if listed:
        return listed
    try:
        from db.scan_watchdog import admin_emails

        return admin_emails()
    except Exception:
        return []


def flag(slot: dt.datetime, status: str, *, session=None) -> Dict[str, Any]:
    """Flag a missed or failed slot once: record it, log it, re-run it, email ops."""
    from db import scan_watchdog as store

    key = slot.astimezone(UTC).isoformat(timespec="minutes")
    if not store.claim_slot(key, status):
        return {"slot": key, "status": status, "flagged": False}
    redispatched = redispatch(session)
    # An error log reaches Sentry (api.monitoring), which is the other place this shows up.
    log.error(json.dumps({"event": "scan_slot_" + status, "slot": key, "redispatched": redispatched}))
    sent = 0
    try:
        from ui.email_utils import send_digest_email

        subject, text, inner = alert_content(slot, status, redispatched)
        for to in _recipients():
            if send_digest_email(to, subject, inner, text):
                sent += 1
    except Exception as e:
        log.warning(json.dumps({"event": "scan_watchdog_email_failed", "error": type(e).__name__}))
    return {"slot": key, "status": status, "flagged": True, "redispatched": redispatched, "emailed": sent}


def check(now: Optional[dt.datetime] = None, *, done: Optional[set] = None, session=None) -> Dict[str, Any]:
    """One pass. `done` holds slots already settled in this process so a pass after the
    verdict does no network call. Returns what happened, for logs and tests."""
    now = now or dt.datetime.now(UTC)
    slot = due_slot(now)
    if slot is None:
        return {"status": "idle"}
    if done is not None and slot in done:
        return {"status": "settled", "slot": slot.isoformat()}
    runs = fetch_runs(session)
    if runs is None:
        return {"status": UNKNOWN, "slot": slot.isoformat()}  # try again next pass
    status = classify(slot, runs)
    if status == RUNNING:
        return {"status": RUNNING, "slot": slot.isoformat()}
    if done is not None:
        done.add(slot)
    if status == OK:
        return {"status": OK, "slot": slot.isoformat()}
    try:
        return flag(slot, status, session=session)
    except Exception as e:  # storage down: don't flag (no claim means it could flag twice)
        log.warning(json.dumps({"event": "scan_watchdog_flag_failed", "error": type(e).__name__}))
        if done is not None:
            done.discard(slot)
        return {"status": UNKNOWN, "slot": slot.isoformat()}


def start() -> None:
    """Start the check in a background thread. Only on Render (the always-on API), and
    only when HSF_SCAN_WATCHDOG_ENABLED isn't 0."""
    if os.environ.get("RENDER", "").strip().lower() != "true" or not enabled():
        return
    done: set = set()

    def run() -> None:
        while True:
            try:
                if enabled():
                    check(done=done)
            except Exception as e:
                log.warning(json.dumps({"event": "scan_watchdog_error", "error": type(e).__name__}))
            time.sleep(POLL_SECONDS)

    threading.Thread(target=run, name="scan-watchdog", daemon=True).start()
