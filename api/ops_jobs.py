"""Scheduled jobs that run inside the always-on API (Render), so they also run on
weekends and don't depend on cron-job.org or GitHub's cron.

Every POLL_SECONDS one thread checks what is due. Each job is claimed once per key
(services.ops_notify.once), so a restart or a second instance never repeats one.

    live check notices  every 10 min: finished "Live acceptance" runs on GitHub. After a
                        promote (push to main) the admins get pass or fail on
                        Telegram/Discord; the daily scheduled run only reports a failure.
    nightly ops note    Tue-Sat 00:05-03:00 UTC (after the 23:45 health run): health
                        status and score, top incidents, scan slots the backup check
                        flagged, API memory, the last live check. Incidents whose code is
                        in HSF_OPS_NOTE_IGNORE (default RETIREMENT_POLICY_EFFECT, the known
                        historical parity finding) are left out.
    weekly recap        Saturday from 14:00 UTC: services.weekly_recap to every verified,
                        opted-in account.
    business email      Monday from 12:00 UTC: services.business_report to the admins.
    daily account mail  from 15:00 UTC: free Pro ended, first-week tips and the 14-day
                        come-back email (services.lifecycle_emails, services.onboarding_emails).

Env (all default on; set to 0 to stop one):
    HSF_OPS_JOBS_ENABLED          the whole thread
    HSF_LIVE_CHECK_NOTIFY         live check notices
    HSF_NIGHTLY_OPS_NOTE          nightly ops note
    HSF_WEEKLY_RECAP_ENABLED      weekly recap (services.weekly_recap)
    HSF_BUSINESS_EMAIL_ENABLED    business email (services.business_report)
    HSF_ONBOARDING_EMAILS_ENABLED tips and come-back emails (services.onboarding_emails)

Nothing here raises into the API.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

log = logging.getLogger("hsf.ops_jobs")

UTC = dt.timezone.utc
POLL_SECONDS = 300
LIVE_CHECK_EVERY = dt.timedelta(minutes=10)
LIVE_CHECK_LOOKBACK = dt.timedelta(hours=3)
LIVE_WORKFLOW = "live-acceptance.yml"
GH_API = "https://api.github.com"


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip() != "0"


def _parse(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _et(now: dt.datetime) -> dt.datetime:
    from analytics import market_calendar as mc

    return now.astimezone(mc.ET)


def iso_week(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


# ---- live check notices ---------------------------------------------------------------------------
def live_check_message(run: Mapping[str, Any]) -> Optional[tuple]:
    """(subject, text) for one finished Live acceptance run, or None when it needs no
    message (a passing scheduled or manual run)."""
    if run.get("status") != "completed":
        return None
    passed = run.get("conclusion") == "success"
    promote = run.get("event") == "push"
    if passed and not promote:
        return None
    if run.get("conclusion") in ("cancelled", "skipped"):
        return None
    sha = str(run.get("head_sha") or "")[:7]
    title = str((run.get("head_commit") or {}).get("message") or run.get("display_title") or "").split("\n")[0][:120]
    what = f"after promote {sha}" if promote else ("daily run" if run.get("event") == "schedule" else "manual run")
    url = run.get("html_url") or ""
    if passed:
        return (f"✅ Live check passed ({what})",
                f"API journey, browser journey and accessibility checks all passed on the live site.\n{title}\n{url}".strip())
    return (f"❌ Live check failed ({what})",
            f"Something on the live site didn't pass. Open the run for the failing step.\n{title}\n{url}".strip())


def fetch_live_runs(session=None) -> Optional[List[Dict[str, Any]]]:
    try:
        from api.scan_watchdog import _repo, _requests, _token

        s = session or _requests()
        headers = {"Accept": "application/vnd.github+json"}
        if _token():
            headers["Authorization"] = f"Bearer {_token()}"
        r = s.get(f"{GH_API}/repos/{_repo()}/actions/workflows/{LIVE_WORKFLOW}/runs",
                  params={"per_page": 10}, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return list(r.json().get("workflow_runs") or [])
    except Exception as e:
        log.warning(json.dumps({"event": "live_runs_fetch_failed", "error": type(e).__name__}))
        return None


def live_check_notices(now: dt.datetime, runs: Optional[List[Mapping[str, Any]]] = None,
                       notify=None, claim=None) -> List[str]:
    """Message each newly finished run once. Returns the subjects sent."""
    from services import ops_notify

    notify = notify or ops_notify.notify_ops
    claim = claim or ops_notify.once
    runs = runs if runs is not None else (fetch_live_runs() or [])
    sent = []
    for run in runs:
        done_at = _parse(run.get("updated_at"))
        if done_at is None or now - done_at > LIVE_CHECK_LOOKBACK:
            continue
        msg = live_check_message(run)
        if msg is None or not claim("live_check", str(run.get("id"))):
            continue
        notify(msg[0], msg[1], email=msg[0].startswith("❌"))
        sent.append(msg[0])
    return sent


def last_live_check(runs: Optional[List[Mapping[str, Any]]]) -> str:
    for run in runs or []:
        if run.get("status") == "completed" and run.get("conclusion") not in ("cancelled", "skipped"):
            when = _parse(run.get("updated_at"))
            stamp = f"{_et(when):%a %I:%M %p} ET".replace(" 0", " ") if when else "?"
            return f"{'passed' if run.get('conclusion') == 'success' else 'FAILED'} ({stamp})"
    return "no recent run"


# ---- nightly ops note -------------------------------------------------------------------------------
def nightly_due(now: dt.datetime) -> Optional[str]:
    """The trading day (ET date) whose evening note is due, or None. Tue-Sat 00:05-03:00
    UTC covers Mon-Fri evenings in New York."""
    if now.weekday() not in (1, 2, 3, 4, 5):
        return None
    start = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if not start <= now < now.replace(hour=3, minute=0, second=0, microsecond=0):
        return None
    return (now - dt.timedelta(hours=6)).astimezone(UTC).date().isoformat()


def nightly_note(health: Optional[Mapping[str, Any]], *, flags: List[tuple], memory: Mapping[str, Any],
                 live: str, now: dt.datetime) -> tuple:
    """(subject, text) for the nightly ops note."""
    lines: List[str] = []
    fresh = False
    if health:
        gen = _parse(health.get("generated_at"))
        fresh = bool(gen and now - gen < dt.timedelta(hours=4))
    if health and fresh:
        status, score = health.get("system_status", "?"), health.get("health_score", "?")
        icon = {"HEALTHY": "🟢", "ACTION_REQUIRED": "🔴"}.get(status, "🟡")
        subject = f"{icon} HSF tonight: {status} {score}/100"
        human = health.get("human_action")
        if human == "HUMAN_ACTION_REQUIRED":
            lines.append("Needs you: yes, see the incidents below.")
        ignore = [c.strip() for c in os.environ.get("HSF_OPS_NOTE_IGNORE", "RETIREMENT_POLICY_EFFECT").split(",") if c.strip()]
        incidents = [i for i in (health.get("incidents") or []) if i.get("severity") != "INFO"
                     and not any(str(i.get("incident_id", "")).endswith(":" + c) for c in ignore)]
        for i in incidents[:3]:
            lines.append(f"• {i.get('subsystem')}: {i.get('summary')}")
        if not incidents:
            lines.append("No incidents.")
    else:
        subject = "🟡 HSF tonight: health report missing"
        lines.append("The 23:45 UTC system health run hasn't saved a report tonight. Check the System health workflow.")
    today_flags = [f for f in flags if str(f[0])[:10] == (now - dt.timedelta(hours=6)).date().isoformat()]
    lines.append(f"Scan slots flagged by the backup check today: {len(today_flags)}"
                 + (" (" + ", ".join(f"{str(f[0])[11:16]} UTC {f[1]}" for f in today_flags[:4]) + ")" if today_flags else ""))
    peak = memory.get("container_peak_mb") or memory.get("rss_peak_mb")
    if peak:
        warn = " ⚠️ near the limit" if memory.get("memory_warning") else ""
        lines.append(f"API memory peak: {peak:.0f} of {memory.get('memory_limit_mb') or 512:.0f} MB{warn}")
    lines.append(f"Last live check: {live}")
    return subject, "\n".join(lines)


def run_nightly_note(now: dt.datetime) -> bool:
    from services import ops_notify

    day = nightly_due(now)
    if not day:
        return False
    try:
        from db.system_health import load_latest

        health = load_latest()
    except Exception:
        health = None
    gen = _parse((health or {}).get("generated_at"))
    if (not gen or now - gen > dt.timedelta(hours=4)) and now.hour < 1:
        return False  # the health run may still be going; wait until 01:00 UTC before saying it's missing
    if not ops_notify.once("nightly_note", day):
        return False
    try:
        from db.scan_watchdog import recent_flags

        flags = recent_flags(10)
    except Exception:
        flags = []
    try:
        from api.memory import memory_status

        memory = memory_status()
    except Exception:
        memory = {}
    subject, text = nightly_note(health, flags=flags, memory=memory, live=last_live_check(fetch_live_runs()), now=now)
    ops_notify.notify_ops(subject, text)
    return True


# ---- weekly and daily jobs --------------------------------------------------------------------------
def weekly_recap_due(now: dt.datetime) -> Optional[str]:
    if now.weekday() == 5 and now.hour >= 14:
        return iso_week(_et(now).date())
    return None


def business_due(now: dt.datetime) -> Optional[str]:
    if now.weekday() == 0 and now.hour >= 12:
        return iso_week(_et(now).date())
    return None


def daily_mail_due(now: dt.datetime) -> Optional[str]:
    return _et(now).date().isoformat() if now.hour >= 15 else None


def _run_once(job: str, key: Optional[str], fn) -> None:
    """Claim then run; release the claim if the job itself crashed so it retries next pass."""
    if not key:
        return
    from services import ops_notify

    if not ops_notify.once(job, key):
        return
    try:
        result = fn()
        log.info(json.dumps({"event": "ops_job_done", "job": job, "key": key, "result": result}, default=str))
    except Exception as e:
        log.warning(json.dumps({"event": "ops_job_failed", "job": job, "error": type(e).__name__}))
        ops_notify.release(job, key)


def tick(now: Optional[dt.datetime] = None, state: Optional[Dict[str, Any]] = None) -> None:
    now = now or dt.datetime.now(UTC)
    state = state if state is not None else {}
    if _on("HSF_LIVE_CHECK_NOTIFY"):
        last = state.get("live_checked")
        if last is None or now - last >= LIVE_CHECK_EVERY:
            state["live_checked"] = now
            live_check_notices(now)
    if _on("HSF_NIGHTLY_OPS_NOTE"):
        run_nightly_note(now)
    if _on("HSF_WEEKLY_RECAP_ENABLED"):
        from services import weekly_recap

        _run_once("weekly_recap", weekly_recap_due(now), lambda: weekly_recap.run(now=now))
    if _on("HSF_BUSINESS_EMAIL_ENABLED"):
        from services import business_report

        _run_once("business_email", business_due(now), lambda: business_report.run(now=now))

    def daily() -> Dict[str, Any]:
        from services import lifecycle_emails, onboarding_emails

        out: Dict[str, Any] = {"trial_ended": lifecycle_emails.run_trial_ended(now=now)}
        if _on("HSF_ONBOARDING_EMAILS_ENABLED"):
            out["onboarding"] = onboarding_emails.run(now=now)
        return out

    _run_once("daily_account_mail", daily_mail_due(now), daily)


def start() -> None:
    """Start the jobs thread. Only on Render, and only when HSF_OPS_JOBS_ENABLED isn't 0."""
    if os.environ.get("RENDER", "").strip().lower() != "true" or not _on("HSF_OPS_JOBS_ENABLED"):
        return
    state: Dict[str, Any] = {}

    def run() -> None:
        time.sleep(90)  # let the deploy settle and the caches warm first
        while True:
            try:
                if _on("HSF_OPS_JOBS_ENABLED"):
                    tick(state=state)
            except Exception as e:
                log.warning(json.dumps({"event": "ops_jobs_error", "error": type(e).__name__}))
            time.sleep(POLL_SECONDS)

    threading.Thread(target=run, name="ops-jobs", daemon=True).start()
