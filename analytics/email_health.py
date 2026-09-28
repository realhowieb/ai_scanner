"""P1-36 — is HSF email actually going out? (pure; no I/O)

Separate from the Run 59 health model on purpose: it never changes that model's
subsystems, statuses or autonomy readiness. It reads the per-run summaries the
email jobs record (db.email_job_runs) and judges the latest trading day:

  digest / evening  HEALTHY  a run sent at least one email
                    WAITING  nothing recorded yet and it isn't due (counts as healthy)
                    WARNING  runs happened but sent nothing (all sends failed,
                             or no account qualified — the skip reasons say why)
                    UNKNOWN  no run recorded yet (not due yet, disabled, or the
                             job didn't reach the recipient loop)
  alerts            WARNING  any alert email failed to send, or alerts fired and
                             email was attempted but none went out
                    HEALTHY  otherwise, when runs were recorded; UNKNOWN if none

On weekends and holidays the latest trading day is judged, so there's no false
alarm on days nothing is supposed to send.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

from analytics import market_calendar as mc

STATUS_ORDER = {"HEALTHY": 0, "WAITING": 0, "UNKNOWN": 1, "WARNING": 2}
# The email a job sends is due after these ET times on a trading day.
DUE_ET = {"digest": _dt.time(10, 30), "evening": _dt.time(18, 30)}
LABELS = {"digest": "Morning digest", "evening": "Evening wrap", "alerts": "Alert emails"}


def _parse(v: Any) -> Optional[_dt.datetime]:
    if isinstance(v, _dt.datetime):
        d = v
    else:
        try:
            d = _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


def judged_day(now: _dt.datetime) -> _dt.date:
    """Today if it's a trading day, else the previous trading day (ET)."""
    d = now.astimezone(mc.ET).date()
    return d if mc.is_trading_day(d) else mc.previous_trading_day(d)


def _top_reasons(skipped: Mapping[str, Any], n: int = 3) -> str:
    items = sorted(((k, int(v or 0)) for k, v in (skipped or {}).items()), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{k}={v}" for k, v in items) or "none recorded"


def _eval_send_job(job: str, runs: List[Mapping[str, Any]], day: _dt.date, now: _dt.datetime) -> Dict[str, Any]:
    if not runs:
        due = _dt.datetime.combine(day, DUE_ET[job], tzinfo=mc.ET)
        if now < due:
            return {"status": "WAITING", "detail": f"not due yet (after {DUE_ET[job].strftime('%H:%M')} ET)"}
        return {"status": "UNKNOWN",
                "detail": f"no run recorded for {day.isoformat()} (job disabled, not reached, or not yet recorded)"}
    sent = sum(int(r["stats"].get("sent") or 0) for r in runs)
    if sent > 0:
        return {"status": "HEALTHY", "detail": f"{sent} sent"}
    failed = sum(int((r["stats"].get("skipped") or {}).get("send_failed") or 0) for r in runs)
    latest = runs[0]["stats"].get("skipped") or {}
    if failed:
        return {"status": "WARNING",
                "detail": f"{failed} send(s) failed and none went out — check the email setup (SMTP / Resend)"}
    return {"status": "WARNING",
            "detail": f"ran {len(runs)} time(s) but sent nothing — no account qualified ({_top_reasons(latest)})"}


def _eval_alerts(runs: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {"status": "UNKNOWN", "detail": "no alert run recorded"}
    fired = sum(int(r["stats"].get("fired") or 0) for r in runs)
    emailed = sum(int(r["stats"].get("emailed") or 0) for r in runs)
    failed = sum(int(r["stats"].get("email_failed") or 0) for r in runs)
    if failed:
        return {"status": "WARNING",
                "detail": f"{failed} alert email(s) failed to send ({emailed} sent, {fired} alerts fired)"}
    return {"status": "HEALTHY", "detail": f"{emailed} emailed, {fired} alerts fired"}


def evaluate(runs: Sequence[Mapping[str, Any]], now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """{status, day, jobs: {job: {label, status, detail, runs}}} for the latest trading day."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    day = judged_day(now)
    by_job: Dict[str, List[Mapping[str, Any]]] = {"digest": [], "evening": [], "alerts": []}
    for r in runs or []:
        at = _parse(r.get("at"))
        if at is None or r.get("job") not in by_job:
            continue
        if at.astimezone(mc.ET).date() == day:
            by_job[r["job"]].append({**r, "stats": r.get("stats") or {}})
    for job in by_job:
        by_job[job].sort(key=lambda r: _parse(r["at"]), reverse=True)
    jobs = {
        "digest": _eval_send_job("digest", by_job["digest"], day, now),
        "evening": _eval_send_job("evening", by_job["evening"], day, now),
        "alerts": _eval_alerts(by_job["alerts"]),
    }
    for job, res in jobs.items():
        res.update(label=LABELS[job], runs=len(by_job[job]))
    status = max((j["status"] for j in jobs.values()), key=lambda s: STATUS_ORDER[s])
    return {"status": status, "day": day.isoformat(), "jobs": jobs}
