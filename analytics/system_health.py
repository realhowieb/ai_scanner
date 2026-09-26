"""Run 59 — HSF autonomous research & operations health control plane (pure).

Answers one question: "Can HSF safely continue operating unattended right now?"

`evaluate(inputs)` turns already-collected telemetry (see `scripts/system_health.py`
collectors) into one deterministic health model: ten subsystems, a system
status, a human-action level, prioritized incidents, automation candidates and
autonomy readiness. It is read-only and pure: no I/O, no scanner/scoring/cohort/
outcome logic, and no effectiveness statistics (anti-peeking guard on output).

Status vocabulary per subsystem:
  HEALTHY · WAITING (expected research/market waiting — counts as healthy) ·
  DEGRADED · ACTION_REQUIRED · UNKNOWN (telemetry missing — never faked HEALTHY)
System status precedence: ACTION_REQUIRED > DEGRADED > UNKNOWN > HEALTHY.
Human action: NO_ACTION < WATCH < AUTOMATIC_RECOVERY_CANDIDATE < HUMAN_ACTION_REQUIRED.
Calendar-aware: weekends/holidays create no scanner/maturation/freshness findings.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from analytics import market_calendar as mc

SCHEMA = "hsf-system-health-1.0"
SUBSYSTEMS = ("universe", "scanner", "research_capture", "maturation", "market_data",
              "cohort_parity", "forward_evidence", "workflows", "database", "artifact_freshness")

STATUS_ORDER = {"HEALTHY": 0, "WAITING": 0, "UNKNOWN": 1, "DEGRADED": 2, "ACTION_REQUIRED": 3}
ACTION_ORDER = {"NO_ACTION": 0, "WATCH": 1, "AUTOMATIC_RECOVERY_CANDIDATE": 2, "HUMAN_ACTION_REQUIRED": 3}
SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
SCORE_PENALTY = {"ACTION_REQUIRED": 25, "DEGRADED": 10, "UNKNOWN": 5}

# Thresholds (operational, not research).
SLOT_EARLY = _dt.timedelta(minutes=10)
SLOT_LATE = _dt.timedelta(minutes=45)
SCAN_LOOKBACK_DAYS = 3
SCAN_MAX_DURATION_MIN = 20
SCAN_STUCK_MIN = 45
UNIVERSE_SUSPICIOUS_PCT = 10.0
UNIVERSE_BROKEN_PCT = 30.0
MATURATION_RUNTIME_WARN_MIN = 15
DAILY_GRACE = _dt.timedelta(hours=3)
DB_LATENCY_WARN_MS = 2000
NULL_RATE_WARN_PCT = 1.0
METADATA_MIN_PCT = 95.0
AVAILABILITY_INFO_SHARE = 0.20

WORKFLOW_SPECS = {
    "scheduled-scans.yml": {"label": "Scheduled market scan", "cadence": "trading_day"},
    "mature-observations.yml": {"label": "Mature observations", "cadence": "trading_day"},
    "refresh-universe.yml": {"label": "Universe refresh", "cadence": "weekly", "max_calendar_days": 8},
    "forward-evidence-readiness.yml": {"label": "Research readiness (Run 56)", "cadence": "trading_day"},
    "maturation-parity-audit.yml": {"label": "Maturation parity audit (Run 58)", "cadence": "manual"},
    "system-health.yml": {"label": "System health (Run 59)", "cadence": "trading_day"},
}
ARTIFACT_SPECS = {
    "forward_evidence_readiness": {"label": "Run 56 readiness", "cadence": "trading_day"},
    "maturation_parity_audit": {"label": "Run 58 maturation parity", "cadence": "calendar", "max_days": 7},
    "latest_scanner_observation": {"label": "Latest scanner research capture", "cadence": "trading_day"},
    "previous_system_health": {"label": "Previous health snapshot", "cadence": "trading_day",
                               "optional": True},
}
RECOVERY_IMPLEMENTED = False  # Run 60


# ---- helpers -----------------------------------------------------------------------
def _parse(v: Any) -> Optional[_dt.datetime]:
    if v is None:
        return None
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _iso(d: Optional[_dt.datetime]) -> Optional[str]:
    return d.isoformat() if d else None


def finding(code: str, severity: str, summary: str, *, evidence: Any = None,
            detected_at: Optional[_dt.datetime] = None, action: str = "",
            human: str = "WATCH", automation: bool = False) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "summary": summary, "evidence": evidence,
            "detected_at": _iso(detected_at), "recommended_action": action,
            "human_action": human, "automation_candidate": automation}


def subsystem(status: str, *, detail: str, reason: str, observed: Any = None, expected: Any = None,
              last_updated: Optional[_dt.datetime] = None, action: str = "none",
              findings: Optional[List[Dict[str, Any]]] = None, metrics: Optional[Dict[str, Any]] = None,
              human: Optional[str] = None) -> Dict[str, Any]:
    findings = findings or []
    sev = [f["severity"] for f in findings]
    if status not in ("UNKNOWN",):
        if "CRITICAL" in sev:
            status = "ACTION_REQUIRED"
        elif "WARNING" in sev and STATUS_ORDER.get(status, 0) < STATUS_ORDER["DEGRADED"]:
            status = "DEGRADED"
    ha = human or max([f["human_action"] for f in findings] + ["NO_ACTION"], key=ACTION_ORDER.get)
    return {"status": status, "detail_state": detail, "reason": reason, "observed_value": observed,
            "expected_value": expected, "last_updated": _iso(last_updated) if isinstance(last_updated, _dt.datetime)
            else last_updated, "recommended_action": action, "human_action": ha,
            "findings": findings, "metrics": metrics or {}}


def unknown(reason: str, action: str = "restore telemetry collection") -> Dict[str, Any]:
    return subsystem("UNKNOWN", detail="UNKNOWN", reason=reason, action=action, human="WATCH")


def _runs(runs: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for r in runs or []:
        c = _parse(r.get("created_at") or r.get("createdAt"))
        if c is None:
            continue
        u = _parse(r.get("updated_at") or r.get("updatedAt"))
        out.append({"created": c, "updated": u, "conclusion": r.get("conclusion"),
                    "status": r.get("status") or ("completed" if r.get("conclusion") else None),
                    "event": r.get("event")})
    return sorted(out, key=lambda r: r["created"])


def _consecutive_failures(runs: List[Dict[str, Any]]) -> int:
    n = 0
    for r in reversed([r for r in runs if r["status"] == "completed"]):
        if r["conclusion"] in ("failure", "timed_out", "startup_failure"):
            n += 1
        else:
            break
    return n


def _stale_trading_days(last: Optional[_dt.datetime], now: _dt.datetime) -> Optional[int]:
    if last is None:
        return None
    return mc.completed_trading_days_since(last, now - DAILY_GRACE)


# ---- subsystems ----------------------------------------------------------------------
def eval_universe(u: Optional[Mapping[str, Any]], prev: Optional[Mapping[str, Any]],
                  now: _dt.datetime) -> Dict[str, Any]:
    if not u:
        return unknown("universe probe unavailable")
    count = int(u.get("symbol_count") or 0)
    source = u.get("source")
    prev_count = ((prev or {}).get("metrics") or {}).get("symbol_count") if prev else None
    change = round(100.0 * (count - prev_count) / prev_count, 2) if prev_count else None
    f: List[Dict[str, Any]] = []
    detail = "HEALTHY"
    if source == "none" or count == 0:
        detail = "BROKEN"
        f.append(finding("UNIVERSE_EMPTY", "CRITICAL", "US_MARKET universe is empty (no live and no cached list)",
                         evidence={"source": source}, detected_at=now, human="HUMAN_ACTION_REQUIRED",
                         action="check Alpaca assets endpoint and credentials; scheduled scans will fail"))
    elif source == "cached":
        detail = "STALE"
        f.append(finding("UNIVERSE_FALLBACK_CACHE", "WARNING", "live universe fetch failed; last-known-good cache in use",
                         evidence={"cached_at": u.get("cached_at")}, detected_at=now,
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="retry the live universe fetch; check Alpaca availability"))
    if change is not None and abs(change) >= UNIVERSE_SUSPICIOUS_PCT and detail != "BROKEN":
        crit = abs(change) >= UNIVERSE_BROKEN_PCT
        detail = "BROKEN" if crit else "SUSPICIOUS"
        f.append(finding("UNIVERSE_SIZE_JUMP", "CRITICAL" if crit else "WARNING",
                         f"universe changed {change:+.1f}% vs previous snapshot ({prev_count} → {count})",
                         evidence={"previous": prev_count, "current": count}, detected_at=now,
                         human="HUMAN_ACTION_REQUIRED" if crit else "WATCH",
                         action="compare exclusion counts with the previous refresh before trusting new scans"))
    if u.get("duplicates") or u.get("malformed"):
        f.append(finding("UNIVERSE_HYGIENE", "WARNING", "duplicate or malformed symbols in the final universe",
                         evidence={"duplicates": u.get("duplicates"), "malformed": u.get("malformed")},
                         detected_at=now, action="inspect data/us_market_universe filters"))
    excl = u.get("exclusions") or {}
    return subsystem("HEALTHY", detail=detail, reason=f"{count} tradable US-listed symbols ({source})",
                     observed=count, expected=f"> 1000 symbols, change < {UNIVERSE_SUSPICIOUS_PCT:.0f}% per refresh",
                     last_updated=_parse(u.get("generated_at")), findings=f,
                     action="none" if not f else f[0]["recommended_action"],
                     metrics={"symbol_count": count, "previous_count": prev_count, "change_pct": change,
                              "source": source, "duplicates": u.get("duplicates", 0),
                              "malformed": u.get("malformed", 0),
                              "preferred_excluded": excl.get("preferred_share"),
                              "inactive_excluded": excl.get("inactive"),
                              "non_tradable_excluded": excl.get("non_tradable"),
                              "provider_assets": u.get("provider_assets")})


def expected_slots(now: _dt.datetime, days: int = SCAN_LOOKBACK_DAYS) -> List[_dt.datetime]:
    out = []
    d0 = now.astimezone(mc.ET).date() - _dt.timedelta(days=days)
    d = d0
    while d <= now.astimezone(mc.ET).date():
        out += [s for s in mc.expected_scan_slots(d) if s + SLOT_LATE <= now]
        d += _dt.timedelta(days=1)
    return out


def eval_scanner(scan_runs: Optional[Sequence[Mapping[str, Any]]], db_runs: Optional[Sequence[Mapping[str, Any]]],
                 now: _dt.datetime) -> Dict[str, Any]:
    if scan_runs is None:
        return unknown("scheduled-scan workflow history unavailable (GitHub API)")
    runs = _runs(scan_runs)
    slots = expected_slots(now)
    matched, missing, failed = [], [], []
    for s in slots:
        hit = [r for r in runs if s - SLOT_EARLY <= r["created"] <= s + SLOT_LATE]
        if not hit:
            missing.append(s)
        elif not any(r["conclusion"] == "success" for r in hit):
            failed.append(s)
        else:
            matched.append(s)
    succ = [r for r in runs if r["conclusion"] == "success"]
    last_ok = succ[-1] if succ else None
    f: List[Dict[str, Any]] = []
    if missing:
        f.append(finding("MISSED_SCAN", "WARNING", f"{len(missing)} expected scan slot(s) had no run",
                         evidence=[_iso(s) for s in missing[-6:]], detected_at=missing[0],
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="check cron-job.org dispatch and PAT; re-dispatch scheduled-scans (force=false)"))
    cf = _consecutive_failures(runs)
    if cf >= 3:
        f.append(finding("FAILED_SCAN", "CRITICAL", f"{cf} consecutive scheduled scan failures",
                         detected_at=runs[-cf]["created"], human="HUMAN_ACTION_REQUIRED",
                         action="open the latest failed scheduled-scans run log"))
    elif failed:
        f.append(finding("FAILED_SCAN", "WARNING", f"{len(failed)} expected slot(s) ran but failed",
                         evidence=[_iso(s) for s in failed[-6:]], detected_at=failed[0],
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="re-dispatch the scan; inspect the failed run if it repeats"))
    today = now.astimezone(mc.ET).date()
    if mc.is_market_open(now) and last_ok and now - last_ok["created"] > _dt.timedelta(hours=3, minutes=30):
        f.append(finding("STALE_SCANNER", "WARNING", "market is open and the last successful scan is > 3.5h old",
                         evidence={"last_success": _iso(last_ok["created"])}, detected_at=last_ok["created"],
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="re-dispatch scheduled-scans"))
    last_day_slots = [s for s in slots if s.astimezone(mc.ET).date() == (max(slots).astimezone(mc.ET).date()
                                                                          if slots else today)]
    if last_day_slots and all(s in missing + failed for s in last_day_slots):
        f.append(finding("STALE_SCANNER", "CRITICAL", "no successful scan on the most recent trading day",
                         evidence={"trading_day": str(last_day_slots[0].astimezone(mc.ET).date())},
                         detected_at=last_day_slots[0], human="HUMAN_ACTION_REQUIRED",
                         action="scanner is down: check workflow, secrets and cron-job.org"))
    running = [r for r in runs if r["status"] in ("in_progress", "queued")]
    for r in running:
        if now - r["created"] > _dt.timedelta(minutes=SCAN_STUCK_MIN):
            f.append(finding("ABNORMAL_DURATION", "WARNING", "a scheduled scan has been running > 45 min",
                             evidence={"started": _iso(r["created"])}, detected_at=r["created"],
                             action="cancel and re-dispatch if stuck", human="AUTOMATIC_RECOVERY_CANDIDATE",
                             automation=True))
    dur = None
    if last_ok and last_ok["updated"]:
        dur = round((last_ok["updated"] - last_ok["created"]).total_seconds() / 60.0, 1)
        if dur > SCAN_MAX_DURATION_MIN:
            f.append(finding("ABNORMAL_DURATION", "WARNING", f"last scan took {dur} min (> {SCAN_MAX_DURATION_MIN})",
                             detected_at=last_ok["created"], action="check provider latency / universe size"))
    recent_db = [r for r in (db_runs or []) if _parse(r.get("created_at"))]
    partial = [r for r in recent_db if (r.get("row_count") or 0) < 10]
    if partial:
        f.append(finding("PARTIAL_SCAN", "WARNING", f"{len(partial)} saved scan(s) with < 10 result rows",
                         evidence=[{"label": r.get("label"), "rows": r.get("row_count")} for r in partial[:5]],
                         detected_at=_parse(partial[0].get("created_at")),
                         action="check price-provider coverage for those runs"))
    detail = "MARKET_CLOSED_EXPECTED_IDLE" if not slots or not mc.is_trading_day(today) and not f else (
        "HEALTHY" if not f else f[0]["code"])
    return subsystem("HEALTHY", detail=detail,
                     reason=(f"{len(matched)}/{len(slots)} expected slots in the last {SCAN_LOOKBACK_DAYS} days "
                             "ran successfully") if slots else "no scans expected in the lookback window",
                     observed=len(matched), expected=len(slots), last_updated=last_ok["created"] if last_ok else None,
                     findings=f, action="none" if not f else f[0]["recommended_action"],
                     metrics={"expected_scans": len(slots), "completed_scans": len(matched),
                              "failed_scans": len(failed), "missing_scans": len(missing),
                              "most_recent_success": _iso(last_ok["created"]) if last_ok else None,
                              "last_duration_min": dur,
                              "recent_saved_runs": [{"label": r.get("label"), "rows": r.get("row_count"),
                                                     "duration_sec": r.get("duration_sec")} for r in recent_db[-6:]],
                              "symbols_attempted": None, "symbols_processed": None,
                              "coverage_note": "per-scan attempted/processed counts are not persisted"})


def eval_research_capture(summary: Optional[Mapping[str, Any]], scan_runs: Optional[Sequence[Mapping[str, Any]]],
                          now: _dt.datetime) -> Dict[str, Any]:
    if summary is None:
        return unknown("research observation summary unavailable (database)")
    f: List[Dict[str, Any]] = []
    per_scan = {(_parse(p.get("scan_time"))): p for p in summary.get("per_scan") or [] if _parse(p.get("scan_time"))}
    runs = [r for r in _runs(scan_runs) if r["conclusion"] == "success"] if scan_runs is not None else []
    window_start = now - _dt.timedelta(days=SCAN_LOOKBACK_DAYS)
    zero = []
    for r in runs:
        if r["created"] < window_start or not mc.is_market_open(r["created"] + _dt.timedelta(minutes=2)):
            continue
        end = (r["updated"] or r["created"]) + _dt.timedelta(minutes=5)
        if not any(r["created"] <= t <= end for t in per_scan):
            zero.append(r["created"])
    if zero:
        f.append(finding("ZERO_OBSERVATIONS", "WARNING",
                         f"{len(zero)} successful regular-session scan(s) captured no research observations",
                         evidence=[_iso(t) for t in zero[-6:]], detected_at=zero[0],
                         action="check HSF_OBSERVATION_CAPTURE / HSF_RESEARCH_CAPTURE and capture logs"))
    thin = [p for p in per_scan.values() if any((p.get("by_cohort") or {}).get(c, 0) == 0
                                                for c in ("CANDIDATE", "NEAR_MISS", "CONTROL"))
            and _parse(p.get("scan_time")) and mc.is_market_open(_parse(p.get("scan_time")))]
    if thin:
        f.append(finding("COHORT_MISSING_IN_SCAN", "WARNING",
                         f"{len(thin)} regular-session scan(s) missing a research cohort",
                         evidence=[{"scan": p.get("scan_time"), "by_cohort": p.get("by_cohort")} for p in thin[:5]],
                         detected_at=_parse(thin[0].get("scan_time")),
                         action="check _capture_research_cohorts output for those scans"))
    for key, code, msg in (("untagged", "MISSING_COHORT_LABEL", "observations without a research cohort label"),
                           ("duplicate_ids", "DUPLICATE_OBSERVATIONS", "duplicate observation ids"),
                           ("malformed", "MALFORMED_OBSERVATIONS", "observations missing symbol/timestamp")):
        if summary.get(key):
            f.append(finding(code, "WARNING", f"{summary[key]} {msg} in the recent window",
                             detected_at=now, action="inspect the capture path; do not rewrite records"))
    md_n, md_pct = summary.get("metadata_scope_n") or 0, summary.get("metadata_block_pct")
    if md_n and md_pct is not None and md_pct < METADATA_MIN_PCT:
        f.append(finding("METADATA_INCOMPLETE", "WARNING",
                         f"Run 57 research_metadata on {md_pct}% of new observations (< {METADATA_MIN_PCT:.0f}%)",
                         evidence={"scope": md_n}, detected_at=now,
                         action="check research_metadata context building in cron_runner"))
    n = summary.get("observations", 0)
    waiting = n == 0 and not zero
    return subsystem("WAITING" if waiting else "HEALTHY",
                     detail="NO_RECENT_SCANS" if waiting else ("HEALTHY" if not f else f[0]["code"]),
                     reason=f"{n} research observations in the last {SCAN_LOOKBACK_DAYS} days",
                     observed=n, expected="every regular-session scan captures all three cohorts",
                     last_updated=summary.get("latest_created_at"), findings=f,
                     action="none" if not f else f[0]["recommended_action"],
                     metrics={"observations": n, "by_cohort": summary.get("by_cohort"),
                              "untagged": summary.get("untagged", 0), "duplicate_ids": summary.get("duplicate_ids", 0),
                              "malformed": summary.get("malformed", 0),
                              "metadata_block_pct": md_pct, "metadata_scope_n": md_n,
                              "scans_in_window": len(per_scan)})


def eval_maturation(report: Optional[Mapping[str, Any]], runs: Optional[Sequence[Mapping[str, Any]]],
                    prev: Optional[Mapping[str, Any]], now: _dt.datetime) -> Dict[str, Any]:
    if report is None and runs is None:
        return unknown("maturation report and workflow history unavailable")
    f: List[Dict[str, Any]] = []
    rs = _runs(runs) if runs is not None else []
    succ = [r for r in rs if r["conclusion"] == "success"]
    last_ok = succ[-1]["created"] if succ else _parse((report or {}).get("generated_at"))
    stale = _stale_trading_days(last_ok, now)
    if stale is not None and stale >= 1:
        f.append(finding("MATURATION_STALE", "CRITICAL" if stale >= 2 else "WARNING",
                         f"no successful maturation for {stale} completed trading day(s)",
                         evidence={"last_success": _iso(last_ok)}, detected_at=last_ok,
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="re-dispatch mature-observations (idempotent)"))
    cf = _consecutive_failures(rs)
    if cf >= 2:
        f.append(finding("MATURATION_FAILING", "CRITICAL" if cf >= 3 else "WARNING",
                         f"{cf} consecutive maturation run failures", detected_at=rs[-cf]["created"],
                         human="HUMAN_ACTION_REQUIRED" if cf >= 3 else "AUTOMATIC_RECOVERY_CANDIDATE",
                         automation=cf < 3, action="inspect the failed mature-observations run"))
    dur = None
    if succ and succ[-1]["updated"]:
        dur = round((succ[-1]["updated"] - succ[-1]["created"]).total_seconds() / 60.0, 1)
        if dur > MATURATION_RUNTIME_WARN_MIN:
            f.append(finding("RUNTIME_PRESSURE", "WARNING", f"last maturation took {dur} min (limit 20)",
                             detected_at=succ[-1]["created"],
                             action="batch outcome saves / review MAX_SYMBOLS before the workflow times out"))
    r = report or {}
    b = r.get("backlog") or {}
    rl = int(r.get("rate_limited_symbols") or 0)
    n429 = int(r.get("alpaca_429_count") or 0)
    if rl > 0:
        f.append(finding("RATE_LIMIT_PRESSURE", "WARNING", f"{rl} symbols skipped after persistent HTTP 429",
                         evidence={"alpaca_429_count": n429}, detected_at=_parse(r.get("generated_at")),
                         human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="let the next run retry; if it repeats, lower request pacing"))
    elif n429 > 0:
        f.append(finding("RATE_LIMIT_RECOVERED", "INFO", f"{n429} HTTP 429 response(s) recovered by retry",
                         detected_at=_parse(r.get("generated_at")), human="NO_ACTION"))
    deferred = int(r.get("symbols_deferred") or b.get("deferred_symbols") or 0)
    ready = b.get("ready_symbols") or r.get("symbols_with_ready_horizons")
    prev_ready = ((prev or {}).get("metrics") or {}).get("ready_symbols") if prev else None
    if deferred > 0:
        growing = prev_ready is not None and ready is not None and ready > 1.25 * prev_ready
        f.append(finding("BACKLOG_GROWING" if growing else "MATURATION_CAP_BINDING", "WARNING",
                         f"{deferred} ready symbols deferred by the per-run cap"
                         + (f" (ready {prev_ready} → {ready})" if growing else ""),
                         evidence={"ready_symbols": ready, "deferred": deferred},
                         detected_at=_parse(r.get("generated_at")), human="WATCH",
                         action="cohort-neutral only while the cap does not bind (Run 58); review MAX_SYMBOLS"))
    retired = int(((r.get("retired") or {}).get("observations")) or 0)
    prev_ret = ((prev or {}).get("metrics") or {}).get("retired_observations") if prev else None
    if retired > max(100, 2 * (prev_ret or 0)) and prev_ret is not None:
        f.append(finding("RETIREMENT_SPIKE", "WARNING", f"{retired} observations retired (previous {prev_ret})",
                         detected_at=_parse(r.get("generated_at")), human="WATCH",
                         action="check whether observations were starved before their window closed"))
    proc = int(r.get("symbols_processed") or b.get("processed_symbols") or 0)
    nod = int(r.get("price_data_unavailable_symbols") or 0)
    if proc and nod / proc >= AVAILABILITY_INFO_SHARE:
        f.append(finding("DATA_AVAILABILITY_LIMITED", "INFO",
                         f"{nod}/{proc} processed symbols had no minute bars (explained by Run 58)",
                         detected_at=_parse(r.get("generated_at")), human="NO_ACTION"))
    if report is None:
        f.append(finding("MATURATION_REPORT_MISSING", "WARNING", "latest scheduled maturation report not found",
                         detected_at=now, human="WATCH", action="check the maturation-report artifact upload"))
    return subsystem("HEALTHY", detail="HEALTHY" if not [x for x in f if x["severity"] != "INFO"] else f[0]["code"],
                     reason=f"last successful maturation {_iso(last_ok) or 'unknown'}",
                     observed={"ready_symbols": ready, "deferred": deferred},
                     expected="≥ 1 successful run per trading day; cap not binding; no persistent 429",
                     last_updated=last_ok, findings=f, action="none" if not f else f[0]["recommended_action"],
                     metrics={"eligible_observations": r.get("eligible_observations"),
                              "matured_this_run": r.get("outcomes_matured", r.get("attached")),
                              "ready_symbols": ready, "ready_observations": b.get("ready_observations"),
                              "deferred_symbols": deferred, "retired_observations": retired,
                              "oldest_pending_age_min": b.get("oldest_pending_age_min"),
                              "last_success": _iso(last_ok), "runtime_min": dur,
                              "alpaca_requests": r.get("alpaca_requests"), "alpaca_429_count": n429,
                              "alpaca_retry_count": r.get("alpaca_retry_count"),
                              "cache_hits": r.get("cache_hits"), "cache_misses": r.get("cache_misses"),
                              "report_schema": r.get("schema")})


def eval_market_data(report: Optional[Mapping[str, Any]], universe: Optional[Mapping[str, Any]],
                     now: _dt.datetime) -> Dict[str, Any]:
    if report is None and universe is None:
        return unknown("no provider telemetry (maturation report and universe probe both unavailable)")
    r = report or {}
    f: List[Dict[str, Any]] = []
    proc = int(r.get("symbols_processed") or 0)
    prov_err = int(r.get("provider_error_symbols") or 0)
    rl = int(r.get("rate_limited_symbols") or 0)
    if proc and (prov_err + rl) / proc >= 0.10:
        f.append(finding("ALPACA_PERSISTENT_FAILURES", "WARNING",
                         f"{prov_err + rl}/{proc} symbols failed at the provider (errors/429)",
                         detected_at=_parse(r.get("generated_at")), human="AUTOMATIC_RECOVERY_CANDIDATE",
                         automation=True, action="retry next cycle; check Alpaca status page if it repeats"))
    uni_ok = universe is not None and universe.get("source") == "live"
    if universe is not None and not uni_ok:
        f.append(finding("ALPACA_ASSETS_UNAVAILABLE", "WARNING", "Alpaca assets endpoint failed for the universe probe",
                         detected_at=now, human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                         action="retry; verify ALPACA credentials"))
    providers = [{
        "provider": "alpaca", "instrumented": True,
        "requests": r.get("alpaca_requests"), "successes": None if r.get("alpaca_requests") is None else
        max(0, int(r.get("alpaca_requests") or 0) - int(r.get("alpaca_retry_count") or 0)),
        "failures": prov_err, "http_429": r.get("alpaca_429_count"), "retries": r.get("alpaca_retry_count"),
        "timeouts": None, "empty_responses": r.get("price_data_unavailable_symbols"),
        "rate_limited_symbols": rl, "assets_endpoint": None if universe is None else universe.get("source"),
        "last_successful_request": r.get("generated_at") if r.get("alpaca_requests") else None,
    }, {
        "provider": "yfinance", "instrumented": False,
        "note": "possible price-fetch fallback in the scan path; per-provider counts are not persisted",
    }]
    return subsystem("HEALTHY", detail="HEALTHY" if not f else f[0]["code"],
                     reason="Alpaca healthy" if not f else f[0]["summary"],
                     observed={"alpaca_requests": r.get("alpaca_requests"), "http_429": r.get("alpaca_429_count")},
                     expected="no persistent provider failures", last_updated=_parse(r.get("generated_at")),
                     findings=f, action="none" if not f else f[0]["recommended_action"],
                     metrics={"providers": providers})


def eval_cohort_parity(parity_report: Optional[Mapping[str, Any]], readiness: Optional[Mapping[str, Any]],
                       now: _dt.datetime) -> Dict[str, Any]:
    if parity_report is None and readiness is None:
        return unknown("Run 58 parity audit and Run 56 readiness both unavailable")
    hist = ((parity_report or {}).get("current_historical") or {}).get("parity")
    rc = (parity_report or {}).get("root_cause") or {}
    fwd = (readiness or {}).get("maturation_parity") or {}
    horizons = ("+5m", "+15m", "+30m", "+60m")
    f: List[Dict[str, Any]] = []
    fwd_rows = {}
    measurable_classes = []
    for h in horizons:
        p = fwd.get(h) or {}
        fwd_rows[h] = {"candidate_coverage": p.get("candidate_maturation_pct"),
                       "near_miss_coverage": p.get("near_miss_maturation_pct"),
                       "control_coverage": p.get("control_maturation_pct"),
                       "parity_gap": p.get("maturation_parity_gap"),
                       "classification": p.get("parity_classification"), "measurable": bool(p.get("measurable"))}
        if p.get("measurable"):
            measurable_classes.append(p.get("parity_classification"))
    if "CRITICAL" in measurable_classes:
        f.append(finding("FORWARD_PARITY_CRITICAL", "CRITICAL",
                         "forward-epoch cohort maturation parity is CRITICAL (> 20 pp)",
                         evidence={h: fwd_rows[h]["parity_gap"] for h in horizons}, detected_at=now,
                         human="HUMAN_ACTION_REQUIRED",
                         action="research-design decision required (Run 58: liquidity-comparable controls, "
                                "SIP feed, or pre-registered matched comparison)"))
    elif "WARNING" in measurable_classes:
        f.append(finding("FORWARD_PARITY_WARNING", "WARNING", "forward-epoch parity gap 10–20 pp",
                         detected_at=now, human="WATCH", action="monitor; see Run 58 root cause"))
    detail = ("COLLECTING" if not measurable_classes else
              ("CRITICAL" if "CRITICAL" in measurable_classes else
               "WARNING" if "WARNING" in measurable_classes else "HEALTHY"))
    status = "WAITING" if not measurable_classes else "HEALTHY"
    hist_note = ("explained" if rc.get("classification") in ("MARKET_DATA_AVAILABILITY_EFFECT",
                                                               "EXPECTED_TEMPORAL_MISSINGNESS")
                 and rc.get("confidence") in ("HIGH", "MODERATE") else "unexplained" if rc else None)
    return subsystem(status, detail=detail,
                     reason=("forward parity not yet measurable" if not measurable_classes
                             else f"forward parity {detail}") + (
                         f"; historical parity {((hist or {}).get('+60m') or {}).get('classification')} "
                         f"({hist_note}: {rc.get('classification')})" if hist else ""),
                     observed={h: fwd_rows[h]["parity_gap"] for h in horizons},
                     expected="≤ 10 pp per horizon once measurable (Run 56 Gate E)",
                     last_updated=(parity_report or {}).get("generated_at"), findings=f,
                     action="none" if not f else f[0]["recommended_action"],
                     metrics={"historical_parity": hist, "historical_root_cause": rc.get("classification"),
                              "historical_root_cause_confidence": rc.get("confidence"),
                              "historical_parity_explained": hist_note == "explained",
                              "forward_epoch_parity": fwd_rows})


_READINESS_WHITELIST_GATES = ("A_trading_days", "B_scan_runs", "C_cohort_clusters", "D_horizon_maturation",
                              "E_maturation_parity", "F_directional_integrity", "G_effective_clusters",
                              "H_research_integrity")


def eval_forward_evidence(readiness: Optional[Mapping[str, Any]], now: _dt.datetime) -> Dict[str, Any]:
    if readiness is None:
        return unknown("Run 56 readiness unavailable")
    state = readiness.get("state")
    limiting = readiness.get("limiting_factor")
    t = readiness.get("time_coverage") or {}
    s = readiness.get("scan_coverage") or {}
    gates = readiness.get("gates") or {}
    mat = [((readiness.get("horizons") or {}).get(h) or {}) for h in ("+5m", "+15m", "+30m", "+60m")]
    min_cov = [min((c.get("maturation_pct") for c in m.values() if c.get("maturation_pct") is not None), default=None)
               for m in mat]
    known_cov = [c for c in min_cov if c is not None]
    f: List[Dict[str, Any]] = []
    if state == "DATA_QUALITY_BLOCKED":
        ext = "FORWARD_EVIDENCE_BLOCKED"
        f.append(finding(ext, "CRITICAL", f"forward evidence blocked: {readiness.get('state_reason')}",
                         evidence={"limiting_factor": limiting}, detected_at=now, human="HUMAN_ACTION_REQUIRED",
                         action="resolve the blocking data-quality issue before any formal evaluation"))
        out_state, status = "DATA_QUALITY_BLOCKED", "HEALTHY"
    elif state == "READY_FOR_RUN55_RERUN":
        f.append(finding("FORMAL_EVALUATION_READY", "INFO", "forward evidence gates pass; formal evaluation may run",
                         detected_at=now, human="HUMAN_ACTION_REQUIRED",
                         action="approve and manually dispatch Signal Effectiveness Analysis (never automatic)"))
        out_state, status = "READY_FOR_FORMAL_EVALUATION", "HEALTHY"
    elif limiting == "NO_FORWARD_DATA":
        out_state, status = "NO_FORWARD_DATA", "WAITING"
    else:
        out_state, status = "COLLECTING", "WAITING"
    days = t.get("completed_forward_trading_days", 0)
    runs = s.get("regular_session_scan_runs", 0)
    return subsystem(status, detail=out_state,
                     reason=f"{out_state}: day {days}/20, runs {runs}/100",
                     observed={"trading_days": days, "scan_runs": runs},
                     expected="Run 56 gates (min 10 days / 50 runs; preferred 20 / 100)",
                     last_updated=readiness.get("generated_at"), findings=f,
                     action="none" if not f else f[0]["recommended_action"],
                     metrics={"forward_evidence_status": out_state,
                              "forward_epoch_start": (readiness.get("epoch") or {}).get("forward_epoch_start_timestamp"),
                              "trading_days_collected": days, "trading_days_min": 10, "trading_days_preferred": 20,
                              "scan_runs_collected": runs, "scan_runs_min": 50, "scan_runs_preferred": 100,
                              "observations_collected": s.get("forward_observations_regular_session"),
                              "min_cohort_maturation_pct_by_horizon": dict(zip(("+5m", "+15m", "+30m", "+60m"), min_cov)),
                              "min_cohort_maturation_pct": min(known_cov) if known_cov else None,
                              "gates": {k: (gates.get(k) or {}).get("status") for k in _READINESS_WHITELIST_GATES},
                              "estimated_trading_days_until_ready": readiness.get("estimated_trading_days_until_ready"),
                              "long_readiness": readiness.get("long_readiness"),
                              "short_readiness": readiness.get("short_readiness"),
                              "formal_evaluation": "HUMAN APPROVAL REQUIRED" if out_state ==
                              "READY_FOR_FORMAL_EVALUATION" else "NOT READY"})


def eval_workflows(workflows: Optional[Mapping[str, Optional[Sequence[Mapping[str, Any]]]]],
                   now: _dt.datetime) -> Dict[str, Any]:
    if workflows is None:
        return unknown("GitHub workflow metadata unavailable")
    rows, f = [], []
    any_known = False
    for name, spec in WORKFLOW_SPECS.items():
        runs_raw = workflows.get(name)
        if runs_raw is None:
            rows.append({"workflow": name, "label": spec["label"], "cadence": spec["cadence"], "status": "UNKNOWN"})
            continue
        any_known = True
        rs = _runs(runs_raw)
        succ = [r for r in rs if r["conclusion"] == "success"]
        fail = [r for r in rs if r["conclusion"] in ("failure", "timed_out", "startup_failure")]
        last_ok = succ[-1]["created"] if succ else None
        status = "HEALTHY"
        freshness = None
        if spec["cadence"] == "trading_day":
            freshness = _stale_trading_days(last_ok, now)
            if last_ok is None:
                status = "NEVER_RUN"
            elif freshness is not None and freshness >= 2:
                status = "WORKFLOW_STALE"
            # Only judge days the returned history actually covers (API pages are bounded).
            gap_days = []
            if rs:
                first = rs[0]["created"].astimezone(mc.ET).date()
                start = max(first, (now - _dt.timedelta(days=5)).astimezone(mc.ET).date())
                for d in mc.trading_days_between(start, now.astimezone(mc.ET).date()):
                    _o, c = mc.session_bounds_utc(d)
                    if c + DAILY_GRACE > now:
                        continue
                    if not any(r["created"].astimezone(mc.ET).date() == d for r in rs):
                        gap_days.append(str(d))
            if gap_days and status == "HEALTHY":
                status = "SCHEDULE_GAP"
        elif spec["cadence"] == "weekly":
            freshness = (now - last_ok).days if last_ok else None
            if last_ok is None:
                status = "NEVER_RUN"
            elif freshness > spec["max_calendar_days"]:
                status = "WORKFLOW_STALE"
        cf = _consecutive_failures(rs)
        if cf >= 2:
            status = "CONSECUTIVE_FAILURES"
        row = {"workflow": name, "label": spec["label"], "cadence": spec["cadence"], "known": True,
               "last_run": _iso(rs[-1]["created"]) if rs else None, "last_success": _iso(last_ok),
               "last_failure": _iso(fail[-1]["created"]) if fail else None,
               "freshness": freshness, "consecutive_failures": cf, "status": status}
        rows.append(row)
        if status == "CONSECUTIVE_FAILURES":
            f.append(finding("CONSECUTIVE_FAILURES", "CRITICAL" if cf >= 3 else "WARNING",
                             f"{spec['label']}: {cf} consecutive failures", evidence=row,
                             detected_at=rs[-cf]["created"], human="HUMAN_ACTION_REQUIRED" if cf >= 3
                             else "AUTOMATIC_RECOVERY_CANDIDATE", automation=cf < 3,
                             action=f"inspect the latest {name} run log"))
        elif status == "WORKFLOW_STALE":
            f.append(finding("WORKFLOW_STALE", "WARNING", f"{spec['label']} has not succeeded recently",
                             evidence=row, detected_at=last_ok, human="AUTOMATIC_RECOVERY_CANDIDATE",
                             automation=True, action=f"re-dispatch {name}"))
        elif status == "SCHEDULE_GAP":
            f.append(finding("SCHEDULE_GAP", "WARNING", f"{spec['label']} skipped a trading day", evidence=row,
                             detected_at=now, human="WATCH", action=f"check the {name} trigger"))
        elif status == "NEVER_RUN" and name != "system-health.yml":
            f.append(finding("WORKFLOW_NEVER_RUN", "WARNING", f"{spec['label']} has no successful run on record",
                             evidence=row, detected_at=now, human="WATCH", action=f"dispatch {name} once"))
    if not any_known:
        return unknown("GitHub workflow metadata unavailable")
    return subsystem("HEALTHY", detail="HEALTHY" if not f else f[0]["code"],
                     reason=f"{sum(1 for r in rows if r['status'] == 'HEALTHY')}/{len(rows)} workflows healthy",
                     observed=len(rows), expected="each workflow fresh for its cadence", findings=f,
                     action="none" if not f else f[0]["recommended_action"], metrics={"workflows": rows})


def eval_database(probe: Optional[Mapping[str, Any]], now: _dt.datetime) -> Dict[str, Any]:
    if probe is None:
        return unknown("database probe did not run")
    f: List[Dict[str, Any]] = []
    if not probe.get("connected"):
        f.append(finding("DATABASE_UNAVAILABLE", "CRITICAL", "cannot connect to the research database",
                         evidence={"error": probe.get("error")}, detected_at=now, human="HUMAN_ACTION_REQUIRED",
                         action="check DATABASE_URL secret and Neon status"))
        return subsystem("ACTION_REQUIRED", detail="UNAVAILABLE", reason="database unreachable", findings=f,
                         action=f[0]["recommended_action"], metrics=dict(probe))
    for key, label in (("observations_accessible", "hsf_observations"), ("outcomes_accessible", "hsf_observation_outcomes")):
        if probe.get(key) is False:
            f.append(finding("TABLE_UNREADABLE", "CRITICAL", f"{label} is not readable", detected_at=now,
                             human="HUMAN_ACTION_REQUIRED", action="check table grants / schema"))
    lat = probe.get("latency_ms")
    if lat is not None and lat > DB_LATENCY_WARN_MS:
        f.append(finding("DATABASE_SLOW", "WARNING", f"query latency {lat} ms", detected_at=now, human="WATCH",
                         action="check Neon compute state"))
    if probe.get("duplicate_outcome_keys"):
        f.append(finding("DUPLICATE_KEYS", "CRITICAL", "duplicate (observation_id, horizon) outcome keys",
                         evidence={"count": probe["duplicate_outcome_keys"]}, detected_at=now,
                         human="HUMAN_ACTION_REQUIRED", action="investigate; do not repair automatically"))
    nulls = {k: v for k, v in (probe.get("null_rates_pct") or {}).items() if v is not None and v > NULL_RATE_WARN_PCT}
    if nulls:
        f.append(finding("REQUIRED_FIELD_NULLS", "WARNING", "required research fields have unexpected nulls",
                         evidence=nulls, detected_at=now, action="inspect recent captures"))
    return subsystem("HEALTHY", detail="HEALTHY" if not f else f[0]["code"],
                     reason=f"connected, {lat} ms", observed=lat, expected=f"< {DB_LATENCY_WARN_MS} ms",
                     last_updated=probe.get("latest_observation_created_at"), findings=f,
                     action="none" if not f else f[0]["recommended_action"], metrics=dict(probe))


def eval_artifacts(artifacts: Optional[Mapping[str, Optional[Mapping[str, Any]]]], now: _dt.datetime) -> Dict[str, Any]:
    if artifacts is None:
        return unknown("artifact metadata unavailable")
    rows, f = [], []
    unknown_required = False
    for name, spec in ARTIFACT_SPECS.items():
        a = artifacts.get(name)
        gen = _parse((a or {}).get("generated_at"))
        if gen is None:
            st = "NOT_APPLICABLE" if spec.get("optional") else "UNKNOWN"
            unknown_required = unknown_required or not spec.get("optional")
            rows.append({"artifact": name, "label": spec["label"], "generated_at": None, "status": st})
            continue
        age_h = round((now - gen).total_seconds() / 3600.0, 1)
        if spec["cadence"] == "trading_day":
            stale_days = _stale_trading_days(gen, now)
            stale = stale_days is not None and stale_days >= 2
            expected = "≤ 1 completed trading day"
        else:
            stale = age_h > spec["max_days"] * 24
            expected = f"≤ {spec['max_days']} days"
        st = "STALE" if stale else "FRESH"
        rows.append({"artifact": name, "label": spec["label"], "generated_at": _iso(gen), "age_hours": age_h,
                     "expected_max_age": expected, "status": st})
        if stale:
            f.append(finding(f"STALE_ARTIFACT_{name.upper()}", "WARNING", f"{spec['label']} is stale ({age_h} h)",
                             evidence=rows[-1], detected_at=gen, human="AUTOMATIC_RECOVERY_CANDIDATE", automation=True,
                             action="re-dispatch the workflow that produces it"))
    status = "UNKNOWN" if unknown_required and not f else "HEALTHY"
    return subsystem(status, detail="HEALTHY" if not f else "STALE",
                     reason=f"{sum(1 for r in rows if r['status'] == 'FRESH')}/{len(rows)} artifacts fresh",
                     findings=f, action="none" if not f else f[0]["recommended_action"],
                     metrics={"artifacts": rows})


# ---- aggregation ------------------------------------------------------------------------
def evaluate(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the full health model. Each evaluator is isolated: an exception in
    one subsystem yields UNKNOWN for it and never blocks the report."""
    now = _parse(inputs.get("now")) or _dt.datetime.now(_dt.timezone.utc)
    prev = inputs.get("previous") or {}
    psub = (prev.get("subsystems") or {}) if isinstance(prev, dict) else {}
    calls: Dict[str, Callable[[], Dict[str, Any]]] = {
        "universe": lambda: eval_universe(inputs.get("universe"), psub.get("universe"), now),
        "scanner": lambda: eval_scanner(inputs.get("scan_runs"), inputs.get("db_runs"), now),
        "research_capture": lambda: eval_research_capture(inputs.get("observations_summary"),
                                                          inputs.get("scan_runs"), now),
        "maturation": lambda: eval_maturation(inputs.get("maturation_report"), inputs.get("maturation_runs"),
                                              psub.get("maturation"), now),
        "market_data": lambda: eval_market_data(inputs.get("maturation_report"), inputs.get("universe"), now),
        "cohort_parity": lambda: eval_cohort_parity(inputs.get("parity"), inputs.get("readiness"), now),
        "forward_evidence": lambda: eval_forward_evidence(inputs.get("readiness"), now),
        "workflows": lambda: eval_workflows(inputs.get("workflows"), now),
        "database": lambda: eval_database(inputs.get("db_probe"), now),
        "artifact_freshness": lambda: eval_artifacts(inputs.get("artifacts"), now),
    }
    subs: Dict[str, Dict[str, Any]] = {}
    for name, fn in calls.items():
        try:
            subs[name] = fn()
        except Exception as e:  # health plane must never crash
            subs[name] = unknown(f"evaluator error: {type(e).__name__}: {e}")
    system = max((s["status"] for s in subs.values()), key=lambda x: STATUS_ORDER.get(x, 1))
    system = "HEALTHY" if system == "WAITING" else system
    human = max((s["human_action"] for s in subs.values()), key=lambda x: ACTION_ORDER.get(x, 0))
    if system == "UNKNOWN" and ACTION_ORDER[human] < ACTION_ORDER["WATCH"]:
        human = "WATCH"
    incidents = build_incidents(subs, now)
    score = max(0, 100 - sum(SCORE_PENALTY.get(s["status"], 0) for s in subs.values()))
    report = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "system_status": system,
        "human_action": human,
        "health_score": score,
        "health_score_note": "secondary to categorical status; never overrides it",
        "subsystems": subs,
        "incidents": incidents,
        "automation_candidates": [
            {"incident_id": i["incident_id"], "subsystem": i["subsystem"], "action": i["recommended_action"]}
            for i in incidents if i["automation_candidate"]],
        "forward_evidence_status": subs["forward_evidence"]["detail_state"],
        "market": {"is_trading_day": mc.is_trading_day(now.astimezone(mc.ET).date()),
                   "market_open": mc.is_market_open(now),
                   "calendar_covered": mc.calendar_covered(now.astimezone(mc.ET).date()),
                   "next_expected_scan": _iso(mc.next_expected_scan(now))},
        "autonomy_readiness": autonomy(subs),
        "collection_errors": inputs.get("collection_errors") or {},
    }
    assert_clean(report)
    return report


def build_incidents(subs: Mapping[str, Mapping[str, Any]], now: _dt.datetime) -> List[Dict[str, Any]]:
    out = []
    for name, s in subs.items():
        for f in s.get("findings") or []:
            det = _parse(f.get("detected_at")) or now
            out.append({"incident_id": f"{name}:{f['code']}", "severity": f["severity"], "subsystem": name,
                        "detected_at": _iso(det), "age_hours": round((now - det).total_seconds() / 3600.0, 1),
                        "summary": f["summary"], "evidence": f.get("evidence"),
                        "recommended_action": f.get("recommended_action"), "human_action": f["human_action"],
                        "automation_candidate": bool(f.get("automation_candidate"))})
    out.sort(key=lambda i: (SEVERITY_ORDER.get(i["severity"], 9), -i["age_hours"], i["incident_id"]))
    return out


def autonomy(subs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    blind = [n for n in ("database", "scanner", "workflows", "maturation") if subs[n]["status"] == "UNKNOWN"]
    to_recovery = [
        "no automatic re-dispatch of stale/missed workflows (scan, maturation, readiness, parity audit)",
        "no automatic maturation retry escalation",
        "no incident notification layer (email/Slack) for HUMAN_ACTION_REQUIRED",
        "no recovery audit trail / guardrails (rate limits, max retries, kill switch)",
    ] + [f"telemetry blind spot: {n} UNKNOWN" for n in blind]
    to_autonomous = to_recovery + [
        "scanner coverage telemetry (symbols attempted/processed) not persisted",
        "yfinance fallback usage not instrumented",
        "calendar covers 2025-2027 only (needs yearly update or Alpaca calendar cross-check)",
        "formal evaluation requires explicit human approval by design (never automatic)",
    ]
    if subs["cohort_parity"]["detail_state"] in ("CRITICAL", "WARNING") or \
            subs["forward_evidence"]["detail_state"] == "DATA_QUALITY_BLOCKED":
        to_autonomous.append("forward-experiment design decision pending (Run 58 control parity)")
    state = "NOT_READY" if blind else ("RECOVERY_READY" if RECOVERY_IMPLEMENTED else "OBSERVABLE")
    return {"state": state, "blockers_to_recovery_ready": to_recovery, "blockers_to_autonomous": to_autonomous}


# ---- anti-peeking --------------------------------------------------------------------------
_FORBIDDEN_TOKENS = {"return", "returns", "win", "wins", "winrate", "pnl", "profit", "spearman", "pearson",
                     "corr", "correlation", "payoff", "mfe", "mae", "sharpe", "edge", "lift", "alpha",
                     "expectancy", "drawdown", "performance", "effectiveness", "effect"}
_FORBIDDEN_TEXT = re.compile(r"(win[ _-]?rate|mean[ _-]return|median[ _-]return|effect[ _-]size|"
                             r"directional[ _-]performance|candidate return|control return)", re.I)


def forbidden_keys(obj: Any, path: str = "") -> List[str]:
    bad: List[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if set(re.split(r"[^a-z0-9]+", str(k).lower())) & _FORBIDDEN_TOKENS:
                bad.append(f"{path}.{k}")
            bad += forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += forbidden_keys(v, f"{path}[{i}]")
    return bad


def forbidden_text(text: str) -> List[str]:
    return _FORBIDDEN_TEXT.findall(text or "")


def assert_clean(report: Mapping[str, Any]) -> None:
    import json
    bad = forbidden_keys(report) + forbidden_text(json.dumps(report, default=str))
    if bad:
        raise ValueError(f"anti-peeking contract violated: {bad[:5]}")


# ---- Markdown ------------------------------------------------------------------------------
_TITLES = {"universe": "Universe", "scanner": "Scanner", "research_capture": "Research Capture",
           "maturation": "Maturation", "market_data": "Market Data", "cohort_parity": "Cohort Parity",
           "forward_evidence": "Forward Evidence", "workflows": "Workflows", "database": "Database",
           "artifact_freshness": "Artifact Freshness"}


def subsystem_label(s: Mapping[str, Any]) -> str:
    if s["status"] == "WAITING":
        return s["detail_state"]
    if s["status"] == "HEALTHY" and s["detail_state"] not in ("HEALTHY", "MARKET_CLOSED_EXPECTED_IDLE"):
        return f"HEALTHY ({s['detail_state']})"
    return s["status"]


def render_markdown(r: Mapping[str, Any]) -> str:
    human = {"NO_ACTION": "NONE", "WATCH": "WATCH", "AUTOMATIC_RECOVERY_CANDIDATE": "NONE (automatable recovery pending)",
             "HUMAN_ACTION_REQUIRED": "REQUIRED"}[r["human_action"]]
    L = ["# HSF SYSTEM HEALTH", "",
         f"**System Status: {r['system_status']}**  ", f"Health Score: {r['health_score']}/100  ",
         f"Human Action: **{human}**  ", f"Generated: {r['generated_at']}", ""]
    for n in SUBSYSTEMS:
        L.append(f"- {_TITLES[n]}: **{subsystem_label(r['subsystems'][n])}** — {r['subsystems'][n]['reason']}")
    fe = r["subsystems"]["forward_evidence"]["metrics"]
    L += ["", "## Forward Evidence", ""]
    if fe:
        L += [f"- Status: **{r['forward_evidence_status']}** (epoch {fe.get('forward_epoch_start')})",
              f"- Trading Days: {fe.get('trading_days_collected')} / {fe.get('trading_days_preferred')} "
              f"(minimum {fe.get('trading_days_min')})",
              f"- Scan Runs: {fe.get('scan_runs_collected')} / {fe.get('scan_runs_preferred')} (minimum {fe.get('scan_runs_min')})",
              f"- Lowest cohort maturation coverage: {fe.get('min_cohort_maturation_pct')}% (≥ 80%, +60m ≥ 70%)",
              f"- Gates: {fe.get('gates')}",
              f"- Formal Evaluation: **{fe.get('formal_evaluation')}**"]
    else:
        L.append("- UNKNOWN (readiness unavailable)")
    L += ["", "## Incidents", ""]
    if not r["incidents"]:
        L.append("None.")
    else:
        if not any(i["human_action"] == "HUMAN_ACTION_REQUIRED" for i in r["incidents"]):
            L.append("None requiring human action.")
        for k, i in enumerate(r["incidents"], 1):
            L += [f"{k}. **{i['incident_id']}** — {i['severity']} · age {i['age_hours']}h · human action "
                  f"{i['human_action']} · automation candidate {'YES' if i['automation_candidate'] else 'NO'}",
                  f"   {i['summary']}. Recommended: {i['recommended_action'] or '—'}"]
    m = r["market"]
    L += ["", f"Next expected scan: {m['next_expected_scan']} (trading day today: {m['is_trading_day']}, "
          f"market open: {m['market_open']})", "",
          "## Autonomy Readiness", "", f"**{r['autonomy_readiness']['state']}**", "", "Blockers to RECOVERY_READY:"]
    L += [f"- {b}" for b in r["autonomy_readiness"]["blockers_to_recovery_ready"]]
    L += ["", "Additional blockers to AUTONOMOUS:"]
    L += [f"- {b}" for b in r["autonomy_readiness"]["blockers_to_autonomous"]
          if b not in r["autonomy_readiness"]["blockers_to_recovery_ready"]]
    prov = r["subsystems"]["market_data"]["metrics"].get("providers") or []
    L += ["", "## Provider Health", "", "| Provider | Instrumented | Requests | 429s | Retries | Failures | Empty | Last success |",
          "|---|---|---|---|---|---|---|---|"]
    for p in prov:
        L.append(f"| {p['provider']} | {p['instrumented']} | {p.get('requests', '—')} | {p.get('http_429', '—')} | "
                 f"{p.get('retries', '—')} | {p.get('failures', '—')} | {p.get('empty_responses', '—')} | "
                 f"{p.get('last_successful_request', '—')} |")
    wf = r["subsystems"]["workflows"]["metrics"].get("workflows") or []
    L += ["", "## Workflow Freshness", "", "| Workflow | Cadence | Last success | Last failure | Status |", "|---|---|---|---|---|"]
    for w in wf:
        L.append(f"| {w['label']} | {w['cadence']} | {w.get('last_success', '—')} | {w.get('last_failure', '—')} | {w['status']} |")
    arts = r["subsystems"]["artifact_freshness"]["metrics"].get("artifacts") or []
    L += ["", "## Artifact Freshness", "", "| Artifact | Generated | Age (h) | Expected | Status |", "|---|---|---|---|---|"]
    for a in arts:
        L.append(f"| {a['label']} | {a.get('generated_at') or '—'} | {a.get('age_hours', '—')} | "
                 f"{a.get('expected_max_age', '—')} | {a['status']} |")
    rc = r["subsystems"]["research_capture"]["metrics"]
    L += ["", "## Research Data Quality", "",
          f"- Recent observations: {rc.get('observations')} by cohort {rc.get('by_cohort')}",
          f"- Untagged {rc.get('untagged')} · duplicates {rc.get('duplicate_ids')} · malformed {rc.get('malformed')} · "
          f"Run 57 metadata {rc.get('metadata_block_pct')}% of {rc.get('metadata_scope_n')}",
          "", "_Operational observability only: no effectiveness statistics are computed or shown._", ""]
    return "\n".join(L)
