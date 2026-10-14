"""Admin analytics for the web Admin page (admins only, read-only).

Ports the Streamlit Admin views (``ui/admin_analytics.py``): product analytics,
signal performance and HSF Score calibration, Stair-Stepper evidence, research
evidence, ML v4 readiness and Autonomous Recovery. The maths stays in
``analytics.admin_research`` and ``analytics.stair_step_research``.

Research reads use a slim JSONB projection (only the fields the summaries use)
instead of whole observation records, so a 90-day view stays small in memory
and in Neon egress. Each answer is cached 10 minutes per period and filter set.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from analytics import admin_research as ar
from api.today import TTLCache

PERIODS = (7, 30, 90, 365)
TTL_S = 600
MAX_RESEARCH_ROWS = 60_000
MAX_STAIR_OBSERVATIONS = 5_000
_cache = TTLCache(24)


class AnalyticsError(ValueError):
    """A refused request (bad period or filter)."""


class AnalyticsUnavailable(AnalyticsError):
    """The database can't be reached."""


def clear_cache() -> None:
    _cache.clear()


def _iso(value: Any) -> Optional[str]:
    parsed = ar.parse_timestamp(value) if value else None
    return parsed.astimezone(dt.timezone.utc).isoformat() if parsed else None


def _period(days: int) -> int:
    if int(days) not in PERIODS:
        raise AnalyticsError(f"Pick a period of {', '.join(map(str, PERIODS))} days.")
    return int(days)


def _days(now: dt.datetime, days: int) -> List[dt.date]:
    start = now.date() - dt.timedelta(days=max(0, days - 1))
    return [start + dt.timedelta(days=i) for i in range(days)]


# ---- product -------------------------------------------------------------------------------------
def product(days: int = 30, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    days = _period(days)
    return _cache.get(("product", days), lambda: _product(days, now), TTL_S)


def _product(days: int, now: Optional[dt.datetime]) -> Dict[str, Any]:
    from db.admin_analytics import load_admin_data

    data = load_admin_data(days=days, include_research=False)
    if not data.get("available"):
        raise AnalyticsUnavailable("Admin analytics are unavailable: the database is not reachable.")
    now = now or dt.datetime.now(dt.timezone.utc)
    users, events, runs = data.get("users") or [], data.get("events") or [], data.get("runs") or []
    span = _days(now, days)

    def day_of(value: Any) -> Optional[dt.date]:
        parsed = ar.parse_timestamp(value)
        return parsed.astimezone(dt.timezone.utc).date() if parsed else None

    def within(value: Any, n: int) -> bool:
        parsed = ar.parse_timestamp(value)
        return bool(parsed and now - dt.timedelta(days=n) <= parsed <= now + dt.timedelta(minutes=5))

    signup_days = Counter(d for u in users if (d := day_of(u.get("created_at"))) is not None)
    running = sum(n for d, n in signup_days.items() if d < span[0])
    users_daily = []
    for d in span:
        running += signup_days.get(d, 0)
        users_daily.append({"day": d.isoformat(), "new": signup_days.get(d, 0), "total": running})

    scheduled, by_user = Counter(), Counter()
    for run in runs:
        d = day_of(run.get("created_at"))
        if d is not None and d >= span[0]:
            (by_user if run.get("username") else scheduled)[d] += 1
    scans_daily = [{"day": d.isoformat(), "scheduled": scheduled.get(d, 0), "user": by_user.get(d, 0)} for d in span]

    plans = Counter(ar.normalize_plan(u.get("tier")) for u in users if u.get("is_active"))
    signups = {str(e["user_hash"]) for e in events if e.get("event_name") == "signup_completed" and e.get("user_hash")}
    paid_events = [e for e in events if e.get("event_name") == "successful_paid_conversion"]
    paid = {str(e["user_hash"]) for e in paid_events if e.get("user_hash")}
    destination = Counter(ar.normalize_plan(e.get("plan")) for e in paid_events)
    active = {str(e["user_hash"]) for e in events if e.get("user_hash")
              and e.get("event_name") in {"first_authenticated_session", "return_session"}
              and within(e.get("occurred_at"), 30)}

    return {
        "loaded_at": data.get("loaded_at"), "days": days, "unavailable": sorted(data.get("errors") or {}),
        "overview": {
            "total_users": len(users),
            "active_users_30d": len(active),
            "new_users_7d": sum(within(u.get("created_at"), 7) for u in users),
            "scans_today": sum(day_of(r.get("created_at")) == now.date() for r in runs),
            "paying_users": sum(bool(u.get("is_active")) and bool(u.get("has_subscription"))
                                and ar.normalize_plan(u.get("tier")) in {"pro", "premium"} for u in users),
        },
        "systems": _systems(data),
        "users_daily": users_daily,
        "plans": [{"plan": p.title(), "users": plans.get(p, 0)} for p in ar.CANONICAL_PLANS],
        "scans_daily": scans_daily,
        "conversions": {
            "signups": len(signups), "paid": len(paid),
            "signup_to_paid": (len(signups & paid) / len(signups)) if signups else None,
            "destinations": [{"plan": p.title(), "users": destination.get(p, 0)} for p in ("pro", "premium")]
            if paid_events else [],
        },
        "recovery": [{"started_at": _iso(r.get("started_at")), "incident": r.get("incident_id"),
                      "action": r.get("action"), "result": r.get("result")} for r in data.get("recovery") or []],
    }


def _systems(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The "HSF AI system status" table, from the persisted health snapshot."""
    health = data.get("health") or {}
    subsystems = health.get("subsystems") or {}

    def sub(name: str) -> Tuple[str, str]:
        item = subsystems.get(name) or {}
        return str(item.get("status") or "UNKNOWN"), str(item.get("reason") or "No persisted status is available.")

    ok = "users" not in (data.get("errors") or {})
    evidence = subsystems.get("forward_evidence") or {}
    rows = [("Product", "HEALTHY" if ok else "UNKNOWN",
             "User and plan data loaded successfully." if ok else "User data unavailable."),
            ("Scanner", *sub("scanner")), ("Research", *sub("research_capture")), ("Maturation", *sub("maturation")),
            ("Evidence", str(evidence.get("detail_state") or health.get("forward_evidence_status") or "UNKNOWN"),
             str(evidence.get("reason") or "No persisted evidence-readiness state is available."))]
    return [{"system": s, "status": st, "reason": why} for s, st, why in rows]


# ---- research reads ------------------------------------------------------------------------------
# Only the fields admin_research reads. A scanner without "triggered" counts as triggered. Every
# column is named: the connection returns dict rows, so two columns with one name would collide.
_SLIM_SQL = """
SELECT o.observation_id, o.symbol, o.timestamp, o.context,
  jsonb_build_object(
    'session', o.record->'session',
    'research_cohort', o.record->'research_cohort',
    'hsf_score', o.record->'hsf_score',
    'models', jsonb_build_object('hsf_score', o.record->'models'->'hsf_score'),
    'market_context', jsonb_build_object(
      'research_cohort', o.record->'market_context'->'research_cohort',
      'hsf_score', o.record->'market_context'->'hsf_score'),
    'research_metadata', CASE WHEN jsonb_typeof(o.record->'research_metadata') = 'object'
      THEN jsonb_build_object('row_features', jsonb_build_object(
        'hsf_score', o.record->'research_metadata'->'row_features'->'hsf_score')) END,
    'scanners', (SELECT jsonb_agg(jsonb_build_object('name', s->'name',
                   'triggered', COALESCE(s->'triggered', 'true'::jsonb)))
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.record->'scanners') = 'array'
                   THEN o.record->'scanners' ELSE '[]'::jsonb END) s)
  ) AS slim,
  oc.horizon,
  CASE WHEN oc.record IS NOT NULL THEN jsonb_build_object(
    'data_status', oc.record->'data_status', 'raw_return', oc.record->'raw_return',
    'directional_return', oc.record->'directional_return', 'mfe', oc.record->'mfe',
    'mae', oc.record->'mae', 'hit', oc.record->'hit', 'evaluation_time', oc.record->'evaluation_time') END AS outcome
FROM hsf_observations o
LEFT JOIN hsf_observation_outcomes oc ON oc.observation_id = o.observation_id
WHERE o.timestamp >= %s AND o.context LIKE %s
ORDER BY o.timestamp DESC LIMIT %s
"""

_STAIR_SQL = """
SELECT o.observation_id, o.record, oc.horizon, oc.record AS outcome
FROM hsf_observations o
LEFT JOIN hsf_observation_outcomes oc ON oc.observation_id = o.observation_id
WHERE o.timestamp >= %s AND o.context = %s
ORDER BY o.timestamp DESC LIMIT %s
"""


def _rows(sql: str, params: tuple) -> List[tuple]:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        raise AnalyticsUnavailable("Admin analytics are unavailable: the database is not reachable.")
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        cur.close()
        return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]
    finally:
        conn.close()


def _json(value: Any) -> Dict[str, Any]:
    from db.admin_analytics import _loads

    return _loads(value)


def scanner_observations(days: int) -> List[Dict[str, Any]]:
    """Scheduled-scan research observations (slim) with their outcomes by horizon."""
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    by_id: Dict[str, Dict[str, Any]] = {}
    for oid, symbol, ts, context, record, horizon, outcome in _rows(
            _SLIM_SQL, (since, ar.SCANNER_CONTEXT_PREFIX + "%", MAX_RESEARCH_ROWS)):
        if not oid:
            continue
        obs = by_id.get(str(oid))
        if obs is None:
            obs = by_id[str(oid)] = {**_json(record), "observation_id": str(oid), "symbol": symbol,
                                     "timestamp": _iso(ts), "context": context}
        out = _json(outcome)
        if horizon is not None and out:
            obs.setdefault("outcomes", {})[str(horizon)] = out
    return list(by_id.values())


def stair_observations(days: int) -> List[Dict[str, Any]]:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    by_id: Dict[str, Dict[str, Any]] = {}
    for oid, record, horizon, outcome in _rows(
            _STAIR_SQL, (since, ar.STAIR_STEPPER_CONTEXT, MAX_STAIR_OBSERVATIONS * 4)):
        if not oid:
            continue
        obs = by_id.setdefault(str(oid), _json(record))
        obs.setdefault("observation_id", str(oid))
        obs.setdefault("context", ar.STAIR_STEPPER_CONTEXT)
        out = _json(outcome)
        if horizon is not None and out:
            obs.setdefault("outcomes", {})[str(horizon)] = out
    return list(by_id.values())


# ---- signal performance --------------------------------------------------------------------------
def _group(g: Mapping[str, Any], a: str, b: Optional[str] = None) -> Dict[str, Any]:
    return {"key": g.get(a), "column": g.get(b) if b else None, "n": int(g.get("n") or 0),
            "matured": int(g.get("matured") or 0), "sufficient": bool(g.get("sufficient")),
            **{k: g.get(k) for k in ("positive_rate", "average_return", "median_return", "average_mfe", "average_mae")}}


def _cells(groups: Sequence[Mapping[str, Any]], a: str, b: str) -> List[Dict[str, Any]]:
    # Pending observations are listed once with no horizon; they hold no outcomes.
    return [_group(g, a, b) for g in groups
            if g.get(a) is not None and g.get(b) is not None or int(g.get("n") or 0)]


def signals(days: int = 30, *, signal: str = "", cohort: str = "", horizon: str = "", session: str = "",
            score_min: Optional[float] = None, score_max: Optional[float] = None, min_n: int = ar.DEFAULT_MIN_EVIDENCE_N,
            score_horizon: str = "") -> Dict[str, Any]:
    days = _period(days)
    if not 1 <= int(min_n) <= 1000:
        raise AnalyticsError("Minimum sample size must be between 1 and 1000.")
    key = ("signals", days, signal, cohort, horizon, session, score_min, score_max, int(min_n), score_horizon)
    return _cache.get(key, lambda: _signals(days, signal, cohort, horizon, session, score_min, score_max,
                                            int(min_n), score_horizon), TTL_S)


def _signals(days: int, signal: str, cohort: str, horizon: str, session: str, score_min: Optional[float],
             score_max: Optional[float], min_n: int, score_horizon: str) -> Dict[str, Any]:
    rows = ar.flatten_research(scanner_observations(days))
    scores = [float(r["hsf_score"]) for r in rows if r.get("hsf_score") is not None]
    horizons = sorted({str(r["horizon"]) for r in rows if r.get("horizon")})

    def keep(r: Mapping[str, Any]) -> bool:
        score = ar.number(r.get("hsf_score"))
        if (score_min is not None or score_max is not None) and score is None:
            return False
        return ((not signal or r.get("signal") == signal) and (not cohort or r.get("cohort") == cohort)
                and (not horizon or r.get("horizon") == horizon) and (not session or r.get("session") == session)
                and (score_min is None or score >= score_min) and (score_max is None or score <= score_max))

    filtered = [r for r in rows if keep(r)]
    summary = ar.performance_summary(filtered, min_n=min_n)
    scored = [r for r in rows if r.get("hsf_score") is not None]
    score_horizon = score_horizon if score_horizon in horizons else (horizons[0] if horizons else "")
    return {
        "loaded_at": _iso(dt.datetime.now(dt.timezone.utc)), "days": days, "min_n": min_n,
        "options": {"signals": sorted({str(r["signal"]) for r in rows}), "cohorts": sorted({str(r["cohort"]) for r in rows}),
                    "horizons": horizons, "sessions": sorted({str(r["session"]) for r in rows}),
                    "score_low": int(min(scores)) if scores else None, "score_high": int(max(scores)) if scores else None},
        "summary": {"observations": len({r.get("observation_id") for r in filtered}), "matured": summary["matured"],
                    "n": summary["n"], "sufficient": summary["sufficient"],
                    **{k: summary[k] for k in ("positive_rate", "average_return", "median_return",
                                               "average_mfe", "average_mae")}},
        "by_signal_horizon": _cells(ar.grouped_performance(rows, ("signal", "horizon"), min_n=min_n), "signal", "horizon"),
        "score_available": bool(scored),
        "score_horizon": score_horizon or None,
        "score_buckets": [_group(g, "score_bucket") for g in ar.grouped_performance(
            [r for r in scored if r.get("horizon") == score_horizon], ("score_bucket",), min_n=min_n)],
        "score_matrix": _cells(ar.grouped_performance(scored, ("score_bucket", "horizon"), min_n=min_n),
                               "score_bucket", "horizon"),
    }


# ---- research evidence + Stair-Stepper -----------------------------------------------------------
def evidence(days: int = 30) -> Dict[str, Any]:
    days = _period(days)
    return _cache.get(("evidence", days), lambda: _evidence(days), TTL_S)


def _evidence(days: int) -> Dict[str, Any]:
    from api.operations import load_snapshot

    funnel = ar.evidence_funnel(scanner_observations(days))
    health = load_snapshot().get("report") or {}
    subsystems = health.get("subsystems") or {}
    forward = subsystems.get("forward_evidence") or {}
    metrics = forward.get("metrics") or {}
    maturation = subsystems.get("maturation") or {}
    stages = [("Captured", "captured"), ("Valid metadata", "valid_metadata"), ("Matured", "matured"),
              ("Usable outcome", "usable_outcome"), ("Included in research", "included_in_research")]
    base = funnel["captured"] or 0
    return {
        "loaded_at": _iso(dt.datetime.now(dt.timezone.utc)), "days": days,
        "funnel": [{"stage": label, "count": funnel[k], "pct": round(funnel[k] / base * 100, 1) if base else None}
                   for label, k in stages],
        "pending": funnel["pending"], "excluded": funnel["excluded"],
        "metadata_completeness": funnel["metadata_completeness"],
        "oldest_pending": funnel["oldest_pending_observation"],
        "latest_observation": funnel["latest_observation"],
        "latest_matured": funnel["latest_matured_observation"],
        "last_maturation_run": _iso((maturation.get("metrics") or {}).get("last_success") or maturation.get("last_updated")),
        "cohorts": [{"cohort": k, "observations": v} for k, v in sorted(funnel["cohorts"].items())],
        "readiness": {
            "state": str(forward.get("detail_state") or health.get("forward_evidence_status") or "UNKNOWN"),
            "reason": forward.get("reason"),
            "trading_days": metrics.get("trading_days_collected"),
            "trading_days_preferred": metrics.get("trading_days_preferred"),
            "scan_runs": metrics.get("scan_runs_collected"),
            "scan_runs_preferred": metrics.get("scan_runs_preferred"),
            "formal_evaluation": metrics.get("formal_evaluation"),
        },
    }


def stair_stepper(days: int = 30) -> Dict[str, Any]:
    days = _period(days)
    return _cache.get(("stair", days), lambda: _stair(days), TTL_S)


def _stair(days: int) -> Dict[str, Any]:
    from analytics.stair_step_research import OUTCOME_HORIZONS, build_research_report

    observations = [o for o in stair_observations(days) if ar.is_stair_stepper(o)]
    report = build_research_report(observations)
    stamps = [t for o in observations if (t := ar.parse_timestamp(o.get("timestamp"))) is not None]
    windows = []
    for item in report.get("window_comparison") or []:
        for h in OUTCOME_HORIZONS:
            out = (item.get("horizons") or {}).get(h) or {}
            windows.append({"window": item.get("window"), "horizon": h, "observations": item.get("observations"),
                            "matured": out.get("n", 0), "positive_rate": out.get("directional_win_rate"),
                            "average_return": out.get("mean_directional_return"),
                            "median_return": out.get("median_directional_return"),
                            "mfe": out.get("mean_mfe"), "mae": out.get("mean_mae")})
    verdict = report.get("best_window_verdict") or {}
    found = verdict.get("status") == "EVIDENCE_AVAILABLE" and verdict.get("best_window") is not None
    return {
        "loaded_at": _iso(dt.datetime.now(dt.timezone.utc)), "days": days,
        "observations": len(observations), "latest": _iso(max(stamps)) if stamps else None,
        "horizons": list(OUTCOME_HORIZONS), "windows": windows,
        "best_window": verdict.get("best_window") if found else None,
        "verdict_reason": verdict.get("reason"),
    }


# ---- ML v4 readiness -----------------------------------------------------------------------------
def ml_readiness() -> Dict[str, Any]:
    """The compact readiness panel from the cached /v1/ml/readiness view."""
    from api import ml_readiness as mlr

    view = mlr.readiness()
    obs, cov, val = view["observations"], view["coverage"], view["validation"]
    return {
        "status": view["status"], "recommendation": view["recommendation"], "generated_at": _iso(view["generated_at"]),
        "matured_signal_days": obs["matured_signal_days"], "matured_rows": obs["matured"],
        "matured_trading_days": cov["matured_trading_days"], "matured_symbols": cov["matured_symbols"],
        "outcome_coverage_recent": cov["outcome_coverage_recent"],
        "walk_forward_folds": val["usable_walk_forward_folds"],
        "gates": [{"gate": g["gate"], "kind": str(g["kind"]).title(), "value": g["value"],
                   "needs": f"{g['operator']} {g['threshold']}", "status": g["status"],
                   "progress": round(float(g["progress"] or 0), 3), "why": g.get("why")} for g in view["gates"]],
        "blocking_reasons": list(view["blocking_reasons"]),
        "estimate_trading_days": (view.get("projection") or {}).get("estimate_trading_days_to_accumulating_targets"),
    }
