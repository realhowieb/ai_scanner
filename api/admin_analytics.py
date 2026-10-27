"""Admin analytics for the web Admin page (admins only, read-only).

Ports the Streamlit Admin views (``ui/admin_analytics.py``): product analytics,
signal performance and HSF Score calibration, Stair-Stepper evidence, research
evidence, ML v4 readiness and Autonomous Recovery. The maths stays in
``analytics.admin_research`` and ``analytics.stair_step_research``.

Research reads never load whole observation records: SQL returns one row per
observation with only the scalars the summaries use (outcomes folded into one
small JSON object), streamed in batches into compact tuples. Signal performance
and research evidence share one cached set per period (one database read per
10 minutes, newest 20,000 observations); each answer is cached 10 minutes.
"""
from __future__ import annotations

import datetime as dt
from array import array
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from analytics import admin_research as ar
from api.today import TTLCache

PERIODS = (7, 30, 90, 365)
TTL_S = 600
MAX_SCANNER_OBSERVATIONS = 20_000  # newest first; the response says when a period holds more
MAX_STAIR_OBSERVATIONS = 3_000
_cache = TTLCache(24)  # computed answers (small)
_datasets = TTLCache(1)  # the compact research set behind them: one period at a time


class AnalyticsError(ValueError):
    """A refused request (bad period or filter)."""


class AnalyticsUnavailable(AnalyticsError):
    """The database can't be reached."""


def clear_cache() -> None:
    _cache.clear()
    _datasets.clear()


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
        "funnel": _funnel(days),
    }


# ---- sign-up funnel ------------------------------------------------------------------------------
FUNNEL_STEPS = (("signed_up", "Signed up"), ("verified", "Verified email"),
                ("engaged", "Saved a stock or alert"), ("paid", "Pays for Pro"))
SOURCE_LABELS = {"reddit": "Reddit", "direct": "Direct", "other_unknown": "Other"}


def _funnel(days: int) -> Optional[Dict[str, Any]]:
    """The sign-up funnel for the period, or None when it can't be loaded (the rest of
    the dashboard still shows)."""
    try:
        from db.admin_analytics import load_signup_funnel

        return signup_funnel(load_signup_funnel(days=days))
    except Exception:
        return None


def signup_funnel(data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Counts per step for accounts created in the period, overall and by sign-up source.
    `paid` means a live subscription on Pro or Premium (a free invite trial is not paid)."""
    if not data.get("available"):
        return None
    from ui.acquisition import user_hash

    sources = data.get("sources") or {}
    totals: Counter = Counter()
    by_source: Dict[str, Counter] = defaultdict(Counter)
    trials = 0
    for a in data.get("accounts") or []:
        reached = {
            "signed_up": True,
            "verified": bool(a.get("verified")),
            "engaged": bool(a.get("engaged")),
            "paid": bool(a.get("has_subscription")) and ar.normalize_plan(a.get("tier")) in {"pro", "premium"},
        }
        src = SOURCE_LABELS.get(sources.get(user_hash(a.get("user")) or "", ""), "Not recorded")
        for key, hit in reached.items():
            if hit:
                totals[key] += 1
                by_source[src][key] += 1
        trials += bool(a.get("trial"))
    first = totals["signed_up"]
    steps = [{"key": k, "label": label, "users": totals[k],
              "of_signups": (totals[k] / first) if first else None} for k, label in FUNNEL_STEPS]
    rows = [{"source": s, **{k: c[k] for k, _ in FUNNEL_STEPS}}
            for s, c in sorted(by_source.items(), key=lambda kv: (-kv[1]["signed_up"], kv[0]))]
    return {"steps": steps, "by_source": rows, "invite_trials": trials,
            "unavailable": sorted((data.get("errors") or {}).keys())}


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
# One row per observation with only the scalars the summaries use, and its outcomes folded into one
# small JSON object, so a Research tab load stays a few MB on a 512 MB instance. Every column is
# named: the connection returns dict rows, so two columns with one name would collide.
_SCANNER_SQL = """
SELECT o.observation_id, o.timestamp,
  o.record->>'session' AS session,
  o.record->>'research_cohort' AS cohort,
  o.record->'market_context'->>'research_cohort' AS context_cohort,
  o.record->>'hsf_score' AS score_a,
  o.record->'models'->>'hsf_score' AS score_b,
  o.record->'market_context'->>'hsf_score' AS score_c,
  o.record->'research_metadata'->'row_features'->>'hsf_score' AS score_d,
  jsonb_typeof(o.record->'research_metadata') = 'object' AS has_metadata,
  ARRAY(SELECT s->>'name' FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.record->'scanners') = 'array'
          THEN o.record->'scanners' ELSE '[]'::jsonb END) s
        WHERE jsonb_typeof(s) = 'object'
          AND COALESCE(s->'triggered', 'true'::jsonb) NOT IN ('false'::jsonb, 'null'::jsonb, '0'::jsonb, '""'::jsonb)
       ) AS signals,
  (SELECT jsonb_object_agg(oc.horizon, jsonb_build_object(
      's', oc.record->'data_status', 'r', oc.record->'raw_return', 'd', oc.record->'directional_return',
      'f', oc.record->'mfe', 'a', oc.record->'mae', 'h', oc.record->'hit', 'e', oc.record->'evaluation_time'))
   FROM hsf_observation_outcomes oc WHERE oc.observation_id = o.observation_id) AS outcomes
FROM hsf_observations o
WHERE o.timestamp >= %s AND o.context LIKE %s
ORDER BY o.timestamp DESC LIMIT %s
"""

_STAIR_SQL = """
SELECT o.observation_id, o.symbol, o.timestamp,
  o.record->'scan_timestamp' AS scan_timestamp,
  o.record->'stair_step' AS stair_step,
  o.record->'market'->'price' AS price,
  (SELECT jsonb_object_agg(oc.horizon, jsonb_build_object(
      'data_status', oc.record->'data_status', 'directional_return', oc.record->'directional_return',
      'mfe', oc.record->'mfe', 'mae', oc.record->'mae'))
   FROM hsf_observation_outcomes oc WHERE oc.observation_id = o.observation_id) AS outcomes
FROM hsf_observations o
WHERE o.timestamp >= %s AND o.context = %s
ORDER BY o.timestamp DESC LIMIT %s
"""


def _stream(sql: str, params: tuple, batch: int = 2000) -> Iterator[tuple]:
    """Rows in batches, so the whole result never sits in Python as one list."""
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        raise AnalyticsUnavailable("Admin analytics are unavailable: the database is not reachable.")
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for r in rows:
                yield tuple(r.values()) if isinstance(r, dict) else tuple(r)
        cur.close()
    finally:
        conn.close()


def _json(value: Any) -> Dict[str, Any]:
    from db.admin_analytics import _loads

    return _loads(value)


# The flattened research rows (one per observation x signal x outcome horizon, and one with no
# horizon while pending), as analytics.admin_research.flatten_research builds them, stored by
# column in typed arrays: about 40 bytes a row instead of ~700 as dicts or tuples.
_CATEGORIES = ("signal", "cohort", "session", "bucket", "horizon")
_INCLUDED_COHORTS = {"CANDIDATE", "NEAR_MISS", "CONTROL"}
_NAN = float("nan")


class ResearchTable:
    def __init__(self) -> None:
        self.oi = array("I")
        self.codes = {c: array("H") for c in _CATEGORIES}
        self.labels: Dict[str, List[Optional[str]]] = {c: [] for c in _CATEGORIES}
        self._index: Dict[str, Dict[Optional[str], int]] = {c: {} for c in _CATEGORIES}
        self.matured = array("b")
        self.positive = array("b")  # -1 unknown, 0 no, 1 yes
        self.num = {c: array("d") for c in ("score", "ret", "mfe", "mae")}

    def __len__(self) -> int:
        return len(self.oi)

    def code(self, cat: str, label: Optional[str]) -> int:
        index = self._index[cat]
        if label not in index:
            index[label] = len(self.labels[cat])
            self.labels[cat].append(label)
        return index[label]

    def add(self, oi: int, cats: Mapping[str, Optional[str]], matured: bool, positive: Optional[bool],
            nums: Mapping[str, Optional[float]]) -> None:
        self.oi.append(oi)
        for c in _CATEGORIES:
            self.codes[c].append(self.code(c, cats[c]))
        self.matured.append(1 if matured else 0)
        self.positive.append(-1 if positive is None else int(bool(positive)))
        for c, col in self.num.items():
            v = nums[c]
            col.append(_NAN if v is None else v)

    def label(self, cat: str, i: int) -> Optional[str]:
        return self.labels[cat][self.codes[cat][i]]

    def value(self, col: str, i: int) -> Optional[float]:
        v = self.num[col][i]
        return None if v != v else v

    def values(self, cat: str) -> List[str]:
        return sorted(str(v) for v in self.labels[cat] if v is not None)


def compact_research(rows: Iterable[tuple], *, limit: int) -> Dict[str, Any]:
    """The research table and the evidence funnel from slim observation rows, in one pass.

    Mirrors ``admin_research.flatten_research`` and ``evidence_funnel`` (tests hold them equal)."""
    table = ResearchTable()
    seen = valid = matured_n = usable_n = included_n = 0
    pending_times: List[dt.datetime] = []
    latest_obs: Optional[dt.datetime] = None
    latest_matured: Optional[dt.datetime] = None
    cohorts: Counter = Counter()
    for oi, (_oid, ts, session, cohort, ctx_cohort, sa, sb, sc, sd, has_meta, signals, outcomes) in enumerate(rows):
        seen += 1
        cohort = str(cohort or ctx_cohort or "UNTAGGED").upper()
        session = str(session or "UNKNOWN").upper()
        score = next((v for v in (ar.number(x) for x in (sa, sb, sc, sd)) if v is not None), None)
        names: List[str] = []
        for name in signals or []:
            name = str(name or "").strip()
            if name and name not in names:
                names.append(name)
        items = [(str(h), o if isinstance(o, Mapping) else {}) for h, o in _json(outcomes).items()] or [(None, {})]
        parsed = []
        for horizon, o in items:
            raw, direction = ar.number(o.get("r")), ar.number(o.get("d"))
            hit = o.get("h") if isinstance(o.get("h"), bool) else None
            positive = hit if hit is not None else (direction > 0 if direction is not None else (
                raw > 0 if raw is not None else None))
            parsed.append((horizon, str(o.get("s") or "PENDING").upper() == "MATURED",
                           direction if direction is not None else raw, ar.number(o.get("f")), ar.number(o.get("a")),
                           positive, o.get("e")))
        for signal in names or ["unclassified"]:
            for horizon, mat, ret, mfe, mae, positive, _e in parsed:
                table.add(oi, {"signal": signal, "cohort": cohort, "session": session,
                               "bucket": ar.score_bucket(score), "horizon": horizon},
                          mat, positive, {"score": score, "ret": ret, "mfe": mfe, "mae": mae})
        valid += bool(has_meta)
        cohorts[cohort] += 1
        stamp = ar.parse_timestamp(ts)
        if stamp and (latest_obs is None or stamp > latest_obs):
            latest_obs = stamp
        mature = [p for p in parsed if p[0] is not None and p[1]]
        if mature:
            matured_n += 1
        elif stamp:
            pending_times.append(stamp)
        if any(p[2] is not None for p in mature):
            usable_n += 1
        if cohort in _INCLUDED_COHORTS and mature:
            included_n += 1
        for p in mature:
            evaluated = ar.parse_timestamp(p[6])
            if evaluated and (latest_matured is None or evaluated > latest_matured):
                latest_matured = evaluated
    funnel = {
        "captured": seen, "valid_metadata": valid, "matured": matured_n, "usable_outcome": usable_n,
        "included_in_research": included_n, "excluded": max(0, seen - included_n), "pending": max(0, seen - matured_n),
        "metadata_completeness": (valid / seen) if seen else None, "cohorts": dict(cohorts),
        "oldest_pending_observation": min(pending_times).isoformat() if pending_times else None,
        "latest_observation": latest_obs.isoformat() if latest_obs else None,
        "latest_matured_observation": latest_matured.isoformat() if latest_matured else None,
    }
    return {"table": table, "funnel": funnel, "observations": seen, "truncated": seen >= limit}


def scanner_research(days: int) -> Dict[str, Any]:
    """The compact scheduled-scan research set for a period. One set is kept (10 minutes) and
    concurrent callers share one load, so the Research tab reads the database once."""
    def load() -> Dict[str, Any]:
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        return compact_research(_stream(_SCANNER_SQL, (since, ar.SCANNER_CONTEXT_PREFIX + "%",
                                                       MAX_SCANNER_OBSERVATIONS)), limit=MAX_SCANNER_OBSERVATIONS)
    return _datasets.get(("scanner", days), load, TTL_S)


def stair_observations(days: int) -> List[Dict[str, Any]]:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    return [{"observation_id": str(oid), "symbol": symbol, "timestamp": _iso(ts), "context": ar.STAIR_STEPPER_CONTEXT,
             "scan_timestamp": scan_ts, "stair_step": stair if isinstance(stair, dict) else _json(stair),
             "market": {"price": price}, "outcomes": _json(outcomes)}
            for oid, symbol, ts, scan_ts, stair, price, outcomes in _stream(
                _STAIR_SQL, (since, ar.STAIR_STEPPER_CONTEXT, MAX_STAIR_OBSERVATIONS)) if oid]


# ---- signal performance --------------------------------------------------------------------------
def _perf(t: ResearchTable, idx: Iterable[int], min_n: int) -> Dict[str, Any]:
    """``admin_research.performance_summary`` over table rows."""
    matured = [i for i in idx if t.matured[i]]
    rets = [v for i in matured if (v := t.value("ret", i)) is not None]
    positives = [t.positive[i] for i in matured if t.positive[i] >= 0]
    mfes = [v for i in matured if (v := t.value("mfe", i)) is not None]
    maes = [v for i in matured if (v := t.value("mae", i)) is not None]
    return {"n": len(rets), "matured": len(matured), "sufficient": len(rets) >= max(0, int(min_n)),
            "positive_rate": (sum(positives) / len(positives)) if positives else None,
            "average_return": mean(rets) if rets else None, "median_return": median(rets) if rets else None,
            "average_mfe": mean(mfes) if mfes else None, "average_mae": mean(maes) if maes else None}


def _grouped(t: ResearchTable, idx: Iterable[int], a: str, b: Optional[str], min_n: int) -> List[Dict[str, Any]]:
    """``admin_research.grouped_performance`` over table rows, as API groups."""
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i in idx:
        groups[(t.label(a, i), t.label(b, i) if b else None)].append(i)
    return [{"key": k[0], "column": k[1], **_perf(t, g, min_n)}
            for k, g in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))]


def _cells(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Pending observations are listed once with no horizon; they hold no outcomes.
    return [g for g in groups if g["key"] is not None and g["column"] is not None or g["n"]]


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
    data = scanner_research(days)
    t: ResearchTable = data["table"]
    every = range(len(t))
    scores = [v for i in every if (v := t.value("score", i)) is not None]
    horizons = t.values("horizon")
    want = {"signal": signal, "cohort": cohort, "horizon": horizon, "session": session}

    def keep(i: int) -> bool:
        if any(v and t.label(c, i) != v for c, v in want.items()):
            return False
        s = t.value("score", i)
        if score_min is None and score_max is None:
            return True
        return s is not None and (score_min is None or s >= score_min) and (score_max is None or s <= score_max)

    filtered = [i for i in every if keep(i)]
    scored = [i for i in every if t.value("score", i) is not None]
    score_horizon = score_horizon if score_horizon in horizons else (horizons[0] if horizons else "")
    return {
        "loaded_at": _iso(dt.datetime.now(dt.timezone.utc)), "days": days, "min_n": min_n,
        "loaded_observations": data["observations"], "truncated": data["truncated"],
        "options": {"signals": t.values("signal"), "cohorts": t.values("cohort"), "horizons": horizons,
                    "sessions": t.values("session"),
                    "score_low": int(min(scores)) if scores else None, "score_high": int(max(scores)) if scores else None},
        "summary": {"observations": len({t.oi[i] for i in filtered}), **_perf(t, filtered, min_n)},
        "by_signal_horizon": _cells(_grouped(t, every, "signal", "horizon", min_n)),
        "score_available": bool(scored),
        "score_horizon": score_horizon or None,
        "score_buckets": _grouped(t, [i for i in scored if t.label("horizon", i) == score_horizon], "bucket", None, min_n),
        "score_matrix": _cells(_grouped(t, scored, "bucket", "horizon", min_n)),
    }


# ---- research evidence + Stair-Stepper -----------------------------------------------------------
def evidence(days: int = 30) -> Dict[str, Any]:
    days = _period(days)
    return _cache.get(("evidence", days), lambda: _evidence(days), TTL_S)


def _evidence(days: int) -> Dict[str, Any]:
    from api.operations import load_snapshot

    data = scanner_research(days)
    funnel = data["funnel"]
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
        "loaded_observations": data["observations"], "truncated": data["truncated"],
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
