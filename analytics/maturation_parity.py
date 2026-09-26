"""Run 58 — cohort maturation parity & missingness audit (pure, read-only).

Answers "do CANDIDATE / NEAR_MISS / CONTROL observations have equal OPPORTUNITY
to mature?". This is research-data completeness only: the module never reads
outcome values. It uses outcome *presence*, observation timing, capture-time
market fields (price/volume, for data availability), and scheduler decisions
(`scripts.mature_observations` trace statuses). Its output passes an anti-peeking
key guard.

Missingness reasons come in two strengths:
  * TRACED: from a dry-run of the real maturation worker (with real market data
    in CI) recording, per (observation, horizon), what the scheduler does now:
    matured-if-run (PENDING_BACKLOG), deferred by cap, rate limited, no bars,
    too few bars, retired, ineligible.
  * INFERRED: without a trace, only structural reasons can be assigned
    (INVALID_ANCHOR, INELIGIBLE_SYMBOL, RETIRED_WINDOW_CLOSED,
    INSUFFICIENT_FUTURE_BARS when a sibling horizon matured). Everything else is
    UNKNOWN; no precision is invented.

Root-cause rules (pre-declared): each mechanism's contribution to the
CANDIDATE−CONTROL coverage gap at a horizon is (control missing rate − candidate
missing rate) for the reasons belonging to it, in percentage points. A mechanism
is material at >= 5 pp. See `root_cause`.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analytics import forward_readiness as fr
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS

SCHEMA = "hsf-maturation-parity-1.0"
HORIZONS = fr.HORIZONS
COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)
PRIMARY_HORIZON = "+60m"
RETIRE_AFTER = fr.RETIRE_AFTER

REASONS = (
    "NOT_YET_ELIGIBLE", "PENDING_BACKLOG", "SCHEDULER_DEFERRED", "RATE_LIMITED",
    "PRICE_DATA_UNAVAILABLE", "INSUFFICIENT_FUTURE_BARS", "RETIRED_WINDOW_CLOSED",
    "INELIGIBLE_SYMBOL", "INVALID_ANCHOR", "PIPELINE_ERROR", "UNKNOWN",
)
# Worker trace status -> missingness reason.
_TRACE_MAP = {
    "MATURED": "PENDING_BACKLOG",           # data exists; a real run would write it now
    "ALREADY_WRITTEN": "PENDING_BACKLOG",
    "DEFERRED": "SCHEDULER_DEFERRED",
    "RATE_LIMITED": "RATE_LIMITED",
    "PRICE_DATA_UNAVAILABLE": "PRICE_DATA_UNAVAILABLE",
    "INSUFFICIENT_FUTURE_BARS": "INSUFFICIENT_FUTURE_BARS",
    "RETIRED": "RETIRED_WINDOW_CLOSED",
    "INELIGIBLE": "INELIGIBLE_SYMBOL",
    "INVALID_TIMESTAMP": "INVALID_ANCHOR",
    "PROVIDER_ERROR": "PIPELINE_ERROR",
    "DATABASE_ERROR": "PIPELINE_ERROR",
    "NOT_READY": "NOT_YET_ELIGIBLE",
}
MECHANISMS = {
    "HISTORICAL_SCHEDULER_STARVATION": ("PENDING_BACKLOG", "SCHEDULER_DEFERRED", "RATE_LIMITED"),
    "MARKET_DATA_AVAILABILITY_EFFECT": ("PRICE_DATA_UNAVAILABLE", "INSUFFICIENT_FUTURE_BARS"),
    "COHORT_COMPOSITION_EFFECT": ("INELIGIBLE_SYMBOL",),
    "RETIREMENT_POLICY_EFFECT": ("RETIRED_WINDOW_CLOSED",),
    "PIPELINE_BUG": ("PIPELINE_ERROR", "INVALID_ANCHOR"),
}
MATERIAL_PP = 5.0
MIN_ELIGIBLE = 100
TEMPORAL_MATERIAL_PP = 10.0
LIQUIDITY_DISPARITY = 0.10   # control median capture $-volume < 10% of candidate's
TOD_BUCKETS = (
    (0, 570, "PRE (<09:30)"), (570, 630, "09:30-10:30"), (630, 720, "10:30-12:00"),
    (720, 840, "12:00-14:00"), (840, 900, "14:00-15:00"), (900, 960, "15:00-16:00"),
    (960, 1440, "POST (>=16:00)"),
)
MARKET_CLOSE_MIN = 16 * 60

# Anti-peeking: no key may name a performance statistic.
# Matched on whole key tokens (so RETIRED_WINDOW_CLOSED is fine, win_rate is not).
_FORBIDDEN_TOKENS = {"return", "returns", "win", "wins", "winrate", "pnl", "profit", "spearman",
                     "pearson", "corr", "correlation", "payoff", "mfe", "mae", "sharpe", "edge",
                     "lift", "alpha", "expectancy", "drawdown", "performance"}


parity_class = fr.parity_class


def _pct(a: int, b: int) -> Optional[float]:
    return round(100.0 * a / b, 2) if b else None


def _minute_et(d: _dt.datetime) -> int:
    e = fr._et(d)
    return e.hour * 60 + e.minute


def _fmt_min(m: Optional[float]) -> Optional[str]:
    if m is None:
        return None
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def _quantile(vals: Sequence[float], q: float) -> Optional[float]:
    v = sorted(vals)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


# ---- Population ---------------------------------------------------------------
def population(observations: Iterable[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    """Explicit-cohort observations (dedup by id). scope: historical (anchored
    before the Run 56 epoch), forward (at/after it), or all."""
    start = fr.epoch_start()
    seen = set()
    out = []
    for o in observations or []:
        if not o or fr.cohort(o) not in COHORTS:
            continue
        oid = str(o.get("observation_id") or "")
        if oid in seen:
            continue
        a = fr.anchor(o)
        if scope == "historical" and (a is None or a >= start):
            continue
        if scope == "forward" and (a is None or a < start):
            continue
        seen.add(oid)
        out.append(o)
    return out


def matured_index(outcomes_by_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, set]:
    """observation_id -> set of MATURED horizons (presence only)."""
    idx: Dict[str, set] = defaultdict(set)
    for oid, ocs in (outcomes_by_id or {}).items():
        for oc in ocs or []:
            if str(oc.get("data_status")) == "MATURED" and oc.get("horizon") in HORIZONS:
                idx[str(oid)].add(oc["horizon"])
    return idx


def trace_index(trace: Optional[Sequence[Mapping[str, Any]]]) -> Dict[Tuple[str, str], str]:
    idx: Dict[Tuple[str, str], str] = {}
    for t in trace or []:
        hs = HORIZONS if t.get("horizon") == "*" else (t.get("horizon"),)
        for h in hs:
            idx[(str(t.get("observation_id")), h)] = str(t.get("status"))
    return idx


def is_settled(o: Mapping[str, Any], h: str, now: _dt.datetime) -> bool:
    a = fr.anchor(o)
    return a is not None and now >= (a + _dt.timedelta(minutes=fr.HORIZON_MIN[h])
                                     + fr.MATURATION_SLACK + fr.MATURATION_GRACE)


def missing_reason(o: Mapping[str, Any], h: str, matured: Mapping[str, set], now: _dt.datetime,
                   traced: Optional[Mapping[Tuple[str, str], str]] = None) -> Tuple[str, str]:
    """(reason, basis) for a settled, unmatured (observation, horizon)."""
    oid = str(o.get("observation_id") or "")
    a = fr.anchor(o)
    if a is None:
        return "INVALID_ANCHOR", "structural"
    if fr._exclusion(str(o.get("symbol") or "")):
        return "INELIGIBLE_SYMBOL", "structural"
    if traced is not None:
        st = traced.get((oid, h))
        if st in _TRACE_MAP:
            return _TRACE_MAP[st], "traced"
        if st == "ALREADY":  # worker saw an outcome the loader did not: inconsistency
            return "PIPELINE_ERROR", "traced"
    if now >= a + RETIRE_AFTER:
        return "RETIRED_WINDOW_CLOSED", "structural"
    if matured.get(oid):
        return "INSUFFICIENT_FUTURE_BARS", "inferred"
    return "UNKNOWN", "inferred"


# ---- Coverage / missingness / parity --------------------------------------------
def coverage(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime,
             traced: Optional[Mapping[Tuple[str, str], str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for h in HORIZONS:
        out[h] = {}
        for c in COHORTS:
            rows = [o for o in pop if fr.cohort(o) == c]
            eligible = mat = 0
            reasons: Counter = Counter()
            basis: Counter = Counter()
            for o in rows:
                oid = str(o.get("observation_id") or "")
                if h in matured.get(oid, ()):
                    eligible += 1
                    mat += 1
                    continue
                if not is_settled(o, h, now):
                    reasons["NOT_YET_ELIGIBLE"] += 1
                    continue
                eligible += 1
                r, b = missing_reason(o, h, matured, now, traced)
                reasons[r] += 1
                basis[b] += 1
            missing = eligible - mat
            backlog = reasons.get("PENDING_BACKLOG", 0)
            out[h][c] = {
                "observations": len(rows),
                "eligible": eligible,
                "matured": mat,
                "unmatured": missing,
                "maturation_pct": _pct(mat, eligible),
                "projected_maturation_pct_after_backlog": (_pct(mat + backlog, eligible)
                                                           if traced is not None else None),
                "missing_reasons": {r: {"count": reasons.get(r, 0),
                                        "pct_of_cohort_eligible": _pct(reasons.get(r, 0), eligible),
                                        "pct_of_missing": _pct(reasons.get(r, 0), missing)}
                                    for r in REASONS if r != "NOT_YET_ELIGIBLE"},
                "not_yet_eligible": reasons.get("NOT_YET_ELIGIBLE", 0),
                "reason_basis": dict(basis),
            }
    return out


def parity(cov: Mapping[str, Any], key: str = "maturation_pct") -> Dict[str, Any]:
    out = {}
    for h in HORIZONS:
        vals = {c: cov[h][c][key] for c in COHORTS}
        known = [v for v in vals.values() if v is not None]
        gap = round(max(known) - min(known), 2) if len(known) == len(COHORTS) else None
        out[h] = {"candidate_coverage": vals[CANDIDATE], "near_miss_coverage": vals[NEAR_MISS],
                  "control_coverage": vals[CONTROL], "parity_gap": gap,
                  "classification": parity_class(gap)}
    return out


# ---- Timing ------------------------------------------------------------------------
def timing(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_cohort": {}, "plus60_by_scan_slot": {}}
    for c in COHORTS:
        rows = [o for o in pop if fr.cohort(o) == c and fr.anchor(o) is not None]
        mins = [_minute_et(fr.anchor(o)) for o in rows]
        buckets = Counter()
        for m in mins:
            for lo, hi, lab in TOD_BUCKETS:
                if lo <= m < hi:
                    buckets[lab] += 1
        n = len(rows)
        cross60 = sum(1 for m in mins if m < MARKET_CLOSE_MIN <= m + 60)
        cross30 = sum(1 for m in mins if m < MARKET_CLOSE_MIN <= m + 30)
        out["by_cohort"][c] = {
            "observations": n,
            "bucket_counts": {lab: buckets.get(lab, 0) for _lo, _hi, lab in TOD_BUCKETS},
            "bucket_pct": {lab: _pct(buckets.get(lab, 0), n) for _lo, _hi, lab in TOD_BUCKETS},
            "p25_time_et": _fmt_min(_quantile(mins, 0.25)),
            "median_time_et": _fmt_min(_quantile(mins, 0.5)),
            "p75_time_et": _fmt_min(_quantile(mins, 0.75)),
            "plus60_crosses_close_pct": _pct(cross60, n),
            "plus30_crosses_close_pct": _pct(cross30, n),
        }
    slots = sorted({fr._et(fr.anchor(o)).strftime("%H:%M") for o in pop if fr.anchor(o)})
    for s in slots:
        out["plus60_by_scan_slot"][s] = {}
        for c in COHORTS:
            rows = [o for o in pop if fr.cohort(o) == c and fr.anchor(o)
                    and fr._et(fr.anchor(o)).strftime("%H:%M") == s]
            el = [o for o in rows if is_settled(o, "+60m", now)
                  or "+60m" in matured.get(str(o.get("observation_id")), ())]
            m = sum(1 for o in el if "+60m" in matured.get(str(o.get("observation_id")), ()))
            out["plus60_by_scan_slot"][s][c] = {"eligible": len(el), "matured": m,
                                                "maturation_pct": _pct(m, len(el))}
    shares = [out["by_cohort"][c]["plus60_crosses_close_pct"] or 0.0 for c in COHORTS]
    out["plus60_crosses_close_spread_pp"] = round(max(shares) - min(shares), 2) if pop else None
    return out


# ---- Symbol composition ------------------------------------------------------------
def symbol_format(sym: str) -> str:
    s = str(sym or "").upper()
    if not s:
        return "EMPTY"
    if re.search(r"[.\-]", s):
        return "CLASS_OR_SUFFIX"
    if any(ch.isdigit() for ch in s):
        return "CONTAINS_DIGIT"
    if len(s) == 5 and s[-1] in "WURQ":
        return "FIVE_LETTER_SPECIAL_SUFFIX"
    return "PLAIN"


def composition(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime) -> Dict[str, Any]:
    syms = {c: [str(o.get("symbol") or "").upper() for o in pop if fr.cohort(o) == c] for c in COHORTS}
    uniq = {c: set(v) for c, v in syms.items()}
    out: Dict[str, Any] = {"by_cohort": {}, "overlap": {
        "candidate_control": len(uniq[CANDIDATE] & uniq[CONTROL]),
        "candidate_near_miss": len(uniq[CANDIDATE] & uniq[NEAR_MISS]),
        "near_miss_control": len(uniq[NEAR_MISS] & uniq[CONTROL])}}
    for c in COHORTS:
        rows = [o for o in pop if fr.cohort(o) == c]
        n = len(rows)
        dv = [float(o["market"]["price"]) * float(o["market"]["volume"]) for o in rows
              if (o.get("market") or {}).get("price") is not None
              and (o.get("market") or {}).get("volume") is not None]
        px = [float(o["market"]["price"]) for o in rows if (o.get("market") or {}).get("price") is not None]
        per_scan = Counter((fr.scan_run(o), str(o.get("symbol") or "").upper()) for o in rows)
        excl = Counter(fr._exclusion(s) or "eligible" for s in syms[c])
        providers = Counter(str((o.get("research_metadata") or {}).get("price_provider") or "not_recorded")
                            for o in rows)
        by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for o in rows:
            by_sym[str(o.get("symbol") or "").upper()].append(o)
        settled_syms = {s: os_ for s, os_ in by_sym.items()
                        if any(is_settled(o, "+5m", now) for o in os_)}
        never = sum(1 for os_ in settled_syms.values()
                    if not any(matured.get(str(o.get("observation_id"))) for o in os_))
        out["by_cohort"][c] = {
            "observations": n,
            "unique_symbols": len(uniq[c]),
            "observations_per_symbol": round(n / len(uniq[c]), 2) if uniq[c] else None,
            "recurring_symbols": sum(1 for v in Counter(syms[c]).values() if v > 1),
            "symbol_format": dict(Counter(symbol_format(s) for s in syms[c])),
            "exclusion_status": dict(excl),
            "duplicate_symbol_within_scan": sum(v - 1 for v in per_scan.values() if v > 1),
            "capture_dollar_volume_quantiles_musd": [None if q is None else round(q / 1e6, 3)
                                                     for q in (_quantile(dv, 0.1), _quantile(dv, 0.5),
                                                               _quantile(dv, 0.9))],
            "capture_price_median": None if not px else round(_quantile(px, 0.5), 2),
            "price_provider": dict(providers),
            "symbols_never_matured_pct": _pct(never, len(settled_syms)),
            "exchange": "not persisted on observations",
            "etf_status": "not persisted on observations",
        }
    cand = (out["by_cohort"][CANDIDATE]["capture_dollar_volume_quantiles_musd"] or [None] * 3)[1]
    ctrl = (out["by_cohort"][CONTROL]["capture_dollar_volume_quantiles_musd"] or [None] * 3)[1]
    out["control_to_candidate_median_dollar_volume_ratio"] = (
        round(ctrl / cand, 4) if cand and ctrl is not None else None)
    return out


# ---- Scheduler fairness --------------------------------------------------------------
def with_outcomes(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set]) -> List[Dict[str, Any]]:
    """Copy observations with `outcomes` presence attached (what the loader does)."""
    out = []
    for o in pop:
        x = dict(o)
        x["outcomes"] = {h: True for h in matured.get(str(o.get("observation_id")), ())}
        out.append(x)
    return out


def cap_allocation(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime,
                   caps: Sequence[int] = (400, 2000)) -> Dict[str, Any]:
    """Which cohorts' ready work the worker's own ordering reaches under each cap
    (scheduling only: a stub fetch returns no bars, nothing is written)."""
    from scripts.mature_observations import mature_observations
    cohort_of = {str(o.get("observation_id")): fr.cohort(o) for o in pop}
    out: Dict[str, Any] = {}
    obs = with_outcomes(pop, matured)
    for cap in caps:
        tr: List[Dict[str, Any]] = []
        rep = mature_observations(obs, now=now, dry_run=True, max_symbols=cap,
                                  fetch_bars_batch=lambda s, a, b: {}, trace=tr,
                                  retire_after=None, exclusion_reason=lambda s: None)
        per: Dict[str, Counter] = {c: Counter() for c in COHORTS}
        for t in tr:
            c = cohort_of.get(t["observation_id"])
            if c and t["status"] not in ("ALREADY", "NOT_READY"):
                per[c]["deferred" if t["status"] == "DEFERRED" else "reached"] += 1
        out[str(cap)] = {
            "ready_symbols": rep["backlog"]["ready_symbols"],
            "deferred_symbols": rep["backlog"]["deferred_symbols"],
            "by_cohort": {c: {"ready_horizons": per[c]["reached"] + per[c]["deferred"],
                              "reached": per[c]["reached"], "deferred": per[c]["deferred"],
                              "reached_pct": _pct(per[c]["reached"], per[c]["reached"] + per[c]["deferred"])}
                          for c in COHORTS},
        }
        # Gap across the cohorts that have ready work at all.
        known = [out[str(cap)]["by_cohort"][c]["reached_pct"] for c in COHORTS
                 if out[str(cap)]["by_cohort"][c]["reached_pct"] is not None]
        out[str(cap)]["reach_gap_pp"] = round(max(known) - min(known), 2) if len(known) >= 2 else None
    return out


def replay(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set],
           run_times: Sequence[_dt.datetime], cap: int, *, slack_min: int = 15) -> Dict[str, Any]:
    """Deterministic replay of the historical schedule under a symbol cap.

    Model: at each real maturation run time the worker orders ready symbols by
    earliest pending anchor and processes the first `cap`. A processed
    (observation, horizon) is treated as matured iff it is matured in the store
    today (the only ground truth for data availability); otherwise it stays
    pending. Output: how often each cohort's still-unmatured work was ever
    ATTEMPTED. Never-attempted work cannot be blamed on missing market data."""
    from analytics.observation_capture import horizon_eligibility
    state: Dict[str, set] = defaultdict(set)
    attempts: Counter = Counter()
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in pop:
        by_sym[str(o.get("symbol") or "").upper()].append(o)
    for t in sorted(run_times):
        plans = []
        for sym, os_ in by_sym.items():
            ready_items = []
            for o in os_:
                a = fr.anchor(o)
                if a is None or a > t:
                    continue
                oid = str(o.get("observation_id"))
                elig = horizon_eligibility(a, t, state[oid], slack_min=slack_min)
                ready_items += [(o, h, a) for h, s in elig.items() if s == "ready"]
            if ready_items:
                plans.append((min(x[2] for x in ready_items), sym, ready_items))
        plans.sort(key=lambda p: (p[0], p[1]))
        for _earliest, _sym, items in plans[:cap]:
            for o, h, _a in items:
                oid = str(o.get("observation_id"))
                attempts[(oid, h)] += 1
                if h in matured.get(oid, ()):
                    state[oid].add(h)
    out: Dict[str, Any] = {"cap": cap, "runs": len(run_times), "by_horizon": {}}
    last = max(run_times) if run_times else None
    for h in HORIZONS:
        out["by_horizon"][h] = {}
        for c in COHORTS:
            pend = [o for o in pop if fr.cohort(o) == c and last is not None
                    and h not in matured.get(str(o.get("observation_id")), ())
                    and fr.anchor(o) is not None
                    and last >= fr.anchor(o) + _dt.timedelta(minutes=fr.HORIZON_MIN[h] + slack_min)]
            never = sum(1 for o in pend if attempts[(str(o.get("observation_id")), h)] == 0)
            out["by_horizon"][h][c] = {"unmatured_eligible": len(pend), "never_attempted": never,
                                       "never_attempted_pct": _pct(never, len(pend))}
    return out


def static_audit(obs_per_day: Optional[float]) -> List[Dict[str, str]]:
    """Code-level fairness review of scripts/mature_observations.py + loader."""
    horizon_days = round(5000 / obs_per_day, 1) if obs_per_day else None
    return [
        {"item": "loader query", "finding": "ORDER BY timestamp DESC LIMIT 5000 (hour-bucketed timestamp; "
         "all cohorts of a scan share it)", "cohort_neutral": "yes, except arbitrary tie order in the one "
         "boundary scan",
         "risk": (f"~{obs_per_day:.0f} research observations/day -> the 5,000 window covers ~{horizon_days} days, "
                  "shorter than the 6-day retirement window; older unmatured rows stop being retried"
                  if obs_per_day else "window length unknown")},
        {"item": "grouping", "finding": "observations grouped by symbol regardless of cohort; one bar series "
         "per symbol serves every cohort's observations", "cohort_neutral": "yes (shared bars)",
         "risk": "none when every ready symbol is processed"},
        {"item": "ordering + cap", "finding": "symbols ordered by earliest PENDING anchor, then the first "
         "max_symbols processed", "cohort_neutral": "only when the cap does not bind",
         "risk": "when the cap binds, persistently failing old symbols stay at the head and a recurring "
         "symbol's newer observations ride along with its oldest pending one; CANDIDATE symbols recur far "
         "more than CONTROL symbols"},
        {"item": "per-run cap", "finding": "400 symbols until b81daaf (2026-09-26), 2,000 after",
         "cohort_neutral": "capacity is counted in symbols, and CONTROL contributes ~1 symbol per observation",
         "risk": "CONTROL-heavy symbol counts make the cap bind on CONTROL first"},
        {"item": "horizons", "finding": "all ready horizons of an observation are processed together",
         "cohort_neutral": "yes", "risk": "none"},
        {"item": "retirement", "finding": "time-based (anchor + 6 days), cohort-agnostic code path",
         "cohort_neutral": "yes by construction", "risk": "inherits any earlier starvation"},
        {"item": "batching / caching / 429", "finding": "100-symbol batches in the same oldest-first order; "
         "persistent 429 trips a circuit breaker for the remaining batches", "cohort_neutral":
         "yes, but the batches cut off by a breaker are the later ones in the ordering", "risk": "same as cap"},
        {"item": "failure retry", "finding": "failed symbols are retried every run until retirement",
         "cohort_neutral": "yes", "risk": "no-data symbols consume capacity each run"},
    ]


# ---- Symbol-level sharing ----------------------------------------------------------------
def symbol_sharing(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime) -> Dict[str, Any]:
    """Same symbol observed by >1 cohort: did one mature where the other could not?"""
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in pop:
        by_sym[str(o.get("symbol") or "").upper()].append(o)
    cases = Counter()
    examples = []
    for sym, os_ in by_sym.items():
        if len({fr.cohort(o) for o in os_}) < 2:
            continue
        for h in HORIZONS:
            for o1 in os_:
                for o2 in os_:
                    if fr.cohort(o1) >= fr.cohort(o2):
                        continue
                    m1 = h in matured.get(str(o1.get("observation_id")), ())
                    m2 = h in matured.get(str(o2.get("observation_id")), ())
                    if m1 == m2 or not (is_settled(o1, h, now) and is_settled(o2, h, now)):
                        continue
                    same_run = fr.scan_run(o1) == fr.scan_run(o2)
                    same_anchor = fr.anchor(o1) == fr.anchor(o2)
                    key = ("SAME_ANCHOR_MISMATCH" if same_anchor else
                           "SAME_RUN_DIFFERENT_ANCHOR" if same_run else "DIFFERENT_SCAN_RUN")
                    cases[key] += 1
                    if len(examples) < 10 and same_anchor:
                        examples.append({"symbol": sym, "horizon": h,
                                         "cohorts": [fr.cohort(o1), fr.cohort(o2)]})
    return {"mismatch_cases": dict(cases), "same_anchor_examples": examples,
            "interpretation": ("SAME_ANCHOR_MISMATCH would mean identical bars served one cohort and not "
                               "the other (a pipeline defect). Mismatches across different scan runs have "
                               "different anchors and therefore different future bars.")}


# ---- Retirement ----------------------------------------------------------------------
def retirement(pop: Sequence[Dict[str, Any]], matured: Mapping[str, set], now: _dt.datetime) -> Dict[str, Any]:
    out = {}
    for c in COHORTS:
        rows = [o for o in pop if fr.cohort(o) == c]
        ret_obs = 0
        ret_h = 0
        ages = []
        for o in rows:
            a = fr.anchor(o)
            if a is None or now < a + RETIRE_AFTER:
                continue
            missing = [h for h in HORIZONS if h not in matured.get(str(o.get("observation_id")), ())]
            if missing:
                ret_obs += 1
                ret_h += len(missing)
                ages.append((now - a).total_seconds() / 86400.0)
        out[c] = {"observations": len(rows), "retired_observations": ret_obs, "retired_horizons": ret_h,
                  "retirement_rate_pct": _pct(ret_obs, len(rows)),
                  "age_days_at_audit_median": None if not ages else round(_quantile(ages, 0.5), 2),
                  "reason": "window closed (anchor + 6 days) with the horizon unmatured"}
    rates = [out[c]["retirement_rate_pct"] or 0.0 for c in COHORTS]
    out["retirement_rate_spread_pp"] = round(max(rates) - min(rates), 2)
    out["active"] = any(out[c]["retired_observations"] for c in COHORTS)
    return out


# ---- Root cause --------------------------------------------------------------------------
def root_cause(cov: Mapping[str, Any], par: Mapping[str, Any], tim: Mapping[str, Any],
               comp: Mapping[str, Any], sharing: Mapping[str, Any], traced: bool,
               horizon: str = PRIMARY_HORIZON) -> Dict[str, Any]:
    cand, ctrl = cov[horizon][CANDIDATE], cov[horizon][CONTROL]
    n_min = min(cov[horizon][c]["eligible"] for c in COHORTS)
    if n_min < MIN_ELIGIBLE:
        return {"classification": "INSUFFICIENT_DATA", "confidence": "LOW", "horizon": horizon,
                "reason": f"min eligible per cohort {n_min} < {MIN_ELIGIBLE}", "components_pp": {}}
    comps = {}
    for mech, reasons in MECHANISMS.items():
        ctrl_rate = sum(ctrl["missing_reasons"][r]["count"] for r in reasons) / max(ctrl["eligible"], 1)
        cand_rate = sum(cand["missing_reasons"][r]["count"] for r in reasons) / max(cand["eligible"], 1)
        comps[mech] = round(100 * (ctrl_rate - cand_rate), 2)
    unk = round(100 * (ctrl["missing_reasons"]["UNKNOWN"]["count"] / max(ctrl["eligible"], 1)
                       - cand["missing_reasons"]["UNKNOWN"]["count"] / max(cand["eligible"], 1)), 2)
    comps["UNATTRIBUTED"] = unk
    temporal = tim.get("plus60_crosses_close_spread_pp")
    comps["EXPECTED_TEMPORAL_MISSINGNESS"] = temporal
    material = [m for m, v in comps.items()
                if m not in ("UNATTRIBUTED", "EXPECTED_TEMPORAL_MISSINGNESS") and v is not None and v >= MATERIAL_PP]
    if temporal is not None and temporal >= TEMPORAL_MATERIAL_PP:
        material.append("EXPECTED_TEMPORAL_MISSINGNESS")
    if sharing.get("mismatch_cases", {}).get("SAME_ANCHOR_MISMATCH") and "PIPELINE_BUG" not in material:
        material.append("PIPELINE_BUG")
    ratio = comp.get("control_to_candidate_median_dollar_volume_ratio")
    composition_driven = ratio is not None and ratio < LIQUIDITY_DISPARITY
    gap = par[horizon]["parity_gap"]
    if gap is not None and gap <= 10:
        cls = "EXPECTED_TEMPORAL_MISSINGNESS"
    elif not material:
        cls = "INSUFFICIENT_DATA" if not traced else "MULTIPLE_CAUSES"
    elif len(material) == 1:
        cls = material[0]
    else:
        cls = "MULTIPLE_CAUSES"
    if not traced:
        conf = "LOW"
    elif n_min >= 300 and abs(unk) < 5:
        conf = "HIGH"
    else:
        conf = "MODERATE"
    return {"classification": cls, "confidence": conf, "horizon": horizon, "parity_gap_pp": gap,
            "material_mechanisms": material, "components_pp": comps,
            "availability_driven_by_composition": composition_driven,
            "control_to_candidate_median_dollar_volume_ratio": ratio,
            "reason_basis": "traced" if traced else "inferred (no scheduler trace)"}


# ---- Report -----------------------------------------------------------------------------
def audit_scope(observations, outcomes_by_id, *, scope: str, now: _dt.datetime,
                trace: Optional[Sequence[Mapping[str, Any]]] = None,
                run_times: Optional[Sequence[_dt.datetime]] = None) -> Dict[str, Any]:
    pop = population(observations, scope)
    if not pop:
        return {"status": "NO_FORWARD_DATA" if scope == "forward" else "NO_DATA", "observations": 0}
    matured = matured_index(outcomes_by_id)
    traced = trace_index(trace) if trace is not None else None
    cov = coverage(pop, matured, now, traced)
    par = parity(cov)
    tim = timing(pop, matured, now)
    comp = composition(pop, matured, now)
    share = symbol_sharing(pop, matured, now)
    ret = retirement(pop, matured, now)
    days = {fr.trading_date(fr.anchor(o)) for o in pop if fr.anchor(o)}
    regular = [o for o in pop if fr.is_regular_session(fr.anchor(o))]
    cov_reg = coverage(regular, matured, now, traced)
    out = {
        "status": "OK", "scope": scope, "observations": len(pop),
        # Run 55's primary population (regular-session anchors) - reproduces its figures.
        "run55_primary_population_coverage": {
            h: {c: {k: cov_reg[h][c][k] for k in ("eligible", "matured", "unmatured", "maturation_pct")}
                for c in COHORTS} for h in HORIZONS},
        "run55_primary_population_parity": parity(cov_reg),
        "cohort_counts": {c: sum(1 for o in pop if fr.cohort(o) == c) for c in COHORTS},
        "coverage": cov, "parity": par,
        "projected_parity_after_backlog": parity(cov, "projected_maturation_pct_after_backlog") if traced else None,
        "timing": tim, "symbol_composition": comp, "symbol_sharing": share, "retirement": ret,
        "scheduler_cap_allocation": cap_allocation(pop, matured, now),
        "scheduler_static_audit": static_audit(len(pop) / len(days) if days else None),
    }
    if run_times:
        out["scheduler_replay_400"] = replay(pop, matured, run_times, 400)
    out["root_cause"] = root_cause(cov, par, tim, comp, share, traced is not None)
    return out


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


def assert_clean(report: Mapping[str, Any]) -> None:
    bad = forbidden_keys(report)
    if bad:
        raise ValueError(f"anti-peeking contract violated: {bad[:5]}")
