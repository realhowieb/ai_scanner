"""ML v4 data readiness: canonical metrics, gates and maturation diagnostics (pure).

Answers one question, from persisted data only: is HSF collecting enough
trustworthy point-in-time observations with matured outcomes to start ML v4
development? It never trains, scores or tunes anything, and it computes no
effectiveness statistic beyond the label class balance a gate needs.

Population (the one ML v4 trains on, per the ML v3 audit)
    Frozen HSF opportunity rows in ``signal_outcomes`` (source='opportunity'),
    built into research records by ``analytics.research_dataset`` and collapsed
    to the audit's ``signal_day`` unit (first observation per ticker per UTC
    day) by ``analytics.ml_v3_audit.build_audit_rows``. Same-day copies share
    one entry close and one outcome, so they are counted, never trained on twice.

Horizons
    1/3/5 trading days are the only stored outcome windows (5 is primary).
    10/15/20-bar outcomes are not collected; they are reported as such and
    never derived. Intraday +5m..+60m outcomes on ``hsf_observations`` belong to
    a different population (Run 56 forward readiness) and are out of scope.

Maturation lifecycle (``maturation_stage``)
    recorded -> window complete -> maturation due -> label written -> eligible.
    "Not mature yet" (WAITING_FOR_WINDOW, MATURATION_DUE) is always kept apart
    from "maturation failed" (every other non-matured category).

Gates (``GATES``) are pre-registered constants with the reason for each
threshold. ACCUMULATING gates fill as clean data accrues; STRUCTURAL gates need
a pipeline change and are never projected. Revise thresholds here, in one place,
as the dataset grows; never tune them against a model result.

Pure and deterministic given ``now``; safe on empty input. Callers do the I/O.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analytics import market_calendar as mc
from analytics import outcome_intelligence as oi
from analytics import research_dataset as rd

SCHEMA = "hsf-ml-readiness-1"
# Opportunity freezes start 2026-09-12; reads begin here so the dataset is read whole.
DATASET_START = _dt.date(2026, 9, 1)
PRIMARY_HORIZON = 5
HORIZONS = (1, 3, 5)
UNCOLLECTED_HORIZONS = (10, 15, 20)
TOP_N_PER_SNAPSHOT = 5          # ui.opportunities.build_opportunities(top_n=5): rows frozen per snapshot

# ---- Maturation timing (mirrors the outcome cron, analytics.signal_outcomes) ---------------------------
CRON_MIN_AGE = _dt.timedelta(days=8)       # db.signal_outcomes.list_pending_outcomes(min_age_days=8)
# A due row is "missed" only after two full trading days of cron slots (6 per day) passed without a write.
MATURATION_GRACE_TRADING_DAYS = 2
RECENT_TRADING_DAYS = 20        # rolling window for "is the pipeline healthy NOW" gates
PROJECTION_TRADING_DAYS = 10    # recent collection rate window for growth projections
MIN_PROJECTION_DAYS = 8         # fewer collecting trading days than this -> no projection
STALE_FREEZE = _dt.timedelta(hours=1)  # frozen more than this after its snapshot -> late freeze
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")

# ---- Lifecycle categories ------------------------------------------------------------------------------
TRAINING_ELIGIBLE = "TRAINING_ELIGIBLE"            # matured, certified, label complete
NOT_CERTIFIED = "NOT_CERTIFIED"                    # matured but fails canonical eligibility (e.g. unknown version)
WAITING_FOR_WINDOW = "WAITING_FOR_WINDOW"          # label window (or cron min age) not complete yet
MATURATION_DUE = "MATURATION_DUE"                  # due, within the grace of the next cron runs
MATURATION_JOB_MISSED = "MATURATION_JOB_MISSED"    # due for > grace and still no write
PREMATURE_LABEL_WRITE = "PREMATURE_LABEL_WRITE"    # cron wrote an empty label before the window closed
LABEL_FROM_INCOMPLETE_BAR = "LABEL_FROM_INCOMPLETE_BAR"  # label written before the window's last close
MISSING_PRICE_DATA = "MISSING_PRICE_DATA"          # written after the window closed, provider had no bars
DATA_PROVIDER_FAILURE = "DATA_PROVIDER_FAILURE"    # as above, but bars exist now (probe): transient failure
INVALID_SYMBOL = "INVALID_SYMBOL"
LABEL_WRITE_FAILURE = "LABEL_WRITE_FAILURE"        # written at/before the observation, or partial horizons
MALFORMED_OBSERVATION = "MALFORMED_OBSERVATION"    # no ticker / time / valid score: cannot be placed
UNKNOWN = "UNKNOWN"

NOT_YET = (WAITING_FOR_WINDOW, MATURATION_DUE)
FAILED = (MATURATION_JOB_MISSED, PREMATURE_LABEL_WRITE, LABEL_FROM_INCOMPLETE_BAR, MISSING_PRICE_DATA,
          DATA_PROVIDER_FAILURE, INVALID_SYMBOL, LABEL_WRITE_FAILURE, MALFORMED_OBSERVATION, UNKNOWN)
RECOVERABLE = (PREMATURE_LABEL_WRITE, DATA_PROVIDER_FAILURE, MATURATION_JOB_MISSED)
CATEGORIES = (TRAINING_ELIGIBLE, NOT_CERTIFIED) + NOT_YET + FAILED

# raw_signal keys a freeze must write for the served-model provenance gate (none is written today).
MODEL_VERSION_KEYS = ("prebreakout_model_version", "model_version")

# ---- States --------------------------------------------------------------------------------------------
NOT_READY, COLLECTING, NEAR_READY, READY = "NOT_READY", "COLLECTING", "NEAR_READY", "READY"
RECOMMENDATION = {READY: "ML_V4_READY", NEAR_READY: "ML_V4_NEAR_READY",
                  COLLECTING: "ML_V4_NOT_READY", NOT_READY: "ML_V4_NOT_READY"}
NEAR_PROGRESS = 0.75             # every accumulating gate at >= 75% of target -> NEAR_READY
ACCUMULATING, STRUCTURAL = "ACCUMULATING", "STRUCTURAL"
PASS, FAIL = "PASS", "FAIL"

# ---- Walk-forward fold spec (ML v3 recommendations section 34, pre-registered for ML v4) ---------------
FOLD_SPEC = {"min_train": 60, "min_val": 30, "min_val_class": 5, "embargo": 1}
HOLDOUT_SPEC = {"min_rows": 100, "min_days": 5, "min_remaining_folds": 3}


@dataclass(frozen=True)
class Gate:
    name: str
    category: str      # DATA_VOLUME, TIME_COVERAGE, ... (the spec's headings)
    kind: str          # ACCUMULATING or STRUCTURAL
    metric: str        # key in the flat metrics dict
    op: str            # ">=" or "<=" or "=="
    threshold: float
    why: str


GATES: Tuple[Gate, ...] = (
    Gate("matured_signal_days", "DATA_VOLUME", ACCUMULATING, "matured_signal_days_5d", ">=", 300,
         "ML v3 audit (rec. 1): >= 300 matured, certified 5-day signal-days support 3+ purged folds of "
         ">= 30 validation rows after >= 60 training rows, plus a final holdout. At 134 the audit had 1 fold."),
    Gate("entry_days", "TIME_COVERAGE", ACCUMULATING, "matured_entry_days_5d", ">=", 30,
         "ML v3 audit (rec. 1): spread over >= 30 entry days, so no single day (one market move) dominates "
         "a fold and day-block bootstrap CIs have enough blocks."),
    Gate("entry_weeks", "TIME_COVERAGE", ACCUMULATING, "matured_entry_weeks_5d", ">=", 6,
         "3 folds + a 5-day holdout need about 4 weeks; 6 ISO weeks keeps at least 2 weeks of training "
         "history before the first fold and spans more than one weekly market regime."),
    Gate("unique_symbols", "SYMBOL_DIVERSITY", ACCUMULATING, "matured_unique_symbols_5d", ">=", 100,
         "With 300 rows, >= 100 tickers keeps the average at <= 3 rows per ticker so a model can't "
         "memorize ticker identity (ML v3: 69 tickers at 134 rows)."),
    Gate("top10_symbol_share", "SYMBOL_DIVERSITY", ACCUMULATING, "matured_top10_share_5d", "<=", 0.30,
         "ML v3: top-10 tickers were 33.6% of rows and excluding them left 13 validation rows. At <= 30% a "
         "concentration-excluded re-run still has most of every fold."),
    Gate("class_balance", "CLASS_BALANCE", ACCUMULATING, "minority_class_share_5d", ">=", 0.20,
         "Primary label return_5d > 0. Below 20% minority, a 30-row validation fold has < 6 minority rows, "
         "under the fold spec's min_val_class = 5 margin, and AUC is unstable."),
    Gate("walk_forward_folds", "WALK_FORWARD_COVERAGE", ACCUMULATING, "usable_walk_forward_folds_5d", ">=", 3,
         "ML v3 rec. 34: expanding folds (>= 60 train, >= 30 val, purge = label window, 1-day embargo); "
         "3 folds are the minimum to say anything about fold stability."),
    Gate("final_holdout", "WALK_FORWARD_COVERAGE", ACCUMULATING, "final_holdout_valid_5d", "==", 1,
         "ML v3 rec. 34: the last 5+ entry days (>= 100 rows) held out once, while 3 folds remain."),
    Gate("collection_active", "DATA_VOLUME", STRUCTURAL, "trading_days_since_last_observation", "<=", 2,
         "Scheduled freezes run 6 times per trading day; 2 completed trading days with no new observation "
         "means collection stopped."),
    Gate("recent_outcome_coverage", "OUTCOME_COVERAGE", STRUCTURAL, "recent_outcome_coverage_5d", ">=", 0.90,
         "Of observations whose 5-day label came due in the last 20 trading days, >= 90% must have a label. "
         "Measured on a recent window so it reflects the pipeline now, not a fixed historical loss."),
    Gate("maturation_backlog", "OUTCOME_COVERAGE", STRUCTURAL, "maturation_overdue", "<=", 0,
         "Any observation due for 2+ trading days without a label means the maturation job is missing runs."),
    Gate("benchmark_coverage", "OUTCOME_COVERAGE", STRUCTURAL, "benchmark_coverage_5d", ">=", 0.90,
         "ML v3 rec. 2: the recommended ML v4 target is beat-SPY (excess_return_5d > 0); it needs SPY on "
         ">= 90% of matured rows (21.6% at the audit)."),
    Gate("point_in_time_integrity", "POINT_IN_TIME_INTEGRITY", STRUCTURAL, "pit_violations", "<=", 0,
         "Look-ahead scan joins, labels written at/before the observation, labels read from an unfinished "
         "bar, and rows recorded before their own timestamp must all be zero."),
    Gate("scan_feature_join_rate", "FEATURE_INTEGRITY", STRUCTURAL, "recent_scan_join_rate", ">=", 0.80,
         "ML v3 rec. 5: price/volume/trend features come from the backward scan join (22% at the audit). "
         "At >= 80% of recent rows, explicit missing-indicators stop dominating those features."),
    Gate("served_model_version", "FEATURE_INTEGRITY", STRUCTURAL, "recent_model_version_rate", ">=", 0.95,
         "ML v3 rec. 3: PreBreakout % and the HSF model component can't be used as point-in-time features "
         "unless the served model version is frozen on the row (0 of 1,041 at the audit)."),
    Gate("ai_confidence_frozen", "FEATURE_INTEGRITY", STRUCTURAL, "recent_ai_confidence_rate", ">=", 0.95,
         "ML v3 rec. 4: AI Confidence is unavailable at prediction time for research unless frozen per row "
         "(the audit could score it point-in-time on 65 rows)."),
)
GATE_BY_NAME = {g.name: g for g in GATES}


# --------------------------------------------------------------------------- helpers
def _num(v: Any) -> Optional[float]:
    return rd._num(v)


def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def _day_close_utc(day: _dt.date) -> _dt.datetime:
    return mc.session_bounds_utc(day)[1]


def _add_trading_days(day: _dt.date, n: int) -> _dt.date:
    d, k = day, 0
    while k < n:
        d += _dt.timedelta(days=1)
        if mc.is_trading_day(d):
            k += 1
    return d


def _trading_days_ago(now: _dt.datetime, n: int) -> _dt.date:
    d, k = now.astimezone(mc.ET).date(), 0
    while k < n:
        d -= _dt.timedelta(days=1)
        if mc.is_trading_day(d):
            k += 1
    return d


def _json(v: Any) -> Dict[str, Any]:
    return rd._json(v)


def model_version_of(row: Mapping[str, Any]) -> Optional[str]:
    raw = _json(row.get("raw_signal"))
    for k in MODEL_VERSION_KEYS:
        if raw.get(k):
            return str(raw[k])
    return None


# --------------------------------------------------------------------------- maturation lifecycle
def maturation_timing(observed_at: Any, horizon: int = PRIMARY_HORIZON) -> Optional[Dict[str, Any]]:
    """When the label window of an observation closes and when the cron is due to score it."""
    obs = rd.to_dt(observed_at)
    if obs is None:
        return None
    entry = rd.entry_day(obs)
    if entry is None:
        return None
    end = rd.label_window_end(entry, horizon)
    complete_at = _day_close_utc(end)
    due_at = max(complete_at, obs + CRON_MIN_AGE)
    due_day = due_at.astimezone(mc.ET).date()
    if not mc.is_trading_day(due_day):
        due_day = _add_trading_days(due_day, 1)
    overdue_at = _day_close_utc(_add_trading_days(due_day, MATURATION_GRACE_TRADING_DAYS))
    return {"entry_day": entry, "window_end": end, "complete_at": complete_at, "due_at": due_at,
            "overdue_at": overdue_at}


def maturation_stage(row: Mapping[str, Any], now: _dt.datetime, *, certified: Optional[bool] = None,
                     price_probe: Optional[Mapping[str, bool]] = None) -> Dict[str, Any]:
    """Classify one frozen opportunity row on the observation -> label lifecycle.

    `price_probe` (optional, from a caller that can ask the provider) maps a
    ticker to whether daily bars exist for it now; it only splits
    MISSING_PRICE_DATA from DATA_PROVIDER_FAILURE."""
    ticker = str(row.get("ticker") or "").strip().upper()
    score = _num(_json(row.get("raw_signal")).get("hsf_score"))
    t = maturation_timing(row.get("fired_at"))
    stages = {"recorded": False, "window_complete": False, "label_written": False, "eligible": False}
    if not ticker or t is None or score is None or not (0 <= score <= 100):
        return {"category": MALFORMED_OBSERVATION, "stages": stages, "timing": t}
    stages["recorded"] = True
    stages["window_complete"] = now >= t["complete_at"]
    computed = rd.to_dt(row.get("outcome_computed_at"))
    obs = rd.to_dt(row.get("fired_at"))
    r1, r3, r5 = (_num(row.get(f"return_{h}d")) for h in HORIZONS)

    def out(cat: str) -> Dict[str, Any]:
        return {"category": cat, "stages": stages, "timing": t}

    if computed is None:
        if now < t["due_at"]:
            return out(WAITING_FOR_WINDOW)
        return out(MATURATION_DUE if now < t["overdue_at"] else MATURATION_JOB_MISSED)
    stages["label_written"] = True
    if computed <= obs:
        return out(LABEL_WRITE_FAILURE)
    if r1 is None and r3 is None and r5 is None:
        if computed < t["complete_at"]:
            return out(PREMATURE_LABEL_WRITE)
        if not SYMBOL_RE.match(ticker):
            return out(INVALID_SYMBOL)
        if price_probe is not None and ticker in price_probe:
            return out(DATA_PROVIDER_FAILURE if price_probe[ticker] else MISSING_PRICE_DATA)
        return out(MISSING_PRICE_DATA)
    if r1 is None or r3 is None or r5 is None:
        return out(LABEL_WRITE_FAILURE)
    if computed < t["complete_at"]:
        return out(LABEL_FROM_INCOMPLETE_BAR)
    if certified is False:
        return out(NOT_CERTIFIED)
    stages["eligible"] = True
    return out(TRAINING_ELIGIBLE)


# --------------------------------------------------------------------------- distributions
def _dist(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(v) if v is not None else "UNKNOWN" for v in values).items()))


def _horizon_block(rows: Sequence[Any], h: int) -> Dict[str, Any]:
    """Counts at horizon h over audit rows (signal-day unit)."""
    mat = [r for r in rows if r.matured(h) and r.certified]
    pos = sum(1 for r in mat if (_num(r.labels.get(f"return_{h}d")) or 0) > 0)
    neg = len(mat) - pos
    with_bench = [r for r in mat if r.labels.get(f"excess_return_{h}d") is not None]
    beat = sum(1 for r in with_bench if r.labels[f"excess_return_{h}d"] > 0)
    status = Counter(r.maturity.get(f"{h}d") for r in rows)
    return {
        "matured": len(mat), "pending": status.get(rd.PENDING, 0), "unavailable": status.get(rd.UNAVAILABLE, 0),
        "invalid": status.get(rd.INVALID, 0), "maturity_pct": _rate(len(mat), len(rows)),
        "positive": pos, "negative": neg, "positive_rate": _rate(pos, len(mat)),
        "minority_class_share": _rate(min(pos, neg), len(mat)),
        "beat_spy": {"with_benchmark": len(with_bench), "positive": beat,
                     "positive_rate": _rate(beat, len(with_bench))},
        "entry_days": len({r.entry_day for r in mat}),
        "unique_symbols": len({r.ticker for r in mat}),
    }


def _fold_block(rows: Sequence[Any], h: int) -> Dict[str, Any]:
    from analytics import ml_v3_audit as mv3

    data = mv3.eligible(rows, h)
    folds = mv3.walk_forward_folds(data, h, **FOLD_SPEC)
    hold = mv3.final_holdout(data, h, **HOLDOUT_SPEC, **FOLD_SPEC)
    return {"eligible_rows": len(data), "usable_folds": len(folds),
            "folds": [{"fold": f.fold, "train_rows": len(f.train_ids), "validation_rows": len(f.val_ids),
                       "validation_start": f.validation_start, "validation_end": f.validation_end} for f in folds],
            "final_holdout": {k: v for k, v in hold.items() if k != "ids"}}


def _concentration(rows: Sequence[Any]) -> Dict[str, Any]:
    c = Counter(r.ticker for r in rows)
    n = len(rows)
    top = c.most_common(10)
    return {"rows": n, "unique_symbols": len(c), "top10": [{"ticker": t, "rows": k} for t, k in top],
            "top10_share": _rate(sum(k for _, k in top), n),
            "top_symbol_share": _rate(top[0][1], n) if top else None}


# --------------------------------------------------------------------------- collection audit
def _slot_of(ts: _dt.datetime) -> Optional[str]:
    """Nearest expected scan slot (HH:MM UTC) within system_health's early/late window."""
    from analytics.system_health import SLOT_EARLY, SLOT_LATE

    for s in mc.expected_scan_slots(ts.astimezone(mc.ET).date()):
        if s - SLOT_EARLY <= ts <= s + SLOT_LATE:
            return s.strftime("%H:%M")
    return None


def collection_audit(raw_rows: Sequence[Mapping[str, Any]], now: _dt.datetime, *,
                     stages: Mapping[int, str]) -> Dict[str, Any]:
    """Expected vs actual collection per trading day, missing scan windows,
    partial snapshots, duplicates, stale freezes, repeated symbols."""
    rows = [r for r in raw_rows if rd.to_dt(r.get("fired_at")) is not None and r.get("ticker")]
    if not rows:
        return {"observations": 0}
    first = min(rd.to_dt(r["fired_at"]) for r in rows)
    start_day = first.astimezone(mc.ET).date()
    today = now.astimezone(mc.ET).date()
    snaps: Dict[_dt.datetime, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        snaps[rd.to_dt(r["fired_at"])].append(r)
    by_day_obs: Dict[_dt.date, int] = Counter()
    by_day_signal: Dict[_dt.date, set] = defaultdict(set)
    by_day_snaps: Dict[_dt.date, set] = defaultdict(set)
    slot_hits: Dict[Tuple[_dt.date, str], int] = Counter()
    off_slot = 0
    for ts_, members in snaps.items():
        d = ts_.astimezone(mc.ET).date()
        by_day_obs[d] += len(members)
        by_day_snaps[d].add(ts_)
        for m in members:
            by_day_signal[d].add(str(m["ticker"]).upper())
        s = _slot_of(ts_)
        if s:
            slot_hits[(d, s)] += 1
        else:
            off_slot += 1
    days = [d for d in mc.trading_days_between(start_day, today)
            if d < today or now >= _day_close_utc(d)]
    expected_per_day = len(mc.SCAN_SLOTS_UTC) * TOP_N_PER_SNAPSHOT
    slot_names = [f"{h:02d}:{m:02d}" for h, m in mc.SCAN_SLOTS_UTC]
    missing = [(d, s) for d in days for s in slot_names if (d, s) not in slot_hits]
    per_slot = {s: {"expected": len(days), "with_snapshot": sum(1 for d in days if (d, s) in slot_hits)}
                for s in slot_names}
    non_trading = sum(n for d, n in by_day_obs.items() if not mc.is_trading_day(d))
    partial = sum(1 for m in snaps.values() if len(m) < TOP_N_PER_SNAPSHOT)
    exact_dups = sum(c - 1 for c in Counter((str(r["ticker"]).upper(), rd.iso(r["fired_at"]))
                                            for r in rows).values() if c > 1)
    signal_days = sum(len(v) for v in by_day_signal.values())
    late = [r for r in rows if rd.to_dt(r.get("created_at")) is not None
            and rd.to_dt(r["created_at"]) - rd.to_dt(r["fired_at"]) > STALE_FREEZE]
    before = [r for r in rows if rd.to_dt(r.get("created_at")) is not None
              and rd.to_dt(r["created_at"]) < rd.to_dt(r["fired_at"]) - _dt.timedelta(minutes=5)]
    # carry-over: signal-days whose ticker was also a signal-day on the previous collecting trading day
    ordered = sorted(d for d in by_day_signal if mc.is_trading_day(d))
    carry = sum(len(by_day_signal[b] & by_day_signal[a]) for a, b in zip(ordered, ordered[1:]))
    carry_base = sum(len(by_day_signal[b]) for b in ordered[1:])
    no_label = Counter(str(r["ticker"]).upper() for r in rows
                       if stages.get(int(r["id"])) in (MISSING_PRICE_DATA, DATA_PROVIDER_FAILURE, INVALID_SYMBOL))
    recent = [d for d in days if d >= _trading_days_ago(now, PROJECTION_TRADING_DAYS)]
    return {
        "observations": len(rows), "snapshots": len(snaps), "first_observation": rd.iso(first),
        "trading_days_in_range": len(days),
        "trading_days_with_observations": sum(1 for d in days if by_day_obs.get(d)),
        "expected_observations_per_trading_day": expected_per_day,
        "expected_basis": f"{len(mc.SCAN_SLOTS_UTC)} scheduled slots x top {TOP_N_PER_SNAPSHOT} opportunities "
                          "(an upper bound: a slot whose snapshot is unchanged freezes nothing new)",
        "actual_observations_per_trading_day": _rate(sum(by_day_obs.get(d, 0) for d in days), len(days)),
        "actual_signal_days_per_trading_day": _rate(sum(len(by_day_signal.get(d, ())) for d in days), len(days)),
        "recent_trading_days": len(recent),
        "recent_observations_per_trading_day": _rate(sum(by_day_obs.get(d, 0) for d in recent), len(recent)),
        "recent_signal_days_per_trading_day": _rate(sum(len(by_day_signal.get(d, ())) for d in recent),
                                                    len(recent)),
        "per_trading_day": [{"day": d.isoformat(), "observations": by_day_obs.get(d, 0),
                             "signal_days": len(by_day_signal.get(d, ())), "snapshots": len(by_day_snaps.get(d, ()))}
                            for d in days],
        "missing_scan_windows": {"count": len(missing), "of": len(days) * len(slot_names), "per_slot": per_slot,
                                 "recent": [f"{d.isoformat()} {s}" for d, s in missing[-12:]]},
        "snapshots_off_schedule": off_slot,
        "partial_snapshots": partial,
        "non_trading_day_observations": non_trading,
        "duplicates": {"exact_same_ticker_same_instant": exact_dups,
                       "same_ticker_same_day_repeats": len(rows) - signal_days,
                       "repeat_rate": _rate(len(rows) - signal_days, len(rows)),
                       "carry_over_from_previous_day": carry,
                       "carry_over_rate": _rate(carry, carry_base)},
        "late_frozen_observations": len(late),
        "recorded_before_timestamp": len(before),
        "symbols_repeatedly_without_labels": [{"ticker": t, "rows": k} for t, k in no_label.most_common(10) if k >= 2],
    }


def effective_samples(rows: Sequence[Any], h: int = PRIMARY_HORIZON) -> int:
    """Matured signal-days left after dropping same-ticker rows whose entry day
    falls inside the previous kept row's label window (greedy, oldest first):
    a rough count of non-overlapping, independent samples."""
    last_end: Dict[str, _dt.date] = {}
    n = 0
    for r in sorted((r for r in rows if r.matured(h) and r.certified), key=lambda r: (r.entry_day, r.observation_id)):
        end = last_end.get(r.ticker)
        if end is not None and r.entry_day <= end:
            continue
        last_end[r.ticker] = r.window_end[h]
        n += 1
    return n


# --------------------------------------------------------------------------- gates
def _passes(op: str, value: Optional[float], threshold: float) -> bool:
    if value is None:
        return False
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    return value == threshold


def _progress(g: Gate, value: Optional[float]) -> Optional[float]:
    if value is None:
        return 0.0
    if g.op == ">=":
        return round(min(1.0, value / g.threshold), 4) if g.threshold else 1.0
    if g.op == "<=":
        if value <= g.threshold:
            return 1.0
        return round(max(0.0, min(1.0, g.threshold / value)), 4) if value else 0.0
    return 1.0 if value == g.threshold else 0.0


def evaluate_gates(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for g in GATES:
        v = metrics.get(g.metric)
        ok = _passes(g.op, v, g.threshold)
        out.append({"gate": g.name, "category": g.category, "kind": g.kind, "metric": g.metric, "value": v,
                    "operator": g.op, "threshold": g.threshold, "status": PASS if ok else FAIL,
                    "progress": _progress(g, v), "why": g.why})
    return out


def decide_status(gates: Sequence[Mapping[str, Any]]) -> str:
    failed = [g for g in gates if g["status"] == FAIL]
    if not failed:
        return READY
    if any(g["kind"] == STRUCTURAL for g in failed):
        return NOT_READY
    if all((g["progress"] or 0) >= NEAR_PROGRESS for g in failed):
        return NEAR_READY
    return COLLECTING


def _is_ratio(metric: str) -> bool:
    return any(k in metric for k in ("rate", "share", "coverage"))


def _show(metric: str, v: Any) -> str:
    if v is None:
        return "unavailable"
    return f"{v:.1%}" if _is_ratio(metric) else f"{v:g}" if isinstance(v, float) else str(v)


def blocking_reasons(gates: Sequence[Mapping[str, Any]]) -> List[str]:
    return [f"{g['gate']}: {_show(g['metric'], g['value'])} (needs {g['operator']} "
            f"{_show(g['metric'], g['threshold'])})" for g in gates if g["status"] == FAIL]


# --------------------------------------------------------------------------- projection
def project(gates: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], collection: Mapping[str, Any],
            recent_coverage: Optional[float]) -> Dict[str, Any]:
    """Rough trading-day estimates per failing accumulating gate, from the
    recent collection rate. Never a guarantee; refuses short or broken windows."""
    days = int(collection.get("recent_trading_days") or 0)
    rate_sd = collection.get("recent_signal_days_per_trading_day")
    collecting_days = sum(1 for d in (collection.get("per_trading_day") or [])[-days:] if d["observations"]) \
        if days else 0
    lag = PRIMARY_HORIZON + 1  # label window + the cron's min age beyond it (~8 calendar days)
    basis = {"window_trading_days": days, "collecting_trading_days": collecting_days,
             "signal_days_per_trading_day": rate_sd, "maturation_success_rate": recent_coverage,
             "maturation_lag_trading_days": lag}
    if collecting_days < MIN_PROJECTION_DAYS or not rate_sd:
        return {"basis": basis, "available": False,
                "reason": f"only {collecting_days} collecting trading days in the last {days}; "
                          f"need >= {MIN_PROJECTION_DAYS} for a rate", "gates": []}
    success = recent_coverage if recent_coverage is not None else 1.0
    eff_rate = rate_sd * success
    pending_sd = int(metrics.get("pending_signal_days_5d") or 0)
    out = []
    for g in gates:
        if g["status"] == PASS:
            continue
        name, v, thr = g["gate"], g["value"] or 0, g["threshold"]
        est, note = None, ""
        if g["kind"] == STRUCTURAL:
            note = "needs a pipeline change; not projected"
        elif name == "matured_signal_days":
            need = max(0.0, thr - v - pending_sd * success)
            est = math.ceil(need / eff_rate) + lag if eff_rate > 0 else None
            note = f"{v} now + {pending_sd} pending x {success:.0%} success, then {eff_rate:.1f}/trading day"
        elif name == "entry_days":
            est = int(thr - v) + lag
            note = "one entry day per collecting trading day"
        elif name == "entry_weeks":
            est = int(thr - v) * 5 + lag
            note = "one ISO week per 5 collecting trading days"
        elif name in ("walk_forward_folds", "final_holdout"):
            sd = GATE_BY_NAME["matured_signal_days"].threshold
            need = max(0.0, sd - (metrics.get("matured_signal_days_5d") or 0) - pending_sd * success)
            est = math.ceil(need / eff_rate) + lag if eff_rate > 0 else None
            note = "folds/holdout are sized by the 300 signal-day target; follows matured_signal_days"
        elif name == "unique_symbols":
            new_rate = metrics.get("recent_new_symbols_per_trading_day")
            est = math.ceil((thr - v) / new_rate) + lag if new_rate else None
            note = (f"{new_rate} new tickers per trading day recently (linear; new-ticker rate usually slows)"
                    if new_rate else "no new tickers recently; not projectable")
        else:
            note = "depends on the mix of future data; not projected"
        out.append({"gate": name, "estimated_trading_days": est, "note": note})
    return {"basis": basis, "available": True, "gates": out,
            "estimate_trading_days_to_accumulating_targets": max(
                (o["estimated_trading_days"] for o in out if o["estimated_trading_days"] is not None), default=None),
            "disclaimer": "An estimate from the recent rate, not a guarantee. Structural gates must be fixed first."}


# --------------------------------------------------------------------------- report
def build_report(raw_rows: Sequence[Mapping[str, Any]], scan_rows: Iterable[Mapping[str, Any]], *,
                 now: Optional[_dt.datetime] = None, spy_closes: Optional[Mapping[_dt.date, float]] = None,
                 price_probe: Optional[Mapping[str, bool]] = None) -> Dict[str, Any]:
    """The canonical readiness report. `raw_rows` are opportunity rows
    (signal_outcomes, source='opportunity'), `scan_rows` the scheduled scan
    records (or the slim index of them) for the backward feature join."""
    from analytics import ml_v3_audit as mv3

    now = now or _dt.datetime.now(_dt.timezone.utc)
    rows = [dict(r) for r in raw_rows or [] if r.get("id") is not None]
    records = rd.build_records(rows, scan_rows or [])
    obs_rows = mv3.build_audit_rows(records, rows, unit="observation")
    sd_rows = mv3.build_audit_rows(records, rows, unit="signal_day")
    cert = {int(r["observation"]["observation_id"]): bool(r["outcome"].certified) for r in records}
    stage_of: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        stage_of[int(r["id"])] = maturation_stage(r, now, certified=cert.get(int(r["id"])), price_probe=price_probe)
    cats = {oid: s["category"] for oid, s in stage_of.items()}
    sd_ids = {r.observation_id for r in sd_rows}

    # --- maturation integrity
    cat_obs = Counter(cats.values())
    cat_sd = Counter(c for oid, c in cats.items() if oid in sd_ids)
    lifecycle = Counter()
    for s in stage_of.values():
        for k, v in s["stages"].items():
            lifecycle[k] += int(bool(v))
    recent_from = _day_close_utc(_trading_days_ago(now, RECENT_TRADING_DAYS))
    due_recent = [oid for oid, s in stage_of.items() if s["timing"] and recent_from <= s["timing"]["due_at"] <= now
                  and s["category"] != MATURATION_DUE]
    labeled_recent = [oid for oid in due_recent if cats[oid] in (TRAINING_ELIGIBLE, NOT_CERTIFIED)]
    recent_coverage = _rate(len(labeled_recent), len(due_recent))
    due_all = [oid for oid, c in cats.items() if c not in NOT_YET and c != MALFORMED_OBSERVATION]
    all_coverage = _rate(sum(1 for oid in due_all if cats[oid] in (TRAINING_ELIGIBLE, NOT_CERTIFIED)), len(due_all))

    # --- horizons
    horizons = {f"{h}d": _horizon_block(sd_rows, h) for h in HORIZONS}
    horizons_obs = {f"{h}d": _horizon_block(obs_rows, h) for h in HORIZONS}
    for h in UNCOLLECTED_HORIZONS:
        horizons[f"{h}_bar"] = {"collected": False,
                                "note": "No 10/15/20-bar outcome is stored for HSF observations; not derived."}
    folds = {f"{h}d": _fold_block(sd_rows, h) for h in HORIZONS}
    mat5 = [r for r in sd_rows if r.matured(PRIMARY_HORIZON) and r.certified]
    conc = _concentration(mat5)
    bench5 = sum(1 for r in mat5 if r.labels.get("benchmark_return_5d") is not None)

    # --- point-in-time integrity
    lookahead = 0
    for rec in records:
        j = rec["features"].join
        if j.get("status") == rd.JOIN_MATCHED and j.get("scan_timestamp"):
            if rd.to_dt(j["scan_timestamp"]) > rd.to_dt(rec["observation"]["observed_at"]):
                lookahead += 1
    collection = collection_audit(rows, now, stages=cats)
    pit = {"lookahead_scan_joins": lookahead,
           "labels_written_at_or_before_observation": sum(1 for r in obs_rows if any(
               r.maturity.get(f"{h}d") == rd.INVALID for h in HORIZONS)),
           "labels_from_incomplete_bar": cat_obs.get(LABEL_FROM_INCOMPLETE_BAR, 0),
           "recorded_before_timestamp": collection.get("recorded_before_timestamp", 0)}
    pit_violations = sum(pit.values())

    # --- feature integrity (recent window = rows observed in the last RECENT_TRADING_DAYS)
    recent_rows = [r for r in rows if rd.to_dt(r["fired_at"]) and rd.to_dt(r["fired_at"]) >= recent_from]
    rec_by_id = {int(r["observation"]["observation_id"]): r for r in records}
    joined = lambda rs_: sum(1 for r in rs_ if rec_by_id.get(int(r["id"])) is not None and  # noqa: E731
                             rec_by_id[int(r["id"])]["features"].join.get("status") == rd.JOIN_MATCHED)
    join_status = _dist(r["features"].join.get("status") for r in records)
    features = {
        "scan_join_rate": _rate(joined(rows), len(rows)),
        "recent_scan_join_rate": _rate(joined(recent_rows), len(recent_rows)),
        "scan_join_status": join_status,
        "model_version_rate": _rate(sum(1 for r in rows if model_version_of(r)), len(rows)),
        "recent_model_version_rate": _rate(sum(1 for r in recent_rows if model_version_of(r)), len(recent_rows)),
        "ai_confidence_rate": _rate(sum(1 for r in rows if _num(r.get("ai_confidence")) is not None), len(rows)),
        "recent_ai_confidence_rate": _rate(sum(1 for r in recent_rows if _num(r.get("ai_confidence")) is not None),
                                           len(recent_rows)),
        "recent_rows": len(recent_rows),
    }

    # --- dimensions (signal-day unit; observation counts alongside)
    regimes = None
    if spy_closes:
        reg = mv3.spy_regime_by_day(spy_closes, {r.entry_day for r in sd_rows})
        regimes = _dist(f"{(reg.get(r.entry_day) or {}).get('trend')}/{(reg.get(r.entry_day) or {}).get('volatility')}"
                        if (reg.get(r.entry_day) or {}).get("trend") else None for r in sd_rows)
    oi_of = {r.observation_id: r.oi_record for r in sd_rows}
    dims = {
        "by_score_range": _dist(oi.score_bucket(_num(r.oi_record.hsf_score) if r.oi_record else None)
                                for r in sd_rows),
        "by_score_range_matured_5d": _dist(oi.score_bucket(_num(r.oi_record.hsf_score) if r.oi_record else None)
                                           for r in mat5),
        "by_status": _dist((oi_of.get(r.observation_id).status if oi_of.get(r.observation_id) else None)
                           for r in sd_rows),
        "by_recommendation": {"available": False,
                              "note": "No separate recommendation is frozen; the HSF status (STRONG/WATCH/CAUTION) "
                                      "is the tier and is reported under by_status."},
        "by_tier": {"available": False, "note": "Same as by_status (HSF status is the tier)."},
        "by_setup": _dist(r.setup for r in sd_rows),
        "by_signal": _dist(s for r in sd_rows for s in (r.oi_record.signals if r.oi_record else ()) or ("none",)),
        "by_source": {"opportunity": len(sd_rows)},
        "by_market_regime": regimes if regimes is not None else {
            "available": False,
            "note": "market_regime is not frozen on observations (capture passes None). The readiness report "
                    "derives it offline from SPY prior closes; the API does not fetch prices."},
        "by_holding_window": {k: v["matured"] for k, v in horizons.items() if "matured" in v},
    }

    # --- growth inputs
    first_seen: Dict[str, _dt.date] = {}
    for r in sorted(sd_rows, key=lambda r: r.observed_at):
        first_seen.setdefault(r.ticker, r.observed_at.astimezone(mc.ET).date())
    recent_day0 = _trading_days_ago(now, PROJECTION_TRADING_DAYS)
    new_syms = sum(1 for d in first_seen.values() if d >= recent_day0)
    rec_days = int(collection.get("recent_trading_days") or 0)

    times = [r.observed_at for r in obs_rows]
    last_obs = max(times) if times else None
    since_last = mc.completed_trading_days_since(last_obs, now) if last_obs else None
    h5 = horizons["5d"]
    metrics = {
        "total_observations": len(rows),
        "signal_days": len(sd_rows),
        "unique_symbols": len({r.ticker for r in sd_rows}),
        "unique_observation_dates": len({r.observed_day for r in sd_rows}),
        "unique_entry_days": len({r.entry_day for r in sd_rows}),
        "matured_observations_5d": horizons_obs["5d"]["matured"],
        "immature_observations": cat_obs.get(WAITING_FOR_WINDOW, 0) + cat_obs.get(MATURATION_DUE, 0),
        "failed_maturation_observations": sum(cat_obs.get(c, 0) for c in FAILED),
        "maturity_pct_observations": _rate(horizons_obs["5d"]["matured"], len(rows)),
        "matured_signal_days_5d": h5["matured"],
        "pending_signal_days_5d": cat_sd.get(WAITING_FOR_WINDOW, 0) + cat_sd.get(MATURATION_DUE, 0),
        "matured_entry_days_5d": h5["entry_days"],
        "matured_entry_weeks_5d": len({tuple(r.entry_day.isocalendar()[:2]) for r in mat5}),
        "matured_unique_symbols_5d": h5["unique_symbols"],
        "matured_top10_share_5d": conc["top10_share"],
        "minority_class_share_5d": h5["minority_class_share"],
        "usable_walk_forward_folds_5d": folds["5d"]["usable_folds"],
        "final_holdout_valid_5d": 1 if folds["5d"]["final_holdout"].get("valid") else 0,
        "effective_independent_samples_5d": effective_samples(sd_rows),
        "trading_days_since_last_observation": since_last,
        "recent_outcome_coverage_5d": recent_coverage,
        "all_time_outcome_coverage_5d": all_coverage,
        "maturation_overdue": cat_obs.get(MATURATION_JOB_MISSED, 0),
        "benchmark_coverage_5d": _rate(bench5, len(mat5)),
        "pit_violations": pit_violations,
        "recent_scan_join_rate": features["recent_scan_join_rate"],
        "recent_model_version_rate": features["recent_model_version_rate"],
        "recent_ai_confidence_rate": features["recent_ai_confidence_rate"],
        "recent_new_symbols_per_trading_day": _rate(new_syms, rec_days),
        "earliest_observation": min(times).isoformat() if times else None,
        "latest_observation": last_obs.isoformat() if last_obs else None,
        "latest_label_written": max((rd.iso(r.get("outcome_computed_at")) for r in rows
                                     if r.get("outcome_computed_at")), default=None),
    }
    gates = evaluate_gates(metrics)
    status = decide_status(gates)
    return {
        "schema": SCHEMA, "generated_at": now.isoformat(), "status": status,
        "recommendation": RECOMMENDATION[status],
        "unit": "signal_day (first observation per ticker per UTC day); observation counts alongside",
        "primary_horizon_days": PRIMARY_HORIZON,
        "metrics": metrics, "gates": gates, "blocking_reasons": blocking_reasons(gates),
        "horizons": horizons, "horizons_observation_unit": horizons_obs, "walk_forward": folds,
        "concentration_5d": conc, "dimensions": dims,
        "maturation": {"by_category_observations": {c: cat_obs.get(c, 0) for c in CATEGORIES},
                       "by_category_signal_days": {c: cat_sd.get(c, 0) for c in CATEGORIES},
                       "lifecycle_observations": dict(lifecycle),
                       "recoverable_observations": sum(cat_obs.get(c, 0) for c in RECOVERABLE),
                       "recent_window_trading_days": RECENT_TRADING_DAYS,
                       "recent_due": len(due_recent), "recent_labeled": len(labeled_recent)},
        "collection": collection, "point_in_time": pit, "feature_integrity": features,
        "projection": project(gates, metrics, collection, recent_coverage),
        "spec": {"fold": FOLD_SPEC, "holdout": HOLDOUT_SPEC, "near_progress": NEAR_PROGRESS,
                 "cron_min_age_days": CRON_MIN_AGE.days, "maturation_grace_trading_days": MATURATION_GRACE_TRADING_DAYS},
    }


def api_view(report: Mapping[str, Any]) -> Dict[str, Any]:
    """The stable /v1/ml/readiness schema (aggregates only: no tickers, ids or returns)."""
    m = report["metrics"]
    proj = report.get("projection") or {}
    return {
        "schema": report["schema"], "status": report["status"], "recommendation": report["recommendation"],
        "generated_at": report["generated_at"], "unit": report["unit"],
        "primary_horizon_days": report["primary_horizon_days"],
        "observations": {"total": m["total_observations"], "signal_days": m["signal_days"],
                         "matured": m["matured_observations_5d"], "matured_signal_days": m["matured_signal_days_5d"],
                         "immature": m["immature_observations"], "failed_maturation": m["failed_maturation_observations"],
                         "maturity_pct": m["maturity_pct_observations"],
                         "effective_independent_samples": m["effective_independent_samples_5d"]},
        "coverage": {"symbols": m["unique_symbols"], "matured_symbols": m["matured_unique_symbols_5d"],
                     "trading_days": m["unique_entry_days"], "matured_trading_days": m["matured_entry_days_5d"],
                     "matured_weeks": m["matured_entry_weeks_5d"],
                     "date_start": m["earliest_observation"], "date_end": m["latest_observation"],
                     "outcome_coverage_recent": m["recent_outcome_coverage_5d"],
                     "outcome_coverage_all_time": m["all_time_outcome_coverage_5d"],
                     "benchmark_coverage": m["benchmark_coverage_5d"]},
        "labels": {k: {kk: v[kk] for kk in ("matured", "positive", "negative", "positive_rate",
                                            "minority_class_share")}
                   for k, v in report["horizons"].items() if "matured" in v},
        "uncollected_horizons": [f"{h}_bar" for h in UNCOLLECTED_HORIZONS],
        "validation": {"usable_walk_forward_folds": m["usable_walk_forward_folds_5d"],
                       "final_holdout_valid": bool(m["final_holdout_valid_5d"]),
                       "by_horizon": {k: v["usable_folds"] for k, v in report["walk_forward"].items()},
                       "fold_spec": report["spec"]["fold"]},
        "maturation": {"by_category": report["maturation"]["by_category_observations"],
                       "recoverable": report["maturation"]["recoverable_observations"],
                       "overdue": m["maturation_overdue"], "latest_label_written": m["latest_label_written"]},
        "integrity": {"point_in_time": report["point_in_time"],
                      "scan_join_rate_recent": m["recent_scan_join_rate"],
                      "model_version_rate_recent": m["recent_model_version_rate"],
                      "ai_confidence_rate_recent": m["recent_ai_confidence_rate"]},
        "gates": [{k: g[k] for k in ("gate", "category", "kind", "metric", "value", "operator", "threshold",
                                     "status", "progress", "why")} for g in report["gates"]],
        "blocking_reasons": list(report["blocking_reasons"]),
        "projection": {"available": proj.get("available", False), "reason": proj.get("reason"),
                       "estimate_trading_days_to_accumulating_targets":
                           proj.get("estimate_trading_days_to_accumulating_targets"),
                       "gates": proj.get("gates") or [], "basis": proj.get("basis")},
    }


# --------------------------------------------------------------------------- monitoring
def monitor(report: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Alerts for the scheduled health run: collection stopped, maturation
    behind, duplicates rising, class imbalance, PIT violations, stale inputs."""
    m = report["metrics"]
    col = report.get("collection") or {}
    alerts: List[Dict[str, Any]] = []

    def add(code: str, severity: str, message: str) -> None:
        alerts.append({"code": code, "severity": severity, "message": message})

    since = m.get("trading_days_since_last_observation")
    if since is None or since > GATE_BY_NAME["collection_active"].threshold:
        add("COLLECTION_STOPPED", "CRITICAL", f"no new HSF observation for {since} completed trading days")
    if m.get("maturation_overdue"):
        add("MATURATION_BEHIND", "WARNING", f"{m['maturation_overdue']} observations are past due without a label")
    cov = m.get("recent_outcome_coverage_5d")
    if cov is not None and cov < GATE_BY_NAME["recent_outcome_coverage"].threshold:
        add("MATURATION_FAILING", "WARNING", f"recent 5-day outcome coverage {cov:.0%} (< 90%)")
    dups = (col.get("duplicates") or {})
    if dups.get("exact_same_ticker_same_instant"):
        add("DUPLICATE_OBSERVATIONS", "WARNING",
            f"{dups['exact_same_ticker_same_instant']} exact duplicate observations (same ticker, same instant)")
    days = [d for d in (col.get("per_trading_day") or []) if d["observations"]]
    recent, prior = days[-5:], days[-25:-5]
    if len(recent) >= 3 and len(prior) >= 5:
        rr = 1 - sum(d["signal_days"] for d in recent) / sum(d["observations"] for d in recent)
        pr = 1 - sum(d["signal_days"] for d in prior) / sum(d["observations"] for d in prior)
        if rr - pr > 0.10:
            add("DUPLICATE_RATE_RISING", "WARNING",
                f"same-ticker same-day repeat rate {rr:.0%} over the last {len(recent)} collecting days "
                f"vs {pr:.0%} before")
    if previous:
        prev_rate = ((previous.get("collection") or {}).get("duplicates") or {}).get("repeat_rate")
        if prev_rate is not None and dups.get("repeat_rate") is not None and dups["repeat_rate"] - prev_rate > 0.10:
            add("DUPLICATE_RATE_RISING", "WARNING",
                f"same-ticker same-day repeat rate rose from {prev_rate:.0%} to {dups['repeat_rate']:.0%}")
    h5 = (report.get("horizons") or {}).get("5d") or {}
    if (h5.get("matured") or 0) >= 50 and (h5.get("minority_class_share") or 0) < GATE_BY_NAME["class_balance"].threshold:
        add("CLASS_IMBALANCE", "WARNING", f"5-day label minority share {h5['minority_class_share']:.0%} on "
                                          f"{h5['matured']} signal-days")
    if m.get("pit_violations"):
        add("PIT_VIOLATION", "CRITICAL", f"{m['pit_violations']} point-in-time violations: {report['point_in_time']}")
    gen = rd.to_dt(report.get("generated_at"))
    last_label = rd.to_dt(m.get("latest_label_written"))
    if gen and last_label and mc.completed_trading_days_since(last_label, gen) > 3 \
            and (report.get("maturation") or {}).get("by_category_observations", {}).get(MATURATION_DUE, 0):
        add("READINESS_INPUTS_STALE", "WARNING",
            f"no outcome label written since {last_label.isoformat()} while observations are due")
    return alerts


# --------------------------------------------------------------------------- markdown
def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1%}" if 0 <= v <= 1 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def render_markdown(report: Mapping[str, Any], *, alerts: Sequence[Mapping[str, Any]] = ()) -> str:
    m, col, mat = report["metrics"], report["collection"], report["maturation"]
    L = ["# ML v4 data readiness report", "",
         f"Generated {report['generated_at']} · schema `{report['schema']}` · unit: {report['unit']}", "",
         f"## Status: **{report['status']}** → recommendation **{report['recommendation']}**", ""]
    if report["blocking_reasons"]:
        L += ["Blocking gates:", ""] + [f"- {b}" for b in report["blocking_reasons"]] + [""]
    L += ["## Gates", "", "| GATE | CATEGORY | KIND | VALUE | NEEDS | STATUS | PROGRESS |", "|---|---|---|---|---|---|---|"]
    for g in report["gates"]:
        thr = g["threshold"]
        L.append(f"| {g['gate']} | {g['category']} | {g['kind']} | {_fmt(g['value'])} | {g['operator']} "
                 f"{_fmt(thr) if isinstance(thr, float) else thr} | {g['status']} | {_fmt(g['progress'])} |")
    L += ["", "Why each threshold exists:", ""] + [f"- **{g['gate']}**: {g['why']}" for g in report["gates"]] + [""]
    L += ["## Dataset", "", "| METRIC | VALUE |", "|---|---|"]
    for k in ("total_observations", "signal_days", "unique_symbols", "unique_observation_dates", "unique_entry_days",
              "matured_observations_5d", "immature_observations", "failed_maturation_observations",
              "maturity_pct_observations", "matured_signal_days_5d", "pending_signal_days_5d",
              "matured_entry_days_5d", "matured_entry_weeks_5d", "matured_unique_symbols_5d",
              "effective_independent_samples_5d", "usable_walk_forward_folds_5d", "earliest_observation",
              "latest_observation", "latest_label_written"):
        L.append(f"| {k} | {_fmt(m.get(k))} |")
    L += ["", "## Labels by horizon (signal-day unit, matured + certified)", "",
          "| HORIZON | MATURED | POSITIVE | NEGATIVE | POSITIVE RATE | MINORITY | BEAT SPY (N) | FOLDS |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in report["horizons"].items():
        if "matured" not in v:
            L.append(f"| {k} | not collected | | | | | | |")
            continue
        L.append(f"| {k} | {v['matured']} | {v['positive']} | {v['negative']} | {_fmt(v['positive_rate'])} | "
                 f"{_fmt(v['minority_class_share'])} | {_fmt(v['beat_spy']['positive_rate'])} "
                 f"({v['beat_spy']['with_benchmark']}) | {report['walk_forward'][k]['usable_folds']} |")
    L += ["", "## Outcome maturation lifecycle (observation unit)", "", "| CATEGORY | OBSERVATIONS | SIGNAL-DAYS |",
          "|---|---|---|"]
    for c, n in mat["by_category_observations"].items():
        L.append(f"| {c} | {n} | {mat['by_category_signal_days'].get(c, 0)} |")
    L += ["", f"Recoverable by re-scoring: {mat['recoverable_observations']}. Recent window "
              f"({mat['recent_window_trading_days']} trading days): {mat['recent_labeled']} of {mat['recent_due']} due "
              f"observations labeled ({_fmt(m['recent_outcome_coverage_5d'])}); all time "
              f"{_fmt(m['all_time_outcome_coverage_5d'])}.", ""]
    L += ["## Collection", "", "| MEASURE | VALUE |", "|---|---|"]
    for k in ("observations", "snapshots", "trading_days_in_range", "trading_days_with_observations",
              "expected_observations_per_trading_day", "actual_observations_per_trading_day",
              "actual_signal_days_per_trading_day", "recent_observations_per_trading_day",
              "recent_signal_days_per_trading_day", "snapshots_off_schedule", "partial_snapshots",
              "non_trading_day_observations", "late_frozen_observations", "recorded_before_timestamp"):
        L.append(f"| {k} | {_fmt(col.get(k))} |")
    d = col.get("duplicates") or {}
    L.append(f"| duplicates (exact / same-day repeats / repeat rate / carry-over rate) | "
             f"{_fmt(d.get('exact_same_ticker_same_instant'))} / {_fmt(d.get('same_ticker_same_day_repeats'))} / "
             f"{_fmt(d.get('repeat_rate'))} / {_fmt(d.get('carry_over_rate'))} |")
    msw = col.get("missing_scan_windows") or {}
    L += ["", f"Expected basis: {col.get('expected_basis')}.", "",
          f"Missing scan windows: {msw.get('count')} of {msw.get('of')} slot-days.", "",
          "| SLOT (UTC) | TRADING DAYS | WITH SNAPSHOT |", "|---|---|---|"]
    for s, v in (msw.get("per_slot") or {}).items():
        L.append(f"| {s} | {v['expected']} | {v['with_snapshot']} |")
    if col.get("symbols_repeatedly_without_labels"):
        L += ["", "Symbols repeatedly without labels: " + ", ".join(
            f"{x['ticker']} ({x['rows']})" for x in col["symbols_repeatedly_without_labels"])]
    L += ["", "## Per trading day", "", "| DAY | OBSERVATIONS | SIGNAL-DAYS | SNAPSHOTS |", "|---|---|---|---|"]
    for row in col.get("per_trading_day") or []:
        L.append(f"| {row['day']} | {row['observations']} | {row['signal_days']} | {row['snapshots']} |")
    pit, fi = report["point_in_time"], report["feature_integrity"]
    L += ["", "## Point-in-time and feature integrity", "", "| CHECK | VALUE |", "|---|---|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in pit.items()]
    L += [f"| {k} | {_fmt(v) if not isinstance(v, dict) else v} |" for k, v in fi.items()]
    dims = report["dimensions"]
    L += ["", "## Distributions (signal-day unit)", ""]
    for k, v in dims.items():
        L.append(f"- **{k}**: {v}")
    c = report["concentration_5d"]
    L += ["", f"Concentration (matured 5d): {c['unique_symbols']} tickers, top-10 share {_fmt(c['top10_share'])}, "
              f"top ticker share {_fmt(c['top_symbol_share'])}.", ""]
    p = report["projection"]
    L += ["## Growth projection", ""]
    L.append(f"Basis: {p['basis']}")
    if not p["available"]:
        L += ["", f"Not projected: {p['reason']}."]
    else:
        L += ["", "| GATE | EST. TRADING DAYS | NOTE |", "|---|---|---|"]
        L += [f"| {g['gate']} | {_fmt(g['estimated_trading_days'])} | {g['note']} |" for g in p["gates"]]
        L += ["", f"Accumulating targets: ~{_fmt(p['estimate_trading_days_to_accumulating_targets'])} trading days. "
                  f"{p['disclaimer']}"]
    if alerts:
        L += ["", "## Monitor alerts", ""] + [f"- {a['severity']} {a['code']}: {a['message']}" for a in alerts]
    return "\n".join(L) + "\n"
