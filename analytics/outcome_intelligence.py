"""Outcome Intelligence: one canonical, benchmark-relative evidence layer (pure).

Answers "how have HSF signals actually performed?" from the existing canonical
outcome store only: frozen HSF Top-Opportunities in ``signal_outcomes``
(source='opportunity'), whose forward 1/3/5-trading-day returns and 5-day MFE/MAE
are written by ``analytics.signal_outcomes`` and whose SPY returns over the same
window are scored by the same function. Nothing here computes an outcome; it only
reads, classifies maturity, and aggregates. Every endpoint aggregates through
``metrics()`` so a metric can't be calculated two different ways.

Rules this module enforces:
  * Point-in-time features only: score, version, setup, signals, status and
    PreBreakout probability come from the frozen signal-time payload
    (``analytics.hsf_calibration.normalize_row``), never from a current scan.
  * Pending (and computed-but-empty "unavailable") outcomes never enter a metric.
  * No default selects a "best" anything: every filter is explicit and echoed.
  * Same-ticker same-day observations share one outcome (same entry close), so the
    default unit is one record per ticker per entry day (the day's FIRST
    observation, chosen by time, never by outcome). ``unit="observation"`` gives
    the raw rows; both counts are always reported.
  * Every aggregate carries its sample sizes and an evidence-quality label; the
    label never replaces the count.

Pure functions; Streamlit/DB free. Safe on empty input.
"""
from __future__ import annotations

import datetime as _dt
import math
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analytics.hsf_calibration import SCORE_BUCKETS, normalize_row

HORIZONS: Tuple[int, ...] = (1, 3, 5)          # trading days, as stored
DEFAULT_HORIZON = 5                            # the pre-declared primary horizon (hsf_calibration, track record)
MFE_MAE_HORIZONS: Tuple[int, ...] = (5,)       # only the 5-day window stores MFE/MAE
UNITS = ("signal_day", "observation")
DEFAULT_UNIT = "signal_day"
PERIODS = ("day", "week", "month")

# Evidence quality, keyed on MATURED sample size. Cut-offs are the repository's
# existing calibration thresholds (hsf_calibration.confidence_label: 10/30/100) so
# the product never shows two different confidence scales.
EVIDENCE_THRESHOLDS: Tuple[Tuple[int, str], ...] = ((100, "STRONG"), (30, "MODERATE"), (10, "LIMITED"))
INSUFFICIENT = "INSUFFICIENT"

PENDING, MATURED, UNAVAILABLE, INVALID = "pending", "matured", "unavailable", "invalid"

DISCLAIMER = ("Historical evidence from matured HSF signals. Returns are close-to-close with no costs or slippage, "
              "and are descriptive, not a forecast. Past outcomes do not guarantee future results.")

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- helpers
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_dt(v: Any) -> Optional[_dt.datetime]:
    if isinstance(v, _dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=_dt.timezone.utc)
    if isinstance(v, str) and v.strip():
        try:
            d = _dt.datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    return None


def evidence_quality(matured: int) -> str:
    """INSUFFICIENT <10, LIMITED 10-29, MODERATE 30-99, STRONG >=100 matured."""
    n = int(matured or 0)
    for floor, label in EVIDENCE_THRESHOLDS:
        if n >= floor:
            return label
    return INSUFFICIENT


def score_bucket(score: Optional[float], buckets: Sequence[Tuple[int, int]] = SCORE_BUCKETS) -> Optional[str]:
    if score is None:
        return None
    for lo, hi in buckets:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return None


def parse_buckets(spec: Optional[str]) -> List[Tuple[int, int]]:
    """'40-49,50-59' -> [(40, 49), (50, 59)]. Canonical buckets when empty.
    Raises ValueError on malformed, out-of-range or overlapping buckets."""
    if not spec or not str(spec).strip():
        return list(SCORE_BUCKETS)
    out: List[Tuple[int, int]] = []
    for part in str(spec).split(","):
        lo_s, sep, hi_s = part.strip().partition("-")
        if not sep:
            raise ValueError(f"bucket '{part.strip()}' must look like 60-69")
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            raise ValueError(f"bucket '{part.strip()}' must use whole numbers") from None
        if not (0 <= lo <= hi <= 100):
            raise ValueError(f"bucket '{part.strip()}' must be within 0-100, low to high")
        out.append((lo, hi))
    if len(out) > 20:
        raise ValueError("at most 20 buckets")
    out.sort()
    for (a_lo, a_hi), (b_lo, _b_hi) in zip(out, out[1:]):
        if b_lo <= a_hi:
            raise ValueError("buckets must not overlap")
    return out


# --------------------------------------------------------------------------- records
class _Slotted:
    """Small read-only record with dict-style access. Slots instead of dicts keep
    the resident dataset small (hsf-api runs in 512 MB)."""

    __slots__ = ()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class HorizonOutcome(_Slotted):
    __slots__ = ("status", "raw_return", "benchmark_return", "excess_return", "mfe", "mae")

    def __init__(self, status, raw_return, benchmark_return, excess_return, mfe, mae):
        self.status, self.raw_return, self.benchmark_return = status, raw_return, benchmark_return
        self.excess_return, self.mfe, self.mae = excess_return, mfe, mae


class OutcomeRecord(_Slotted):
    __slots__ = ("observation_id", "ticker", "observed_at", "observed_day", "entry_day", "hsf_score", "score_bucket",
                 "score_version", "setup", "signals", "status", "prebreakout_prob", "certified", "benchmark_symbol",
                 "entry_price", "outcome_price", "horizons", "observations_that_day")

    def __init__(self, **kw: Any):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))



def _intern(v: Any) -> Any:
    return sys.intern(v) if isinstance(v, str) else v


_SIGNAL_SETS: Dict[Tuple[str, ...], Tuple[str, ...]] = {}


def canonical_record(row: Dict[str, Any]) -> OutcomeRecord:
    """One frozen opportunity row -> the canonical outcome record.

    Signal-time fields come only from the frozen payload (via the existing
    hsf_calibration.normalize_row). Outcome fields come only from the outcome
    columns. Per-horizon status:
      pending      outcome not computed yet
      unavailable  computed, but no price outcome was available for this horizon
      invalid      outcome timestamp at/before the observation (lookahead guard)
      matured      computed, return present
    """
    base = normalize_row(row)
    observed = _to_dt(row.get("fired_at"))
    computed = _to_dt(row.get("outcome_computed_at"))
    entry_day = observed.date() if observed else None  # the outcome engine's own entry key
    horizons: Dict[int, HorizonOutcome] = {}
    for h in HORIZONS:
        raw = _num(row.get(f"return_{h}d"))
        bench = _num(row.get(f"benchmark_return_{h}d"))
        if computed is None:
            status = PENDING
        elif observed is not None and computed <= observed:
            status = INVALID
        elif raw is None:
            status = UNAVAILABLE
        else:
            status = MATURED
        ok = status == MATURED
        horizons[h] = HorizonOutcome(
            status,
            raw if ok else None,
            bench if ok else None,
            (raw - bench) if ok and bench is not None else None,
            _num(row.get("mfe_5d")) if ok and h in MFE_MAE_HORIZONS else None,
            _num(row.get("mae_5d")) if ok and h in MFE_MAE_HORIZONS else None,
        )
    score = base.get("hsf_score")
    sigs = tuple(sorted({str(s).strip().lower() for s in base.get("signals") or [] if str(s).strip()}))
    if len(_SIGNAL_SETS) < 4096:  # a handful of real combinations; bounded regardless
        sigs = _SIGNAL_SETS.setdefault(sigs, tuple(_intern(x) for x in sigs))
    return OutcomeRecord(
        observation_id=str(row.get("id")) if row.get("id") is not None else None,
        ticker=_intern(str(row.get("ticker") or "").strip().upper()),
        observed_at=observed,
        observed_day=observed.date().isoformat() if observed else None,
        entry_day=entry_day,
        hsf_score=score,
        score_bucket=_intern(score_bucket(score)),
        score_version=_intern(base.get("score_version")),
        setup=_intern(base.get("primary_setup")),
        signals=sigs,
        status=_intern(base.get("status")),
        prebreakout_prob=base.get("prob"),
        certified=_certified(base, row, observed, computed),
        benchmark_symbol=_intern(row.get("benchmark_symbol")),
        entry_price=None,     # not stored by the outcome engine (documented gap)
        outcome_price=None,
        horizons=horizons,
    )


def _certified(base: Dict[str, Any], row: Dict[str, Any], observed, computed) -> bool:
    """Matured AND passes the existing canonical eligibility rule for HSF
    opportunity observations (known score version, valid ticker/time/score)."""
    if computed is None or observed is None or computed <= observed:
        return False
    if _num(row.get("return_1d")) is None:
        return False
    score = base.get("hsf_score")
    if score is None or not (0 <= score <= 100):
        return False
    try:
        from analytics.opportunity_outcomes import is_eligible_opportunity_observation

        return bool(is_eligible_opportunity_observation({
            "snapshot_time": observed, "ticker": base.get("ticker"), "score": score,
            "status": base.get("status"), "score_version": base.get("score_version")}))
    except Exception:
        return False


def build_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonical records for every row, oldest first. Rows without a ticker or a
    timestamp can't be placed in time and are dropped (counted by the caller)."""
    out = [canonical_record(r) for r in rows or []]
    out = [r for r in out if r.ticker and r.observed_at is not None]
    out.sort(key=_time_key)
    per_day: Dict[Tuple[str, Any], int] = {}
    for r in out:
        per_day[(r.ticker, r.entry_day)] = per_day.get((r.ticker, r.entry_day), 0) + 1
    for r in out:  # frozen rows for this ticker on this entry day (whole dataset)
        r.observations_that_day = per_day[(r.ticker, r.entry_day)]
    return out


def dedupe_signal_days(records: Sequence[Any]) -> List[Any]:
    """One record per (ticker, entry day): the day's FIRST observation by time.
    All same-day copies share one entry close and so one outcome; counting each
    would weight a ticker by how many snapshots ran that day. Input order is kept
    (build_records sorts oldest first)."""
    kept: Dict[Tuple[str, Any], Any] = {}
    order = records if _is_sorted(records) else sorted(records, key=_time_key)
    for r in order:
        kept.setdefault((r.ticker, r.entry_day), r)
    return list(kept.values())


def _time_key(r: Any):
    return (r.observed_at, r.observation_id or "")


def _is_sorted(records: Sequence[Dict[str, Any]]) -> bool:
    return all(_time_key(a) <= _time_key(b) for a, b in zip(records, records[1:]))


# --------------------------------------------------------------------------- filters
FILTER_KEYS = ("ticker", "setup", "signal", "min_score", "max_score", "score_bucket", "score_version",
               "start_date", "end_date", "certified_only", "matured_only")


def normalize_filters(**kw: Any) -> Dict[str, Any]:
    """Every filter key, explicit, with None/False when not requested. Raises
    ValueError on contradictory input."""
    f: Dict[str, Any] = {k: None for k in FILTER_KEYS}
    f["certified_only"] = bool(kw.get("certified_only") or False)
    f["matured_only"] = bool(kw.get("matured_only") or False)
    for k in ("ticker", "setup", "signal", "score_bucket", "score_version"):
        v = kw.get(k)
        if v is not None and str(v).strip():
            f[k] = str(v).strip()
    if f["ticker"]:
        f["ticker"] = f["ticker"].upper()
    if f["signal"]:
        f["signal"] = f["signal"].lower()
    for k in ("min_score", "max_score"):
        v = _num(kw.get(k))
        if v is not None:
            if not 0 <= v <= 100:
                raise ValueError(f"{k} must be between 0 and 100")
            f[k] = v
    if f["min_score"] is not None and f["max_score"] is not None and f["min_score"] > f["max_score"]:
        raise ValueError("min_score must not exceed max_score")
    if f["score_bucket"]:
        try:
            ((lo, hi),) = parse_buckets(f["score_bucket"])
        except ValueError as e:
            raise ValueError(f"score_bucket: {e}") from None
        f["score_bucket"] = f"{lo}-{hi}"
    for k in ("start_date", "end_date"):
        v = kw.get(k)
        if isinstance(v, _dt.datetime):
            v = v.date()
        if isinstance(v, str) and v.strip():
            try:
                v = _dt.date.fromisoformat(v.strip())
            except ValueError:
                raise ValueError(f"{k} must be YYYY-MM-DD") from None
        f[k] = v.isoformat() if isinstance(v, _dt.date) else None
    if f["start_date"] and f["end_date"] and f["start_date"] > f["end_date"]:
        raise ValueError("start_date must not be after end_date")
    return f


def _match(r: Dict[str, Any], f: Dict[str, Any], horizon: Optional[int]) -> bool:
    if f["ticker"] and r.ticker != f["ticker"]:
        return False
    if f["setup"] and str(r.setup or "").lower() != f["setup"].lower():
        return False
    if f["signal"] and f["signal"] not in r.signals:
        return False
    s = r.hsf_score
    if f["min_score"] is not None and (s is None or s < f["min_score"]):
        return False
    if f["max_score"] is not None and (s is None or s > f["max_score"]):
        return False
    if f["score_bucket"]:
        lo, hi = (int(x) for x in f["score_bucket"].split("-"))
        if s is None or not lo <= s <= hi:
            return False
    if f["score_version"] and str(r.get("score_version") or "") != f["score_version"]:
        return False
    day = r.observed_day
    if f["start_date"] and day < f["start_date"]:
        return False
    if f["end_date"] and day > f["end_date"]:
        return False
    if f["certified_only"] and not r.certified:
        return False
    if f["matured_only"]:
        hs = [horizon] if horizon is not None else list(HORIZONS)
        if not any(r.horizons[h].status == MATURED for h in hs):
            return False
    return True


def select(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, unit: str = DEFAULT_UNIT,
           horizon: Optional[int] = None) -> List[Dict[str, Any]]:
    """Filter first (on point-in-time fields), then collapse to the unit. Filtering
    before collapsing means a filter on score picks the day's first observation
    that matches it, and the reported raw count is the filtered raw count."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}")
    chosen = [r for r in records if _match(r, filters, horizon)]
    return dedupe_signal_days(chosen) if unit == "signal_day" else list(chosen)


# --------------------------------------------------------------------------- metrics
def _r(v: Optional[float], nd: int = 6) -> Optional[float]:
    return None if v is None else round(v, nd)


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def wilson_interval(successes: int, n: int, z: float = Z95) -> Optional[Tuple[float, float]]:
    """95% Wilson score interval for a proportion. Assumes independent trials."""
    if n <= 0:
        return None
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_interval(xs: List[float], z: float = Z95) -> Optional[Tuple[float, float]]:
    """Normal-approximation 95% interval for a mean; only reported from 30
    observations (below that the approximation isn't trustworthy)."""
    if len(xs) < 30:
        return None
    m, sd = statistics.fmean(xs), statistics.stdev(xs)
    half = z * sd / math.sqrt(len(xs))
    return (m - half, m + half)


def _ci(pair: Optional[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    return None if pair is None else {"low": round(pair[0], 6), "high": round(pair[1], 6)}


def metrics(records: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    """The ONE aggregation every endpoint uses, for one horizon.

    Counts cover every record in the group; return/benchmark/MFE metrics use only
    MATURED records (pending, unavailable and invalid never enter a metric).
    win_rate = share of matured with raw_return > 0; benchmark_beat_rate = share of
    matured-with-benchmark with excess_return > 0. Each rate's own denominator is
    returned beside it.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {', '.join(map(str, HORIZONS))}")
    hs = [r.horizons[horizon] for r in records]
    by_status = {s: sum(1 for x in hs if x.status == s) for s in (MATURED, PENDING, UNAVAILABLE, INVALID)}
    mat = [x for x in hs if x.status == MATURED]
    raw = [x.raw_return for x in mat]
    bench_pairs = [x for x in mat if x.benchmark_return is not None]
    bench = [x.benchmark_return for x in bench_pairs]
    excess = [x.excess_return for x in bench_pairs]
    mfe = [x.mfe for x in mat if x.mfe is not None]
    mae = [x.mae for x in mat if x.mae is not None]
    wins = sum(1 for v in raw if v > 0)
    beats = sum(1 for v in excess if v > 0)
    days = {r.entry_day for r, x in zip(records, hs) if x.status == MATURED}
    n = len(mat)
    return {
        "horizon": horizon,
        "sample_size": len(records),
        "matured_count": n,
        "pending_count": by_status[PENDING],
        "unavailable_count": by_status[UNAVAILABLE],
        "invalid_count": by_status[INVALID],
        "distinct_days": len(days),
        "evidence_quality": evidence_quality(n),
        "average_return": _r(_mean(raw)),
        "median_return": _r(_median(raw)),
        "average_return_ci95": _ci(mean_interval(raw)),
        "win_count": wins,
        "win_rate": _r(wins / n if n else None, 4),
        "win_rate_ci95": _ci(wilson_interval(wins, n)),
        "benchmark_count": len(bench_pairs),
        "average_benchmark_return": _r(_mean(bench)),
        "median_benchmark_return": _r(_median(bench)),
        "average_excess_return": _r(_mean(excess)),
        "median_excess_return": _r(_median(excess)),
        "benchmark_beat_count": beats,
        "benchmark_beat_rate": _r(beats / len(excess) if excess else None, 4),
        "benchmark_beat_rate_ci95": _ci(wilson_interval(beats, len(excess))),
        "mfe_count": len(mfe),
        "average_mfe": _r(_mean(mfe)),
        "median_mfe": _r(_median(mfe)),
        "mae_count": len(mae),
        "average_mae": _r(_mean(mae)),
        "median_mae": _r(_median(mae)),
    }


def coverage(records: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    """How complete the evidence is: matured records missing a benchmark or MFE/MAE."""
    mat = [r.horizons[horizon] for r in records if r.horizons[horizon]["status"] == MATURED]
    n = len(mat)
    miss_b = sum(1 for x in mat if x.benchmark_return is None)
    mfe_applicable = horizon in MFE_MAE_HORIZONS
    miss_m = sum(1 for x in mat if x.mfe is None or x.mae is None) if mfe_applicable else n
    return {
        "matured": n,
        "missing_benchmark": miss_b,
        "benchmark_coverage": _r((n - miss_b) / n if n else None, 4),
        "mfe_mae_available_for_horizon": mfe_applicable,
        "missing_mfe_mae": miss_m,
        "mfe_mae_coverage": _r((n - miss_m) / n if n else None, 4),
    }


def date_range(records: Sequence[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    ts = [r["observed_at"] for r in records]
    return {"start": min(ts).isoformat() if ts else None, "end": max(ts).isoformat() if ts else None}


def version_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        k = str(r.get("score_version") or "unknown")
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def score_distribution(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores = sorted(r["hsf_score"] for r in records if r.get("hsf_score") is not None)
    counts = {f"{lo}-{hi}": sum(1 for s in scores if lo <= s <= hi) for lo, hi in SCORE_BUCKETS}
    return {"scored": len(scores), "unscored": len(records) - len(scores),
            "min": scores[0] if scores else None, "max": scores[-1] if scores else None,
            "median": _r(_median(scores), 2), "bucket_counts": counts}


# --------------------------------------------------------------------------- views
def _envelope(records_raw: Sequence[Dict[str, Any]], chosen: Sequence[Dict[str, Any]], filters: Dict[str, Any],
              unit: str, horizon: Optional[int]) -> Dict[str, Any]:
    versions = version_counts(chosen)
    warnings: List[str] = []
    if len(versions) > 1 and not filters.get("score_version"):
        warnings.append("Mixed HSF score versions in this sample; filter by score_version to compare like with like.")
    return {
        "filters": dict(filters),
        "unit": unit,
        "horizon": horizon,
        "raw_observations": sum(1 for r in records_raw if _match(r, filters, horizon)),
        "date_range": date_range(chosen),
        "score_versions": versions,
        "warnings": warnings,
        "evidence_thresholds": {label: floor for floor, label in EVIDENCE_THRESHOLDS},
        "disclaimer": DISCLAIMER,
    }


def summary(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, horizon: int = DEFAULT_HORIZON,
            unit: str = DEFAULT_UNIT) -> Dict[str, Any]:
    chosen = select(records, filters, unit=unit, horizon=horizon)
    m = metrics(chosen, horizon)
    return {
        **_envelope(records, chosen, filters, unit, horizon),
        "total_observations": len(chosen),
        "matured_observations": m["matured_count"],
        "pending_observations": m["pending_count"],
        "unavailable_observations": m["unavailable_count"],
        "certified_observations": sum(1 for r in chosen if r["certified"]),
        "metrics": m,
        "coverage": coverage(chosen, horizon),
    }


def by_score(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, horizon: int = DEFAULT_HORIZON,
             unit: str = DEFAULT_UNIT, buckets: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """Every bucket, always, in score order (empty and weak buckets included)."""
    bks = list(buckets or SCORE_BUCKETS)
    chosen = select(records, filters, unit=unit, horizon=horizon)
    rows = []
    for lo, hi in bks:
        grp = [r for r in chosen if r.get("hsf_score") is not None and lo <= r["hsf_score"] <= hi]
        rows.append({"bucket": f"{lo}-{hi}", "min_score": lo, "max_score": hi, **metrics(grp, horizon)})
    unbucketed = [r for r in chosen if score_bucket(r.get("hsf_score"), bks) is None]
    return {**_envelope(records, chosen, filters, unit, horizon),
            "buckets": rows,
            "unbucketed_count": len(unbucketed),
            "calibration": calibration_view(rows)}


# Metrics checked for monotonicity across score buckets (higher score -> better).
MONOTONIC_METRICS = ("win_rate", "median_return", "median_excess_return", "benchmark_beat_rate", "median_mfe")


def calibration_view(bucket_rows: Sequence[Dict[str, Any]], *, min_quality: str = "LIMITED") -> Dict[str, Any]:
    """Is each metric non-decreasing as score rises? Only buckets with at least
    LIMITED evidence are compared; every inversion (a lower bucket beating the
    next higher one) is listed. Exposes evidence, fixes nothing."""
    floor = dict((label, f) for f, label in EVIDENCE_THRESHOLDS).get(min_quality, 10)
    out: Dict[str, Any] = {"min_matured_per_bucket": floor, "metrics": {}}
    for key in MONOTONIC_METRICS:
        n_key = "benchmark_count" if key in ("median_excess_return", "benchmark_beat_rate") else (
            "mfe_count" if key == "median_mfe" else "matured_count")
        pts = [(b["bucket"], b[key]) for b in bucket_rows if b.get(key) is not None and b.get(n_key, 0) >= floor]
        inversions = [{"lower_bucket": a[0], "higher_bucket": b[0], "lower_value": a[1], "higher_value": b[1]}
                      for a, b in zip(pts, pts[1:]) if a[1] > b[1]]
        out["metrics"][key] = {
            "buckets_compared": [p[0] for p in pts],
            "monotonic": (not inversions) if len(pts) >= 2 else None,
            "inversions": inversions,
        }
    return out


def by_horizon(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *,
               unit: str = DEFAULT_UNIT) -> Dict[str, Any]:
    """Every supported horizon, in horizon order. No horizon is singled out."""
    rows = []
    chosen_any: List[Dict[str, Any]] = []
    shared = None if filters.get("matured_only") else select(records, filters, unit=unit)
    for h in HORIZONS:
        chosen = shared if shared is not None else select(records, filters, unit=unit, horizon=h)
        chosen_any = chosen if len(chosen) > len(chosen_any) else chosen_any
        rows.append({**metrics(chosen, h), "coverage": coverage(chosen, h)})
    return {**_envelope(records, chosen_any, filters, unit, None), "horizons": rows}


def by_group(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, horizon: int = DEFAULT_HORIZON,
             unit: str = DEFAULT_UNIT, group_by: str = "setup") -> Dict[str, Any]:
    """Per canonical setup (primary_setup) or per signal label. Ordered by sample
    size, then name; never by performance. A record carries several signals, so
    signal groups overlap (stated in the response)."""
    if group_by not in ("setup", "signal"):
        raise ValueError("group_by must be setup or signal")
    chosen = select(records, filters, unit=unit, horizon=horizon)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in chosen:
        keys = [r.get("setup") or "unlabeled"] if group_by == "setup" else (r["signals"] or ["none"])
        for k in keys:
            groups.setdefault(str(k), []).append(r)
    rows = []
    for name, grp in groups.items():
        supported = [h for h in HORIZONS if any(r["horizons"][h]["status"] == MATURED for r in grp)]
        rows.append({"name": name, **metrics(grp, horizon), "score_distribution": score_distribution(grp),
                     "supported_horizons": supported})
    rows.sort(key=lambda x: (-x["sample_size"], x["name"]))
    return {**_envelope(records, chosen, filters, unit, horizon), "group_by": group_by,
            "groups_overlap": group_by == "signal", "groups": rows}


def _period_key(d: _dt.date, period: str) -> str:
    if period == "day":
        return d.isoformat()
    if period == "week":
        iso = d.isocalendar()
        return (d - _dt.timedelta(days=iso[2] - 1)).isoformat()  # Monday of the ISO week
    return d.replace(day=1).isoformat()


def timeseries(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, period: str = "week",
               horizon: int = DEFAULT_HORIZON, unit: str = DEFAULT_UNIT) -> Dict[str, Any]:
    """Every period that has observations, oldest first (gaps are simply absent)."""
    if period not in PERIODS:
        raise ValueError("period must be day, week or month")
    chosen = select(records, filters, unit=unit, horizon=horizon)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in chosen:
        groups.setdefault(_period_key(r["observed_at"].date(), period), []).append(r)
    points = []
    for key in sorted(groups):
        m = metrics(groups[key], horizon)
        points.append({"period_start": key, "observation_count": m["sample_size"],
                       **{k: m[k] for k in ("matured_count", "pending_count", "evidence_quality", "average_return",
                                            "median_return", "average_excess_return", "median_excess_return",
                                            "win_rate", "benchmark_beat_rate", "benchmark_count")}})
    return {**_envelope(records, chosen, filters, unit, horizon), "period": period, "points": points}


def observation_view(r: Dict[str, Any]) -> Dict[str, Any]:
    """Public shape of one canonical record (observation level)."""
    return {
        "observation_id": r["observation_id"],
        "ticker": r["ticker"],
        "observed_at": r["observed_at"].isoformat() if r["observed_at"] else None,
        "entry_day": r["entry_day"].isoformat() if r["entry_day"] else None,
        "hsf_score": r["hsf_score"],
        "score_bucket": r["score_bucket"],
        "score_version": r["score_version"],
        "setup": r["setup"],
        "signals": list(r["signals"]),
        "status": r["status"],
        "prebreakout_prob": r["prebreakout_prob"],
        "certified": r["certified"],
        "benchmark_symbol": r["benchmark_symbol"],
        "entry_price": r["entry_price"],
        "outcome_price": r["outcome_price"],
        "observations_that_day": r.get("observations_that_day"),
        "outcomes": [{"horizon": h, **{k: (_r(v) if isinstance(v, float) else v)
                                       for k, v in r["horizons"][h].as_dict().items()}}
                     for h in HORIZONS],
    }


def page_of(chosen: Sequence[Dict[str, Any]], page: int, page_size: int) -> Dict[str, Any]:
    """Newest first, 1-based pages."""
    newest = sorted(chosen, key=lambda r: (r["observed_at"], r["observation_id"] or ""), reverse=True)
    start = (max(1, int(page)) - 1) * int(page_size)
    return {"page": max(1, int(page)), "page_size": int(page_size), "total": len(newest),
            "items": [observation_view(r) for r in newest[start:start + int(page_size)]]}


def query(records: Sequence[Dict[str, Any]], filters: Dict[str, Any], *, horizon: int = DEFAULT_HORIZON,
          unit: str = DEFAULT_UNIT, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    chosen = select(records, filters, unit=unit, horizon=horizon)
    return {**_envelope(records, chosen, filters, unit, horizon), "metrics": metrics(chosen, horizon),
            "coverage": coverage(chosen, horizon), "observations": page_of(chosen, page, page_size)}


def symbol(records: Sequence[Dict[str, Any]], ticker: str, filters: Dict[str, Any], *,
           horizon: Optional[int] = None, unit: str = DEFAULT_UNIT, page: int = 1,
           page_size: int = 50) -> Dict[str, Any]:
    """A ticker's history: aggregate per horizon (or the one requested) + pages."""
    f = {**filters, "ticker": str(ticker).strip().upper()}
    hs = [horizon] if horizon is not None else list(HORIZONS)
    records = [r for r in records if r["ticker"] == f["ticker"]]
    chosen = select(records, f, unit=unit, horizon=horizon)
    return {**_envelope(records, chosen, f, unit, horizon), "ticker": f["ticker"],
            "horizons": [{**metrics(select(records, f, unit=unit, horizon=h), h)} for h in hs],
            "observations": page_of(chosen, page, page_size)}


def ai_evidence_text(view: Dict[str, Any]) -> Optional[str]:
    """Plain-text evidence block for the AI layer from a symbol() view: sample
    size, filters, date range and benchmark-relative metrics for EVERY horizon
    (never a best subset). None when nothing has matured."""
    rows = [h for h in view.get("horizons") or [] if h.get("matured_count")]
    if not rows:
        return None
    dr = view.get("date_range") or {}
    active = {k: v for k, v in (view.get("filters") or {}).items() if v not in (None, False) and k != "ticker"}

    def pct(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v * 100:+.2f}%"

    def share(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v * 100:.0f}%"

    lines = [f"Historical HSF evidence for {view.get('ticker')} (unit: {view.get('unit')}; "
             f"filters: {active or 'none'}; observed {str(dr.get('start'))[:10]} to {str(dr.get('end'))[:10]}; "
             f"benchmark SPY, same window):"]
    for h in rows:
        lines.append(
            f"- {h['horizon']} trading days: {h['matured_count']} matured ({h['evidence_quality']} evidence), "
            f"{h['pending_count']} pending; median return {pct(h['median_return'])}, "
            f"median excess vs SPY {pct(h['median_excess_return'])} (n={h['benchmark_count']}), "
            f"win rate {share(h['win_rate'])}, beat SPY {share(h['benchmark_beat_rate'])}"
            + (f", avg MFE {pct(h['average_mfe'])}, avg MAE {pct(h['average_mae'])}" if h.get("mfe_count") else ""))
    lines.append("This is descriptive history of past signals, not a forecast or guarantee.")
    return "\n".join(lines)


