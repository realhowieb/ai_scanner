"""Run 55 — research evidence & signal effectiveness (pure, read-only).

Measures what the EXISTING scanner predicts, using production research
observations (Run 47 cohorts) and their matured outcomes. It never changes
scoring, ranking, tiers, cohorts, outcomes, or any stored record: it only reads
already-loaded dicts and returns a report.

Every decision rule below lives in `CRITERIA` and the verdict functions, and was
fixed before the production results were viewed (see
docs/RUN55_SIGNAL_EFFECTIVENESS.md). Nothing is tuned against this dataset.

Measurement conventions (pre-declared):
  * Primary population: explicitly tagged (Run 47+) observations with MATURED
    outcomes, anchored in the regular session (09:30-16:00 ET, weekdays). Legacy
    untagged rows are counted in readiness and excluded from every analysis.
  * If a (scan run, symbol) pair carries more than one cohort, the highest-priority
    cohort is kept (CANDIDATE > NEAR_MISS > CONTROL) and the others are dropped.
  * directional_return: the stored value when present, otherwise derived from
    raw_return and the observation's direction (LONG = raw, SHORT = -raw). CONTROL
    rows carry no direction and are measured long, the convention the maturation
    worker already uses. A win is directional_return > 0.
  * Returns are winsorized per horizon at the pooled 0.5th/99.5th percentiles of
    the primary population, so one micro-cap spike cannot create or erase an edge.
    Unwinsorized means are also reported.
  * Cohort differences and score correlations get 95% CIs from a cluster
    bootstrap over scan runs (observations in one run share market moves).
    Per-group CIs are i.i.d. approximations (Wilson for win rate, normal for the
    mean) and are likely too narrow; they are descriptive.
  * MFE/MAE: only +60m records are used. The worker computes excursions over the
    largest horizon matured in the same batch, so on shorter-horizon records the
    window varies; +60m is the only horizon whose window is always 60 bars.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS
from analytics.scanner_performance import wilson_interval

SCHEMA = "hsf-run55-signal-effectiveness-1.0"
HORIZONS = ("+5m", "+15m", "+30m", "+60m")
PRIMARY_HORIZON = "+60m"
COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)
_COHORT_PRIORITY = {CANDIDATE: 0, NEAR_MISS: 1, CONTROL: 2}

# ---- Pre-declared criteria (fixed before viewing production results) --------
CRITERIA: Dict[str, Any] = {
    "min_n_any": 30,              # below this nothing is estimated as evidence
    "min_n_moderate": 100,        # per arm, for MODERATE_EVIDENCE
    "min_n_strong": 300,          # per arm, for STRONG_EVIDENCE
    "min_clusters": 10,           # distinct scan runs behind a comparison
    "min_clusters_strong": 20,
    "meaningful_effect": 0.0025,  # 25 bp directional-return difference
    "edge_min_horizons": 3,       # of 4: a single lucky horizon cannot pass
    "no_edge_min_powered_horizons": 3,
    "winsor_pct": 0.5,            # two-sided, pooled per horizon
    "bootstrap_resamples": 1000,
    "bootstrap_resamples_feature": 500,
    "seed": 55,
    "min_bucket_n": 30,
    "monotonic_min_scored_n": 100,
    "monotonic_no_min_scored_n": 300,
    "reversal_mfe": 0.01,         # "moved strongly": +60m MFE >= 1%
    "regime_min_presence": 0.8,
    "score_buckets_fixed": ["<50", "50-59", "60-69", "70-79", "80-89", "90-100"],
    "primary_horizon": "+60m",
    "primary_population": ("explicitly tagged observations, regular-session anchors "
                           "(09:30-16:00 ET), MATURED outcomes, cohort overlap deduped"),
    "readiness": {
        "min_modern_matured_observations": 300,
        "min_cohort_primary_n": 30,
        "cohort_primary_n_conditional": 100,
        "min_scan_runs": 20,
        "stored_directional_coverage": 0.95,
        "excursion_coverage": 0.95,
        "max_invalid_rate": 0.01,
        "min_short_n": 30,
    },
}
FIXED_SCORE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (-math.inf, 50.0, "<50"), (50.0, 60.0, "50-59"), (60.0, 70.0, "60-69"),
    (70.0, 80.0, "70-79"), (80.0, 90.0, "80-89"), (90.0, math.inf, "90-100"),
)
TOD_BUCKETS: Tuple[Tuple[int, int, str], ...] = (
    (0, 570, "PRE (<09:30)"), (570, 630, "09:30-10:30"), (630, 720, "10:30-12:00"),
    (720, 840, "12:00-14:00"), (840, 900, "14:00-15:00"), (900, 960, "15:00-16:00"),
    (960, 1440, "POST (>=16:00)"),
)
NUMERIC_FEATURES = ("score", "rvol", "gap_pct", "chg_pct", "atr_pct", "price",
                    "volume", "dollar_volume", "trigger_count")
FLAG_FEATURES = ("is_breakout", "gap_up", "gap_down", "unusual_vol", "momentum")
# Requested diagnostics the scheduled capture does not persist (Run 53B contract).
UNAVAILABLE_FEATURES = {
    "rsi": "not captured by the scheduled scan observation",
    "ema_alignment": "not captured",
    "ema9_ema21": "ema_cross is computed by scan/breakout.py but not mapped into the observation",
    "breakout_distance": "BreakoutPos20D is not mapped into the observation",
    "confirmation_count / agreement": "Day Trader intel is computed at render time, never persisted",
    "conflict_count / conflict flags": "Day Trader conflicts are computed at render time, never persisted",
    "prebreakout": "scheduled observations carry no PreBreakout score",
    "vwap / adx / supertrend / ewo": "not in the scheduled breakout result frame",
}


# ---- Parsing helpers ---------------------------------------------------------
def _num(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _parse_dt(v: Any) -> Optional[_dt.datetime]:
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _et(d: _dt.datetime) -> _dt.datetime:
    try:
        from zoneinfo import ZoneInfo
        return d.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - tzdata missing
        return d.astimezone(_dt.timezone(_dt.timedelta(hours=-4)))


def is_explicit(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("research_cohort")
                or (rec.get("market_context") or {}).get("research_cohort"))


def cohort(rec: Dict[str, Any]) -> Optional[str]:
    c = rec.get("research_cohort") or (rec.get("market_context") or {}).get("research_cohort")
    return str(c).upper() if c else None


def direction(rec: Dict[str, Any]) -> Optional[str]:
    """First scanner direction, normalized to LONG/SHORT (None when absent)."""
    for s in rec.get("scanners") or []:
        v = str((s or {}).get("direction") or "").strip().lower()
        if v:
            return "SHORT" if v in {"short", "bearish", "sell"} else "LONG"
    return None


def scan_run(rec: Dict[str, Any]) -> str:
    mc = rec.get("market_context") or {}
    return str(mc.get("scan_id") or rec.get("scan_timestamp") or rec.get("timestamp") or "")


def anchor(rec: Dict[str, Any]) -> Optional[_dt.datetime]:
    return _parse_dt(rec.get("scan_timestamp") or rec.get("timestamp"))


def tod_bucket(d: Optional[_dt.datetime]) -> Optional[str]:
    if d is None:
        return None
    e = _et(d)
    minute = e.hour * 60 + e.minute
    for lo, hi, label in TOD_BUCKETS:
        if lo <= minute < hi:
            return label
    return None


def is_regular_session(d: Optional[_dt.datetime]) -> bool:
    if d is None:
        return False
    e = _et(d)
    minute = e.hour * 60 + e.minute
    return e.weekday() < 5 and 570 <= minute < 960


def features(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Point-in-time features actually persisted on the observation."""
    mkt = rec.get("market") or {}
    ind = rec.get("indicators") or {}
    scanners = [s for s in (rec.get("scanners") or []) if isinstance(s, dict)]
    names = {str(s.get("name")) for s in scanners if s.get("triggered", True)}
    breakout = next((s for s in scanners if s.get("name") == "breakout"), None)
    score = _num((breakout or {}).get("score"))
    if score is None:
        score = next((_num(s.get("score")) for s in scanners if _num(s.get("score")) is not None), None)
    price, volume = _num(mkt.get("price")), _num(mkt.get("volume"))
    meta = (breakout or {}).get("meta") or {}
    return {
        "score": score,
        "rvol": _num(ind.get("rvol")),
        "gap_pct": _num(ind.get("gap_pct")),
        "chg_pct": _num(ind.get("chg_pct")),
        "atr_pct": _num(ind.get("atr_pct")),
        "price": price,
        "volume": volume,
        "dollar_volume": price * volume if price is not None and volume is not None else None,
        "trigger_count": len(names) if scanners else None,
        "is_breakout": bool(meta.get("is_breakout")) if breakout is not None else None,
        "gap_up": ("gap_up" in names) if scanners else None,
        "gap_down": ("gap_down" in names) if scanners else None,
        "unusual_vol": ("unusual_vol" in names) if scanners else None,
        "momentum": ("momentum" in names) if scanners else None,
    }


def tier(rec: Dict[str, Any]) -> Optional[str]:
    """Production Day Trader setup tier, if a record ever carries it."""
    for src in (rec.get("dt_intel"), rec.get("intelligence"), rec.get("market_context")):
        if isinstance(src, dict):
            q = src.get("quality") or src.get("setup_quality") or src.get("tier")
            if q:
                return str(q).upper()
    return None


# ---- Record building ---------------------------------------------------------
def build_records(observations: Sequence[Dict[str, Any]],
                  outcomes_by_id: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Flatten (observation x MATURED horizon) into analysis records.

    Returns {"records": [...], "dropped_overlap": n, "transform": {...}}. Only
    explicitly tagged observations produce records; legacy rows never do."""
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped = 0
    for rec in observations or []:
        if not rec or not is_explicit(rec):
            continue
        c = cohort(rec)
        if c not in _COHORT_PRIORITY:
            continue
        key = (scan_run(rec), str(rec.get("symbol") or "").upper())
        prev = best.get(key)
        if prev is None:
            best[key] = rec
        else:
            dropped += 1
            if _COHORT_PRIORITY[c] < _COHORT_PRIORITY[cohort(prev)]:
                best[key] = rec

    records: List[Dict[str, Any]] = []
    transform = {"checked": 0, "mismatches": 0, "stored": 0, "derived": 0}
    for rec in best.values():
        oid = str(rec.get("observation_id") or "")
        a = anchor(rec)
        d = direction(rec)
        feats = features(rec)
        seen_h = set()
        for oc in outcomes_by_id.get(oid) or []:
            h = oc.get("horizon")
            if h not in HORIZONS or h in seen_h or str(oc.get("data_status")) != "MATURED":
                continue
            raw = _num(oc.get("raw_return"))
            if raw is None:
                continue
            seen_h.add(h)
            stored = _num(oc.get("directional_return"))
            if stored is not None:
                transform["stored"] += 1
                if d is not None:
                    transform["checked"] += 1
                    expected = -raw if d == "SHORT" else raw
                    if abs(stored - expected) > 1e-6:
                        transform["mismatches"] += 1
                dr, src = stored, "stored"
            else:
                transform["derived"] += 1
                dr, src = (-raw if d == "SHORT" else raw), "derived"
            records.append({
                "observation_id": oid, "symbol": str(rec.get("symbol") or "").upper(),
                "cohort": cohort(rec), "scan_run": scan_run(rec), "anchor": a,
                "regular": is_regular_session(a), "tod": tod_bucket(a),
                "direction": d or "LONG", "direction_known": d is not None,
                "horizon": h, "raw": raw, "dr": dr, "dr_source": src,
                "mfe": _num(oc.get("mfe")), "mae": _num(oc.get("mae")),
                "features": feats, "tier": tier(rec),
                "regime": (rec.get("market_context") or {}).get("market_regime"),
            })
    return {"records": records, "dropped_overlap": dropped, "transform": transform}


def winsor_limits(records: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    pct = CRITERIA["winsor_pct"]
    out = {}
    for h in HORIZONS:
        vals = [r["dr"] for r in records if r["horizon"] == h]
        if len(vals) >= 2:
            lo, hi = np.percentile(vals, [pct, 100 - pct])
            out[h] = (float(lo), float(hi))
    return out


def apply_winsor(records: List[Dict[str, Any]], limits: Dict[str, Tuple[float, float]]) -> None:
    for r in records:
        lo, hi = limits.get(r["horizon"], (-math.inf, math.inf))
        r["drw"] = min(max(r["dr"], lo), hi)


# ---- Statistics ----------------------------------------------------------------
def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def summarize(rows: Sequence[Dict[str, Any]], *, excursions: bool = True) -> Dict[str, Any]:
    """Descriptive stats over records (drw = winsorized directional return)."""
    vals = np.array([r["drw"] for r in rows], dtype=float)
    n = int(vals.size)
    out: Dict[str, Any] = {"n": n}
    if n == 0:
        out.update({"win_rate": None, "win_rate_ci": [None, None], "mean": None,
                    "mean_ci": [None, None], "mean_unwinsorized": None, "median": None,
                    "sd": None, "p25": None, "p75": None, "payoff_ratio": None})
    else:
        wins = int((vals > 0).sum())
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else None
        half = 1.96 * sd / math.sqrt(n) if sd is not None else None
        pos, neg = vals[vals > 0], vals[vals < 0]
        payoff = (float(pos.mean()) / abs(float(neg.mean()))
                  if pos.size and neg.size else None)
        out.update({
            "win_rate": _r(wins / n, 4), "win_rate_ci": list(wilson_interval(wins, n)),
            "mean": _r(mean), "mean_ci": [_r(mean - half), _r(mean + half)] if half is not None else [None, None],
            "mean_unwinsorized": _r(float(np.mean([r["dr"] for r in rows]))),
            "median": _r(float(np.median(vals))), "sd": _r(sd),
            "p25": _r(float(np.percentile(vals, 25))), "p75": _r(float(np.percentile(vals, 75))),
            "payoff_ratio": _r(payoff, 3),
        })
    if excursions:
        out.update(excursion_stats(rows))
    return out


def excursion_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mfe = np.array([r["mfe"] for r in rows if r.get("mfe") is not None], dtype=float)
    mae = np.array([r["mae"] for r in rows if r.get("mae") is not None], dtype=float)
    ratio = (float(mfe.mean()) / abs(float(mae.mean()))
             if mfe.size and mae.size and mae.mean() != 0 else None)
    return {
        "excursion_n": int(min(mfe.size, mae.size)),
        "mfe_mean": _r(float(mfe.mean())) if mfe.size else None,
        "mfe_median": _r(float(np.median(mfe))) if mfe.size else None,
        "mae_mean": _r(float(mae.mean())) if mae.size else None,
        "mae_median": _r(float(np.median(mae))) if mae.size else None,
        "mfe_mae_ratio": _r(ratio, 3),
    }


def _cluster_weights(clusters: Sequence[str], B: int, seed: int):
    uniq = sorted(set(clusters))
    idx = {c: i for i, c in enumerate(uniq)}
    rng = np.random.default_rng(seed)
    C = len(uniq)
    W = rng.multinomial(C, [1.0 / C] * C, size=B) if C else np.zeros((B, 0))
    return idx, W


def cluster_bootstrap_diff(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]], *,
                           B: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """95% CI for mean(a)-mean(b) and win(a)-win(b), resampling scan runs."""
    B = B or CRITERIA["bootstrap_resamples"]
    seed = CRITERIA["seed"] if seed is None else seed
    idx, W = _cluster_weights([r["scan_run"] for r in list(a) + list(b)], B, seed)
    C = len(idx)
    if C < 2 or not a or not b:
        return {"diff_mean_ci": [None, None], "diff_win_ci": [None, None]}

    def sums(rows):
        s, w, n = np.zeros(C), np.zeros(C), np.zeros(C)
        for r in rows:
            i = idx[r["scan_run"]]
            s[i] += r["drw"]
            w[i] += 1.0 if r["drw"] > 0 else 0.0
            n[i] += 1.0
        return s, w, n
    sa, wa, na = sums(a)
    sb, wb, nb = sums(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = (W @ sa) / (W @ na) - (W @ sb) / (W @ nb)
        dw = (W @ wa) / (W @ na) - (W @ wb) / (W @ nb)
    dm, dw = dm[np.isfinite(dm)], dw[np.isfinite(dw)]
    ci = (lambda x: [_r(float(np.percentile(x, 2.5))), _r(float(np.percentile(x, 97.5)))]
          if x.size else [None, None])
    return {"diff_mean_ci": ci(dm), "diff_win_ci": ci(dw)}


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties (vectorized, same result as scipy's 'average')."""
    n = x.size
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], n]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.size < 3:
        return None
    return _corr(_rankdata(xa), _rankdata(ya))


def correlation_with_ci(rows: Sequence[Dict[str, Any]], key, *, B: Optional[int] = None,
                        seed: Optional[int] = None) -> Dict[str, Any]:
    """Spearman (cluster-bootstrap CI) + Pearson between a feature and drw."""
    B = B or CRITERIA["bootstrap_resamples"]
    seed = CRITERIA["seed"] if seed is None else seed
    pts = [(key(r), r["drw"], r["scan_run"]) for r in rows if key(r) is not None]
    n = len(pts)
    out = {"n": n, "spearman": None, "spearman_ci": [None, None], "pearson": None,
           "clusters": len({p[2] for p in pts})}
    if n < 3:
        return out
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    out["spearman"], out["pearson"] = _r(spearman(x, y), 4), _r(_corr(x, y), 4)
    by_c: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(pts):
        by_c[p[2]].append(i)
    keys = sorted(by_c)
    if len(keys) < 2:
        return out
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), size=len(keys))
        ii = np.concatenate([by_c[keys[k]] for k in pick])
        s = spearman(x[ii], y[ii])
        if s is not None:
            stats.append(s)
    if stats:
        out["spearman_ci"] = [_r(float(np.percentile(stats, 2.5)), 4),
                              _r(float(np.percentile(stats, 97.5)), 4)]
    return out


def ci_sign(ci: Sequence[Optional[float]]) -> str:
    lo, hi = (list(ci) + [None, None])[:2]
    if lo is None or hi is None:
        return "UNKNOWN"
    if lo > 0:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    return "NONE"


def evidence_label(min_n: int, clusters: int, sign: str) -> str:
    c = CRITERIA
    if min_n < c["min_n_any"] or clusters < c["min_clusters"]:
        return "INSUFFICIENT"
    resolved = sign in ("POSITIVE", "NEGATIVE")
    if resolved and min_n >= c["min_n_strong"] and clusters >= c["min_clusters_strong"]:
        return "STRONG_EVIDENCE"
    if resolved and min_n >= c["min_n_moderate"]:
        return "MODERATE_EVIDENCE"
    return "WEAK_EVIDENCE"


def compare(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A minus B with cluster-bootstrap CIs, power, and an evidence label."""
    sa, sb = summarize(a, excursions=False), summarize(b, excursions=False)
    boot = cluster_bootstrap_diff(a, b)
    clusters = len({r["scan_run"] for r in list(a) + list(b)})
    min_n = min(sa["n"], sb["n"])
    out: Dict[str, Any] = {
        "a_n": sa["n"], "b_n": sb["n"], "clusters": clusters,
        "diff_mean": _r(sa["mean"] - sb["mean"]) if sa["n"] and sb["n"] else None,
        "diff_median": _r(sa["median"] - sb["median"]) if sa["n"] and sb["n"] else None,
        "diff_win_rate": _r(sa["win_rate"] - sb["win_rate"], 4) if sa["n"] and sb["n"] else None,
        **boot,
    }
    sign = ci_sign(boot["diff_mean_ci"])
    out["sign"] = sign
    out["evidence"] = evidence_label(min_n, clusters, sign)
    lo, hi = boot["diff_mean_ci"]
    meaningful = CRITERIA["meaningful_effect"]
    if lo is not None and hi is not None and min_n > 0:
        se = (hi - lo) / (2 * 1.96)
        mde = 2.8 * se  # 80% power, alpha 0.05, two-sided
        out["mde"] = _r(mde)
        req = math.ceil(min_n * (mde / meaningful) ** 2) if meaningful > 0 else None
        out["required_n_per_arm"] = req
        out["additional_needed_per_arm"] = max(0, req - min_n) if req is not None else None
        out["powered"] = bool(mde <= meaningful and min_n >= CRITERIA["min_n_moderate"]
                              and clusters >= CRITERIA["min_clusters"])
    else:
        out.update({"mde": None, "required_n_per_arm": None,
                    "additional_needed_per_arm": None, "powered": False})
    return out


# ---- Analyses ------------------------------------------------------------------
def _by(records: Iterable[Dict[str, Any]], **eq) -> List[Dict[str, Any]]:
    return [r for r in records if all(r.get(k) == v for k, v in eq.items())]


def readiness(observations: Sequence[Dict[str, Any]],
              outcomes_by_id: Dict[str, List[Dict[str, Any]]],
              built: Dict[str, Any], primary: Sequence[Dict[str, Any]], *,
              now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    rc = CRITERIA["readiness"]
    obs = [o for o in observations or [] if o]
    explicit = [o for o in obs if is_explicit(o)]
    ids = {str(o.get("observation_id") or "") for o in obs}
    by_cohort = Counter(cohort(o) or "LEGACY_INFERRED" for o in obs)
    explicit_by_cohort = Counter(cohort(o) for o in explicit)

    pit = invalid_obs = 0
    matured_obs = unmatured_obs = retire_eligible = 0
    coverage: Dict[str, Dict[str, int]] = {c: {h: 0 for h in HORIZONS} for c in COHORTS}
    out_total = out_dr = out_mfe = out_mae = out_invalid = extreme = 0
    for o in explicit:
        mkt = o.get("market") or {}
        for k in ("price", "volume"):
            v = mkt.get(k)
            if v is not None and (_num(v) is None or _num(v) < 0):
                invalid_obs += 1
                break
        a = anchor(o)
        ocs = [x for x in outcomes_by_id.get(str(o.get("observation_id") or "")) or []
               if str(x.get("data_status")) == "MATURED"]
        hs = {x.get("horizon") for x in ocs}
        if ocs:
            matured_obs += 1
        else:
            unmatured_obs += 1
        if a is not None and now >= a + _dt.timedelta(days=6) and not set(HORIZONS) <= hs:
            retire_eligible += 1
        for h in HORIZONS:
            if h in hs and cohort(o) in coverage:
                coverage[cohort(o)][h] += 1
        for x in ocs:
            out_total += 1
            ev = _parse_dt(x.get("evaluation_time"))
            if a is not None and ev is not None and ev <= a:
                pit += 1
            raw = _num(x.get("raw_return"))
            if raw is None:
                out_invalid += 1
            elif abs(raw) > 0.5:
                extreme += 1
            out_dr += _num(x.get("directional_return")) is not None
            out_mfe += _num(x.get("mfe")) is not None
            out_mae += _num(x.get("mae")) is not None

    dup = conflicting = orphans = 0
    for oid, ocs in (outcomes_by_id or {}).items():
        if oid not in ids:
            orphans += len(ocs)
        per_h: Dict[Any, List[Any]] = defaultdict(list)
        for x in ocs:
            per_h[x.get("horizon")].append(_num(x.get("raw_return")))
        for vals in per_h.values():
            if len(vals) > 1:
                dup += len(vals) - 1
                if len(set(vals)) > 1:
                    conflicting += 1

    frac = (lambda a, b: round(a / b, 4) if b else None)
    primary_n = {c: len(_by(primary, cohort=c, horizon=PRIMARY_HORIZON)) for c in COHORTS}
    direction_n = Counter(r["direction"] for r in primary if r["horizon"] == PRIMARY_HORIZON
                          and r["direction_known"])
    runs = {scan_run(o) for o in explicit}
    invalid_rate = frac(invalid_obs + out_invalid, len(explicit) + out_total) or 0.0

    gates = [
        ("modern_matured_observations", matured_obs >= rc["min_modern_matured_observations"],
         "INSUFFICIENT", f"{matured_obs} >= {rc['min_modern_matured_observations']}"),
        ("cohort_primary_min", min(primary_n.values()) >= rc["min_cohort_primary_n"],
         "INSUFFICIENT", f"min cohort n at {PRIMARY_HORIZON} = {min(primary_n.values())} >= {rc['min_cohort_primary_n']}"),
        ("point_in_time", pit == 0, "INSUFFICIENT", f"{pit} violations"),
        ("conflicting_outcomes", conflicting == 0, "INSUFFICIENT", f"{conflicting} conflicting duplicates"),
        ("invalid_values", invalid_rate <= rc["max_invalid_rate"], "INSUFFICIENT",
         f"invalid rate {invalid_rate} <= {rc['max_invalid_rate']}"),
        ("cohort_primary_powered", min(primary_n.values()) >= rc["cohort_primary_n_conditional"],
         "CONDITIONALLY_READY", f"min cohort n at {PRIMARY_HORIZON} >= {rc['cohort_primary_n_conditional']}"),
        ("scan_runs", len(runs) >= rc["min_scan_runs"], "CONDITIONALLY_READY",
         f"{len(runs)} distinct scan runs >= {rc['min_scan_runs']}"),
        ("stored_directional_return", (frac(out_dr, out_total) or 0) >= rc["stored_directional_coverage"],
         "CONDITIONALLY_READY", f"stored coverage {frac(out_dr, out_total)} (derived from raw_return otherwise)"),
        ("excursion_coverage", (frac(min(out_mfe, out_mae), out_total) or 0) >= rc["excursion_coverage"],
         "CONDITIONALLY_READY", f"MFE/MAE coverage {frac(min(out_mfe, out_mae), out_total)}"),
        ("short_sample", direction_n.get("SHORT", 0) >= rc["min_short_n"], "CONDITIONALLY_READY",
         f"SHORT n at {PRIMARY_HORIZON} = {direction_n.get('SHORT', 0)}"),
    ]
    failed = [g for g in gates if not g[1]]
    verdict = ("INSUFFICIENT" if any(g[2] == "INSUFFICIENT" for g in failed)
               else "CONDITIONALLY_READY" if failed else "READY")
    return {
        "verdict": verdict,
        "gates": [{"gate": g[0], "passed": bool(g[1]), "severity_if_failed": g[2], "detail": g[3]}
                  for g in gates],
        "failed_gates": [g[0] for g in failed],
        "total_observations": len(obs),
        "explicitly_tagged": len(explicit),
        "legacy_inferred": len(obs) - len(explicit),
        "observations_by_cohort_all": dict(by_cohort),
        "observations_by_cohort_explicit": {c: explicit_by_cohort.get(c, 0) for c in COHORTS},
        "unique_symbols": len({str(o.get("symbol") or "").upper() for o in explicit}),
        "unique_scan_runs": len(runs),
        "matured_observations": matured_obs,
        "unmatured_observations": unmatured_obs,
        "retirement_eligible_unmatured": retire_eligible,
        "matured_outcome_records": out_total,
        "coverage_by_cohort_horizon": coverage,
        "primary_population_n_by_cohort": primary_n,
        "directional_return_stored_coverage": frac(out_dr, out_total),
        "mfe_coverage": frac(out_mfe, out_total),
        "mae_coverage": frac(out_mae, out_total),
        "duplicate_outcomes": dup,
        "conflicting_outcomes": conflicting,
        "orphan_outcomes": orphans,
        "point_in_time_violations": pit,
        "invalid_observations": invalid_obs,
        "invalid_outcomes": out_invalid,
        "extreme_returns_abs_gt_50pct": extreme,
        "cohort_overlap_dropped": built["dropped_overlap"],
        "direction_n_primary": {"LONG": direction_n.get("LONG", 0), "SHORT": direction_n.get("SHORT", 0),
                                "UNKNOWN_(CONTROL)": primary_n[CONTROL]},
    }


def cohort_effectiveness(primary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_horizon": {}}
    for h in HORIZONS:
        groups = {c: _by(primary, cohort=c, horizon=h) for c in COHORTS}
        out["by_horizon"][h] = {
            "cohorts": {c: summarize(groups[c], excursions=(h == PRIMARY_HORIZON)) for c in COHORTS},
            "candidate_minus_control": compare(groups[CANDIDATE], groups[CONTROL]),
            "candidate_minus_near_miss": compare(groups[CANDIDATE], groups[NEAR_MISS]),
            "near_miss_minus_control": compare(groups[NEAR_MISS], groups[CONTROL]),
        }
    return out


def score_buckets(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-declared bucketing: fixed buckets, merging any bucket with
    n < min_bucket_n into its upper neighbour (the last merges down). If fewer
    than 3 buckets survive, fall back to score quintiles. Counts use the primary
    horizon's scored observations (no outcome values involved)."""
    min_n = CRITERIA["min_bucket_n"]
    scores = [r["features"]["score"] for r in scored
              if r["horizon"] == PRIMARY_HORIZON and r["features"]["score"] is not None]
    bins = [[lo, hi, label] for lo, hi, label in FIXED_SCORE_BUCKETS]
    count = (lambda lo, hi: sum(1 for s in scores if lo <= s < hi))
    i = 0
    while i < len(bins) and len(bins) > 1:
        if count(bins[i][0], bins[i][1]) < min_n:
            j = i + 1 if i + 1 < len(bins) else i - 1
            lo, hi = min(bins[i][0], bins[j][0]), max(bins[i][1], bins[j][1])
            label = f"{bins[min(i, j)][2].split('-')[0]}-{bins[max(i, j)][2].split('-')[-1]}"
            bins[min(i, j)] = [lo, hi, label]
            del bins[max(i, j)]
            i = 0
            continue
        i += 1
    mode = "fixed"
    if sum(1 for b in bins if count(b[0], b[1]) >= min_n) < 3 and len(scores) >= 3 * min_n:
        qs = [float(q) for q in np.percentile(scores, [20, 40, 60, 80])]
        edges = [-math.inf] + qs + [math.inf]
        bins = []
        for k in range(5):
            lo, hi = edges[k], edges[k + 1]
            bins.append([lo, hi, f"Q{k + 1} [{_fmt(lo)}, {_fmt(hi)})"])
        mode = "quintile"
    return {"mode": mode, "buckets": [(b[0], b[1], b[2]) for b in bins],
            "scored_n": len(scores),
            "score_quantiles": ([_r(float(q), 3) for q in np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])]
                                if scores else None)}


def _fmt(x: float) -> str:
    return "-inf" if x == -math.inf else "+inf" if x == math.inf else f"{x:.1f}"


def assign_bucket(score: Optional[float], buckets: Sequence[Tuple[float, float, str]]) -> Optional[str]:
    if score is None:
        return None
    for lo, hi, label in buckets:
        if lo <= score < hi:
            return label
    return None


def score_monotonicity(primary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in primary if r["cohort"] in (CANDIDATE, NEAR_MISS)
              and r["features"]["score"] is not None]
    spec = score_buckets(scored)
    buckets = spec["buckets"]
    per_h: Dict[str, Any] = {}
    sig_pos = sig_neg = not_pos = 0
    for h in HORIZONS:
        rows = _by(scored, horizon=h)
        stats = []
        for lo, hi, label in buckets:
            b = [r for r in rows if assign_bucket(r["features"]["score"], buckets) == label]
            stats.append({"bucket": label, **summarize(b, excursions=(h == PRIMARY_HORIZON))})
        means = [s["mean"] for s in stats if s["n"] >= CRITERIA["min_bucket_n"] and s["mean"] is not None]
        violations = sum(1 for x, y in zip(means, means[1:]) if y < x)
        corr = correlation_with_ci(rows, lambda r: r["features"]["score"])
        sign = ci_sign(corr["spearman_ci"])
        sig_pos += sign == "POSITIVE"
        sig_neg += sign == "NEGATIVE"
        not_pos += sign in ("NONE", "NEGATIVE")
        per_h[h] = {"buckets": stats, "adjacent_violations": violations,
                    "comparable_buckets": len(means), **corr, "sign": sign}
    n = spec["scored_n"]
    need = CRITERIA["edge_min_horizons"]
    prim_viol = per_h[PRIMARY_HORIZON]["adjacent_violations"]
    if n < CRITERIA["monotonic_min_scored_n"]:
        verdict, why = "INCONCLUSIVE", f"scored n {n} < {CRITERIA['monotonic_min_scored_n']}"
    elif sig_pos >= need and prim_viol <= 1:
        verdict, why = "YES", f"Spearman CI > 0 at {sig_pos}/4 horizons; {prim_viol} adjacent violation(s) at {PRIMARY_HORIZON}"
    elif sig_neg >= 2 or (n >= CRITERIA["monotonic_no_min_scored_n"] and not_pos >= need):
        verdict, why = "NO", f"Spearman CI > 0 at only {sig_pos}/4 horizons (negative at {sig_neg}) with scored n {n}"
    else:
        verdict, why = "INCONCLUSIVE", f"Spearman CI > 0 at {sig_pos}/4 horizons; scored n {n}"
    return {"verdict": verdict, "reason": why, "population": "CANDIDATE + NEAR_MISS with BreakoutScore",
            "bucket_spec": {**spec, "buckets": [b[2] for b in buckets]}, "by_horizon": per_h}


def tier_effectiveness(primary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in primary if r["tier"]]
    total = len(_by(primary, horizon=PRIMARY_HORIZON))
    prim = [r for r in rows if r["horizon"] == PRIMARY_HORIZON]
    if len(prim) < CRITERIA["min_n_any"]:
        return {"verdict": "INCONCLUSIVE", "available": False,
                "reason": ("production Day Trader tiers (Strong/Developing/Weak) are computed at "
                           "render time from VWAP/ADX/SuperTrend/EWO and are not persisted on "
                           f"observations; {len(prim)} of {total} primary observations carry a tier"),
                "tier_n": len(prim)}
    order = ["WEAK", "DEVELOPING", "STRONG"]
    by_h = {}
    sep = 0
    for h in HORIZONS:
        hr = _by(rows, horizon=h)
        by_h[h] = {t: summarize(_by(hr, tier=t), excursions=(h == PRIMARY_HORIZON)) for t in order}
        comp = compare(_by(hr, tier="STRONG"), _by(hr, tier="WEAK"))
        by_h[h]["strong_minus_weak"] = comp
        sep += comp["sign"] == "POSITIVE"
    shares = Counter(r["tier"] for r in prim)
    verdict = "YES" if sep >= CRITERIA["edge_min_horizons"] else (
        "NO" if all(by_h[h]["strong_minus_weak"]["powered"] for h in HORIZONS) else "INCONCLUSIVE")
    return {"verdict": verdict, "available": True, "by_horizon": by_h,
            "share": {t: round(shares.get(t, 0) / len(prim), 4) for t in order}}


def direction_analysis(primary: Sequence[Dict[str, Any]], transform: Dict[str, int]) -> Dict[str, Any]:
    rows = [r for r in primary if r["direction_known"]]
    by_h = {h: {d: summarize(_by(rows, horizon=h, direction=d), excursions=(h == PRIMARY_HORIZON))
                for d in ("LONG", "SHORT")} for h in HORIZONS}
    short_n = by_h[PRIMARY_HORIZON]["SHORT"]["n"]
    if short_n < CRITERIA["readiness"]["min_short_n"]:
        verdict = "INSUFFICIENT"
        why = f"SHORT n at {PRIMARY_HORIZON} = {short_n}"
    else:
        comps = {h: compare(_by(rows, horizon=h, direction="LONG"), _by(rows, horizon=h, direction="SHORT"))
                 for h in HORIZONS}
        by_h.update({f"{h}_long_minus_short": comps[h] for h in HORIZONS})
        diff = sum(1 for c in comps.values() if c["sign"] != "NONE" and c["sign"] != "UNKNOWN")
        verdict = "MATERIAL_DIFFERENCE" if diff >= CRITERIA["edge_min_horizons"] else "NO_MATERIAL_DIFFERENCE"
        why = f"LONG-SHORT CI excludes 0 at {diff}/4 horizons"
    return {"verdict": verdict, "reason": why, "by_horizon": by_h,
            "short_transform_check": {**transform, "ok": transform["mismatches"] == 0}}


def time_of_day(explicit_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [b[2] for b in TOD_BUCKETS]
    by_h = {}
    for h in HORIZONS:
        hr = _by(explicit_records, horizon=h)
        rows = {}
        for lab in labels:
            b = _by(hr, tod=lab)
            rows[lab] = {c: summarize(_by(b, cohort=c), excursions=(h == PRIMARY_HORIZON))
                         for c in (CANDIDATE, CONTROL)}
            rows[lab]["candidate_minus_control"] = compare(_by(b, cohort=CANDIDATE), _by(b, cohort=CONTROL))
        by_h[h] = rows
    prim = by_h[PRIMARY_HORIZON]
    usable = [(lab, prim[lab][CANDIDATE]) for lab in labels[1:-1]
              if prim[lab][CANDIDATE]["n"] >= CRITERIA["min_n_any"]]
    if len(usable) < 2:
        verdict = "INSUFFICIENT"
        why = f"{len(usable)} regular-session bucket(s) with candidate n >= {CRITERIA['min_n_any']}"
    else:
        lo_b = min(usable, key=lambda t: t[1]["mean"])
        hi_b = max(usable, key=lambda t: t[1]["mean"])
        overlap = lo_b[1]["mean_ci"][1] >= hi_b[1]["mean_ci"][0]
        spread = hi_b[1]["mean"] - lo_b[1]["mean"]
        material = (not overlap) and spread >= CRITERIA["meaningful_effect"]
        verdict = "MATERIAL" if material else "NOT_MATERIAL"
        why = (f"candidate mean spread {spread:+.4%} between {lo_b[0]} and {hi_b[0]} "
               f"({'non-overlapping' if not overlap else 'overlapping'} CIs)")
    populated = {lab: prim[lab][CANDIDATE]["n"] + prim[lab][CONTROL]["n"] for lab in labels}
    return {"verdict": verdict, "reason": why, "population": "explicit observations, all sessions",
            "n_by_bucket_primary": populated, "by_horizon": by_h}


def regime(primary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prim = _by(primary, horizon=PRIMARY_HORIZON)
    present = sum(1 for r in prim if r["regime"])
    share = present / len(prim) if prim else 0.0
    if not prim or share < CRITERIA["regime_min_presence"]:
        return {"verdict": "REGIME ANALYSIS UNAVAILABLE", "present_share": round(share, 4),
                "reason": ("market_context.market_regime is not populated at capture "
                           "(scheduler passes no point-in-time regime); not reconstructed "
                           "to avoid leakage")}
    by_h = {h: {g: summarize([r for r in _by(primary, horizon=h) if r["regime"] == g])
                for g in sorted({r["regime"] for r in prim})} for h in HORIZONS}
    return {"verdict": "AVAILABLE", "present_share": round(share, 4), "by_horizon": by_h}


def feature_diagnostics(primary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pop = [r for r in primary if r["cohort"] in (CANDIDATE, NEAR_MISS)]
    B = CRITERIA["bootstrap_resamples_feature"]
    out: Dict[str, Any] = {"population": "CANDIDATE + NEAR_MISS", "numeric": {}, "flags": {},
                           "unavailable": UNAVAILABLE_FEATURES,
                           "multiple_comparisons_note": (
                               f"{len(NUMERIC_FEATURES) + len(FLAG_FEATURES)} features x 4 horizons; "
                               "expect ~1 in 20 spurious CI exclusions. Diagnostic only.")}
    for f in NUMERIC_FEATURES:
        key = (lambda r, f=f: r["features"].get(f))
        per_h = {}
        signs = []
        for h in HORIZONS:
            rows = [r for r in _by(pop, horizon=h) if key(r) is not None]
            corr = correlation_with_ci(rows, key, B=B)
            win = [key(r) for r in rows if r["drw"] > 0]
            lose = [key(r) for r in rows if r["drw"] <= 0]
            q_stats = []
            if len(rows) >= 4 * CRITERIA["min_n_any"]:
                qs = np.percentile([key(r) for r in rows], [25, 50, 75])
                edges = [-math.inf] + [float(q) for q in qs] + [math.inf]
                for k in range(4):
                    b = [r for r in rows if edges[k] <= key(r) < edges[k + 1]]
                    q_stats.append({"bucket": f"Q{k + 1}", "range": [_fmt(edges[k]), _fmt(edges[k + 1])],
                                    **summarize(b, excursions=False)})
            per_h[h] = {**corr, "sign": ci_sign(corr["spearman_ci"]),
                        "winners_mean": _r(float(np.mean(win))) if win else None,
                        "winners_median": _r(float(np.median(win))) if win else None,
                        "losers_mean": _r(float(np.mean(lose))) if lose else None,
                        "losers_median": _r(float(np.median(lose))) if lose else None,
                        "quartiles": q_stats}
            signs.append((per_h[h]["sign"], corr["spearman"], corr["n"]))
        out["numeric"][f] = {"by_horizon": per_h, "classification": _classify_feature(signs)}
    for f in FLAG_FEATURES:
        per_h = {}
        signs = []
        for h in HORIZONS:
            rows = _by(pop, horizon=h)
            yes = [r for r in rows if r["features"].get(f) is True]
            no = [r for r in rows if r["features"].get(f) is False]
            comp = compare(yes, no)
            per_h[h] = comp
            signs.append((comp["sign"], comp["diff_mean"], min(comp["a_n"], comp["b_n"])))
        out["flags"][f] = {"by_horizon": per_h, "classification": _classify_feature(signs)}
    return out


def _classify_feature(signs: Sequence[Tuple[str, Optional[float], int]]) -> str:
    """USEFUL / INVERSE: CI excludes 0 at the primary horizon and the point
    estimate has the same sign at >= 3 of 4 horizons. NEUTRAL: primary n >= 100
    and CI includes 0. Otherwise INSUFFICIENT."""
    prim_sign, _, prim_n = signs[HORIZONS.index(PRIMARY_HORIZON)]
    if prim_n < CRITERIA["min_n_moderate"]:
        return "INSUFFICIENT"
    pos = sum(1 for _s, v, _n in signs if v is not None and v > 0)
    neg = sum(1 for _s, v, _n in signs if v is not None and v < 0)
    if prim_sign == "POSITIVE" and pos >= CRITERIA["edge_min_horizons"]:
        return "USEFUL"
    if prim_sign == "NEGATIVE" and neg >= CRITERIA["edge_min_horizons"]:
        return "INVERSE"
    return "NEUTRAL"


def excursion_quality(primary: Sequence[Dict[str, Any]], score_spec: Dict[str, Any]) -> Dict[str, Any]:
    p60 = _by(primary, horizon=PRIMARY_HORIZON)
    by_obs: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in primary:
        by_obs[r["observation_id"]][r["horizon"]] = r["drw"]
    thr = CRITERIA["reversal_mfe"]

    def patterns(rows):
        n = len(rows)
        rev = sum(1 for r in rows if r["mfe"] is not None and r["mfe"] >= thr and r["drw"] <= 0)
        both = [by_obs[r["observation_id"]] for r in rows if "+5m" in by_obs[r["observation_id"]]]
        early_adverse = sum(1 for o in both if o["+5m"] < 0 and o.get("+60m", 0) > 0)
        early_fade = sum(1 for o in both if o["+5m"] > 0 and o.get("+60m", 0) <= 0)
        return {"n": n,
                "strong_move_then_reversal_share": _r(rev / n, 4) if n else None,
                "early_adverse_then_winner_share": _r(early_adverse / len(both), 4) if both else None,
                "early_favourable_then_loser_share": _r(early_fade / len(both), 4) if both else None}
    buckets = [tuple(b) for b in score_spec.get("_buckets", [])]
    out = {"horizon": PRIMARY_HORIZON,
           "note": "+60m records only: shorter-horizon MFE/MAE windows vary with batch composition",
           "by_cohort": {c: {**excursion_stats(_by(p60, cohort=c)), **patterns(_by(p60, cohort=c))}
                         for c in COHORTS}, "by_score_bucket": {}}
    scored = [r for r in p60 if r["cohort"] in (CANDIDATE, NEAR_MISS) and r["features"]["score"] is not None]
    for lo, hi, label in buckets:
        rows = [r for r in scored if assign_bucket(r["features"]["score"], buckets) == label]
        out["by_score_bucket"][label] = {**excursion_stats(rows), **patterns(rows),
                                         "mean_return": summarize(rows, excursions=False)["mean"]}
    return out


# ---- Verdicts ----------------------------------------------------------------------
def primary_verdict(ready: Dict[str, Any], coh: Dict[str, Any], mono: Dict[str, Any],
                    tiers: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-declared overall verdict. See module docstring / CRITERIA."""
    need = CRITERIA["edge_min_horizons"]
    cc = {h: coh["by_horizon"][h]["candidate_minus_control"] for h in HORIZONS}
    cn = {h: coh["by_horizon"][h]["candidate_minus_near_miss"] for h in HORIZONS}
    strong = ("MODERATE_EVIDENCE", "STRONG_EVIDENCE")
    pos = [h for h in HORIZONS if cc[h]["sign"] == "POSITIVE" and cc[h]["evidence"] in strong]
    neg = [h for h in HORIZONS if cc[h]["sign"] == "NEGATIVE" and cc[h]["evidence"] in strong]
    powered = [h for h in HORIZONS if cc[h]["powered"]]
    cn_neg = [h for h in HORIZONS if cn[h]["sign"] == "NEGATIVE" and cn[h]["evidence"] in strong]
    facts = {"candidate_vs_control_positive": pos, "candidate_vs_control_negative": neg,
             "candidate_vs_control_powered": powered, "candidate_vs_near_miss_negative": cn_neg,
             "score_monotonicity": mono["verdict"], "tier_separation": tiers["verdict"],
             "readiness": ready["verdict"]}
    if ready["verdict"] == "INSUFFICIENT":
        v, why = "INSUFFICIENT_EVIDENCE", "readiness INSUFFICIENT: " + ", ".join(ready["failed_gates"])
    elif len(pos) >= need and not neg and len(cn_neg) < 2:
        v, why = "EDGE_DETECTED", f"CANDIDATE beats CONTROL (CI > 0, >= MODERATE) at {len(pos)}/4 horizons"
    elif len(pos) >= need:
        v, why = "INSUFFICIENT_EVIDENCE", (f"CANDIDATE beats CONTROL at {len(pos)}/4 horizons but "
                                           f"contradicted (neg vs control {neg}, neg vs near-miss {cn_neg})")
    elif len(neg) >= 2 or (len(powered) >= CRITERIA["no_edge_min_powered_horizons"] and len(pos) <= 1):
        v, why = "NO_EDGE_DETECTED", (f"CANDIDATE vs CONTROL: positive at {len(pos)}/4, negative at "
                                      f"{len(neg)}/4, powered (MDE <= {CRITERIA['meaningful_effect']:.2%}) "
                                      f"at {len(powered)}/4 horizons")
    else:
        v, why = "INSUFFICIENT_EVIDENCE", (f"CANDIDATE vs CONTROL positive at {len(pos)}/4 horizons; "
                                           f"only {len(powered)}/4 horizons powered to rule out a "
                                           f"{CRITERIA['meaningful_effect']:.2%} effect")
    used = [cc[h]["evidence"] for h in HORIZONS]
    order = ["INSUFFICIENT", "WEAK_EVIDENCE", "MODERATE_EVIDENCE", "STRONG_EVIDENCE"]
    if v == "INSUFFICIENT_EVIDENCE":
        quality = "INSUFFICIENT"
    else:
        basis = pos if v == "EDGE_DETECTED" else (neg or powered)
        quality = min((cc[h]["evidence"] for h in basis), key=order.index) if basis else "WEAK_EVIDENCE"
        if v == "NO_EDGE_DETECTED" and quality in ("INSUFFICIENT",):
            quality = "WEAK_EVIDENCE"
    return {"verdict": v, "reason": why, "evidence_quality": quality, "facts": facts,
            "candidate_vs_control_evidence_by_horizon": dict(zip(HORIZONS, used))}


def run56_recommendation(ready: Dict[str, Any], verdict: Dict[str, Any], mono: Dict[str, Any],
                         tiers: Dict[str, Any], feats: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic decision matrix (pre-declared)."""
    integrity = {"point_in_time", "conflicting_outcomes", "invalid_values"}
    useful = sorted([f for f, v in {**feats["numeric"], **feats["flags"]}.items()
                     if v["classification"] in ("USEFUL", "INVERSE")])
    notes = ["Parts 4 and 8 (tiers, conflicts, EMA/RSI/PreBreakout) cannot be evaluated "
             "until the scheduled capture persists those point-in-time fields."]
    if integrity & set(ready["failed_gates"]):
        code, name = "G", "INVESTIGATE_DATA_QUALITY"
        why = "integrity gate failed: " + ", ".join(sorted(integrity & set(ready["failed_gates"])))
    elif verdict["verdict"] == "INSUFFICIENT_EVIDENCE":
        code, name = "F", "COLLECT_MORE_DATA"
        why = verdict["reason"]
    elif verdict["verdict"] == "EDGE_DETECTED":
        if mono["verdict"] == "YES":
            code, name, why = "A", "PRESERVE_SCORING", "edge detected and score ordering is monotonic"
        elif mono["verdict"] == "NO":
            code, name, why = "B", "RECALIBRATE_SCORE", "selection has edge but higher scores do not rank better"
        else:
            code, name, why = "A", "PRESERVE_SCORING", "edge detected; score ordering not yet resolvable"
    else:  # NO_EDGE_DETECTED
        code, name = "E", "IMPROVE_FEATURE_SET"
        why = ("selection shows no edge over CONTROL; " + (
            f"individual features with a consistent association: {', '.join(useful)}" if useful
            else "no persisted feature shows a consistent association; the persisted set is too narrow"))
    return {"code": code, "action": name, "reason": why, "notes": notes,
            "feature_leads": useful, "do_not_implement_in_run55": True}


def analyze(observations: Sequence[Dict[str, Any]],
            outcomes_by_id: Dict[str, List[Dict[str, Any]]], *,
            now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Full Run 55 report from already-loaded observations + outcome records."""
    built = build_records(observations, outcomes_by_id or {})
    records = built["records"]
    primary = [r for r in records if r["regular"]]
    limits = winsor_limits(primary)
    apply_winsor(records, limits)
    ready = readiness(observations, outcomes_by_id or {}, built, primary, now=now)
    coh = cohort_effectiveness(primary)
    mono = score_monotonicity(primary)
    tiers = tier_effectiveness(primary)
    dirn = direction_analysis(primary, built["transform"])
    tod = time_of_day(records)
    reg = regime(primary)
    feats = feature_diagnostics(primary)
    spec = dict(mono["bucket_spec"])
    spec["_buckets"] = score_buckets([r for r in primary if r["cohort"] in (CANDIDATE, NEAR_MISS)
                                      and r["features"]["score"] is not None])["buckets"]
    exc = excursion_quality(primary, spec)
    verdict = primary_verdict(ready, coh, mono, tiers)
    rec56 = run56_recommendation(ready, verdict, mono, tiers, feats)
    return {
        "schema": SCHEMA,
        "generated_at": (now or _dt.datetime.now(_dt.timezone.utc)).isoformat(),
        "criteria": CRITERIA,
        "winsor_limits": {h: [_r(a), _r(b)] for h, (a, b) in limits.items()},
        "primary_population": {"records": len(primary), "all_explicit_records": len(records)},
        "readiness": ready,
        "cohorts": coh,
        "score_monotonicity": mono,
        "tiers": tiers,
        "direction": dirn,
        "time_of_day": tod,
        "regime": reg,
        "features": feats,
        "excursions": exc,
        "verdict": verdict,
        "run56": rec56,
    }
