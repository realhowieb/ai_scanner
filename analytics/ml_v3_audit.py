"""ML v3 leakage-safe walk-forward audit (research only, no production effect).

Consumes the canonical research dataset (``analytics.research_dataset``) and the
canonical outcome semantics (``analytics.outcome_intelligence``). Nothing here
writes to a database, retrains or replaces a production model, or changes a
score. Every model it fits lives only in memory for one fold.

Unit of analysis
    One row per (ticker, UTC observation day): the day's FIRST frozen
    observation, exactly Outcome Intelligence's default ``signal_day`` unit.
    Same-day copies share one entry close and one outcome, so counting them
    would weight a ticker by how many snapshots ran that day.

Time keys (per row)
    entry_day      first trading day on/after the UTC fire date (entry bar)
    window_end[h]  the trading day h bars after entry: the last bar the
                   h-day label reads (``research_dataset.label_window_end``)

Walk-forward (expanding window), per horizon h
    Rows are ordered by entry_day. A validation block is a run of consecutive
    entry days holding at least ``min_val`` matured rows with both classes.
    Training rows for a block that starts on day V are rows with
        entry_day < V                         (chronological)
        window_end[h] < V                     (PURGE: no training label reads
                                               a bar on or after V)
        window_end[h] < V - embargo days      (EMBARGO: an extra gap of
                                               ``embargo`` trading days)
    so no training label uses price information from the validation period.
    The next block starts after the current block. An optional FINAL HOLDOUT
    (the most recent block) is reserved before any fold is built and is never
    seen by the walk-forward.

Probabilities for score-type inputs (HSF Score, a stored model %) come from a
one-feature logistic map fitted on the TRAINING fold only, so their Brier and
log loss are out of sample too.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analytics import outcome_intelligence as oi
from analytics import research_dataset as rd
from analytics import research_schema as rs

AUDIT_VERSION = "hsf-ml-v3-audit-1"
HORIZONS = (1, 3, 5)
UNSUPPORTED_HORIZONS = (10, 15, 20)
SEED = 20261008

# ----------------------------------------------------------------------------- features
NUMERIC_FEATURES = (
    "hsf_score", "hsf_signals_component", "hsf_model_component", "hsf_momentum_component",
    "hsf_fading_penalty", "n_signals", "fading", "chg_pct", "gap_pct", "breakout_score", "prebreakout_prob",
    "snapshot_rank", "snapshot_size", "price", "volume", "rvol_20", "volatility_20d_pct", "scan_gap_pct",
    "scan_chg_pct", "scanner_breakout_score", "is_breakout", "trend_10d_pct", "trend_20d_pct",
    "breakout_pos_20d", "dollar_vol_20", "rs_vs_spy", "scanner_rank",
)
CATEGORICAL_FEATURES = ("primary_setup", "hsf_status", "ema_cross", "pattern_tag")
SIGNAL_NAMES = ("breakout", "golden_cross", "prebreakout", "gapper", "gainer", "loser")
EXCLUDED_FEATURES = {
    "hsf_score_version": "constant ('1.0') across the whole dataset; carries no information",
    "signals": "expanded into one 0/1 column per known signal (signal__*)",
}

# Feature groups for importance roll-up and ablation (only groups whose columns exist).
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "hsf_score": ("hsf_score", "hsf_signals_component", "hsf_model_component", "hsf_momentum_component",
                  "hsf_fading_penalty"),
    "momentum": ("chg_pct", "scan_chg_pct", "trend_10d_pct", "trend_20d_pct"),
    "volume": ("volume", "rvol_20", "dollar_vol_20"),
    "ema_trend": ("ema_cross", "breakout_pos_20d"),
    "volatility_gap": ("volatility_20d_pct", "gap_pct", "scan_gap_pct"),
    "prebreakout": ("prebreakout_prob",),
    "setup_metadata": ("primary_setup", "hsf_status", "n_signals", "fading", "signal__", "breakout_score",
                       "scanner_breakout_score", "is_breakout", "pattern_tag"),
    "rank": ("snapshot_rank", "snapshot_size", "scanner_rank"),
    "market_context": ("rs_vs_spy",),
    "price_level": ("price",),
}
# Categories the run spec names that HSF never stored (reported, not run).
MISSING_GROUPS = {
    "rsi": "RSI-14 is never computed by scheduled scans or persisted (research_schema.UNAVAILABLE_FEATURES).",
    "market_regime": "No regime/breadth/VIX value is frozen at observation time.",
}

# Features the production AI Confidence model uses, mapped to research columns.
AI_CONFIDENCE_FEATURE_MAP = {
    "Trend10D%": "trend_10d_pct", "Trend20D%": "trend_20d_pct", "VolRel20": "rvol_20",
    "DollarVol20": "dollar_vol_20", "BreakoutScore": "scanner_breakout_score", "GapPct": "scan_gap_pct",
}

# Point-in-time / leakage classes used in the leakage audit.
SAFE_STORED_C = "SAFE_STORED"
SAFE_ASOF_JOIN = "SAFE_ASOF_JOIN"
REVIEW = "REVIEW"
LEAKAGE = "LEAKAGE"
MISSING_C = "MISSING"


# ----------------------------------------------------------------------------- labels
@dataclass(frozen=True)
class LabelDef:
    key: str
    description: str
    horizons: Tuple[int, ...]
    fn: Callable[[Mapping[str, Any], int], Optional[int]]
    needs_benchmark: bool = False


def _ret(lab: Mapping[str, Any], h: int) -> Optional[float]:
    return lab.get(f"return_{h}d")


def _exc(lab: Mapping[str, Any], h: int) -> Optional[float]:
    return lab.get(f"excess_return_{h}d")


def _binary(v: Optional[float], thr: float, *, strict: bool = True) -> Optional[int]:
    if v is None:
        return None
    return int(v > thr) if strict else int(v >= thr)


def _label_f(lab: Mapping[str, Any], h: int) -> Optional[int]:
    mfe, mae = lab.get("mfe_5d"), lab.get("mae_5d")
    if h != 5 or mfe is None or mae is None:
        return None
    return int(mfe >= 0.04 and mae > -0.02)


def _label_e(lab: Mapping[str, Any], h: int) -> Optional[int]:
    r, e = _ret(lab, h), _exc(lab, h)
    if r is None or e is None:
        return None
    return int(r > 0 and e > 0)


LABELS: Dict[str, LabelDef] = {
    "A_return_gt_0": LabelDef("A_return_gt_0", "return_h > 0 (Outcome Intelligence win)", HORIZONS,
                              lambda lab, h: _binary(_ret(lab, h), 0.0)),
    "B_return_ge_4pct": LabelDef("B_return_ge_4pct", "return_h >= +4% (production models' upside threshold)",
                                 HORIZONS, lambda lab, h: _binary(_ret(lab, h), 0.04, strict=False)),
    "C_excess_gt_0": LabelDef("C_excess_gt_0", "excess_return_h > 0 (beats SPY; OI benchmark beat)", HORIZONS,
                              lambda lab, h: _binary(_exc(lab, h), 0.0), needs_benchmark=True),
    "D_excess_ge_2pct": LabelDef("D_excess_ge_2pct", "excess_return_h >= +2% vs SPY", HORIZONS,
                                 lambda lab, h: _binary(_exc(lab, h), 0.02, strict=False), needs_benchmark=True),
    "E_abs_and_rel_win": LabelDef("E_abs_and_rel_win", "return_h > 0 AND excess_return_h > 0", HORIZONS,
                                  _label_e, needs_benchmark=True),
    "F_clean_path_5d": LabelDef("F_clean_path_5d",
                                "mfe_5d >= +4% AND mae_5d > -2% (risk-adjusted; order-free proxy of the "
                                "production '+4% before -2%' rule, stricter because it also needs no -2% "
                                "touch after the +4%)", (5,), _label_f),
}
PRIMARY_LABEL = "A_return_gt_0"


# ----------------------------------------------------------------------------- dataset
@dataclass
class AuditRow:
    observation_id: int
    ticker: str
    observed_at: _dt.datetime
    observed_day: str
    entry_day: _dt.date
    window_end: Dict[int, _dt.date]
    features: Dict[str, Any]
    labels: Dict[str, Any]
    maturity: Dict[str, str]
    certified: bool
    setup: Optional[str]
    join_status: Optional[str]
    join_lag_s: Optional[float]
    scoring_version: Optional[str]
    oi_record: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def matured(self, h: int) -> bool:
        return self.maturity.get(f"{h}d") == rd.MATURED


def _trading_days_before(day: _dt.date, n: int) -> _dt.date:
    from analytics import market_calendar as mc

    d, k = day, 0
    while k < n:
        d -= _dt.timedelta(days=1)
        if mc.is_trading_day(d):
            k += 1
    return d


def build_audit_rows(records: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]], *,
                     unit: str = "signal_day") -> List[AuditRow]:
    """Research records -> audit rows, oldest first, collapsed to ``unit``.

    ``records`` come from ``research_dataset.build_records`` (or a built
    dataset's ``records``); ``raw_rows`` are the same opportunity rows, used
    only to build the canonical Outcome Intelligence record for metrics."""
    oi_by_id = {str(r.observation_id): r for r in oi.build_records([dict(x) for x in raw_rows])}
    rows: List[AuditRow] = []
    for rec in records:
        o, snap, out = rec["observation"], rec["features"], rec["outcome"]
        obs = rd.to_dt(o["observed_at"])
        entry = rd.entry_day(o["observed_at"])
        if obs is None or entry is None:
            continue
        rows.append(AuditRow(
            observation_id=int(o["observation_id"]), ticker=o["ticker"], observed_at=obs,
            observed_day=obs.date().isoformat(), entry_day=entry,
            window_end={h: rd.label_window_end(entry, h) for h in HORIZONS},
            features=dict(snap.values), labels=dict(out.values), maturity=dict(out.maturity),
            certified=bool(out.certified), setup=o.get("setup"), join_status=snap.join.get("status"),
            join_lag_s=snap.join.get("lag_seconds"), scoring_version=o.get("scoring_version"),
            oi_record=oi_by_id.get(str(o["observation_id"]))))
    rows.sort(key=lambda r: (r.observed_at, r.observation_id))
    if unit == "signal_day":
        seen, kept = set(), []
        for r in rows:
            key = (r.ticker, r.observed_day)
            if key not in seen:
                seen.add(key)
                kept.append(r)
        rows = kept
    elif unit != "observation":
        raise ValueError("unit must be 'signal_day' or 'observation'")
    return rows


def eligible(rows: Sequence[AuditRow], h: int, label: str = PRIMARY_LABEL) -> List[Tuple[AuditRow, int]]:
    """Matured, certified rows at horizon h with a defined label value."""
    ld = LABELS[label]
    out = []
    for r in rows:
        if h not in ld.horizons or not r.matured(h) or not r.certified:
            continue
        y = ld.fn(r.labels, h)
        if y is not None:
            out.append((r, int(y)))
    return out


# ----------------------------------------------------------------------------- folds
@dataclass
class Fold:
    fold: int
    horizon: int
    train_ids: List[int]
    val_ids: List[int]
    train_start: Optional[str]
    train_end: Optional[str]
    validation_start: str
    validation_end: str
    purged: int
    embargoed: int
    embargo_days: int


def purge_train(candidates: Sequence[AuditRow], val_start: _dt.date, h: int, embargo: int) -> Tuple[List[AuditRow], int, int]:
    """Apply the purge + embargo rule against a validation block starting on
    entry day ``val_start``. Returns (kept, purged_count, embargoed_count)."""
    cutoff = _trading_days_before(val_start, embargo) if embargo > 0 else val_start
    kept, purged, embargoed = [], 0, 0
    for r in candidates:
        if r.entry_day >= val_start:
            continue  # never chronological training data
        end = r.window_end[h]
        if end >= val_start:
            purged += 1
        elif embargo > 0 and end >= cutoff:
            embargoed += 1
        else:
            kept.append(r)
    return kept, purged, embargoed


def walk_forward_folds(data: Sequence[Tuple[AuditRow, int]], h: int, *, min_train: int = 60, min_val: int = 30,
                       min_val_class: int = 5, embargo: int = 1,
                       exclude_ids: Iterable[int] = ()) -> List[Fold]:
    """Expanding-window folds over ``data`` (rows with labels), deterministic."""
    excl = set(exclude_ids)
    data = [(r, y) for r, y in data if r.observation_id not in excl]
    by_day: Dict[_dt.date, List[Tuple[AuditRow, int]]] = defaultdict(list)
    for r, y in data:
        by_day[r.entry_day].append((r, y))
    days = sorted(by_day)
    rows_only = [r for r, _ in data]
    folds: List[Fold] = []
    i = 0
    while i < len(days):
        # grow the validation block until it is big enough
        j, block = i, []
        while j < len(days):
            block.extend(by_day[days[j]])
            j += 1
            ys = [y for _, y in block]
            if len(block) >= min_val and min(sum(ys), len(ys) - sum(ys)) >= min_val_class:
                break
        ys = [y for _, y in block]
        if len(block) < min_val or min(sum(ys), len(ys) - sum(ys)) < min_val_class:
            break
        vstart = days[i]
        train, purged, emb = purge_train(rows_only, vstart, h, embargo)
        label_of = {r.observation_id: y for r, y in data}
        tys = [label_of[r.observation_id] for r in train]
        if len(train) >= min_train and 0 < sum(tys) < len(tys):
            folds.append(Fold(
                fold=len(folds) + 1, horizon=h,
                train_ids=[r.observation_id for r in train], val_ids=[r.observation_id for r, _ in block],
                train_start=min(r.entry_day for r in train).isoformat(),
                train_end=max(r.entry_day for r in train).isoformat(),
                validation_start=vstart.isoformat(), validation_end=days[j - 1].isoformat(),
                purged=purged, embargoed=emb, embargo_days=embargo))
            i = j
        else:
            # not enough clean training history yet: this block can't be validated;
            # move the start forward one day (its rows become future training data)
            i += 1
    return folds


def final_holdout(data: Sequence[Tuple[AuditRow, int]], h: int, *, min_rows: int = 100, min_days: int = 5,
                  min_remaining_folds: int = 3, **fold_kw) -> Dict[str, Any]:
    """Reserve the most recent block as an untouched holdout, only if it is big
    enough AND the remaining history still supports ``min_remaining_folds``."""
    by_day: Dict[_dt.date, int] = Counter(r.entry_day for r, _ in data)
    days = sorted(by_day)
    acc, chosen = 0, []
    for d in reversed(days):
        chosen.append(d)
        acc += by_day[d]
        if acc >= min_rows and len(chosen) >= min_days:
            break
    if acc < min_rows or len(chosen) < min_days:
        return {"valid": False, "reason": f"only {acc} matured rows over {len(chosen)} days available for a "
                                          f"holdout (need >= {min_rows} rows and >= {min_days} days)"}
    start = min(chosen)
    hold = [r.observation_id for r, _ in data if r.entry_day >= start]
    rest = [(r, y) for r, y in data if r.entry_day < start]
    remaining = walk_forward_folds(rest, h, **fold_kw)
    if len(remaining) < min_remaining_folds:
        return {"valid": False, "reason": f"reserving {len(hold)} rows from {start} leaves {len(remaining)} "
                                          f"walk-forward folds (need >= {min_remaining_folds})"}
    return {"valid": True, "start": start.isoformat(), "ids": hold, "rows": len(hold)}


# ----------------------------------------------------------------------------- encoding
class Encoder:
    """Numeric + one-hot encoding fitted on the TRAINING fold only (vocab and
    medians never see validation rows)."""

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], signals: bool = True):
        self.numeric = [c for c in numeric]
        self.categorical = [c for c in categorical]
        self.signals = signals
        self.vocab: Dict[str, List[str]] = {}
        self.medians: Dict[str, float] = {}
        self.columns: List[str] = []

    @staticmethod
    def _num(v: Any) -> float:
        if v is None:
            return float("nan")
        if isinstance(v, bool):
            return float(v)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return f if math.isfinite(f) else float("nan")

    def fit(self, rows: Sequence[AuditRow]) -> "Encoder":
        for c in self.categorical:
            cnt = Counter(str(r.features.get(c)) for r in rows if r.features.get(c) is not None)
            self.vocab[c] = sorted(k for k, n in cnt.items() if n >= 3)
        for c in self.numeric:
            vals = [self._num(r.features.get(c)) for r in rows]
            vals = [v for v in vals if not math.isnan(v)]
            self.medians[c] = float(statistics.median(vals)) if vals else 0.0
        self.columns = list(self.numeric)
        for c in self.categorical:
            self.columns += [f"{c}={v}" for v in self.vocab[c]]
        if self.signals:
            self.columns += [f"signal__{s}" for s in SIGNAL_NAMES]
        return self

    def transform(self, rows: Sequence[AuditRow], *, impute: bool) -> np.ndarray:
        X = np.full((len(rows), len(self.columns)), np.nan)
        for i, r in enumerate(rows):
            k = 0
            for c in self.numeric:
                v = self._num(r.features.get(c))
                X[i, k] = (self.medians[c] if math.isnan(v) else v) if impute else v
                k += 1
            for c in self.categorical:
                val = r.features.get(c)
                for v in self.vocab[c]:
                    X[i, k] = 1.0 if (val is not None and str(val) == v) else 0.0
                    k += 1
            if self.signals:
                sig = set(r.features.get("signals") or ())
                for s in SIGNAL_NAMES:
                    X[i, k] = 1.0 if s in sig else 0.0
                    k += 1
        return X


def feature_set(drop_groups: Sequence[str] = (), only: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str], bool]:
    """(numeric, categorical, use_signals) after removing ablated groups."""
    numeric, categorical, signals = list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES), True
    if only is not None:
        numeric = [c for c in numeric if c in only]
        categorical = [c for c in categorical if c in only]
        signals = "signal__" in only
    for g in drop_groups:
        cols = FEATURE_GROUPS[g]
        numeric = [c for c in numeric if c not in cols]
        categorical = [c for c in categorical if c not in cols]
        if "signal__" in cols:
            signals = False
    return numeric, categorical, signals


# ----------------------------------------------------------------------------- models
def _sk():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return LogisticRegression, StandardScaler


class Model:
    name = "model"
    kind = "probability"
    hyperparameters: Dict[str, Any] = {}

    def fit(self, train: Sequence[AuditRow], y: np.ndarray) -> "Model":
        return self

    def predict(self, rows: Sequence[AuditRow]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def available(self, r: AuditRow) -> bool:
        return True


class Majority(Model):
    name = "majority_baseline"

    def fit(self, train, y):
        self.p = float(np.mean(y))
        return self

    def predict(self, rows):
        return np.full(len(rows), self.p)


class RandomBaseline(Model):
    name = "random_baseline"

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def predict(self, rows):
        # deterministic per observation so folds/reruns are reproducible
        return np.array([int(hashlib.sha256(f"{self.seed}:{r.observation_id}".encode()).hexdigest()[:8], 16)
                         / 0xFFFFFFFF for r in rows])


class ScoreModel(Model):
    """A stored score (HSF Score, PreBreakout % as shown, AI Confidence) mapped
    to a probability by a 1-feature logistic fitted on the training fold."""
    kind = "score"

    def __init__(self, name: str, column: str, source: str = "features"):
        self.name, self.column, self.source = name, column, source
        self.hyperparameters = {"input": column, "probability_map": "train-fold logistic on the score"}

    def _value(self, r: AuditRow) -> Optional[float]:
        v = (r.features if self.source == "features" else r.extra).get(self.column)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def available(self, r):
        return self._value(r) is not None

    def fit(self, train, y):
        LogisticRegression, _ = _sk()
        x = np.array([[self._value(r)] for r in train], dtype=float)
        self.mu, self.sd = float(x.mean()), float(x.std() or 1.0)
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit((x - self.mu) / self.sd, y)
        return self

    def raw(self, rows):
        return np.array([self._value(r) for r in rows], dtype=float)

    def predict(self, rows):
        x = self.raw(rows).reshape(-1, 1)
        return self.lr.predict_proba((x - self.mu) / self.sd)[:, 1]


class Logistic(Model):
    name = "logistic_regression"

    def __init__(self, numeric=None, categorical=None, signals=True, name=None):
        n, c, s = feature_set()
        self.numeric = list(numeric if numeric is not None else n)
        self.categorical = list(categorical if categorical is not None else c)
        self.signals = signals
        if name:
            self.name = name
        self.hyperparameters = {"C": 1.0, "penalty": "l2", "max_iter": 2000, "scaling": "train-fold StandardScaler",
                                "imputation": "train-fold median", "class_weight": None}

    def fit(self, train, y):
        LogisticRegression, StandardScaler = _sk()
        self.enc = Encoder(self.numeric, self.categorical, self.signals).fit(train)
        X = self.enc.transform(train, impute=True)
        self.keep = np.nanstd(X, axis=0) > 0
        self.sc = StandardScaler().fit(X[:, self.keep])
        self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(self.sc.transform(X[:, self.keep]), y)
        return self

    def predict(self, rows):
        X = self.enc.transform(rows, impute=True)
        return self.lr.predict_proba(self.sc.transform(X[:, self.keep]))[:, 1]


XGB_PRODUCTION_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                             objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=42,
                             n_jobs=1)


class XGB(Model):
    def __init__(self, name: str, numeric=None, categorical=None, signals=True, params=None):
        n, c, s = feature_set()
        self.name = name
        self.numeric = list(numeric if numeric is not None else n)
        self.categorical = list(categorical if categorical is not None else c)
        self.signals = signals
        self.params = dict(params or XGB_PRODUCTION_PARAMS)
        self.hyperparameters = dict(self.params)

    def fit(self, train, y):
        from xgboost import XGBClassifier

        self.enc = Encoder(self.numeric, self.categorical, self.signals).fit(train)
        X = self.enc.transform(train, impute=False)
        self.m = XGBClassifier(**self.params).fit(X, y)
        return self

    def matrix(self, rows):
        return self.enc.transform(rows, impute=False)

    def predict(self, rows):
        return self.m.predict_proba(self.matrix(rows))[:, 1]


def production_feature_xgb() -> XGB:
    """Retrained XGBoost with the production AI Confidence features + params."""
    return XGB("xgb_retrained_production_features", numeric=list(AI_CONFIDENCE_FEATURE_MAP.values()),
               categorical=[], signals=False)


def safe_subset_xgb(leakage_classes: Mapping[str, str]) -> XGB:
    """XGBoost on every feature the leakage audit did not mark LEAKAGE/REVIEW."""
    bad = {f for f, c in leakage_classes.items() if c in (LEAKAGE, REVIEW)}
    numeric = [c for c in NUMERIC_FEATURES if c not in bad]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in bad]
    return XGB("xgb_leakage_safe_subset", numeric=numeric, categorical=categorical,
               params={**XGB_PRODUCTION_PARAMS, "n_estimators": 200, "max_depth": 3, "min_child_weight": 5})


# ----------------------------------------------------------------------------- metrics
def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    both = 0 < y.sum() < len(y)
    pred = (p > threshold).astype(int)
    out = {
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4) if len(y) else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if both else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4) if pred.sum() else None,
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4) if y.sum() else None,
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4) if pred.sum() and y.sum() else None,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "threshold": round(float(threshold), 4),
        "predicted_positive": int(pred.sum()),
    }
    return out


def financial_metrics(rows: Sequence[AuditRow], h: int) -> Dict[str, Any]:
    """Canonical Outcome Intelligence metrics (oi.metrics) for these rows."""
    recs = [r.oi_record for r in rows if r.oi_record is not None]
    m = oi.metrics(recs, h)
    keep = ("matured_count", "win_rate", "win_rate_ci95", "average_return", "median_return", "average_return_ci95",
            "benchmark_count", "average_benchmark_return", "median_benchmark_return", "average_excess_return",
            "median_excess_return", "benchmark_beat_rate", "benchmark_beat_rate_ci95", "mfe_count", "median_mfe",
            "average_mfe", "mae_count", "median_mae", "average_mae", "distinct_days", "evidence_quality")
    return {k: m.get(k) for k in keep}


def top_fraction(rows: Sequence[AuditRow], p: np.ndarray, frac: float = 1 / 3) -> List[AuditRow]:
    """Rows in the top `frac` of predictions (ties broken by observation id, so
    a constant predictor selects a deterministic, arbitrary subset)."""
    order = sorted(range(len(rows)), key=lambda i: (-float(p[i]), rows[i].observation_id))
    k = max(1, int(round(len(rows) * frac)))
    return [rows[i] for i in order[:k]]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Optional[float]:
    if len(y) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(y) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return round(total, 4)


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                    "mean_predicted": round(float(np.mean(p[m])), 4) if m.any() else None,
                    "observed_rate": round(float(np.mean(y[m])), 4) if m.any() else None})
    return out


def day_block_bootstrap_auc(y: np.ndarray, p: np.ndarray, days: Sequence[Any], *, n_boot: int = 1000,
                            seed: int = SEED) -> Optional[Dict[str, float]]:
    """95% CI for ROC-AUC resampling whole entry days (rows of one day are
    correlated through the shared market move)."""
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, d in enumerate(days):
        groups[d].append(i)
    keys = sorted(groups)
    if len(keys) < 3 or not (0 < y.sum() < len(y)):
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in pick for i in groups[keys[k]]]
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, p[idx]))
    if len(vals) < n_boot * 0.5:
        return None
    return {"low": round(float(np.percentile(vals, 2.5)), 4), "high": round(float(np.percentile(vals, 97.5)), 4),
            "method": f"entry-day block bootstrap, {len(vals)} resamples"}


def summarize(values: Sequence[Optional[float]], *, worst: str = "min") -> Dict[str, Optional[float]]:
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"folds": 0, "mean": None, "median": None, "std": None, "worst": None, "best": None}
    return {"folds": len(v), "mean": round(statistics.fmean(v), 4), "median": round(statistics.median(v), 4),
            "std": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0,
            "worst": round(min(v) if worst == "min" else max(v), 4),
            "best": round(max(v) if worst == "min" else min(v), 4)}


# ----------------------------------------------------------------------------- walk-forward engine
def run_walk_forward(data: Sequence[Tuple[AuditRow, int]], folds: Sequence[Fold], model_factory: Callable[[], Model],
                     h: int) -> Dict[str, Any]:
    """Fit/predict one model on every fold. Rows the model can't score (e.g. a
    stored % that is absent) are excluded from BOTH train and validation for
    that model and the count is reported, never silently."""
    by_id = {r.observation_id: (r, y) for r, y in data}
    fold_out, pooled = [], {"y": [], "p": [], "ids": [], "days": [], "fold": []}
    excluded_unscorable = 0
    for f in folds:
        m = model_factory()
        tr = [by_id[i] for i in f.train_ids]
        va = [by_id[i] for i in f.val_ids]
        tr_ok = [(r, y) for r, y in tr if m.available(r)]
        va_ok = [(r, y) for r, y in va if m.available(r)]
        excluded_unscorable += (len(tr) - len(tr_ok)) + (len(va) - len(va_ok))
        ytr = np.array([y for _, y in tr_ok], dtype=int)
        yva = np.array([y for _, y in va_ok], dtype=int)
        base = {"fold": f.fold, "train_n": len(tr_ok), "validation_n": len(va_ok),
                "validation_start": f.validation_start, "validation_end": f.validation_end}
        if len(tr_ok) < 20 or len(va_ok) < 10 or not (0 < ytr.sum() < len(ytr)):
            fold_out.append({**base, "skipped": "too few scorable rows or one class in training"})
            continue
        m.fit([r for r, _ in tr_ok], ytr)
        p = np.asarray(m.predict([r for r, _ in va_ok]), dtype=float)
        thr = float(ytr.mean())
        cm = classification_metrics(yva, p, thr)
        sel = top_fraction([r for r, _ in va_ok], p)
        fold_out.append({**base, **cm, "financial_top_tercile": financial_metrics(sel, h),
                         "financial_all_validation": financial_metrics([r for r, _ in va_ok], h)})
        for (r, y), pp in zip(va_ok, p):
            pooled["y"].append(int(y))
            pooled["p"].append(float(pp))
            pooled["ids"].append(r.observation_id)
            pooled["days"].append(r.entry_day.isoformat())
            pooled["fold"].append(f.fold)
    ok = [x for x in fold_out if "skipped" not in x]
    summary = {k: summarize([x.get(k) for x in ok], worst="max" if k in ("brier", "log_loss") else "min")
               for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "log_loss", "brier")}
    y_all, p_all = np.array(pooled["y"], dtype=int), np.array(pooled["p"], dtype=float)
    pooled_metrics = classification_metrics(y_all, p_all, float(y_all.mean()) if len(y_all) else 0.5) \
        if len(y_all) else None
    if pooled_metrics:
        pooled_metrics["roc_auc_ci95"] = day_block_bootstrap_auc(y_all, p_all, pooled["days"])
        pooled_metrics["ece"] = ece(y_all, p_all)
        sel_ids = set()
        for fo in sorted(set(pooled["fold"])):
            idx = [i for i, ff in enumerate(pooled["fold"]) if ff == fo]
            rows_f = [by_id[pooled["ids"][i]][0] for i in idx]
            sel_ids.update(r.observation_id for r in top_fraction(rows_f, p_all[idx]))
        pooled_metrics["financial_top_tercile"] = financial_metrics([by_id[i][0] for i in sel_ids], h)
        pooled_metrics["financial_all_validation"] = financial_metrics([by_id[i][0] for i in pooled["ids"]], h)
    return {"folds": fold_out, "summary": summary, "pooled": pooled_metrics, "predictions": pooled,
            "excluded_unscorable_rows": excluded_unscorable}


# ----------------------------------------------------------------------------- score calibration
def score_calibration(rows: Sequence[AuditRow]) -> List[Dict[str, Any]]:
    """Canonical score buckets x horizon, via oi.metrics; plus monotonicity."""
    out = []
    for h in HORIZONS:
        recs = [r for r in rows if r.matured(h) and r.oi_record is not None]
        bucket_rows = []
        for lo, hi in oi.SCORE_BUCKETS:
            grp = [r for r in recs if oi.score_bucket(r.features.get("hsf_score")) == f"{lo}-{hi}"]
            m = financial_metrics(grp, h)
            bucket_rows.append({"horizon": h, "bucket": f"{lo}-{hi}", "sample_size": len(grp), **m})
        view = oi.calibration_view([{**b, "matured_count": b["matured_count"] or 0,
                                     "benchmark_count": b["benchmark_count"] or 0,
                                     "mfe_count": b["mfe_count"] or 0} for b in bucket_rows])
        for b in bucket_rows:
            b["monotonicity"] = {k: v["monotonic"] for k, v in view["metrics"].items()}
            b["inversions"] = [f"{k}: {i['lower_bucket']} ({i['lower_value']}) > {i['higher_bucket']} "
                               f"({i['higher_value']})" for k, v in view["metrics"].items() for i in v["inversions"]]
        xs = [r.features["hsf_score"] for r in recs if r.features.get("hsf_score") is not None]
        ys = [r.labels.get(f"return_{h}d") for r in recs if r.features.get("hsf_score") is not None]
        rho = spearman(xs, ys)
        for b in bucket_rows:
            b["spearman_score_vs_return"] = rho
        out.extend(bucket_rows)
    return out


def spearman(x: Sequence[float], y: Sequence[Optional[float]]) -> Optional[Dict[str, float]]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 10:
        return None
    from scipy.stats import spearmanr

    r = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {"rho": round(float(r.statistic), 4), "p_value": round(float(r.pvalue), 4), "n": len(pairs)}


# ----------------------------------------------------------------------------- leakage audit
def leakage_table(rows: Sequence[AuditRow]) -> List[Dict[str, Any]]:
    """FEATURE / SOURCE / SCHEMA VERSION / COVERAGE / PIT STATUS / LEAKAGE RISK
    for every schema feature, plus the automated checks behind each class."""
    specs = {f.name: f for f in rs.feature_schema()}
    n = len(rows)
    out = []
    for name, spec in specs.items():
        cov = sum(1 for r in rows if r.features.get(name) not in (None, (), [])) / n if n else 0.0
        status = SAFE_ASOF_JOIN if spec.source == rs.SRC_SCAN or spec.source == rs.SRC_SCAN_META else SAFE_STORED_C
        risk, notes = "LOW", []
        if rs.looks_like_outcome(name):
            status, risk = LEAKAGE, "HIGH"
            notes.append("name matches an outcome pattern")
        if name == "prebreakout_prob":
            status, risk = REVIEW, "MEDIUM"
            notes.append("stored value as shown is point-in-time, but the served model version was never "
                         "recorded and the model's inputs changed 2026-10-07/08 (train/serve skew fix)")
        if name == "hsf_model_component":
            status, risk = REVIEW, "MEDIUM"
            notes.append("max(BreakoutScore, PreBreakout %): inherits the PreBreakout version gap")
        if name == "hsf_score":
            notes.append("value shown at fire time (frozen); formula version 1.0 throughout")
        if name in ("snapshot_rank", "snapshot_size"):
            status = SAFE_STORED_C
            notes.append("reconstructed from rows frozen at the same instant only")
        if spec.source in (rs.SRC_SCAN, rs.SRC_SCAN_META):
            notes.append("backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h")
        if cov == 0:
            status, risk = MISSING_C, "N/A"
        out.append({"feature": name, "source": spec.source, "stored_path": spec.path,
                    "schema_version": rs.FEATURE_SCHEMA_VERSION, "coverage": round(cov, 4),
                    "point_in_time_status": status, "leakage_risk": risk, "notes": "; ".join(notes)})
    for u in rs.UNAVAILABLE_FEATURES:
        out.append({"feature": u["name"], "source": "-", "stored_path": "-", "schema_version": None,
                    "coverage": 0.0, "point_in_time_status": MISSING_C if u["pit"] != rs.UNSAFE_CURRENT_VALUE else LEAKAGE,
                    "leakage_risk": "HIGH (if joined)" if u["pit"] == rs.UNSAFE_CURRENT_VALUE else "N/A",
                    "notes": u["reason"]})
    return out


def join_integrity(rows: Sequence[AuditRow]) -> Dict[str, Any]:
    """Every MATCHED scan join must have been written at or before observed_at."""
    lags = sorted(r.join_lag_s for r in rows if r.join_status == rd.JOIN_MATCHED and r.join_lag_s is not None)
    neg = sum(1 for v in lags if v < 0)
    q = (lambda k: lags[min(len(lags) - 1, int(k * (len(lags) - 1)))]) if lags else (lambda k: None)
    return {"matched": len(lags), "negative_lag_violations": neg,
            "lag_seconds": {"min": lags[0] if lags else None, "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
                            "p95": q(0.95), "max": lags[-1] if lags else None},
            "status_counts": dict(Counter(str(r.join_status) for r in rows))}


def single_feature_auc_scan(data: Sequence[Tuple[AuditRow, int]], *, flag: float = 0.85) -> List[Dict[str, Any]]:
    """In-sample single-feature AUC. A feature that alone separates the label
    almost perfectly is a classic leakage symptom (reported, not removed)."""
    from sklearn.metrics import roc_auc_score

    y = np.array([lab for _, lab in data], dtype=int)
    out = []
    if not (0 < y.sum() < len(y)):
        return out
    for c in NUMERIC_FEATURES:
        vals = np.array([Encoder._num(r.features.get(c)) for r, _ in data])
        ok = ~np.isnan(vals)
        if ok.sum() < 30 or len(set(y[ok])) < 2 or np.nanstd(vals) == 0:
            continue
        a = float(roc_auc_score(y[ok], vals[ok]))
        out.append({"feature": c, "n": int(ok.sum()), "auc": round(a, 4),
                    "suspicious": max(a, 1 - a) >= flag})
    return out


# ----------------------------------------------------------------------------- importance / ablation
def fold_importance(model: XGB, rows: Sequence[AuditRow], y: np.ndarray, *, repeats: int = 5,
                    seed: int = SEED) -> Dict[str, Dict[str, float]]:
    """Gain, permutation (validation ROC-AUC drop) and mean |SHAP| per column."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    X = model.matrix(rows)
    cols = model.enc.columns
    base = roc_auc_score(y, model.m.predict_proba(X)[:, 1])
    gain = model.m.get_booster().get_score(importance_type="gain")
    gain_vec = {cols[int(k[1:])]: v for k, v in gain.items()} if gain and next(iter(gain)).startswith("f") else {
        k: v for k, v in gain.items()}
    contribs = model.m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    shap = np.abs(contribs).mean(axis=0)
    rng = np.random.default_rng(seed)
    out = {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - roc_auc_score(y, model.m.predict_proba(Xp)[:, 1]))
        out[c] = {"gain": float(gain_vec.get(c, 0.0)), "permutation_auc_drop": float(np.mean(drops)),
                  "shap_mean_abs": float(shap[j])}
    return out


def column_group(col: str) -> str:
    base = col.split("=")[0]
    for g, cols in FEATURE_GROUPS.items():
        if base in cols or (col.startswith("signal__") and "signal__" in cols):
            return g
    return "other"


def importance_report(data: Sequence[Tuple[AuditRow, int]], folds: Sequence[Fold], leakage: Mapping[str, str],
                      h: int) -> List[Dict[str, Any]]:
    by_id = {r.observation_id: (r, y) for r, y in data}
    per_fold: List[Dict[str, Dict[str, float]]] = []
    for f in folds:
        tr = [by_id[i] for i in f.train_ids]
        va = [by_id[i] for i in f.val_ids]
        yv = np.array([y for _, y in va])
        if not (0 < yv.sum() < len(yv)):
            continue
        m = safe_subset_xgb(leakage).fit([r for r, _ in tr], np.array([y for _, y in tr]))
        per_fold.append(fold_importance(m, [r for r, _ in va], yv))
    cols = sorted({c for d in per_fold for c in d})
    out = []
    for c in cols:
        perm = [d[c]["permutation_auc_drop"] for d in per_fold if c in d]
        gain = [d[c]["gain"] for d in per_fold if c in d]
        shap = [d[c]["shap_mean_abs"] for d in per_fold if c in d]
        mean_perm = statistics.fmean(perm) if perm else 0.0
        pos_share = sum(1 for v in perm if v > 0) / len(perm) if perm else 0.0
        base = c.split("=")[0]
        if leakage.get(base) in (LEAKAGE, REVIEW):
            cls = "LEAKAGE RISK"
        elif len(perm) >= 2 and mean_perm > 0.005 and pos_share < 0.6:
            cls = "UNSTABLE"
        elif mean_perm >= 0.02 and pos_share >= 0.75:
            cls = "HIGH VALUE"
        elif mean_perm >= 0.005 and pos_share >= 0.6:
            cls = "MODERATE VALUE"
        elif (statistics.fmean(gain) if gain else 0) > 0 and abs(mean_perm) < 0.001:
            cls = "REDUNDANT"
        else:
            cls = "LOW VALUE"
        out.append({"horizon": h, "feature": c, "group": column_group(c), "folds": len(perm),
                    "gain_mean": round(statistics.fmean(gain), 4) if gain else None,
                    "permutation_auc_drop_mean": round(mean_perm, 4),
                    "permutation_auc_drop_std": round(statistics.pstdev(perm), 4) if len(perm) > 1 else None,
                    "permutation_positive_fold_share": round(pos_share, 2),
                    "shap_mean_abs": round(statistics.fmean(shap), 4) if shap else None,
                    "classification": cls,
                    "leakage_status": leakage.get(base, "derived (one-hot / signal flag)")})
    out.sort(key=lambda r: -r["permutation_auc_drop_mean"])
    return out


# ----------------------------------------------------------------------------- regimes / concentration / stability
def spy_regime_by_day(spy_closes: Mapping[_dt.date, float], days: Iterable[_dt.date]) -> Dict[_dt.date, Dict[str, Any]]:
    """Regime per entry day from SPY closes STRICTLY BEFORE that day (the prior
    session's close is the newest bar a regime label may read)."""
    closes = sorted((d, c) for d, c in spy_closes.items() if c is not None)
    out = {}
    for day in days:
        hist = [c for d, c in closes if d < day]
        if len(hist) < 21:
            out[day] = {"trend": None, "volatility": None}
            continue
        ret20 = hist[-1] / hist[-21] - 1
        rets = [hist[i] / hist[i - 1] - 1 for i in range(len(hist) - 20, len(hist))]
        vol = statistics.pstdev(rets) * math.sqrt(252)
        out[day] = {"trend": "bullish" if ret20 > 0.02 else ("bearish" if ret20 < -0.02 else "sideways"),
                    "volatility": "high" if vol >= 0.20 else "low", "spy_ret20": round(ret20, 4),
                    "spy_vol20_annual": round(vol, 4)}
    return out


def concentration(rows: Sequence[AuditRow]) -> Dict[str, Any]:
    c = Counter(r.ticker for r in rows)
    n = len(rows)
    top = c.most_common(10)
    return {"rows": n, "unique_tickers": len(c),
            "top_ticker": {"ticker": top[0][0], "rows": top[0][1], "share": round(top[0][1] / n, 4)} if top else None,
            "top10_share": round(sum(v for _, v in top) / n, 4) if n else None,
            "top10": [{"ticker": t, "rows": v} for t, v in top],
            "setups": dict(Counter(r.setup or "UNKNOWN" for r in rows).most_common()),
            "days": len({r.entry_day for r in rows}),
            "max_day_share": round(max(Counter(r.entry_day for r in rows).values()) / n, 4) if n else None}


def subset_auc(pred: Mapping[str, list], keep_ids: Iterable[int]) -> Optional[Dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    keep = set(keep_ids)
    idx = [i for i, oid in enumerate(pred["ids"]) if oid in keep]
    y = np.array([pred["y"][i] for i in idx], dtype=int)
    p = np.array([pred["p"][i] for i in idx], dtype=float)
    if len(y) < 20 or not (0 < y.sum() < len(y)):
        return {"n": len(y), "roc_auc": None}
    return {"n": len(y), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "roc_auc_ci95": day_block_bootstrap_auc(y, p, [pred["days"][i] for i in idx])}


def temporal_buckets(pred: Mapping[str, list], by_id: Mapping[int, AuditRow], h: int, *, period: str = "week",
                     min_n: int = 20) -> List[Dict[str, Any]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, oid in enumerate(pred["ids"]):
        d = by_id[oid].entry_day
        key = f"{d.isocalendar().year}-W{d.isocalendar().week:02d}" if period == "week" else d.strftime("%Y-%m")
        groups[key].append(i)
    out = []
    for key in sorted(groups):
        idx = groups[key]
        y = np.array([pred["y"][i] for i in idx], dtype=int)
        p = np.array([pred["p"][i] for i in idx], dtype=float)
        both = 0 < y.sum() < len(y)
        fin = financial_metrics([by_id[pred["ids"][i]] for i in idx], h)
        out.append({"period": key, "n": len(idx), "sufficient": len(idx) >= min_n,
                    "roc_auc": round(float(roc_auc_score(y, p)), 4) if both and len(idx) >= min_n else None,
                    "pr_auc": round(float(average_precision_score(y, p)), 4) if both and len(idx) >= min_n else None,
                    "brier": round(float(brier_score_loss(y, p)), 4),
                    "win_rate": fin["win_rate"], "median_excess_return": fin["median_excess_return"],
                    "benchmark_beat_rate": fin["benchmark_beat_rate"]})
    return out


def stability_verdict(values: Sequence[Optional[float]]) -> str:
    v = [x for x in values if x is not None]
    if len(v) < 3:
        return "INSUFFICIENT (fewer than 3 evaluable periods)"
    slope = np.polyfit(range(len(v)), v, 1)[0]
    spread = max(v) - min(v)
    if spread < 0.08:
        return "STABLE"
    if slope > 0.02:
        return "IMPROVING"
    if slope < -0.02:
        return "DEGRADING"
    return "VOLATILE / REGIME DEPENDENT"


# ----------------------------------------------------------------------------- calibration (offline only)
def offline_calibration(pred: Mapping[str, list]) -> Dict[str, Any]:
    """Platt and isotonic fitted ONLY on earlier folds' out-of-sample
    predictions, applied to the next fold. Fold 1 has no history and is left
    out of all three columns so the comparison is like for like."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    folds = sorted(set(pred["fold"]))
    y_all = np.array(pred["y"], dtype=int)
    p_all = np.array(pred["p"], dtype=float)
    f_all = np.array(pred["fold"])
    raw_y, raw_p, platt_p, iso_p = [], [], [], []
    for fo in folds[1:]:
        hist = f_all < fo
        cur = f_all == fo
        if hist.sum() < 30 or len(set(y_all[hist])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000).fit(p_all[hist].reshape(-1, 1), y_all[hist])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_all[hist], y_all[hist])
        raw_y += list(y_all[cur])
        raw_p += list(p_all[cur])
        platt_p += list(lr.predict_proba(p_all[cur].reshape(-1, 1))[:, 1])
        iso_p += list(iso.predict(p_all[cur]))
    if not raw_y:
        return {"evaluated_rows": 0, "note": "needs at least two folds with 30+ earlier predictions"}
    y = np.array(raw_y)
    res = {"evaluated_rows": len(y), "folds_evaluated": len(folds) - 1}
    for name, p in (("raw", raw_p), ("platt", platt_p), ("isotonic", iso_p)):
        p = np.clip(np.array(p), 1e-6, 1 - 1e-6)
        res[name] = {"brier": round(float(brier_score_loss(y, p)), 4), "ece": ece(y, p),
                     "reliability": reliability_bins(y, p)}
    return res


# ----------------------------------------------------------------------------- drift baseline
def distribution(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    n_all = len(values)
    if len(v) == 0:
        return {"n": n_all, "non_null": 0}
    qs = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": n_all, "non_null": int(len(v)), "null_rate": round(1 - len(v) / n_all, 4) if n_all else None,
            "mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
            "p05": round(float(qs[0]), 6), "p25": round(float(qs[1]), 6), "p50": round(float(qs[2]), 6),
            "p75": round(float(qs[3]), 6), "p95": round(float(qs[4]), 6)}


def drift_baseline(rows: Sequence[AuditRow], predictions: Mapping[str, Mapping[str, list]]) -> Dict[str, Any]:
    feats = {c: distribution([Encoder._num(r.features.get(c)) if r.features.get(c) is not None else None
                              for r in rows]) for c in NUMERIC_FEATURES}
    labels = {}
    for h in HORIZONS:
        mat = [r for r in rows if r.matured(h)]
        labels[f"{h}d"] = {"matured": len(mat),
                           "return": distribution([r.labels.get(f"return_{h}d") for r in mat]),
                           "excess_return": distribution([r.labels.get(f"excess_return_{h}d") for r in mat]),
                           "win_rate": round(sum(1 for r in mat if (r.labels.get(f"return_{h}d") or 0) > 0) / len(mat), 4)
                           if mat else None}
    return {"rows": len(rows),
            "window": {"first": rows[0].observed_at.isoformat() if rows else None,
                       "last": rows[-1].observed_at.isoformat() if rows else None},
            "features": feats, "hsf_score": feats["hsf_score"],
            "predictions": {k: distribution(v["p"]) for k, v in predictions.items()},
            "labels": labels,
            "mfe_5d": distribution([r.labels.get("mfe_5d") for r in rows if r.matured(5)]),
            "mae_5d": distribution([r.labels.get("mae_5d") for r in rows if r.matured(5)]),
            "setups": dict(Counter(r.setup or "UNKNOWN" for r in rows).most_common()),
            "tickers_top20": dict(Counter(r.ticker for r in rows).most_common(20)),
            "unique_tickers": len({r.ticker for r in rows})}


# ----------------------------------------------------------------------------- legacy reproduction
def legacy_split_diagnostics(frame, *, horizon_days: int = 5, validation_fraction: float = 0.2) -> Dict[str, Any]:
    """Explain the production training split on the legacy (runs-table) frame.

    ``frame`` columns: Symbol, Timestamp (UTC), run_label, the 6 AI Confidence
    features, ForwardReturnHit. Pure: no I/O, no model."""
    import pandas as pd

    from analytics import market_calendar as mc

    df = frame.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df[df["Timestamp"].notna()]
    order = df["Timestamp"].sort_values(kind="mergesort").index
    split_at = max(1, int(len(order) * (1 - validation_fraction)))
    tr, va = order[:split_at], order[split_at:]
    val_start = df.loc[va, "Timestamp"].min()
    vstart_day = val_start.date()

    def end_day(ts):
        d = ts.date()
        while not mc.is_trading_day(d):
            d += _dt.timedelta(days=1)
        return rd.label_window_end(d, horizon_days)

    ends = df.loc[tr, "Timestamp"].map(end_day)
    overlapping = int((ends >= vstart_day).sum())
    snap = df["run_label"].astype(str).eq("daily_snapshot") if "run_label" in df.columns else pd.Series(False, index=df.index)
    feats = [c for c in df.columns if c in AI_CONFIDENCE_FEATURE_MAP]
    key = df["Symbol"].astype(str) + "|" + df["Timestamp"].dt.date.astype(str)
    vec = df[feats].round(6).astype(str).agg("|".join, axis=1) if feats else pd.Series("", index=df.index)
    tr_keys = set(key.loc[tr])
    tr_vecs = set((key + "#" + vec).loc[tr])
    return {
        "rows": int(len(df)), "train_rows": int(len(tr)), "validation_rows": int(len(va)),
        "train_range": [df.loc[tr, "Timestamp"].min().isoformat(), df.loc[tr, "Timestamp"].max().isoformat()],
        "validation_range": [val_start.isoformat(), df.loc[va, "Timestamp"].max().isoformat()],
        "daily_snapshot_rows": int(snap.sum()),
        "rows_per_symbol_day": round(float(len(df) / max(1, key.nunique())), 2),
        "exact_duplicate_feature_rows": int((key + "#" + vec).duplicated().sum()),
        "train_rows_label_window_overlaps_validation": overlapping,
        "validation_rows_whose_symbol_day_is_in_train": int(key.loc[va].isin(tr_keys).sum()),
        "validation_rows_identical_to_a_train_row": int((key + "#" + vec).loc[va].isin(tr_vecs).sum()),
        "positive_rate_train": round(float(df.loc[tr, "ForwardReturnHit"].mean()), 4),
        "positive_rate_validation": round(float(df.loc[va, "ForwardReturnHit"].mean()), 4),
        "preprocessing": "fillna(0.0) per column (no statistics fitted, so nothing crosses the split); "
                         "isotonic calibration map fitted on the SAME validation predictions the AUC is reported on",
        "split_method": f"chronological {int((1 - validation_fraction) * 100)}/{int(validation_fraction * 100)} "
                        "by row Timestamp, no purge, no embargo, no dedup",
    }


def legacy_variants(frame, *, horizon_days: int = 5, embargo: int = 1, params: Optional[Mapping[str, Any]] = None,
                    n_splits: int = 5) -> List[Dict[str, Any]]:
    """Re-evaluate the production AI Confidence recipe on the same legacy rows
    under progressively stricter validation. Same features, same params."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from analytics import market_calendar as mc

    params = dict(params or XGB_PRODUCTION_PARAMS)
    feats = list(AI_CONFIDENCE_FEATURE_MAP)
    df = frame.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df[df["Timestamp"].notna()].sort_values(["Timestamp", "Symbol"], kind="mergesort").reset_index(drop=True)
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = df["ForwardReturnHit"].astype(int).to_numpy()

    def entry(ts):
        d = ts.date()
        while not mc.is_trading_day(d):
            d += _dt.timedelta(days=1)
        return d

    df["entry"] = df["Timestamp"].map(entry)
    df["wend"] = df["entry"].map(lambda d: rd.label_window_end(d, horizon_days))
    entry_a = df["entry"].to_numpy()
    wend_a = df["wend"].to_numpy()

    def fit_auc(tr_idx, va_idx):
        tr_idx, va_idx = np.asarray(tr_idx, dtype=int), np.asarray(va_idx, dtype=int)
        if len(tr_idx) < 50 or len(va_idx) < 20:
            return None
        ytr, yva = y[tr_idx], y[va_idx]
        if len(set(ytr)) < 2 or len(set(yva)) < 2:
            return None
        m = XGBClassifier(**params).fit(X[tr_idx], ytr)
        return round(float(roc_auc_score(yva, m.predict_proba(X[va_idx])[:, 1])), 4)

    out = []
    n = len(df)
    split = int(n * 0.8)
    tr, va = np.arange(split), np.arange(split, n)
    out.append({"variant": "legacy_reproduction (chronological 80/20, all rows)", "train_n": len(tr),
                "validation_n": len(va), "roc_auc": fit_auc(tr, va)})
    rtr, rva = train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)
    out.append({"variant": "random 80/20 split (reference only, NOT a benchmark)", "train_n": len(rtr),
                "validation_n": len(rva), "roc_auc": fit_auc(rtr, rva)})
    vstart = entry_a[split]
    cutoff = _trading_days_before(vstart, embargo)
    keep_tr = tr[(wend_a[tr] < cutoff) & (entry_a[tr] < vstart)]
    out.append({"variant": f"chronological 80/20 + purge (label window) + {embargo}-day embargo",
                "train_n": len(keep_tr), "validation_n": len(va), "roc_auc": fit_auc(keep_tr, va)})
    nosnap = (~df["run_label"].astype(str).eq("daily_snapshot")).to_numpy() if "run_label" in df.columns \
        else np.ones(n, bool)
    day_key = (df["Symbol"].astype(str) + "|" + df["entry"].astype(str)).to_numpy()
    first = (~pd.Series(day_key).duplicated()).to_numpy() & nosnap
    d_tr = keep_tr[first[keep_tr]]
    train_keys = set(day_key[d_tr])
    d_va = np.array([i for i in va if first[i] and day_key[i] not in train_keys], dtype=int)
    out.append({"variant": "... + one row per symbol/entry day (no daily-snapshot copies)", "train_n": len(d_tr),
                "validation_n": len(d_va), "roc_auc": fit_auc(d_tr, d_va)})
    # expanding walk-forward on the deduped rows, purge + embargo per fold
    ded = np.where(first)[0]
    days = sorted(set(entry_a[ded]))
    bounds = np.linspace(0, len(days), n_splits + 2, dtype=int)
    fold_aucs = []
    for k in range(1, n_splits + 1):
        vdays = days[bounds[k]:bounds[k + 1]]
        if not vdays:
            continue
        vs, ve = vdays[0], vdays[-1]
        cut = _trading_days_before(vs, embargo)
        trk = ded[(entry_a[ded] < vs) & (wend_a[ded] < cut)]
        vak = ded[(entry_a[ded] >= vs) & (entry_a[ded] <= ve)]
        if len(trk) < 50 or len(vak) < 30:
            continue
        a = fit_auc(trk, vak)
        fold_aucs.append({"fold": k, "validation_start": vs.isoformat(), "validation_end": ve.isoformat(),
                          "train_n": len(trk), "validation_n": len(vak), "roc_auc": a})
    s = summarize([f["roc_auc"] for f in fold_aucs])
    out.append({"variant": f"expanding walk-forward ({len(fold_aucs)} folds), deduped, purge + embargo",
                "train_n": None, "validation_n": sum(f["validation_n"] for f in fold_aucs),
                "roc_auc": s["mean"], "fold_summary": s, "folds": fold_aucs})
    return out


# ----------------------------------------------------------------------------- experiment records
def experiment_record(*, experiment_id: str, manifest: Mapping[str, Any], model: Model, label: str, h: int,
                      folds: Sequence[Fold], purge: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "audit_version": AUDIT_VERSION,
        "dataset_version": manifest.get("dataset_version"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "audit_slice_fingerprint": manifest.get("audit_slice_fingerprint"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "label_schema_version": manifest.get("label_schema_version"),
        "git_revision": manifest.get("git_revision"),
        "model": model.name, "model_kind": model.kind, "hyperparameters": model.hyperparameters,
        "label": label, "horizon_days": h,
        "train_windows": [[f.train_start, f.train_end] for f in folds],
        "validation_windows": [[f.validation_start, f.validation_end] for f in folds],
        "purge": dict(purge),
        "metrics": {"summary": result.get("summary"), "pooled": {k: v for k, v in (result.get("pooled") or {}).items()},
                    "folds": result.get("folds")},
        "excluded_unscorable_rows": result.get("excluded_unscorable_rows"),
    }


def slice_fingerprint(rows: Sequence[AuditRow]) -> str:
    """Fingerprint of exactly what the audit consumed (ids, features, labels,
    maturity) — stable while matured outcomes don't change."""
    payload = [{"id": r.observation_id, "f": {k: rd._canon(v) for k, v in sorted(r.features.items())},
                "y": {k: rd._canon(v) for k, v in sorted(r.labels.items())}, "m": r.maturity,
                "c": r.certified} for r in rows]
    return rd.fingerprint({"audit_version": AUDIT_VERSION, "rows": payload})


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=1, sort_keys=True, default=str)


# ----------------------------------------------------------------------------- legacy v1-era recipes
def _binary_flag(series):
    import pandas as pd

    def one(v):
        if isinstance(v, str):
            return 1 if v.strip().lower() in ("1", "true", "yes", "y", "t") else 0
        try:
            return 1 if (v is not None and not pd.isna(v) and float(v) != 0) else 0
        except (TypeError, ValueError):
            return 0
    return series.map(one).astype(int)


def future_breakout_label(frame, horizon_scans: int = 3):
    """Vectorized, exact ``ml_prebreakout.add_future_breakout_label``.

    The reference looks at the next ``horizon_scans`` rows of the WHOLE frame
    (sorted by Symbol, Timestamp) and labels 1 when any of them has the same
    symbol AND any of them is a breakout. The two conditions need not hold on
    the same row, so a symbol's last rows can inherit the next symbol's
    breakout. That quirk is reproduced on purpose (``label_bleed`` marks it).
    Also returns the latest timestamp among the rows the label reads."""
    import pandas as pd

    df = frame.sort_values(["Symbol", "Timestamp"], kind="mergesort").reset_index(drop=True).copy()
    df["IsBreakout"] = _binary_flag(df["IsBreakout"])
    any_brk = pd.Series(False, index=df.index)
    same_sym = pd.Series(False, index=df.index)
    same_sym_brk = pd.Series(False, index=df.index)
    last_ts = df["Timestamp"].copy()
    for k in range(1, horizon_scans + 1):
        brk_k = df["IsBreakout"].shift(-k).fillna(0).astype(int).astype(bool)
        sym_k = df["Symbol"].shift(-k).eq(df["Symbol"])
        any_brk |= brk_k
        same_sym |= sym_k
        same_sym_brk |= brk_k & sym_k
        ts_k = df["Timestamp"].shift(-k)
        last_ts = last_ts.where(ts_k.isna() | (ts_k <= last_ts), ts_k)
    df["FutureBreakout"] = (any_brk & same_sym).astype(int)
    df["label_bleed"] = (df["FutureBreakout"].astype(bool) & ~same_sym_brk).astype(int)
    df["label_window_end_ts"] = last_ts
    return df


def legacy_v1_recipes(frame, *, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Reproduce the two July-era recipes on today's runs history and re-test
    them under chronological, purged, deduplicated validation.

    PreBreakout v1 (prebreakout-xgb-v1, AUC 0.936-0.962): label = IsBreakout in
        any of the next 3 scan rows of the symbol; 6 scanner features incl.
        BreakoutScore; sklearn train_test_split(test_size=0.2, stratify=y,
        random_state=42).
    AI Confidence v1 (Jul, AUC 0.998): label = IsBreakout of the SAME row;
        same 6 features; same random split.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    params = dict(params or XGB_PRODUCTION_PARAMS)
    feats = list(AI_CONFIDENCE_FEATURE_MAP)
    df = frame.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df[df["Timestamp"].notna()]
    df = future_breakout_label(df)
    df = df.sort_values(["Timestamp", "Symbol"], kind="mergesort").reset_index(drop=True)
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    n = len(df)

    def auc_for(y, tr, va):
        if len(tr) < 50 or len(va) < 20 or len(set(y[tr])) < 2 or len(set(y[va])) < 2:
            return None
        m = XGBClassifier(**params).fit(X[tr], y[tr])
        return round(float(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])), 4)

    def single(y, col):
        v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        if len(set(y)) < 2:
            return None
        return round(float(roc_auc_score(y, v)), 4)

    out: Dict[str, Any] = {"rows": n, "symbols": int(df["Symbol"].nunique()),
                           "range": [df["Timestamp"].min().isoformat(), df["Timestamp"].max().isoformat()]
                           if n else None}
    if n < 100:
        out["skipped"] = "fewer than 100 legacy rows"
        return out
    nxt = df.sort_values(["Symbol", "Timestamp"]).groupby("Symbol")["Timestamp"].diff().dt.total_seconds() / 60
    out["median_minutes_between_scans_of_a_symbol"] = round(float(nxt.median()), 1) if nxt.notna().any() else None
    split = int(n * 0.8)
    tr_c, va_c = np.arange(split), np.arange(split, n)
    vstart = df.loc[split, "Timestamp"]
    snap = df["run_label"].astype(str).eq("daily_snapshot").to_numpy() if "run_label" in df.columns else np.zeros(n, bool)
    for name, ycol, desc in (("prebreakout_v1", "FutureBreakout", "IsBreakout in any of the next 3 scan rows"),
                             ("ai_confidence_v1", "IsBreakout", "IsBreakout of the same row")):
        y = df[ycol].astype(int).to_numpy()
        strat = y if len(set(y)) > 1 else None
        tr_r, va_r = train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=strat)
        res = {"label": desc, "positive_rate": round(float(y.mean()), 4),
               "reproduced_random_split_auc": auc_for(y, tr_r, va_r),
               "chronological_80_20_auc": auc_for(y, tr_c, va_c),
               "single_feature_auc_BreakoutScore": single(y, "BreakoutScore")}
        if ycol == "FutureBreakout":
            cur = df["IsBreakout"].astype(int).to_numpy()
            res["positives_from_cross_symbol_bleed"] = int(df["label_bleed"].sum())
            res["share_of_positives_already_breakout_now"] = round(float(cur[y == 1].mean()), 4) if y.sum() else None
            res["auc_of_current_IsBreakout_flag_alone"] = round(float(roc_auc_score(y, cur)), 4) if len(set(y)) > 1 else None
            ends = df["label_window_end_ts"]
            purge_ok = np.array([i for i in tr_c if pd.notna(ends[i]) and ends[i] < vstart])
            res["chronological_purged_auc"] = auc_for(y, purge_ok, va_c)
            keep = ~snap
            res["chronological_purged_no_snapshot_copies_auc"] = auc_for(
                y, np.array([i for i in purge_ok if keep[i]]), np.array([i for i in va_c if keep[i]]))
            res["purged_training_rows"] = int(len(tr_c) - len(purge_ok))
        out[name] = res
    out["split_facts"] = {"train_rows": int(split), "validation_rows": int(n - split),
                          "validation_start": vstart.isoformat(), "daily_snapshot_rows": int(snap.sum())}
    return out
