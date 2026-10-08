"""Versioned feature and label schemas for the HSF research dataset (pure).

The research layer keeps a hard boundary at ``observed_at``: features are what
HSF knew at that moment, labels are what happened afterwards. This module owns
the two contracts and the two record types that carry them, so training code
has to join features and labels explicitly and can never receive them in one
mixed dictionary.

  * ``FEATURE_SCHEMA`` (version ``FEATURE_SCHEMA_VERSION``): every feature a
    research row may carry, with its persisted source, type, nullability and
    point-in-time classification. Adding, dropping or redefining a feature means
    a new schema version; the old one stays importable.
  * ``LABEL_SCHEMA`` (version ``LABEL_SCHEMA_VERSION``): the outcome definitions
    that already exist in production (1/3/5 trading-day close returns, 5-day
    MFE/MAE, SPY benchmark and excess return). No new labels are introduced.
  * ``FeatureSnapshot`` / ``OutcomeRecord``: frozen records. A snapshot refuses
    any key that is not a declared feature, and any key that reads like an
    outcome, so a label can't slip into a feature payload by accident.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

FEATURE_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1

# Point-in-time classifications (P0 audit).
SAFE_STORED = "SAFE_STORED"
SAFE_RECONSTRUCTABLE = "SAFE_RECONSTRUCTABLE"
UNSAFE_CURRENT_VALUE = "UNSAFE_CURRENT_VALUE"
MISSING = "MISSING"
UNKNOWN = "UNKNOWN"
PIT_CLASSES = (SAFE_STORED, SAFE_RECONSTRUCTABLE, UNSAFE_CURRENT_VALUE, MISSING, UNKNOWN)

# Market-context readiness classes (P1 audit).
AVAILABLE_STORED = "AVAILABLE_STORED"
SAFE_TO_DERIVE = "SAFE_TO_DERIVE"
NOT_AVAILABLE = "NOT_AVAILABLE"
UNSAFE = "UNSAFE"

# Persisted sources.
SRC_OPPORTUNITY = "signal_outcomes(source='opportunity')"
SRC_SCAN = "hsf_observations(context='scheduled:*')"
SRC_SCAN_META = "hsf_observations.research_metadata (Run 57+)"
SRC_SNAPSHOT_DERIVED = "derived from the same frozen snapshot"

# Anything that reads like hindsight can never be a feature name or key.
_OUTCOME_PATTERN = re.compile(
    r"(return|outcome|mfe|mae|future|matur|evaluation|horizon|excess|benchmark|"
    r"label|hit\b|win|pnl|profit|forward|realized|certified|exit|outcome_price)", re.I)


def looks_like_outcome(name: str) -> bool:
    return bool(_OUTCOME_PATTERN.search(str(name)))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    type: str          # float | int | str | bool | list[str]
    source: str
    path: str          # where the stored value lives
    pit: str           # SAFE_STORED | SAFE_RECONSTRUCTABLE
    description: str
    nullable: bool = True
    point_in_time: bool = True


@dataclass(frozen=True)
class LabelSpec:
    name: str
    type: str
    source: str
    horizon_days: Optional[int]
    description: str
    nullable: bool = True


_FEATURES_V1: Tuple[FeatureSpec, ...] = (
    # HSF state frozen on the opportunity row at fire time.
    FeatureSpec("hsf_score", "float", SRC_OPPORTUNITY, "raw_signal.hsf_score", SAFE_STORED,
                "HSF Score shown at observation time (0-100)."),
    FeatureSpec("hsf_score_version", "str", SRC_OPPORTUNITY, "raw_signal.score_version", SAFE_STORED,
                "Score formula version recorded at fire time (null = legacy row, canonically 1.0)."),
    FeatureSpec("hsf_signals_component", "float", SRC_OPPORTUNITY, "raw_signal.score_components.signals_component",
                SAFE_STORED, "Confluence part of the HSF Score."),
    FeatureSpec("hsf_model_component", "float", SRC_OPPORTUNITY, "raw_signal.score_components.model_component",
                SAFE_STORED, "Breakout/PreBreakout model part of the HSF Score."),
    FeatureSpec("hsf_momentum_component", "float", SRC_OPPORTUNITY, "raw_signal.score_components.momentum_component",
                SAFE_STORED, "Momentum part of the HSF Score."),
    FeatureSpec("hsf_fading_penalty", "float", SRC_OPPORTUNITY, "raw_signal.score_components.fading_penalty",
                SAFE_STORED, "Fading penalty applied to the HSF Score."),
    FeatureSpec("primary_setup", "str", SRC_OPPORTUNITY, "raw_signal.primary_setup", SAFE_STORED,
                "Primary setup label at fire time."),
    FeatureSpec("hsf_status", "str", SRC_OPPORTUNITY, "raw_signal.status", SAFE_STORED,
                "STRONG / WATCH / CAUTION at fire time."),
    FeatureSpec("signals", "list[str]", SRC_OPPORTUNITY, "indicators.signals", SAFE_STORED,
                "Confirming signal lists the ticker appeared in.", nullable=False),
    FeatureSpec("n_signals", "int", SRC_OPPORTUNITY, "indicators.n_signals", SAFE_STORED,
                "Number of positive confirming signals."),
    FeatureSpec("fading", "bool", SRC_OPPORTUNITY, "indicators.fading", SAFE_STORED,
                "Ticker was in the day's losers list."),
    FeatureSpec("chg_pct", "float", SRC_OPPORTUNITY, "indicators.chg_pct", SAFE_STORED,
                "Day change % from the gapper/mover list (null when the ticker was in neither)."),
    FeatureSpec("gap_pct", "float", SRC_OPPORTUNITY, "indicators.gap_pct", SAFE_STORED,
                "Gap % from the gapper list (null when not a gapper)."),
    FeatureSpec("breakout_score", "float", SRC_OPPORTUNITY, "setup_score", SAFE_STORED,
                "Breakout setup score when the ticker was a top setup."),
    FeatureSpec("prebreakout_prob", "float", SRC_OPPORTUNITY, "prebreakout_prob", SAFE_STORED,
                "PreBreakout probability shown at fire time (picks only). Served model version not recorded."),
    FeatureSpec("snapshot_rank", "int", SRC_SNAPSHOT_DERIVED, "rank by hsf_score within the same fired_at",
                SAFE_RECONSTRUCTABLE,
                "1 = highest HSF Score among the opportunities frozen for the same snapshot; ties by ticker."),
    FeatureSpec("snapshot_size", "int", SRC_SNAPSHOT_DERIVED, "count of rows with the same fired_at",
                SAFE_RECONSTRUCTABLE, "How many opportunities were frozen for that snapshot."),
    # Scanner state frozen per scheduled scan (joined backward in time).
    FeatureSpec("price", "float", SRC_SCAN, "record.market.price", SAFE_STORED,
                "Last price in the scheduled scan (scanner 'Last')."),
    FeatureSpec("volume", "float", SRC_SCAN, "record.market.volume", SAFE_STORED,
                "Day volume in the scheduled scan."),
    FeatureSpec("rvol_20", "float", SRC_SCAN, "record.indicators.rvol", SAFE_STORED,
                "Volume relative to the 20-day average (scanner 'VolRel20')."),
    FeatureSpec("volatility_20d_pct", "float", SRC_SCAN, "record.indicators.atr_pct", SAFE_STORED,
                "20-day daily-return volatility % (scanner 'Volatility20D%')."),
    FeatureSpec("scan_gap_pct", "float", SRC_SCAN, "record.indicators.gap_pct", SAFE_STORED,
                "Gap % as computed by the scan."),
    FeatureSpec("scan_chg_pct", "float", SRC_SCAN, "record.indicators.chg_pct", SAFE_STORED,
                "Change % as computed by the scan."),
    FeatureSpec("scanner_breakout_score", "float", SRC_SCAN, "record.scanners[name=breakout].score", SAFE_STORED,
                "BreakoutScore of the scan row."),
    FeatureSpec("is_breakout", "bool", SRC_SCAN, "record.scanners[name=breakout].meta.is_breakout", SAFE_STORED,
                "Scanner IsBreakout flag."),
    FeatureSpec("trend_10d_pct", "float", SRC_SCAN_META, "record.research_metadata.row_features.trend_10d_pct",
                SAFE_STORED, "10-day trend % (scanner 'Trend10D%')."),
    FeatureSpec("trend_20d_pct", "float", SRC_SCAN_META, "record.research_metadata.row_features.trend_20d_pct",
                SAFE_STORED, "20-day trend % (scanner 'Trend20D%')."),
    FeatureSpec("breakout_pos_20d", "float", SRC_SCAN_META,
                "record.research_metadata.row_features.breakout_pos_20d", SAFE_STORED,
                "Position of price within the 20-day range."),
    FeatureSpec("dollar_vol_20", "float", SRC_SCAN_META, "record.research_metadata.row_features.dollar_vol_20",
                SAFE_STORED, "20-day average dollar volume."),
    FeatureSpec("rs_vs_spy", "float", SRC_SCAN_META, "record.research_metadata.row_features.rs_vs_spy",
                SAFE_STORED, "Relative strength vs SPY as computed by the scan."),
    FeatureSpec("ema_cross", "str", SRC_SCAN_META, "record.research_metadata.row_features.ema_cross",
                SAFE_STORED, "EMA cross tag from the scan (no EMA values are stored)."),
    FeatureSpec("pattern_tag", "str", SRC_SCAN_META, "record.research_metadata.row_features.pattern_tag",
                SAFE_STORED, "Pattern tag from the scan."),
    FeatureSpec("scanner_rank", "int", SRC_SCAN_META, "record.research_metadata.rank_at_observation",
                SAFE_STORED, "Rank in the scan's BreakoutScore-ordered result rows."),
)

FEATURE_SCHEMAS: Mapping[int, Tuple[FeatureSpec, ...]] = MappingProxyType({1: _FEATURES_V1})

# Features research would want that HSF never persisted. They are NOT schema
# columns (an all-null column is not a feature); coverage reports them at 0.
UNAVAILABLE_FEATURES: Tuple[Dict[str, str], ...] = (
    {"name": "rsi_14", "pit": MISSING,
     "reason": "Scheduled scans never compute RSI; Day Trader computes it at render time and doesn't store it. "
               "Reconstructable from raw daily bars, but no historical bars are persisted."},
    {"name": "ema9 / ema21 values", "pit": MISSING,
     "reason": "Only the EMA cross tag is stored. Values are reconstructable from raw bars, which aren't persisted."},
    {"name": "adx / vwap / supertrend / ewo / day_trader_tier", "pit": MISSING,
     "reason": "Computed only when the Day Trader page renders; never persisted for scheduled scans."},
    {"name": "prev_close / vol_avg_20 / high_20 / spark_10d", "pit": SAFE_STORED,
     "reason": "Stored in per-scan runs.results_json for 90 days only (then pruned). Not read by schema v1."},
    {"name": "prebreakout_model_version (served)", "pit": UNKNOWN,
     "reason": "The served model version lives in the loaded bundle and is not frozen on observations. "
               "hsf_observations.versions holds the code constant, which can differ from the served champion."},
    {"name": "sector / industry / market_cap / fundamentals", "pit": UNSAFE_CURRENT_VALUE,
     "reason": "Only current metadata exists. Joining it to history would leak today's values."},
    {"name": "market_regime / breadth", "pit": MISSING,
     "reason": "Regime is computed only in the Market Brief UI and never frozen (REGIME_CAPTURE_UNAVAILABLE)."},
    {"name": "universe membership list", "pit": UNSAFE_CURRENT_VALUE,
     "reason": "Universe files are overwritten by refresh jobs; only the universe name is frozen."},
)

_LABELS_V1: Tuple[LabelSpec, ...] = (
    LabelSpec("return_1d", "float", "signal_outcomes.return_1d", 1,
              "Close-to-close return from the entry close (first daily close on/after the UTC fire date)."),
    LabelSpec("return_3d", "float", "signal_outcomes.return_3d", 3, "Same entry, 3 trading-day close."),
    LabelSpec("return_5d", "float", "signal_outcomes.return_5d", 5, "Same entry, 5 trading-day close."),
    LabelSpec("mfe_5d", "float", "signal_outcomes.mfe_5d", 5,
              "Max favorable excursion: highest High of the 5 bars after the entry bar vs entry close."),
    LabelSpec("mae_5d", "float", "signal_outcomes.mae_5d", 5,
              "Max adverse excursion: lowest Low of the 5 bars after the entry bar vs entry close."),
    LabelSpec("benchmark_return_1d", "float", "signal_outcomes.benchmark_return_1d (when present)", 1,
              "SPY return over the same window, same scoring function (Outcome Intelligence)."),
    LabelSpec("benchmark_return_3d", "float", "signal_outcomes.benchmark_return_3d (when present)", 3,
              "SPY return over the same 3-day window."),
    LabelSpec("benchmark_return_5d", "float", "signal_outcomes.benchmark_return_5d (when present)", 5,
              "SPY return over the same 5-day window."),
    LabelSpec("excess_return_1d", "float", "return_1d - benchmark_return_1d", 1,
              "Null unless both returns exist; never zero-filled."),
    LabelSpec("excess_return_3d", "float", "return_3d - benchmark_return_3d", 3, "As above, 3 days."),
    LabelSpec("excess_return_5d", "float", "return_5d - benchmark_return_5d", 5, "As above, 5 days."),
)

LABEL_SCHEMAS: Mapping[int, Tuple[LabelSpec, ...]] = MappingProxyType({1: _LABELS_V1})
HORIZONS_DAYS = (1, 3, 5)


def feature_schema(version: int = FEATURE_SCHEMA_VERSION) -> Tuple[FeatureSpec, ...]:
    try:
        return FEATURE_SCHEMAS[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unknown feature_schema_version: {version!r}") from None


def label_schema(version: int = LABEL_SCHEMA_VERSION) -> Tuple[LabelSpec, ...]:
    try:
        return LABEL_SCHEMAS[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unknown label_schema_version: {version!r}") from None


def feature_names(version: int = FEATURE_SCHEMA_VERSION) -> Tuple[str, ...]:
    return tuple(f.name for f in feature_schema(version))


def label_names(version: int = LABEL_SCHEMA_VERSION) -> Tuple[str, ...]:
    return tuple(lab.name for lab in label_schema(version))


def schema_as_dict(version: int = FEATURE_SCHEMA_VERSION) -> List[Dict[str, Any]]:
    return [{"name": f.name, "type": f.type, "source": f.source, "path": f.path, "nullable": f.nullable,
             "point_in_time": f.point_in_time, "classification": f.pit, "description": f.description}
            for f in feature_schema(version)]


def label_schema_as_dict(version: int = LABEL_SCHEMA_VERSION) -> List[Dict[str, Any]]:
    return [{"name": lab.name, "type": lab.type, "source": lab.source, "horizon_days": lab.horizon_days,
             "nullable": lab.nullable, "description": lab.description} for lab in label_schema(version)]


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()})


@dataclass(frozen=True)
class FeatureSnapshot:
    """What HSF knew about one observation at ``observed_at``. Nothing else.

    ``values`` holds exactly the declared features of ``feature_schema_version``
    (missing ones are None, never 0). ``join`` describes the temporal join that
    supplied the scan features (status, lag, which scan); it is provenance, not
    a feature."""
    observation_id: int
    ticker: str
    observed_at: str
    feature_schema_version: int
    values: Mapping[str, Any]
    join: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = set(feature_names(self.feature_schema_version))
        keys = set(self.values)
        extra = keys - allowed
        if extra:
            raise ValueError(f"not in feature schema v{self.feature_schema_version}: {sorted(extra)}")
        bad = [k for k in keys | set(self.join) if looks_like_outcome(k)]
        if bad:
            raise ValueError(f"outcome-like keys may not enter a feature snapshot: {sorted(bad)}")
        full = {name: self.values.get(name) for name in feature_names(self.feature_schema_version)}
        object.__setattr__(self, "values", _freeze(full))
        object.__setattr__(self, "join", MappingProxyType(dict(self.join)))

    def vector(self) -> List[Any]:
        """Values in schema order (the row of a feature matrix)."""
        return [self.values[n] for n in feature_names(self.feature_schema_version)]

    def to_dict(self) -> Dict[str, Any]:
        return {"observation_id": self.observation_id, "ticker": self.ticker, "observed_at": self.observed_at,
                "feature_schema_version": self.feature_schema_version,
                "features": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.values.items()},
                "join": dict(self.join)}


@dataclass(frozen=True)
class OutcomeRecord:
    """What happened after ``observed_at``. Kept apart from FeatureSnapshot."""
    observation_id: int
    ticker: str
    observed_at: str
    label_schema_version: int
    values: Mapping[str, Any]
    maturity: Mapping[str, str]          # horizon label -> PENDING | MATURED | UNAVAILABLE
    certified: bool
    outcome_computed_at: Optional[str]
    entry_day: Optional[str]
    label_window_end: Optional[str]

    def __post_init__(self) -> None:
        allowed = set(label_names(self.label_schema_version))
        extra = set(self.values) - allowed
        if extra:
            raise ValueError(f"not in label schema v{self.label_schema_version}: {sorted(extra)}")
        full = {name: self.values.get(name) for name in label_names(self.label_schema_version)}
        object.__setattr__(self, "values", MappingProxyType(full))
        object.__setattr__(self, "maturity", MappingProxyType(dict(self.maturity)))

    def vector(self) -> List[Any]:
        return [self.values[n] for n in label_names(self.label_schema_version)]

    def to_dict(self) -> Dict[str, Any]:
        return {"observation_id": self.observation_id, "ticker": self.ticker, "observed_at": self.observed_at,
                "label_schema_version": self.label_schema_version, "labels": dict(self.values),
                "maturity": dict(self.maturity), "certified": self.certified,
                "outcome_computed_at": self.outcome_computed_at, "entry_day": self.entry_day,
                "label_window_end": self.label_window_end}


def join_features_labels(snapshots: Iterable[FeatureSnapshot], outcomes: Iterable[OutcomeRecord]
                         ) -> List[Tuple[FeatureSnapshot, Optional[OutcomeRecord]]]:
    """The one sanctioned way to put features next to labels: an explicit join by
    observation_id that keeps the two records separate objects."""
    by_id = {o.observation_id: o for o in outcomes}
    pairs = []
    for s in snapshots:
        o = by_id.get(s.observation_id)
        if o is not None and o.observed_at != s.observed_at:
            raise ValueError(f"observed_at mismatch for observation {s.observation_id}")
        pairs.append((s, o))
    return pairs
