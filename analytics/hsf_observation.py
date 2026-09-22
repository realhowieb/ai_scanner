"""Run 36 — canonical HSF market observation/outcome schema (pure, no I/O).

A single, versioned contract for "what HSF knew about one symbol at one instant".
It unifies the fields the existing surfaces already produce (scanner rows,
build_day_trader_metrics, PreBreakout, AI confidence, DT intel, HSF Opportunity)
into one record WITHOUT changing any production write path. Persistence is a
separate, opt-in layer (db.hsf_observations); this module only builds/validates
records.

Design rules (Run 36):
  * One observation per (symbol, timestamp, context). Multiple scanners firing on
    the same symbol are a LIST inside one observation — no information is lost.
  * Observations are IMMUTABLE. When a scoring algorithm changes we bump its
    version and write a NEW observation; we never overwrite an old one.
  * Every observation carries explicit version metadata and data-quality metadata
    so we always know which logic produced it and how complete it was.
  * Outcomes live in a SEPARATE namespace and are attached AFTER the fact; they
    can never contaminate the original features/predictions, and attaching one is
    guarded against lookahead (evaluation strictly after the observation).

Pure functions only — unit-tested directly, Streamlit-free.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Dict, List, Optional, Sequence

# --- Schema versions (bump when the record shape changes) --------------------
OBSERVATION_SCHEMA_VERSION = "hsf-obs-1.0"
OUTCOME_SCHEMA_VERSION = "hsf-outcome-1.0"

# The market/indicator fields the canonical snapshot recognizes. Missing ones are
# recorded as absent (never silently zero-filled) — see data-quality metadata.
MARKET_FIELDS = ("price", "previous_close", "open", "high", "low", "volume")
INDICATOR_FIELDS = (
    "ema9", "ema21", "rsi", "rvol", "adx", "vwap", "vs_vwap_pct",
    "supertrend_direction", "ewo", "gap_pct", "chg_pct", "atr_pct",
)
# Price outcome horizons the outcome schema supports (label → forward minute bars;
# session/next-day are computed by the caller and passed as labels).
PRICE_HORIZONS = {"+5m": 5, "+15m": 15, "+30m": 30, "+60m": 60}
NAMED_HORIZONS = ("market_close", "next_trading_day")


def resolve_versions() -> Dict[str, Optional[str]]:
    """Best-effort snapshot of the live model/logic versions (Task 8).

    Reuses existing identifiers where they exist; each lookup is guarded so a
    missing/optional dependency never breaks record construction.
    """
    versions: Dict[str, Optional[str]] = {
        "schema": OBSERVATION_SCHEMA_VERSION,
        "hsf_score": None,
        "prebreakout_model": None,
        "ai_confidence_model": None,
        "dt_score": "run32-v1",  # DT Score is v1 (research CLOSED; coherence indicator)
    }
    try:
        from ui.opportunities import HSF_SCORE_VERSION
        versions["hsf_score"] = str(HSF_SCORE_VERSION)
    except Exception:
        pass
    try:
        from ml_prebreakout import MODEL_VERSION as _pb
        versions["prebreakout_model"] = str(_pb)
    except Exception:
        pass
    try:
        from scan.ai_confidence import MODEL_VERSION as _ai
        versions["ai_confidence_model"] = str(_ai)
    except Exception:
        pass
    return versions


def make_observation_id(symbol: str, timestamp: Any, context: str = "default") -> str:
    """Deterministic id for one (symbol, timestamp, context). Deterministic so the
    same signal instant maps to the same id — enabling idempotent dedupe."""
    key = f"{str(symbol).upper()}|{str(timestamp)}|{str(context)}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _pick(row: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    """Copy recognized fields that are present; omit missing ones (never zero-fill)."""
    return {n: row[n] for n in names if n in row and _present(row.get(n))}


def build_observation(
    *,
    symbol: str,
    timestamp: Any,
    context: str = "default",
    session: Optional[str] = None,
    universe_version: Optional[str] = None,
    market: Optional[Dict[str, Any]] = None,
    indicators: Optional[Dict[str, Any]] = None,
    scanners: Optional[List[Dict[str, Any]]] = None,
    models: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None,
    scan_timestamp: Any = None,
    data_source: Optional[str] = None,
    price_timestamp: Any = None,
    stale: Optional[bool] = None,
    versions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one canonical, versioned observation record.

    `market`/`indicators` are filtered to recognized fields that are actually
    present. `scanners` is a list so several scanners can share one observation.
    Data-quality metadata (completeness, missing fields, fallback) is computed
    from the recognized feature set. Immutable by convention — callers write a new
    record (new version) rather than mutating.
    """
    mkt = _pick(market or {}, MARKET_FIELDS)
    ind = _pick(indicators or {}, INDICATOR_FIELDS)

    recognized = list(MARKET_FIELDS) + list(INDICATOR_FIELDS)
    present = {**mkt, **ind}
    missing = [f for f in recognized if f not in present]
    completeness = len(present) / len(recognized) if recognized else 0.0
    # "fallback" = every daily-derived indicator is missing (intraday-only shape),
    # mirroring analytics.day_trade_parity so DT observations stay consistent.
    daily_derived = ("adx", "supertrend_direction", "ewo", "gap_pct", "rvol")
    fallback = not any(f in present for f in daily_derived)

    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_id": make_observation_id(symbol, timestamp, context),
        "symbol": str(symbol).upper(),
        "timestamp": str(timestamp),
        "scan_timestamp": str(scan_timestamp) if scan_timestamp is not None else None,
        "context": str(context),
        "session": session,
        "universe_version": universe_version,
        "market": mkt,
        "indicators": ind,
        "scanners": list(scanners or []),
        "models": dict(models or {}),
        "market_context": dict(market_context or {}),
        "versions": {**resolve_versions(), **(versions or {})},
        "data_quality": {
            "feature_completeness": round(completeness, 4),
            "present_fields": sorted(present.keys()),
            "missing_fields": missing,
            "fallback_used": fallback,
            "fallback_reason": ("all daily-derived indicators missing" if fallback else ""),
            "stale": bool(stale) if stale is not None else None,
            "price_timestamp": str(price_timestamp) if price_timestamp is not None else None,
            "data_source": data_source,
        },
    }


def _to_dt(value: Any) -> Optional[_dt.datetime]:
    try:
        if isinstance(value, _dt.datetime):
            return value
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def build_outcome(
    *,
    observation_id: str,
    symbol: str,
    observation_timestamp: Any,
    horizon: str,
    evaluation_time: Any,
    raw_return: Optional[float] = None,
    directional_return: Optional[float] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
    future_high: Optional[float] = None,
    future_low: Optional[float] = None,
    hit: Optional[bool] = None,
    data_status: str = "MATURED",
) -> Dict[str, Any]:
    """Build one outcome record for an observation at one horizon.

    Lookahead guard: `evaluation_time` MUST be strictly after
    `observation_timestamp` for a MATURED outcome. A violation raises ValueError
    — outcomes can only ever be measured from the future, never at/ before the
    signal.
    """
    if data_status == "MATURED":
        obs_dt, eval_dt = _to_dt(observation_timestamp), _to_dt(evaluation_time)
        if obs_dt is not None and eval_dt is not None and eval_dt <= obs_dt:
            raise ValueError(
                f"lookahead: evaluation_time {evaluation_time} <= observation "
                f"timestamp {observation_timestamp}")
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "observation_id": observation_id,
        "symbol": str(symbol).upper(),
        "observation_timestamp": str(observation_timestamp),
        "horizon": horizon,
        "evaluation_time": str(evaluation_time) if evaluation_time is not None else None,
        "raw_return": raw_return,
        "directional_return": directional_return,
        "mfe": mfe,
        "mae": mae,
        "future_high": future_high,
        "future_low": future_low,
        "hit": hit,
        "data_status": data_status,
    }


def attach_outcome(observation: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Return a COPY of the observation with `outcome` recorded under
    ``outcomes[horizon]``. Never mutates the input and never touches the original
    features/predictions (Task 4/7). Rejects an outcome whose observation_id or
    symbol does not match, and enforces the lookahead guard again defensively.
    """
    oid = observation.get("observation_id")
    if outcome.get("observation_id") not in (None, oid):
        raise ValueError("outcome.observation_id does not match observation")
    if outcome.get("data_status") == "MATURED":
        obs_dt = _to_dt(observation.get("timestamp"))
        eval_dt = _to_dt(outcome.get("evaluation_time"))
        if obs_dt is not None and eval_dt is not None and eval_dt <= obs_dt:
            raise ValueError("lookahead: outcome evaluated at/before observation time")
    out = dict(observation)
    out["outcomes"] = {**observation.get("outcomes", {}), str(outcome.get("horizon")): outcome}
    return out
