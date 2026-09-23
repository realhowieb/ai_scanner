"""Run 46 — canonical observation integrity, completeness & research readiness.

Read-only audit + deterministic research export over the Run 36 canonical
observation records (as captured by Run 38A). It measures what HSF actually
stored: field completeness, validity, duplicates (exact vs conflicting),
point-in-time safety, provenance, and scan-health linkage — and produces a
research export that carries FEATURES KNOWN AT TIME T only (never outcomes).

Pure, deterministic, no I/O. It changes no scoring/scanner/ML/intelligence and
never mutates or deletes historical data — it only describes and (on export)
copies point-in-time-safe fields.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from analytics.hsf_observation import INDICATOR_FIELDS, MARKET_FIELDS

SCHEMA_VERSION = "hsf-obs-integrity-1.0"

REQUIRED_IDENTITY = ("observation_id", "symbol", "timestamp", "schema_version")
# Fields that must NEVER appear inside a point-in-time feature record.
_OUTCOME_MARKERS = ("raw_return", "directional_return", "mfe", "mae",
                    "future_high", "future_low", "return_", "outcome")
_VALID_DIRECTIONS = {"long", "short", "bullish", "bearish", "neutral", "mixed"}


def schema_inventory() -> List[Dict[str, Any]]:
    """Static classification of the canonical observation schema (Task 3)."""
    def f(name, typ, group, source, pit=True, nullable=True):
        return {"field": name, "type": typ, "group": group, "source": source,
                "point_in_time_safe": pit, "nullable": nullable}
    inv = [
        f("observation_id", "str", "IDENTITY", "hash(symbol|ts|context)", nullable=False),
        f("symbol", "str", "IDENTITY", "scan row", nullable=False),
        f("timestamp", "iso", "IDENTITY", "hour-bucketed scan time", nullable=False),
        f("scan_timestamp", "iso", "IDENTITY", "precise scan start"),
        f("context", "str", "IDENTITY", "scheduled:<universe>"),
        f("session", "str", "CONTEXT", "market session"),
        f("universe_version", "str", "CONTEXT", "universe name"),
    ]
    inv += [f(m, "float", "MARKET_STATE", "scan row") for m in MARKET_FIELDS]
    inv += [f(i, "float/str", "TECHNICAL", "scan row / indicators") for i in INDICATOR_FIELDS]
    inv += [
        f("scanners[]", "list", "HSF_INTELLIGENCE", "derive_scanner_triggers"),
        f("models.prebreakout", "obj", "ML", "ml_prebreakout"),
        f("models.ai_confidence", "obj", "ML", "scan.ai_confidence"),
        f("market_context.market_regime", "str", "CONTEXT", "regime"),
        f("market_context.coverage_health", "str", "CONTEXT", "Run 37 coverage"),
        f("market_context.scan_id", "str", "CONTEXT", "scan run id"),
        f("versions", "obj", "PROVENANCE", "schema/model versions"),
        f("data_quality", "obj", "PROVENANCE", "completeness/fallback/stale"),
        f("outcomes", "obj", "OUTCOMES", "maturation (SEPARATE table)", pit=False),
    ]
    return inv


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        return fv
    except (TypeError, ValueError):
        return None


def _finite(v: Any) -> bool:
    fv = _num(v)
    return fv is not None and math.isfinite(fv)


def validate_observation(obs: Dict[str, Any]) -> List[str]:
    """Lightweight validity checks (Task 8). Returns a list of issue strings
    (empty = valid). Validation only — never recalculates any signal."""
    issues: List[str] = []
    for k in REQUIRED_IDENTITY:
        if not obs.get(k):
            issues.append(f"missing_identity:{k}")
    sym = str(obs.get("symbol") or "")
    if sym and (not sym.replace(".", "").replace("-", "").isalnum() or len(sym) > 10):
        issues.append("malformed_ticker")
    mkt = obs.get("market") or {}
    ind = obs.get("indicators") or {}
    price = _num(mkt.get("price"))
    if price is not None and price <= 0:
        issues.append("invalid_price")
    vol = _num(mkt.get("volume"))
    if vol is not None and vol < 0:
        issues.append("invalid_volume")
    rvol = _num(ind.get("rvol"))
    if rvol is not None and rvol < 0:
        issues.append("invalid_rvol")
    rsi = _num(ind.get("rsi"))
    if rsi is not None and not (0 <= rsi <= 100):
        issues.append("invalid_rsi")
    for name, val in list(mkt.items()) + list(ind.items()):
        fv = _num(val)
        if fv is not None and not math.isfinite(fv):
            issues.append(f"nonfinite:{name}")
    for m in (obs.get("models") or {}).values():
        p = _num((m or {}).get("probability")) if isinstance(m, dict) else None
        if p is not None and not (0 <= p <= 1) and not (0 <= p <= 100):
            issues.append("invalid_probability")
    for s in (obs.get("scanners") or []):
        d = str(s.get("direction") or "long").lower()
        if d not in _VALID_DIRECTIONS:
            issues.append(f"invalid_direction:{d}")
    return issues


def _parse(ts: Any) -> Optional[_dt.datetime]:
    try:
        d = ts if isinstance(ts, _dt.datetime) else _dt.datetime.fromisoformat(
            str(ts).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def check_point_in_time(obs: Dict[str, Any]) -> List[str]:
    """Verify no outcome/future info contaminates the feature record (Task 4).

    Outcomes may live under the separate `outcomes` key, but a matured outcome's
    evaluation_time must be strictly after the observation timestamp, and no
    outcome marker may appear inside market/indicators/scanners."""
    violations: List[str] = []
    feature_blob = str({k: obs.get(k) for k in ("market", "indicators", "scanners", "models")}).lower()
    for marker in _OUTCOME_MARKERS:
        if marker in feature_blob:
            violations.append(f"outcome_marker_in_features:{marker}")
    anchor = _parse(obs.get("scan_timestamp") or obs.get("timestamp"))
    for h, oc in (obs.get("outcomes") or {}).items():
        if not isinstance(oc, dict):
            continue
        if str(oc.get("data_status")) == "MATURED":
            et = _parse(oc.get("evaluation_time"))
            if anchor is not None and et is not None and et <= anchor:
                violations.append(f"lookahead_outcome:{h}")
    return violations


def duplicate_analysis(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact vs conflicting duplicates keyed by the logical id (Task 6). Does NOT
    drop anything — only reports."""
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for o in observations:
        by_id.setdefault(str(o.get("observation_id")), []).append(o)
    exact = conflicting = 0
    conflicts: List[str] = []
    for oid, group in by_id.items():
        if len(group) <= 1:
            continue
        # Compare feature payloads (ignore volatile keys like created_at/outcomes).
        payloads = {_feature_fingerprint(g) for g in group}
        if len(payloads) == 1:
            exact += len(group) - 1
        else:
            conflicting += len(group) - 1
            conflicts.append(oid)
    return {"total_rows": len(observations), "unique_logical": len(by_id),
            "exact_duplicates": exact, "conflicting_duplicates": conflicting,
            "conflicting_ids": conflicts[:20]}


def _feature_fingerprint(obs: Dict[str, Any]) -> str:
    keys = ("symbol", "timestamp", "market", "indicators", "scanners", "models")
    import json
    return json.dumps({k: obs.get(k) for k in keys}, sort_keys=True, default=str)


def completeness_report(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-field population rate over important fields (Task 7)."""
    n = len(observations)
    fields = ["symbol", "timestamp", "scan_timestamp", *(f"market.{m}" for m in MARKET_FIELDS),
              *(f"indicators.{i}" for i in INDICATOR_FIELDS),
              "models.prebreakout", "models.ai_confidence", "market_context.coverage_health"]
    counts: Counter = Counter()
    for o in observations:
        for f in fields:
            if _present_path(o, f):
                counts[f] += 1
    return {"n": n, "fields": {f: {"populated": counts[f],
                                    "pct": round(counts[f] / n, 4) if n else None}
                              for f in fields}}


def _present_path(obs: Dict[str, Any], path: str) -> bool:
    cur: Any = obs
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if cur is None:
        return False
    if isinstance(cur, float) and not math.isfinite(cur):
        return False
    if isinstance(cur, str) and cur.strip() == "":
        return False
    return True


def capture_rate(observations_persisted: int, successfully_evaluated: Optional[int]) -> Optional[float]:
    if not successfully_evaluated:
        return None
    return round(observations_persisted / successfully_evaluated, 4)


def dataset_health_report(
    observations: Sequence[Dict[str, Any]], *,
    successfully_evaluated: Optional[int] = None,
) -> Dict[str, Any]:
    """Full observation-dataset health report (Task 13)."""
    n = len(observations)
    tickers = {str(o.get("symbol")) for o in observations}
    scan_ids = {str((o.get("market_context") or {}).get("scan_id")) for o in observations}
    health = Counter(str((o.get("market_context") or {}).get("coverage_health") or "unknown")
                     for o in observations)
    invalid = sum(1 for o in observations if validate_observation(o))
    pit_violations = sum(1 for o in observations if check_point_in_time(o))
    dup = duplicate_analysis(observations)
    comp = completeness_report(observations)
    required_complete = sum(1 for o in observations
                            if all(o.get(k) for k in REQUIRED_IDENTITY))
    timestamps = sorted(str(o.get("timestamp")) for o in observations if o.get("timestamp"))

    # Dataset health: FAILED if leakage/required-field breaks; DEGRADED if
    # material invalid/conflicting/duplicate; else HEALTHY.
    state = "HEALTHY"
    if pit_violations or (n and required_complete / n < 0.99):
        state = "FAILED"
    elif dup["conflicting_duplicates"] or (n and invalid / n > 0.02):
        state = "DEGRADED"

    return {
        "schema": SCHEMA_VERSION,
        "period": {"first": timestamps[0] if timestamps else None,
                   "last": timestamps[-1] if timestamps else None},
        "observations": n,
        "unique_tickers": len(tickers),
        "scan_runs": len([s for s in scan_ids if s and s != "None"]),
        "capture_rate": capture_rate(n, successfully_evaluated),
        "scan_health_distribution": dict(health),
        "duplicates": dup,
        "required_field_complete_pct": round(required_complete / n, 4) if n else None,
        "invalid_rows": invalid,
        "point_in_time_violations": pit_violations,
        "completeness": comp,
        "dataset_health": state,
    }


def research_export(
    observations: Sequence[Dict[str, Any]], *,
    healthy_only: bool = False, start: Optional[str] = None, end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Deterministic, point-in-time-safe research export (Task 16).

    Stable column set, deterministic ordering (by timestamp, symbol). Outcome
    fields are EXCLUDED by default — the export carries features known at T only.
    Optional HEALTHY-scan filter and date range."""
    rows: List[Dict[str, Any]] = []
    for o in observations:
        ts = str(o.get("timestamp") or "")
        if start and ts[:10] < start:
            continue
        if end and ts[:10] > end:
            continue
        ctx = o.get("market_context") or {}
        if healthy_only and str(ctx.get("coverage_health")) != "HEALTHY":
            continue
        mkt, ind = o.get("market") or {}, o.get("indicators") or {}
        models = o.get("models") or {}
        row = {
            "observation_id": o.get("observation_id"),
            "symbol": o.get("symbol"),
            "timestamp": o.get("timestamp"),
            "scan_timestamp": o.get("scan_timestamp"),
            "session": o.get("session"),
            "universe": o.get("universe_version"),
            "scan_id": ctx.get("scan_id"),
            "coverage_health": ctx.get("coverage_health"),
            "market_regime": ctx.get("market_regime"),
            "schema_version": o.get("schema_version"),
            "feature_completeness": (o.get("data_quality") or {}).get("feature_completeness"),
            "scanner_names": sorted(str(s.get("name")) for s in (o.get("scanners") or [])
                                    if s.get("triggered", True)),
        }
        for m in MARKET_FIELDS:
            row[f"market_{m}"] = mkt.get(m)
        for i in INDICATOR_FIELDS:
            row[f"ind_{i}"] = ind.get(i)
        row["prebreakout_probability"] = (models.get("prebreakout") or {}).get("probability")
        row["ai_confidence"] = (models.get("ai_confidence") or {}).get("confidence")
        # NOTE: no outcome fields — by design (Run 47 consumes outcomes separately).
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("timestamp") or ""), str(r.get("symbol") or "")))
    return rows


EXPORT_COLUMNS = ["observation_id", "symbol", "timestamp", "scan_timestamp",
                  "session", "universe", "scan_id", "coverage_health",
                  "market_regime", "schema_version", "feature_completeness",
                  "scanner_names", *(f"market_{m}" for m in MARKET_FIELDS),
                  *(f"ind_{i}" for i in INDICATOR_FIELDS),
                  "prebreakout_probability", "ai_confidence"]


def render_health_text(rep: Dict[str, Any]) -> str:
    d = rep["duplicates"]
    return "\n".join([
        "HSF OBSERVATION DATASET HEALTH",
        f"Period: {rep['period']['first']} → {rep['period']['last']}",
        f"Observations: {rep['observations']}  Unique tickers: {rep['unique_tickers']}  "
        f"Scan runs: {rep['scan_runs']}",
        f"Capture rate: {rep['capture_rate']}",
        f"Scan health: {rep['scan_health_distribution']}",
        f"Exact duplicates: {d['exact_duplicates']}  Conflicting: {d['conflicting_duplicates']}",
        f"Required-field complete: {rep['required_field_complete_pct']}",
        f"Invalid rows: {rep['invalid_rows']}  PIT violations: {rep['point_in_time_violations']}",
        f"Dataset health: {rep['dataset_health']}",
    ])
