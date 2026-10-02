"""Pure contracts and diagnostics for the read-only Observation Explorer."""
from __future__ import annotations

import csv
import datetime as dt
import io
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from analytics import admin_research as ar
from analytics.forward_readiness import FORWARD_EPOCH, PREVIOUS_EPOCHS
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS

CANONICAL_COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)
DATASET_SCANNER = "scanner"
DATASET_STAIR_STEPPER = "stair_stepper"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_EXPORT_ROWS = 5_000


def epoch_specs() -> list[dict[str, Any]]:
    """Chronological immutable epoch boundaries from the registered research policy."""
    previous = sorted(PREVIOUS_EPOCHS, key=lambda item: item["forward_epoch_start_timestamp"])
    current_start = str(FORWARD_EPOCH["forward_epoch_start_timestamp"])
    specs: list[dict[str, Any]] = []
    for index, item in enumerate(previous):
        start = str(item["forward_epoch_start_timestamp"])
        end = str(previous[index + 1]["forward_epoch_start_timestamp"]) if index + 1 < len(previous) else current_start
        specs.append({
            "key": f"previous:{item.get('run') or start}",
            "label": f"Previous — {item.get('run') or start}",
            "start": start,
            "end": end,
            "control_design": item.get("control_design"),
            "current": False,
        })
    specs.append({
        "key": "current",
        "label": "Current — Run 59B",
        "start": current_start,
        "end": None,
        "control_design": FORWARD_EPOCH.get("control_design"),
        "current": True,
    })
    return specs


def epoch_for_timestamp(value: Any) -> dict[str, Any] | None:
    timestamp = ar.parse_timestamp(value)
    if timestamp is None:
        return None
    for spec in reversed(epoch_specs()):
        start = ar.parse_timestamp(spec["start"])
        end = ar.parse_timestamp(spec.get("end"))
        if start and timestamp >= start and (end is None or timestamp < end):
            return spec
    return None


def normalize_symbols(value: Any) -> list[str]:
    raw = value if isinstance(value, (list, tuple, set)) else str(value or "").replace(";", ",").split(",")
    symbols: list[str] = []
    for item in raw:
        symbol = "".join(char for char in str(item).strip().upper() if char.isalnum() or char in ".-")[:16]
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:50]


def clamp_page(page: Any, page_size: Any) -> tuple[int, int]:
    try:
        safe_page = max(1, int(page))
    except (TypeError, ValueError):
        safe_page = 1
    try:
        safe_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
    except (TypeError, ValueError):
        safe_size = DEFAULT_PAGE_SIZE
    return safe_page, safe_size


def derived_status(outcomes: Mapping[str, Any] | None, *, horizon: str | None = None) -> str:
    outcomes = outcomes if isinstance(outcomes, Mapping) else {}
    selected = [outcomes.get(horizon)] if horizon else list(outcomes.values())
    selected = [value for value in selected if isinstance(value, Mapping)]
    if any(str(value.get("data_status") or "").upper() in {"EXCLUDED", "UNAVAILABLE"}
           for value in selected):
        return "EXCLUDED"
    matured = [value for value in selected if str(value.get("data_status") or "").upper() == "MATURED"]
    if not matured:
        return "PENDING"
    if any(ar.number(value.get("directional_return")) is not None
           or ar.number(value.get("raw_return")) is not None for value in matured):
        return "USABLE"
    return "MATURED"


def observation_summary(observation: Mapping[str, Any], *, horizon: str | None = None) -> dict[str, Any]:
    metadata = observation.get("research_metadata") or {}
    context = observation.get("market_context") or {}
    outcomes = observation.get("outcomes") or {}
    outcome = outcomes.get(horizon) if horizon and isinstance(outcomes, Mapping) else None
    if not isinstance(outcome, Mapping):
        outcome = {}
    epoch = epoch_for_timestamp(observation.get("timestamp"))
    return {
        "observation_id": observation.get("observation_id"),
        "timestamp": observation.get("timestamp"),
        "symbol": observation.get("symbol"),
        "cohort": ar.research_cohort(observation),
        "signals": ", ".join(ar.signal_names(observation)),
        "hsf_score": ar.hsf_score(observation),
        "rank": metadata.get("rank_at_observation"),
        "scan_id": context.get("scan_id") or metadata.get("scan_id"),
        "research_epoch": epoch.get("label") if epoch else "Pre-registered epochs",
        "control_design": context.get("control_design"),
        "session": observation.get("session") or metadata.get("session"),
        "status": derived_status(outcomes, horizon=horizon),
        "horizon": horizon,
        "forward_return": ar.number(outcome.get("directional_return"))
        if ar.number(outcome.get("directional_return")) is not None else ar.number(outcome.get("raw_return")),
        "mfe": ar.number(outcome.get("mfe")),
        "mae": ar.number(outcome.get("mae")),
        "matured_at": outcome.get("evaluation_time"),
    }


def cohort_overlap(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count scan-local symbol overlap between each canonical cohort pair."""
    by_key: dict[tuple[str, str], set[str]] = {}
    for observation in observations:
        context = observation.get("market_context") or {}
        scan_id = str(context.get("scan_id") or "")
        symbol = str(observation.get("symbol") or "").upper()
        if not scan_id or not symbol:
            continue
        by_key.setdefault((scan_id, symbol), set()).add(ar.research_cohort(observation))
    pairs = {
        "candidate_near_miss": {CANDIDATE, NEAR_MISS},
        "candidate_control": {CANDIDATE, CONTROL},
        "near_miss_control": {NEAR_MISS, CONTROL},
    }
    counts = {name: sum(pair.issubset(cohorts) for cohorts in by_key.values()) for name, pair in pairs.items()}
    return {**counts, "status": "PASS" if not any(counts.values()) else "OVERLAP_DETECTED"}


def bounded_csv(rows: Iterable[Mapping[str, Any]], *, maximum: int = MAX_EXPORT_ROWS) -> dict[str, Any]:
    maximum = max(1, min(MAX_EXPORT_ROWS, int(maximum)))
    selected = []
    truncated = False
    for row in rows:
        if len(selected) >= maximum:
            truncated = True
            break
        selected.append(dict(row))
    fields = list(dict.fromkeys(key for row in selected for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    if fields:
        writer.writeheader()
        writer.writerows(selected)
    return {"csv": buffer.getvalue(), "rows": len(selected), "truncated": truncated}


def cohort_counts(observations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(ar.research_cohort(observation) for observation in observations)
    return {cohort: counts.get(cohort, 0) for cohort in CANONICAL_COHORTS}
