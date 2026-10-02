"""Pure aggregations for the Admin research-intelligence dashboard.

This module deliberately contains no database or Streamlit calls. Missing
outcomes stay missing, scanner and Stair-Stepper observations stay separate,
and every evidence decision carries its sample size.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_MIN_EVIDENCE_N = 30
SCANNER_CONTEXT_PREFIX = "scheduled:"
STAIR_STEPPER_CONTEXT = "day_trader:stair_stepper"
CANONICAL_PLANS = ("basic", "pro", "premium", "admin")


def normalize_plan(value: Any) -> str:
    """Return the product's canonical plan key without reviving Free/Basic drift."""
    plan = str(value or "basic").strip().lower()
    aliases = {"free": "basic", "starter": "basic", "plus": "pro"}
    plan = aliases.get(plan, plan)
    return plan if plan in CANONICAL_PLANS else "basic"


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_timestamp(value: Any) -> dt.datetime | None:
    try:
        parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def is_stair_stepper(observation: Mapping[str, Any]) -> bool:
    return str(observation.get("context") or "") == STAIR_STEPPER_CONTEXT


def is_scanner_research(observation: Mapping[str, Any]) -> bool:
    return str(observation.get("context") or "").startswith(SCANNER_CONTEXT_PREFIX)


def research_cohort(observation: Mapping[str, Any]) -> str:
    value = observation.get("research_cohort") or (
        observation.get("market_context") or {}
    ).get("research_cohort")
    return str(value or "UNTAGGED").upper()


def hsf_score(observation: Mapping[str, Any]) -> float | None:
    """Read only explicitly persisted canonical HSF score fields.

    BreakoutScore and scanner/model scores are intentionally not substituted.
    """
    candidates = (
        observation.get("hsf_score"),
        (observation.get("models") or {}).get("hsf_score"),
        (observation.get("market_context") or {}).get("hsf_score"),
        ((observation.get("research_metadata") or {}).get("row_features") or {}).get("hsf_score"),
    )
    for candidate in candidates:
        value = number(candidate)
        if value is not None:
            return value
    return None


def score_bucket(value: Any) -> str | None:
    score = number(value)
    if score is None:
        return None
    if score < 50:
        return "<50"
    if score < 60:
        return "50-59"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    if score < 90:
        return "80-89"
    return "90-100"


def signal_names(observation: Mapping[str, Any]) -> list[str]:
    names = []
    for scanner in observation.get("scanners") or []:
        if not isinstance(scanner, Mapping) or not scanner.get("triggered", True):
            continue
        name = str(scanner.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names or ["unclassified"]


def flatten_research(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per observation x signal x matured outcome.

    Pending observations are represented once with a null horizon so the funnel
    can count them, but performance aggregations ignore those rows.
    """
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if not is_scanner_research(observation):
            continue
        outcomes = observation.get("outcomes") or {}
        outcome_items = list(outcomes.items()) if isinstance(outcomes, Mapping) else []
        if not outcome_items:
            outcome_items = [(None, None)]
        for signal in signal_names(observation):
            for horizon, outcome in outcome_items:
                outcome = outcome if isinstance(outcome, Mapping) else {}
                status = str(outcome.get("data_status") or "PENDING").upper()
                raw_return = number(outcome.get("raw_return"))
                directional_return = number(outcome.get("directional_return"))
                hit = outcome.get("hit") if isinstance(outcome.get("hit"), bool) else None
                positive = hit if hit is not None else (
                    directional_return > 0 if directional_return is not None else (
                        raw_return > 0 if raw_return is not None else None
                    )
                )
                rows.append({
                    "observation_id": observation.get("observation_id"),
                    "timestamp": observation.get("timestamp"),
                    "symbol": observation.get("symbol"),
                    "context": observation.get("context"),
                    "cohort": research_cohort(observation),
                    "session": str(observation.get("session") or "UNKNOWN").upper(),
                    "signal": signal,
                    "hsf_score": hsf_score(observation),
                    "score_bucket": score_bucket(hsf_score(observation)),
                    "horizon": str(horizon) if horizon is not None else None,
                    "data_status": status,
                    "raw_return": raw_return,
                    "directional_return": directional_return,
                    "mfe": number(outcome.get("mfe")),
                    "mae": number(outcome.get("mae")),
                    "positive": positive,
                    "evaluation_time": outcome.get("evaluation_time"),
                })
    return rows


def matured_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("data_status") or "").upper() == "MATURED"]


def performance_summary(rows: Sequence[Mapping[str, Any]], *, min_n: int = DEFAULT_MIN_EVIDENCE_N) -> dict[str, Any]:
    matured = matured_rows(rows)
    returns = [
        number(row.get("directional_return"))
        if number(row.get("directional_return")) is not None
        else number(row.get("raw_return"))
        for row in matured
    ]
    returns = [value for value in returns if value is not None]
    positives = [row.get("positive") for row in matured if isinstance(row.get("positive"), bool)]
    mfes = [value for row in matured if (value := number(row.get("mfe"))) is not None]
    maes = [value for row in matured if (value := number(row.get("mae"))) is not None]
    n = len(returns)
    return {
        "n": n,
        "matured": len(matured),
        "sufficient": n >= max(0, int(min_n)),
        "positive_rate": (sum(bool(value) for value in positives) / len(positives)) if positives else None,
        "average_return": mean(returns) if returns else None,
        "median_return": median(returns) if returns else None,
        "average_mfe": mean(mfes) if mfes else None,
        "average_mae": mean(maes) if maes else None,
    }


def grouped_performance(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    *,
    min_n: int = DEFAULT_MIN_EVIDENCE_N,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output.append({
            **dict(zip(group_keys, values)),
            **performance_summary(group, min_n=min_n),
        })
    return output


def evidence_funnel(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scanner = [row for row in observations if is_scanner_research(row)]
    captured = len(scanner)
    valid_metadata = sum(
        isinstance(row.get("research_metadata"), Mapping) for row in scanner
    )
    matured = 0
    usable = 0
    included = 0
    pending_times = []
    latest_matured = None
    for row in scanner:
        outcomes = row.get("outcomes") or {}
        mature_outcomes = [
            value for value in outcomes.values()
            if isinstance(value, Mapping) and str(value.get("data_status") or "").upper() == "MATURED"
        ] if isinstance(outcomes, Mapping) else []
        if mature_outcomes:
            matured += 1
        else:
            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp:
                pending_times.append(timestamp)
        if any(number(value.get("directional_return")) is not None or number(value.get("raw_return")) is not None
               for value in mature_outcomes):
            usable += 1
        if research_cohort(row) in {"CANDIDATE", "NEAR_MISS", "CONTROL"} and mature_outcomes:
            included += 1
        for value in mature_outcomes:
            evaluated = parse_timestamp(value.get("evaluation_time"))
            if evaluated and (latest_matured is None or evaluated > latest_matured):
                latest_matured = evaluated
    latest_observation = max(
        (value for row in scanner if (value := parse_timestamp(row.get("timestamp"))) is not None),
        default=None,
    )
    return {
        "captured": captured,
        "valid_metadata": valid_metadata,
        "matured": matured,
        "usable_outcome": usable,
        "included_in_research": included,
        "excluded": max(0, captured - included),
        "pending": max(0, captured - matured),
        "metadata_completeness": (valid_metadata / captured) if captured else None,
        "cohorts": dict(Counter(research_cohort(row) for row in scanner)),
        "oldest_pending_observation": min(pending_times).isoformat() if pending_times else None,
        "latest_observation": latest_observation.isoformat() if latest_observation else None,
        "latest_matured_observation": latest_matured.isoformat() if latest_matured else None,
    }


def best_supported_group(
    groups: Sequence[Mapping[str, Any]],
    *,
    metric: str = "median_return",
    min_n: int = DEFAULT_MIN_EVIDENCE_N,
) -> Mapping[str, Any] | None:
    eligible = [
        group for group in groups
        if int(group.get("n") or 0) >= min_n and number(group.get(metric)) is not None
    ]
    return max(eligible, key=lambda group: number(group.get(metric)) or -math.inf) if eligible else None
