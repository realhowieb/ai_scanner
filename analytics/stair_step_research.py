"""Outcome research for the Day Trader Stair-stepper.

This module is deliberately separate from the production detector: it records
the detector's existing outputs, matures future outcomes, and summarizes the
evidence. It never changes qualification, HSF Score, or scanner ranking.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from analytics.hsf_observation import build_observation, build_outcome
from analytics.stair_step import WINDOW_OPTIONS, is_stair_stepper

CONTEXT = "day_trader:stair_stepper"
DETECTION_VERSION = "stair-stepper-research-1.0"
OUTCOME_HORIZONS = {"+5m": 5, "+10m": 10, "+15m": 15, "+30m": 30}
DEDUP_INTERVAL_MINUTES = 30
MAX_TARGET_LAG_MINUTES = 2
MIN_ROLE_SAMPLE = 30
MIN_ROLE_DAYS = 5

_ET = ZoneInfo("America/New_York")


def _parse_ts(value: Any) -> Optional[dt.datetime]:
    try:
        parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _bucket_timestamp(value: Any) -> Optional[str]:
    parsed = _parse_ts(value)
    if parsed is None:
        return None
    parsed = parsed.astimezone(dt.timezone.utc)
    minute = (parsed.minute // DEDUP_INTERVAL_MINUTES) * DEDUP_INTERVAL_MINUTES
    return parsed.replace(minute=minute, second=0, microsecond=0).isoformat()


def _market_session(value: Any) -> str:
    parsed = _parse_ts(value)
    if parsed is None:
        return "UNKNOWN"
    local = parsed.astimezone(_ET)
    minute = local.hour * 60 + local.minute
    if 4 * 60 <= minute < 9 * 60 + 30:
        return "PREMARKET"
    if 9 * 60 + 30 <= minute < 16 * 60:
        return "REGULAR"
    if 16 * 60 <= minute < 20 * 60:
        return "AFTERHOURS"
    return "CLOSED"


def _observation_id(symbol: str, window: int, direction: str, bucket: str) -> str:
    identity = f"{symbol.upper()}|{window}|{direction}|{bucket}|{DETECTION_VERSION}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def build_qualifying_observations(
    rows_by_window: Mapping[int, Sequence[Dict[str, Any]]],
    *,
    r2_min: float,
    max_pullback_pct: float,
    min_trend_pct_per_hour: float,
    data_source: str = "alpaca_minute_bars",
    data_feed: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build immutable canonical observations for every qualifying window.

    Direction is evaluated as ``either`` for research so the UI's display filter
    cannot suppress valid down detections. All other thresholds are exactly the
    values used by the visible Stair-stepper check.
    """
    observations: List[Dict[str, Any]] = []
    for window in WINDOW_OPTIONS:
        for row in rows_by_window.get(window, ()):
            if not is_stair_stepper(
                row,
                r2_min=r2_min,
                direction="either",
                max_pullback_pct=max_pullback_pct,
                min_trend_pct_per_hour=min_trend_pct_per_hour,
            ):
                continue
            symbol = str(row.get("ticker") or "").strip().upper()
            direction = str(row.get("direction") or "").lower()
            observed_at = _parse_ts(row.get("as_of"))
            bucket = _bucket_timestamp(observed_at)
            price = _number(row.get("current_price"))
            if not symbol or direction not in {"up", "down"} or observed_at is None or bucket is None or price is None:
                continue
            session = _market_session(observed_at)
            canonical_direction = "long" if direction == "up" else "short"
            thresholds = {
                "r2_min": float(r2_min),
                "max_pullback_pct": float(max_pullback_pct),
                "min_trend_pct_per_hour": float(min_trend_pct_per_hour),
            }
            details = {
                "window": int(window),
                "direction": direction,
                "r2": _number(row.get("r2")),
                "slope_per_minute": _number(row.get("slope_per_minute")),
                "trend_pct_per_hour": _number(row.get("trend_pct_per_hour")),
                "current_price": price,
                "fitted_price": _number(row.get("fitted_price")),
                "max_pullback_pct": _number(row.get("max_pullback_pct")),
                "coverage": _number(row.get("coverage")),
                "bars": int(row.get("bars") or 0),
                "thresholds": thresholds,
                "detection_bucket": bucket,
                "dedupe_interval_minutes": DEDUP_INTERVAL_MINUTES,
                "trading_date": observed_at.astimezone(_ET).date().isoformat(),
                "market_session": session,
            }
            observation = build_observation(
                symbol=symbol,
                timestamp=bucket,
                context=CONTEXT,
                session=session,
                market={"price": price},
                scanners=[{
                    "name": "stair_stepper",
                    "version": DETECTION_VERSION,
                    "triggered": True,
                    "score": details["r2"],
                    "direction": canonical_direction,
                    "meta": details,
                }],
                scan_timestamp=observed_at.isoformat(),
                data_source=data_source,
                price_timestamp=observed_at.isoformat(),
                versions={"stair_stepper": DETECTION_VERSION},
            )
            observation["observation_id"] = _observation_id(symbol, window, direction, bucket)
            observation["stair_step"] = details
            observation["outcome_horizons"] = dict(OUTCOME_HORIZONS)
            observation["data_quality"]["data_feed"] = data_feed
            observations.append(observation)
    return observations


def capture_qualifying_observations(
    rows_by_window: Mapping[int, Sequence[Dict[str, Any]]],
    **kwargs,
) -> Dict[str, int]:
    """Best-effort batch persistence. Failure never changes the UI result."""
    observations = build_qualifying_observations(rows_by_window, **kwargs)
    if not observations:
        return {"attempted": 0, "written": 0, "duplicates": 0, "failed": 0}
    try:
        from db.hsf_observations import save_observations_batch

        return save_observations_batch(observations)
    except Exception:
        return {"attempted": len(observations), "written": 0, "duplicates": 0,
                "failed": len(observations)}


def is_stair_step_observation(observation: Mapping[str, Any]) -> bool:
    return observation.get("context") == CONTEXT and isinstance(observation.get("stair_step"), dict)


def horizons_for_observation(observation: Mapping[str, Any]) -> Dict[str, int]:
    if not is_stair_step_observation(observation):
        return {}
    requested = observation.get("outcome_horizons") or OUTCOME_HORIZONS
    return {h: int(requested[h]) for h in OUTCOME_HORIZONS if h in requested}


def _same_session_bars(observation: Mapping[str, Any], bars: Sequence[Mapping[str, Any]]):
    anchor = _parse_ts(observation.get("scan_timestamp") or observation.get("timestamp"))
    if anchor is None:
        return anchor, []
    trading_date = anchor.astimezone(_ET).date()
    session = str((observation.get("stair_step") or {}).get("market_session")
                  or observation.get("session") or _market_session(anchor))
    clean = []
    for raw in bars or []:
        bar_time = _parse_ts(raw.get("t"))
        close = _number(raw.get("c"))
        if bar_time is None or close is None or bar_time <= anchor:
            continue
        if bar_time.astimezone(_ET).date() != trading_date or _market_session(bar_time) != session:
            continue
        high = _number(raw.get("h"))
        low = _number(raw.get("l"))
        clean.append({
            "t": bar_time,
            "c": close,
            "h": high if high is not None else close,
            "l": low if low is not None else close,
        })
    clean.sort(key=lambda row: row["t"])
    return anchor, clean


def _excursions(entry: float, bars: Sequence[Mapping[str, Any]], direction: str) -> tuple[float, float]:
    favorable: List[float] = []
    adverse: List[float] = []
    for bar in bars:
        high = float(bar["h"])
        low = float(bar["l"])
        if direction == "down":
            favorable.append((entry - low) / entry)
            adverse.append((entry - high) / entry)
        else:
            favorable.append((high - entry) / entry)
            adverse.append((low - entry) / entry)
    return max([0.0, *favorable]), min([0.0, *adverse])


def compute_stair_step_outcomes(
    observation: Mapping[str, Any],
    bars: Sequence[Mapping[str, Any]],
    *,
    horizons: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Mature regular/pre/post-session outcomes without crossing sessions.

    The first bar at or after each wall-clock horizon is accepted only within a
    two-minute lag. Missing bars remain missing; they never become zero returns.
    """
    if not is_stair_step_observation(observation):
        return []
    details = observation.get("stair_step") or {}
    entry = _number((observation.get("market") or {}).get("price") or details.get("current_price"))
    direction = str(details.get("direction") or "").lower()
    anchor, future = _same_session_bars(observation, bars)
    if entry is None or entry <= 0 or direction not in {"up", "down"} or anchor is None or not future:
        return []
    wanted = set(horizons or OUTCOME_HORIZONS)
    outcomes: List[Dict[str, Any]] = []
    for label, minutes in OUTCOME_HORIZONS.items():
        if label not in wanted:
            continue
        target = anchor + dt.timedelta(minutes=minutes)
        outcome_bar = next((bar for bar in future if bar["t"] >= target), None)
        if outcome_bar is None:
            continue
        lag = (outcome_bar["t"] - target).total_seconds() / 60.0
        if lag > MAX_TARGET_LAG_MINUTES:
            continue
        path = [bar for bar in future if bar["t"] <= outcome_bar["t"]]
        raw_return = (float(outcome_bar["c"]) - entry) / entry
        directional_return = raw_return if direction == "up" else -raw_return
        mfe, mae = _excursions(entry, path, direction)
        outcome = build_outcome(
            observation_id=str(observation.get("observation_id") or ""),
            symbol=str(observation.get("symbol") or ""),
            observation_timestamp=anchor,
            horizon=label,
            evaluation_time=outcome_bar["t"],
            raw_return=round(raw_return, 8),
            directional_return=round(directional_return, 8),
            mfe=round(mfe, 8),
            mae=round(mae, 8),
            future_high=max(float(bar["h"]) for bar in path),
            future_low=min(float(bar["l"]) for bar in path),
            hit=directional_return > 0,
        )
        outcome.update({
            "target_time": target.isoformat(),
            "source_bar_time": outcome_bar["t"].isoformat(),
            "target_lag_minutes": round(lag, 3),
            "market_session": details.get("market_session"),
        })
        outcomes.append(outcome)
    return outcomes


def _outcome(observation: Mapping[str, Any], horizon: str) -> Optional[Mapping[str, Any]]:
    value = (observation.get("outcomes") or {}).get(horizon)
    return value if isinstance(value, Mapping) and value.get("data_status") == "MATURED" else None


def _summary(rows: Sequence[Mapping[str, Any]], horizon: str) -> Dict[str, Any]:
    outcomes = [out for row in rows if (out := _outcome(row, horizon)) is not None]
    returns = [_number(out.get("directional_return")) for out in outcomes]
    returns = [value for value in returns if value is not None]
    mfes = [_number(out.get("mfe")) for out in outcomes]
    maes = [_number(out.get("mae")) for out in outcomes]
    return {
        "n": len(returns),
        "mean_directional_return": mean(returns) if returns else None,
        "median_directional_return": median(returns) if returns else None,
        "directional_win_rate": (sum(value > 0 for value in returns) / len(returns)) if returns else None,
        "mean_mfe": mean([value for value in mfes if value is not None]) if any(v is not None for v in mfes) else None,
        "mean_mae": mean([value for value in maes if value is not None]) if any(v is not None for v in maes) else None,
    }


def _group_performance(groups: Mapping[Any, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    representatives = [sorted(rows, key=lambda row: int((row.get("stair_step") or {}).get("window") or 999))[0]
                       for rows in groups.values() if rows]
    return {h: _summary(representatives, h) for h in OUTCOME_HORIZONS}


def _window_comparison(observations: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for window in WINDOW_OPTIONS:
        rows = [row for row in observations if (row.get("stair_step") or {}).get("window") == window]
        directions = [str((row.get("stair_step") or {}).get("direction")) for row in rows]
        r2_values = [_number((row.get("stair_step") or {}).get("r2")) for row in rows]
        days = {str((row.get("stair_step") or {}).get("trading_date")) for row in rows}
        output.append({
            "window": window,
            "observations": len(rows),
            "unique_symbols": len({row.get("symbol") for row in rows}),
            "trading_days": len(days - {"None"}),
            "up": directions.count("up"),
            "down": directions.count("down"),
            "median_r2": median([v for v in r2_values if v is not None]) if any(v is not None for v in r2_values) else None,
            "horizons": {h: _summary(rows, h) for h in OUTCOME_HORIZONS},
        })
    return output


def _overlap(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        details = row.get("stair_step") or {}
        grouped[(row.get("symbol"), details.get("detection_bucket"))].append(row)
    combinations: Counter[str] = Counter()
    single = multiple = agree = disagree = 0
    agreement_groups: Dict[Any, List[Mapping[str, Any]]] = {}
    disagreement_groups: Dict[Any, List[Mapping[str, Any]]] = {}
    for key, rows in grouped.items():
        windows = sorted({int((row.get("stair_step") or {}).get("window")) for row in rows})
        combinations["+".join(map(str, windows))] += 1
        if len(windows) == 1:
            single += 1
            continue
        multiple += 1
        directions = {str((row.get("stair_step") or {}).get("direction")) for row in rows}
        if len(directions) == 1:
            agree += 1
            agreement_groups[key] = rows
        else:
            disagree += 1
            disagreement_groups[key] = rows
    total = len(grouped)
    return {
        "setup_intervals": total,
        "single_window_pct": single / total if total else None,
        "multiple_window_pct": multiple / total if total else None,
        "common_combinations": [{"windows": key, "count": count}
                                for key, count in combinations.most_common(15)],
        "multi_window_agree": agree,
        "multi_window_disagree": disagree,
        "agreement_outcomes": _group_performance(agreement_groups),
        "disagreement_outcomes": _group_performance(disagreement_groups),
    }


def _consensus(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        details = row.get("stair_step") or {}
        grouped[(row.get("symbol"), details.get("detection_bucket"), details.get("direction"))].append(row)
    by_combination: Dict[str, Dict[Any, List[Mapping[str, Any]]]] = defaultdict(dict)
    for key, rows in grouped.items():
        windows = sorted({int((row.get("stair_step") or {}).get("window")) for row in rows})
        if len(windows) < 2:
            continue
        label = "+".join(map(str, windows))
        by_combination[label][key] = rows
    return {
        "combinations": [{
            "windows": label,
            "setups": len(groups),
            "horizons": _group_performance(groups),
        } for label, groups in sorted(by_combination.items(), key=lambda item: (-len(item[1]), item[0]))],
    }


def _bucket_analysis(observations: Sequence[Mapping[str, Any]], field: str,
                     buckets: Sequence[tuple[float, float, str]]) -> List[Dict[str, Any]]:
    output = []
    for low, high, label in buckets:
        rows = []
        for row in observations:
            value = _number((row.get("stair_step") or {}).get(field))
            if value is not None and low <= value < high:
                rows.append(row)
        output.append({"bucket": label, "observations": len(rows),
                       "by_window": {str(w): {h: _summary([
                           row for row in rows if (row.get("stair_step") or {}).get("window") == w
                       ], h) for h in OUTCOME_HORIZONS} for w in WINDOW_OPTIONS}})
    return output


def _episodes(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_symbol_direction: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        details = row.get("stair_step") or {}
        by_symbol_direction[(row.get("symbol"), details.get("direction"))].append(row)
    episodes = []
    for (symbol, direction), rows in by_symbol_direction.items():
        ordered = sorted(rows, key=lambda row: _parse_ts(row.get("scan_timestamp")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
        current: List[Mapping[str, Any]] = []
        for row in ordered:
            timestamp = _parse_ts(row.get("scan_timestamp"))
            previous = _parse_ts(current[-1].get("scan_timestamp")) if current else None
            if current and timestamp and previous and timestamp - previous > dt.timedelta(minutes=60):
                episodes.append((symbol, direction, current))
                current = []
            current.append(row)
        if current:
            episodes.append((symbol, direction, current))
    records = []
    first_rows = []
    confirmation_rows = []
    confirmation_delays = []
    confirmation_moves = []
    first_counts: Counter[int] = Counter()
    for symbol, direction, rows in episodes:
        first_by_window: Dict[int, Mapping[str, Any]] = {}
        for row in rows:
            window = int((row.get("stair_step") or {}).get("window"))
            first_by_window.setdefault(window, row)
        ordered = sorted(
            first_by_window.values(),
            key=lambda row: _parse_ts(row.get("scan_timestamp"))
            or dt.datetime.max.replace(tzinfo=dt.timezone.utc),
        )
        if not ordered:
            continue
        first = ordered[0]
        first_rows.append(first)
        first_time = _parse_ts(first.get("scan_timestamp"))
        first_window = int((first.get("stair_step") or {}).get("window"))
        first_counts[first_window] += 1
        confirmations = []
        for index, row in enumerate(ordered[1:]):
            timestamp = _parse_ts(row.get("scan_timestamp"))
            price0 = _number((first.get("market") or {}).get("price"))
            price1 = _number((row.get("market") or {}).get("price"))
            delay = (
                (timestamp - first_time).total_seconds() / 60.0
                if timestamp and first_time else None
            )
            price_move = ((price1 - price0) / price0) if price0 and price1 is not None else None
            if index == 0:
                confirmation_rows.append(row)
                if delay is not None:
                    confirmation_delays.append(delay)
                if price_move is not None:
                    confirmation_moves.append(price_move)
            confirmations.append({
                "window": int((row.get("stair_step") or {}).get("window")),
                "minutes_after_first": delay,
                "price_move_from_first": price_move,
                "outcomes": {h: _outcome(row, h) for h in OUTCOME_HORIZONS},
            })
        records.append({
            "symbol": symbol,
            "direction": direction,
            "first_window": first_window,
            "first_at": first.get("scan_timestamp"),
            "confirmations": confirmations,
            "first_detection_outcomes": {h: _outcome(first, h) for h in OUTCOME_HORIZONS},
        })
    return {
        "episodes": len(records),
        "first_window_counts": dict(sorted(first_counts.items())),
        "confirmed_episodes": len(confirmation_rows),
        "median_minutes_to_first_confirmation": (
            median(confirmation_delays) if confirmation_delays else None
        ),
        "median_price_move_to_first_confirmation": (
            median(confirmation_moves) if confirmation_moves else None
        ),
        "first_detection_performance": {
            horizon: _summary(first_rows, horizon) for horizon in OUTCOME_HORIZONS
        },
        "first_confirmation_performance": {
            horizon: _summary(confirmation_rows, horizon) for horizon in OUTCOME_HORIZONS
        },
        "records": records[:500],
    }


def _daily_stability(rows: Sequence[Mapping[str, Any]], horizon: str) -> Optional[float]:
    by_day: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        outcome = _outcome(row, horizon)
        value = _number(outcome.get("directional_return")) if outcome else None
        if value is not None:
            by_day[str((row.get("stair_step") or {}).get("trading_date"))].append(value)
    daily_means = [mean(values) for values in by_day.values() if values]
    return pstdev(daily_means) if len(daily_means) >= 2 else None


def _verdict(comparison: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
             episode_report: Mapping[str, Any]) -> Dict[str, Any]:
    eligible = []
    for entry in comparison:
        horizon = (entry.get("horizons") or {}).get("+15m") or {}
        if horizon.get("n", 0) >= MIN_ROLE_SAMPLE and entry.get("trading_days", 0) >= MIN_ROLE_DAYS:
            eligible.append(entry)
    if not eligible:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "best_window": None,
            "roles": {},
            "reason": f"Requires at least {MIN_ROLE_SAMPLE} matured +15m outcomes across {MIN_ROLE_DAYS} trading days per window.",
        }
    # Deterministic evidence ranking across every available horizon: median
    # return, win rate, risk/reward, and the weakest horizon all matter. No one
    # unusually large mean return can make a window win.
    def score(entry):
        summaries = [entry["horizons"][h] for h in OUTCOME_HORIZONS]
        medians = [s.get("median_directional_return") or 0.0 for s in summaries]
        win_rates = [s.get("directional_win_rate") or 0.0 for s in summaries]
        risk_reward = [
            (s.get("mean_mfe") or 0.0) - abs(s.get("mean_mae") or 0.0)
            for s in summaries
        ]
        return (
            mean(medians),
            min(medians),
            mean(win_rates),
            mean(risk_reward),
            -int(entry["window"]),
        )

    best = max(eligible, key=score)
    stability = {}
    for entry in eligible:
        window = int(entry["window"])
        window_rows = [row for row in observations
                       if (row.get("stair_step") or {}).get("window") == window]
        stability[window] = _daily_stability(window_rows, "+15m")
    stable_windows = [window for window, value in stability.items() if value is not None]
    most_consistent = min(stable_windows, key=lambda window: (stability[window], window)) if stable_windows else None
    confirmation_candidates = [
        entry for entry in eligible
        if entry["horizons"]["+30m"].get("n", 0) >= MIN_ROLE_SAMPLE
    ]
    confirmation = max(
        confirmation_candidates,
        key=lambda entry: (
            (entry["horizons"]["+30m"].get("directional_win_rate") or 0.0),
            (entry["horizons"]["+30m"].get("median_directional_return") or 0.0),
            entry["horizons"]["+30m"].get("n", 0),
        ),
    ) if confirmation_candidates else None
    first_counts = {int(window): int(count) for window, count in
                    (episode_report.get("first_window_counts") or {}).items()}
    eligible_windows = {int(entry["window"]) for entry in eligible}
    fastest_counts = {window: count for window, count in first_counts.items()
                      if window in eligible_windows}
    fastest = max(fastest_counts, key=lambda window: (fastest_counts[window], -window)) \
        if fastest_counts else None
    roles = {
        "BEST_RISK_REWARD": best["window"],
    }
    if confirmation is not None:
        roles["BEST_CONFIRMATION"] = confirmation["window"]
    if most_consistent is not None:
        roles["MOST_CONSISTENT"] = most_consistent
    if fastest is not None:
        roles["FASTEST_SIGNAL"] = fastest
    return {
        "status": "EVIDENCE_AVAILABLE",
        "best_window": best["window"],
        "roles": roles,
        "daily_return_std_by_window": stability,
        "reason": "Ranked across 5/10/15/30m median returns, weakest-horizon return, win rate, and MFE-minus-MAE after evidence gates.",
    }


def build_research_report(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in observations if is_stair_step_observation(row)]
    timestamps = [_parse_ts(row.get("scan_timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    comparison = _window_comparison(rows)
    overlap = _overlap(rows)
    episodes = _episodes(rows)
    matured_pairs = sum(1 for row in rows for horizon in OUTCOME_HORIZONS if _outcome(row, horizon))
    matured_observations = sum(
        any(_outcome(row, horizon) for horizon in OUTCOME_HORIZONS) for row in rows
    )
    return {
        "schema_version": "stair-step-research-1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OK" if rows else "NO_OBSERVATIONS",
        "methodology": {
            "context": CONTEXT,
            "detection_version": DETECTION_VERSION,
            "windows": list(WINDOW_OPTIONS),
            "outcome_horizons_minutes": dict(OUTCOME_HORIZONS),
            "dedupe_interval_minutes": DEDUP_INTERVAL_MINUTES,
            "maximum_target_lag_minutes": MAX_TARGET_LAG_MINUTES,
            "minimum_role_sample": MIN_ROLE_SAMPLE,
            "minimum_role_trading_days": MIN_ROLE_DAYS,
        },
        "data_coverage": {
            "observation_start": min(timestamps).isoformat() if timestamps else None,
            "observation_end": max(timestamps).isoformat() if timestamps else None,
            "trading_days": len({value.astimezone(_ET).date() for value in timestamps}),
            "total_observations": len(rows),
            "unique_symbols": len({row.get("symbol") for row in rows}),
            "matured_observations": matured_observations,
            "matured_horizon_pairs": matured_pairs,
            "missing_horizon_pairs": len(rows) * len(OUTCOME_HORIZONS) - matured_pairs,
            "failed_maturations": None,
        },
        "window_comparison": comparison,
        "window_overlap": overlap,
        "early_signal_vs_confirmation": episodes,
        "multi_window_consensus": _consensus(rows),
        "r2_analysis": _bucket_analysis(rows, "r2", (
            (0.80, 0.85, "0.80-0.849"), (0.85, 0.90, "0.85-0.899"),
            (0.90, 0.95, "0.90-0.949"), (0.95, float("inf"), "0.95+"),
        )),
        "pullback_analysis": _bucket_analysis(rows, "max_pullback_pct", (
            (0.0, 0.25, "0-0.249"), (0.25, 0.50, "0.25-0.499"),
            (0.50, 0.75, "0.50-0.749"), (0.75, 1.01, "0.75-1.00"),
            (1.01, float("inf"), ">1.00"),
        )),
        "best_window_verdict": _verdict(comparison, rows, episodes),
        "limitations": [
            "Observations exist only when a user runs the on-demand Stair-stepper check.",
            "IEX coverage can be sparse; missing horizon bars remain missing rather than becoming zero returns.",
            "Roles remain evidence-gated and do not alter the production selector or qualification thresholds.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    coverage = report.get("data_coverage") or {}
    verdict = report.get("best_window_verdict") or {}
    lines = [
        "# Day Trade Stair-Stepper Outcome Validation",
        "",
        f"**Status:** {report.get('status')}",
        f"**Verdict:** {verdict.get('status')}",
        "",
        "## Data coverage",
        f"- Observation period: {coverage.get('observation_start')} to {coverage.get('observation_end')}",
        f"- Trading days: {coverage.get('trading_days')}",
        f"- Observations: {coverage.get('total_observations')}",
        f"- Unique symbols: {coverage.get('unique_symbols')}",
        f"- Matured observations: {coverage.get('matured_observations')}",
        f"- Matured horizon pairs: {coverage.get('matured_horizon_pairs')}",
        f"- Missing horizon pairs: {coverage.get('missing_horizon_pairs')}",
        "- Failed maturations: tracked in the companion maturation report; not inferred from missing bars",
        "",
        "## Window comparison",
        "| Window | Obs | Symbols | Up | Down | Median R² | 5m win | 10m win | 15m win | 30m win |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report.get("window_comparison") or []:
        horizons = entry.get("horizons") or {}
        rates = [((horizons.get(h) or {}).get("directional_win_rate")) for h in OUTCOME_HORIZONS]
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"  # noqa: E731
        lines.append(
            f"| {entry.get('window')} | {entry.get('observations')} | {entry.get('unique_symbols')} | "
            f"{entry.get('up')} | {entry.get('down')} | {fmt(entry.get('median_r2'))} | "
            + " | ".join(fmt(value) for value in rates) + " |"
        )
    lines += [
        "",
        "## Overlap and confirmation",
        f"- Single-window setups: {(report.get('window_overlap') or {}).get('single_window_pct')}",
        f"- Multi-window setups: {(report.get('window_overlap') or {}).get('multiple_window_pct')}",
        f"- Episodes analyzed: {(report.get('early_signal_vs_confirmation') or {}).get('episodes')}",
        f"- Confirmed episodes: {(report.get('early_signal_vs_confirmation') or {}).get('confirmed_episodes')}",
        f"- Median minutes to first confirmation: {(report.get('early_signal_vs_confirmation') or {}).get('median_minutes_to_first_confirmation')}",
        f"- Median price move to first confirmation: {(report.get('early_signal_vs_confirmation') or {}).get('median_price_move_to_first_confirmation')}",
        "",
        "## Multi-window consensus",
        f"- Combinations evaluated: {len((report.get('multi_window_consensus') or {}).get('combinations') or [])}",
        "",
        "## Best-window verdict",
        f"- Status: {verdict.get('status')}",
        f"- Best window: {verdict.get('best_window')}",
        f"- Window roles: {verdict.get('roles')}",
        f"- Reason: {verdict.get('reason')}",
        "",
        "## Recommendation",
        ("- Continue collecting observations; do not surface a production best-window recommendation."
         if verdict.get("status") != "EVIDENCE_AVAILABLE"
         else "- Treat the reported roles as research evidence only; validate stability before any UI recommendation."),
        "",
        "## Limitations",
    ]
    lines.extend(f"- {item}" for item in report.get("limitations") or [])
    lines.append("")
    return "\n".join(lines)
