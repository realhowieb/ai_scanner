"""Vendor-neutral automation snapshot export for scheduled scanner runs.

This module adapts already-computed scanner rows into a compact JSON schema.
It intentionally does not fetch market data, score models, rank rows, or call
external AI services.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is present in app/runtime tests
    pd = None  # type: ignore[assignment]


SCHEMA_VERSION = "1.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "automation"
LATEST_FILENAME = "latest_scan.json"
STATUS_FILENAME = "status.json"
DEFAULT_HISTORY_DAYS = 30

SENSITIVE_ENV_TOKENS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASS",
    "PASSWORD",
    "DATABASE_URL",
    "DSN",
)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: Any | None = None) -> str | None:
    """Return an ISO UTC timestamp or None for unavailable values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat()
    return str(value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value)) if pd is not None else value is None
    except (TypeError, ValueError):
        return value is None


def safe_json_value(value: Any) -> Any:
    """Convert values to strict JSON-safe primitives; NaN/Infinity become null."""
    if _is_missing(value):
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        return iso_utc(value)
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    return str(value)


def _num(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and not _is_missing(row.get(name)):
            try:
                value = float(row.get(name))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _int(row: dict[str, Any], *names: str) -> int | None:
    value = _num(row, *names)
    return int(value) if value is not None else None


def _text(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if not _is_missing(value):
            text = str(value).strip()
            return text or None
    return None


def _probability(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _num(row, name)
        if value is None:
            continue
        if name.endswith("%") or name in {"AI Confidence", "AIConfidence"}:
            return value / 100.0
        return value
    return None


def dataframe_records(results: Any) -> list[dict[str, Any]]:
    if results is None:
        return []
    if pd is not None and isinstance(results, pd.DataFrame):
        return [dict(row) for row in results.to_dict(orient="records")]
    if isinstance(results, list):
        return [dict(row) for row in results if isinstance(row, dict)]
    return []


def candidate_from_row(row: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    symbol = (_text(row, "Ticker", "Symbol", "symbol") or "").upper()
    pre_score = _num(row, "PreBreakoutScore", "prebreakout_score")
    pre_prob = _probability(row, "PreBreakoutProb", "PreBreakoutProb%", "prebreakout_ml_probability")
    breakout_prob = _probability(row, "BreakoutProb", "AI Confidence", "AIConfidence", "breakout_ml_probability")
    breakout_score = _num(row, "BreakoutScore", "Breakout %", "breakout_score")

    candidate = {
        "symbol": symbol or None,
        "price": _num(row, "Last", "Close", "Price", "price"),
        "percent_change": _num(row, "PctChange", "PercentChange", "percent_change"),
        "ema9": _num(row, "EMA9", "ema9"),
        "ema21": _num(row, "EMA21", "ema21"),
        "ema_spread_pct": _num(row, "EMA9_21SpreadPct", "EMASpreadPct", "ema_spread_pct"),
        "ema_signal": _text(row, "EMACross", "EMASignal", "ema_signal"),
        "rsi14": _num(row, "RSI14", "RSI", "rsi14"),
        "rvol": _num(row, "VolRel20", "RelVol", "RVOL", "rvol"),
        "volume": _int(row, "Volume", "volume"),
        "avg_volume": _int(row, "VolAvg20", "AvgVolume", "AverageVolume", "avg_volume"),
        "prebreakout_score": pre_score,
        "breakout_score": breakout_score,
        "prebreakout_ml_probability": pre_prob,
        "breakout_ml_probability": breakout_prob,
        "scanner_signal": _text(row, "PatternTag", "ScannerSignal", "scanner_signal"),
        "rank": rank,
        "data_quality": {"valid": True, "warnings": []},
    }
    _validate_candidate(candidate, source_row=row)
    return safe_json_value(candidate)


def _validate_candidate(candidate: dict[str, Any], *, source_row: dict[str, Any]) -> None:
    warnings = candidate["data_quality"]["warnings"]
    if not candidate.get("symbol"):
        warnings.append("missing_symbol")
    volume = candidate.get("volume")
    avg_volume = candidate.get("avg_volume")
    if volume is not None and volume < 0:
        warnings.append("negative_volume")
    if avg_volume is not None and avg_volume < 0:
        warnings.append("negative_avg_volume")
    rsi = candidate.get("rsi14")
    if rsi is not None and not (0 <= float(rsi) <= 100):
        warnings.append("rsi_out_of_range")
    for key in ("prebreakout_ml_probability", "breakout_ml_probability"):
        value = candidate.get(key)
        if value is not None and not (0 <= float(value) <= 1):
            warnings.append(f"{key}_out_of_range")
    if candidate.get("breakout_score") is not None:
        if candidate.get("price") is None:
            warnings.append("score_present_missing_price")
        if candidate.get("rvol") is None:
            warnings.append("score_present_missing_rvol")
    for key, value in source_row.items():
        if isinstance(value, float) and not math.isfinite(value):
            warnings.append(f"non_finite_source_value:{key}")
    candidate["data_quality"]["valid"] = "missing_symbol" not in warnings


def _score_distribution(values: list[float | None]) -> dict[str, Any]:
    usable = [round(float(v), 6) for v in values if v is not None and math.isfinite(float(v))]
    if not usable:
        return {
            "unique_scores": 0,
            "duplicate_score_ratio": 0.0,
            "warning": None,
        }
    counts = Counter(usable)
    unique = len(counts)
    most_common = counts.most_common(1)[0][1]
    duplicate_ratio = 1.0 - (unique / len(usable))
    warning = None
    if len(usable) >= 10 and (duplicate_ratio >= 0.8 or most_common / len(usable) >= 0.6):
        warning = "suspicious_score_clustering"
    return {
        "unique_scores": unique,
        "duplicate_score_ratio": round(duplicate_ratio, 6),
        "most_common_score_count": most_common,
        "sample_count": len(usable),
        "warning": warning,
    }


def build_diagnostics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    pre_scores = [c.get("prebreakout_score") for c in candidates]
    pre_probs = [c.get("prebreakout_ml_probability") for c in candidates]
    breakout_probs = [c.get("breakout_ml_probability") for c in candidates]
    symbols = [c.get("symbol") for c in candidates if c.get("symbol")]
    duplicate_symbols = sorted([symbol for symbol, count in Counter(symbols).items() if count > 1])
    return {
        "duplicate_symbols": duplicate_symbols,
        "prebreakout_score_distribution": _score_distribution(pre_scores),
        "prebreakout_ml_probability_distribution": _score_distribution(pre_probs),
        "breakout_ml_probability_distribution": _score_distribution(breakout_probs),
    }


def validate_snapshot(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "generated_at_utc", "scan", "models", "summary", "candidates"):
        if key not in snapshot:
            errors.append(f"missing_top_level:{key}")
    candidates = snapshot.get("candidates")
    if candidates is not None and not isinstance(candidates, list):
        errors.append("candidates_not_list")
    try:
        encoded = json.dumps(snapshot, allow_nan=False)
        json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json:{type(exc).__name__}")
    return errors


def collect_model_provenance() -> dict[str, dict[str, Any]]:
    """Best-effort model metadata without retraining or requiring availability."""
    return {
        "prebreakout": _prebreakout_model_provenance(),
        "ai_confidence": _ai_confidence_model_provenance(),
    }


def _prebreakout_model_provenance() -> dict[str, Any]:
    try:
        import ml_prebreakout

        bundle = ml_prebreakout.load_prebreakout_model()
        metadata = dict(bundle.get("metadata") or {}) if isinstance(bundle, dict) else {}
        return safe_json_value(
            {
                "version": (bundle or {}).get("model_version") if isinstance(bundle, dict) else getattr(ml_prebreakout, "MODEL_VERSION", None),
                "trained_at": (bundle or {}).get("trained_at") if isinstance(bundle, dict) else metadata.get("trained_at"),
                "artifact": (bundle or {}).get("source") if isinstance(bundle, dict) else str(getattr(ml_prebreakout, "MODEL_PATH", "")),
                "feature_schema_version": metadata.get("feature_schema_version"),
                "feature_count": len((bundle or {}).get("features") or [] if isinstance(bundle, dict) else []),
            }
        )
    except Exception as exc:
        return {"version": None, "trained_at": None, "artifact": None, "warning": f"unavailable:{type(exc).__name__}"}


def _ai_confidence_model_provenance() -> dict[str, Any]:
    try:
        from scan.ai_confidence import MODEL_VERSION, load_ai_confidence_bundle

        _model, metadata, warning = load_ai_confidence_bundle()
        return safe_json_value(
            {
                "version": metadata.get("model_version") or MODEL_VERSION,
                "trained_at": metadata.get("trained_at"),
                "artifact": metadata.get("source") or "local",
                "feature_schema_version": metadata.get("feature_schema_version"),
                "feature_count": len(metadata.get("feature_names") or []),
                "warning": warning,
            }
        )
    except Exception as exc:
        return {"version": None, "trained_at": None, "artifact": None, "warning": f"unavailable:{type(exc).__name__}"}


def github_run_metadata(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    return {
        "github_run_id": env.get("GITHUB_RUN_ID"),
        "github_run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "github_workflow": env.get("GITHUB_WORKFLOW"),
        "github_job": env.get("GITHUB_JOB"),
        "github_ref": env.get("GITHUB_REF"),
        "github_repository": env.get("GITHUB_REPOSITORY"),
        "git_sha": env.get("GITHUB_SHA") or env.get("RENDER_GIT_COMMIT") or env.get("COMMIT_SHA"),
    }


def build_run_id(
    *,
    universe: str,
    scan_type: str,
    started_at_utc: dt.datetime,
    env: dict[str, str] | None = None,
) -> str:
    env = os.environ if env is None else env
    run_id = env.get("GITHUB_RUN_ID")
    attempt = env.get("GITHUB_RUN_ATTEMPT")
    suffix = f"-{attempt}" if attempt else ""
    if run_id:
        return f"github-{run_id}{suffix}-{scan_type.lower()}-{universe.lower()}"
    stamp = started_at_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"local-{stamp}-{scan_type.lower()}-{universe.lower()}"


def build_snapshot(
    results: Any,
    *,
    universe: str,
    scan_type: str,
    market_session: str | None,
    started_at_utc: dt.datetime,
    completed_at_utc: dt.datetime | None = None,
    duration_seconds: float | None = None,
    symbols_requested: int | None = None,
    symbols_processed: int | None = None,
    symbols_skipped: int | None = None,
    warnings: Iterable[str] | None = None,
    errors: Iterable[str] | None = None,
    model_metadata: dict[str, dict[str, Any]] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed_at_utc = completed_at_utc or utc_now()
    records = dataframe_records(results)
    candidates = [candidate_from_row(row, rank=i + 1) for i, row in enumerate(records)]
    diagnostics = build_diagnostics(candidates)
    warnings_list = [str(w) for w in (warnings or []) if str(w).strip()]
    errors_list = [str(e) for e in (errors or []) if str(e).strip()]
    if diagnostics["duplicate_symbols"]:
        warnings_list.append("duplicate_symbols_detected")
    for key, dist in diagnostics.items():
        if isinstance(dist, dict) and dist.get("warning"):
            warnings_list.append(f"{key}:{dist['warning']}")

    run_id = build_run_id(
        universe=universe,
        scan_type=scan_type,
        started_at_utc=started_at_utc,
        env=env,
    )
    env_meta = github_run_metadata(env)
    git_sha = env_meta.get("git_sha")
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": iso_utc(completed_at_utc),
        "scan": {
            "run_id": run_id,
            "scan_type": scan_type,
            "universe": universe,
            "market_session": market_session,
            "started_at_utc": iso_utc(started_at_utc),
            "completed_at_utc": iso_utc(completed_at_utc),
            "duration_seconds": round(float(duration_seconds or 0.0), 3),
            "status": "success",
            "github_run_id": env_meta.get("github_run_id"),
            "github_run_attempt": env_meta.get("github_run_attempt"),
            "github_workflow": env_meta.get("github_workflow"),
            "github_job": env_meta.get("github_job"),
            "github_ref": env_meta.get("github_ref"),
            "github_repository": env_meta.get("github_repository"),
            "git_sha": git_sha,
            "symbols_requested": symbols_requested,
            "symbols_processed": symbols_processed,
            "symbols_skipped": symbols_skipped,
        },
        "models": model_metadata or collect_model_provenance(),
        "summary": {
            "symbols_scanned": symbols_processed if symbols_processed is not None else symbols_requested,
            "symbols_requested": symbols_requested,
            "symbols_skipped": symbols_skipped,
            "candidates_found": len(candidates),
            "prebreakout_candidates": sum(1 for c in candidates if c.get("prebreakout_ml_probability") is not None),
            "breakout_candidates": sum(1 for c in candidates if c.get("breakout_score") is not None),
            "errors": len(errors_list),
            "warnings": len(warnings_list),
        },
        "diagnostics": diagnostics,
        "warnings": warnings_list,
        "errors": errors_list,
        "candidates": candidates,
    }
    return safe_json_value(snapshot)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp_path.write_text(encoded, encoding="utf-8")
    os.replace(tmp_path, path)


def _history_path(root: Path, snapshot: dict[str, Any]) -> Path:
    completed = str(snapshot["scan"]["completed_at_utc"])
    day = completed[:10] if len(completed) >= 10 else utc_now().date().isoformat()
    run_id = str(snapshot["scan"]["run_id"])
    safe_run_id = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in run_id)
    return root / "history" / day / f"{safe_run_id}.json"


def prune_history(root: Path, *, retention_days: int = DEFAULT_HISTORY_DAYS, now: dt.datetime | None = None) -> list[Path]:
    history_root = root / "history"
    if not history_root.exists():
        return []
    now = now or utc_now()
    cutoff = now.date() - dt.timedelta(days=max(0, int(retention_days)))
    removed: list[Path] = []
    for child in history_root.iterdir():
        if not child.is_dir():
            continue
        try:
            child_date = dt.date.fromisoformat(child.name)
        except ValueError:
            continue
        if child_date >= cutoff:
            continue
        for file in child.glob("*.json"):
            try:
                file.unlink()
                removed.append(file)
            except OSError:
                pass
        try:
            child.rmdir()
        except OSError:
            pass
    return removed


def build_status(snapshot: dict[str, Any], *, status: str = "success") -> dict[str, Any]:
    return safe_json_value(
        {
            "last_attempt_at_utc": snapshot.get("generated_at_utc"),
            "last_success_at_utc": snapshot.get("generated_at_utc") if status == "success" else None,
            "status": status,
            "run_id": (snapshot.get("scan") or {}).get("run_id"),
            "git_sha": (snapshot.get("scan") or {}).get("git_sha"),
            "snapshot": LATEST_FILENAME,
            "candidate_count": (snapshot.get("summary") or {}).get("candidates_found", 0),
            "warning_count": (snapshot.get("summary") or {}).get("warnings", 0),
            "error_count": (snapshot.get("summary") or {}).get("errors", 0),
        }
    )


def publish_snapshot(
    snapshot: dict[str, Any],
    *,
    root: Path | str = DEFAULT_ROOT,
    retention_days: int | None = None,
) -> dict[str, Any]:
    """Validate and publish latest/status/history automation artifacts."""
    root_path = Path(root)
    validation_errors = validate_snapshot(snapshot)
    if validation_errors:
        raise ValueError(f"automation snapshot validation failed: {', '.join(validation_errors)}")

    latest_path = root_path / LATEST_FILENAME
    status_path = root_path / STATUS_FILENAME
    history_path = _history_path(root_path, snapshot)
    write_json_atomic(history_path, snapshot)
    write_json_atomic(latest_path, snapshot)
    status = build_status(snapshot)
    write_json_atomic(status_path, status)
    days = DEFAULT_HISTORY_DAYS if retention_days is None else int(retention_days)
    removed = prune_history(root_path, retention_days=days)
    return {
        "ok": True,
        "latest_path": str(latest_path),
        "status_path": str(status_path),
        "history_path": str(history_path),
        "removed_history_files": len(removed),
        "candidate_count": len(snapshot.get("candidates") or []),
        "warning_count": int((snapshot.get("summary") or {}).get("warnings") or 0),
        "error_count": int((snapshot.get("summary") or {}).get("errors") or 0),
    }


def publish_scan_results(
    results: Any,
    *,
    universe: str,
    scan_type: str,
    market_session: str | None,
    started_at_utc: dt.datetime,
    completed_at_utc: dt.datetime | None = None,
    duration_seconds: float | None = None,
    symbols_requested: int | None = None,
    symbols_processed: int | None = None,
    symbols_skipped: int | None = None,
    warnings: Iterable[str] | None = None,
    errors: Iterable[str] | None = None,
    model_metadata: dict[str, dict[str, Any]] | None = None,
    root: Path | str = DEFAULT_ROOT,
    retention_days: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    snapshot = build_snapshot(
        results,
        universe=universe,
        scan_type=scan_type,
        market_session=market_session,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_seconds=duration_seconds,
        symbols_requested=symbols_requested,
        symbols_processed=symbols_processed,
        symbols_skipped=symbols_skipped,
        warnings=warnings,
        errors=errors,
        model_metadata=model_metadata,
        env=env,
    )
    return publish_snapshot(snapshot, root=root, retention_days=retention_days)
