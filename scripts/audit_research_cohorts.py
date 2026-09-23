#!/usr/bin/env python3
"""Run 53 — read-only research-dataset integrity audit.

Measures the Run 47 research cohorts (CANDIDATE / NEAR_MISS / CONTROL) as they
sit in the live store and reports per-cohort integrity metrics for effectiveness
analysis. STRICTLY read-only: it never writes observations, outcomes, scans, or
scanners, and never changes scoring/ranking — it only counts and validates what
production already captured.

    python -m scripts.audit_research_cohorts [--limit N] [--out DIR]

Per cohort it reports: observations, distinct symbols, distinct scan runs,
observations/run, first/latest observation timestamp, matured/unmatured outcome
counts, observations with missing recognized features, duplicate observation_ids,
conflicting duplicates (same (symbol,timestamp,context) → different record),
invalid values (negative/NaN price or volume), and point-in-time violations
(an attached outcome evaluated at/before its observation timestamp). It also
separates EXPLICITLY-tagged cohort rows from LEGACY-inferred candidates so old
pre-cohort observations cannot silently contaminate modern analysis (Task 9).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS, cohort_of

ROOT = Path(__file__).resolve().parents[1]
_COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)

# Run 53B: minimum analysis contract. Optional fields remain visible in the audit,
# but their absence never makes an otherwise valid cohort/outcome row defective.
FIELD_CONTRACT = (
    {"field": "observation_id", "group": "IDENTITY", "capture": True, "run54": True},
    {"field": "symbol", "group": "IDENTITY", "capture": True, "run54": True},
    {"field": "timestamp", "group": "IDENTITY", "capture": True, "run54": True},
    {"field": "market_context.scan_id", "group": "IDENTITY", "capture": True, "run54": True},
    {"field": "research_cohort", "group": "IDENTITY", "capture": True, "run54": True},
    {"field": "direction", "group": "POINT_IN_TIME", "capture": True, "run54": True},
    {"field": "market.price", "group": "POINT_IN_TIME", "capture": True, "run54": True},
    {"field": "scanner_score", "group": "POINT_IN_TIME", "capture": False, "run54": False},
    {"field": "rank", "group": "POINT_IN_TIME", "capture": False, "run54": False},
    {"field": "models.prebreakout.probability", "group": "POINT_IN_TIME", "capture": False, "run54": False},
    {"field": "models.ai_confidence.confidence", "group": "POINT_IN_TIME", "capture": False, "run54": False},
)
OUTCOME_CONTRACT = (
    "horizon", "evaluation_time", "raw_return", "directional_return", "mfe", "mae",
)


def _direction(rec: Dict[str, Any]) -> Optional[str]:
    for scanner in rec.get("scanners") or []:
        value = str(scanner.get("direction") or "").strip().lower()
        if value:
            return "SHORT" if value in {"short", "bearish"} else "LONG"
    return None


def _field_value(rec: Dict[str, Any], field: str) -> Any:
    if field == "research_cohort":
        return (rec.get("research_cohort")
                or (rec.get("market_context") or {}).get("research_cohort"))
    if field == "direction":
        return _direction(rec)
    if field == "scanner_score":
        for scanner in rec.get("scanners") or []:
            if scanner.get("score") is not None:
                return scanner.get("score")
        return None
    value: Any = rec
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _present(value: Any) -> bool:
    return value is not None and value != "" and not (
        isinstance(value, float) and value != value)


def _field_classification(field: str, *, explicit: bool) -> tuple[str, str]:
    spec = next((s for s in FIELD_CONTRACT if s["field"] == field), None)
    if field.startswith("outcome."):
        return "EXPECTED_NULL", "required only after the specific horizon matures"
    if field == "research_cohort" and not explicit:
        return "LEGACY_SCHEMA", "legacy Candidate rows predate explicit Run-47 tags"
    if spec and spec["run54"]:
        return "ACTUAL_DATA_DEFECT", "required for the minimum Run-54 contract"
    if spec:
        return "OPTIONAL", "required only for analyses that use this feature"
    return "OPTIONAL", "canonical completeness field, not a Run-54 requirement"


def _outcome_complete(rec: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
    return (_present((rec.get("market") or {}).get("price"))
            and all(_present(outcome.get(field)) for field in OUTCOME_CONTRACT))


def _parse_dt(v: Any) -> Optional[_dt.datetime]:
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _is_explicit(rec: Dict[str, Any]) -> bool:
    """True when the row carries an explicit cohort tag (post-Run-47); False for a
    legacy untagged observation that cohort_of() would infer as CANDIDATE."""
    return bool(rec.get("research_cohort")
                or (rec.get("market_context") or {}).get("research_cohort"))


def _scan_run(rec: Dict[str, Any]) -> str:
    mc = rec.get("market_context") or {}
    return str(mc.get("scan_id") or rec.get("scan_timestamp") or rec.get("timestamp") or "")


def _invalid_values(rec: Dict[str, Any]) -> bool:
    mkt = rec.get("market") or {}
    for k in ("price", "volume"):
        v = mkt.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            return True
        if f != f or f < 0:  # NaN or negative
            return True
    return False


def _empty_cohort_stats() -> Dict[str, Any]:
    return {
        "observations": 0, "distinct_symbols": set(), "distinct_runs": set(),
        "first_ts": None, "latest_ts": None, "matured": 0, "unmatured": 0,
        "missing_fields_obs": 0, "invalid_values": 0, "pit_violations": 0,
        "explicit": 0, "legacy_inferred": 0,
        "field_missing": Counter(), "canonical_missing": Counter(),
        "modern_field_missing": Counter(), "outcome_field_missing": Counter(),
        "matured_outcome_records": 0,
        "modern_matured": 0, "modern_unmatured": 0,
        "maturity": defaultdict(lambda: defaultdict(lambda: {
            "observations": 0, "analysis_eligible": 0,
        })),
        "direction_examples": defaultdict(list),
    }


def audit_cohorts(observations: List[Dict[str, Any]],
                  outcomes_by_id: Optional[Dict[str, List[Dict[str, Any]]]] = None
                  ) -> Dict[str, Any]:
    """Pure aggregation over already-loaded observations. `outcomes_by_id` maps
    observation_id -> list of that observation's outcome records (from the separate
    outcomes table — never mixed into features). Each outcome's evaluation_time is
    checked against the observation timestamp for point-in-time violations."""
    outcomes_by_id = outcomes_by_id or {}
    stats: Dict[str, Dict[str, Any]] = {c: _empty_cohort_stats() for c in _COHORTS}
    id_seen: Dict[str, str] = {}          # observation_id -> record hash (dupe check)
    key_records: Dict[tuple, str] = {}    # (symbol,timestamp,context) -> record hash
    duplicate_ids = 0
    conflicting_duplicates = 0
    # Per scan_run_id, which cohorts each symbol appeared in (overlap check, Part 2).
    run_symbol_cohorts: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for rec in observations or []:
        if not rec:
            continue
        cohort = cohort_of(rec)
        s = stats.setdefault(cohort, _empty_cohort_stats())
        s["observations"] += 1
        s["explicit" if _is_explicit(rec) else "legacy_inferred"] += 1
        sym = str(rec.get("symbol") or "").upper()
        s["distinct_symbols"].add(sym)
        run_id = _scan_run(rec)
        s["distinct_runs"].add(run_id)
        run_symbol_cohorts[run_id][sym].add(cohort)

        ts = _parse_dt(rec.get("timestamp"))
        if ts is not None:
            s["first_ts"] = ts if s["first_ts"] is None else min(s["first_ts"], ts)
            s["latest_ts"] = ts if s["latest_ts"] is None else max(s["latest_ts"], ts)

        oid = str(rec.get("observation_id") or "")
        rec_hash = json.dumps(rec, sort_keys=True, default=str)
        if oid in id_seen:
            duplicate_ids += 1
            if id_seen[oid] != rec_hash:
                conflicting_duplicates += 1  # same id, different content (mutation)
        else:
            id_seen[oid] = rec_hash
        key = (sym, str(rec.get("timestamp")), str(rec.get("context")))
        if key in key_records and key_records[key] != rec_hash:
            conflicting_duplicates += 1
        key_records.setdefault(key, rec_hash)

        canonical_missing = rec.get("data_quality", {}).get("missing_fields") or []
        if canonical_missing:
            s["missing_fields_obs"] += 1
        s["canonical_missing"].update(str(f) for f in canonical_missing)
        for spec in FIELD_CONTRACT:
            if not _present(_field_value(rec, spec["field"])):
                s["field_missing"][spec["field"]] += 1
                if _is_explicit(rec):
                    s["modern_field_missing"][spec["field"]] += 1
        if _invalid_values(rec):
            s["invalid_values"] += 1

        obs_outcomes = outcomes_by_id.get(oid) or []
        if obs_outcomes:
            s["matured"] += 1
        else:
            s["unmatured"] += 1
        if _is_explicit(rec):
            s["modern_matured" if obs_outcomes else "modern_unmatured"] += 1
        # Point-in-time: an outcome must evaluate strictly AFTER the observation.
        for oc in obs_outcomes:
            eval_dt = _parse_dt(oc.get("evaluation_time"))
            if ts is not None and eval_dt is not None and eval_dt <= ts:
                s["pit_violations"] += 1
            if not _is_explicit(rec) or str(oc.get("data_status")) != "MATURED":
                continue
            s["matured_outcome_records"] += 1
            for field in OUTCOME_CONTRACT:
                if not _present(oc.get(field)):
                    s["outcome_field_missing"][field] += 1
            horizon = str(oc.get("horizon") or "UNKNOWN")
            direction = _direction(rec) or "UNKNOWN"
            cell = s["maturity"][horizon][direction]
            cell["observations"] += 1
            complete = _outcome_complete(rec, oc)
            if complete:
                cell["analysis_eligible"] += 1
            if len(s["direction_examples"][direction]) < 3:
                entry = (rec.get("market") or {}).get("price")
                raw = oc.get("raw_return")
                future = (float(entry) * (1.0 + float(raw))
                          if _present(entry) and _present(raw) else None)
                s["direction_examples"][direction].append({
                    "symbol": sym, "horizon": horizon, "entry_price": entry,
                    "future_price_derived": round(future, 6) if future is not None else None,
                    "raw_return": raw, "directional_return": oc.get("directional_return"),
                    "mfe": oc.get("mfe"), "mae": oc.get("mae"),
                })

    # Finalize sets → counts + ISO timestamps.
    out_cohorts = {}
    for c, s in stats.items():
        obs = s["observations"]
        runs = len(s["distinct_runs"])
        explicit = s["explicit"]
        field_rows = []
        combined_fields = set(s["canonical_missing"]) | {x["field"] for x in FIELD_CONTRACT}
        for field in sorted(combined_fields):
            missing = (s["field_missing"].get(field, 0)
                       if any(x["field"] == field for x in FIELD_CONTRACT)
                       else s["canonical_missing"].get(field, 0))
            classification, reason = _field_classification(
                field, explicit=not (field == "research_cohort" and missing == s["legacy_inferred"]))
            field_rows.append({
                "field": field, "missing_count": missing,
                "missing_pct": round(100.0 * missing / obs, 2) if obs else 0.0,
                "classification": classification,
                "required_for_capture": bool(next((x["capture"] for x in FIELD_CONTRACT
                                                     if x["field"] == field), False)),
                "required_for_maturation": field in {"market.price", "direction"},
                "required_for_run54": bool(next((x["run54"] for x in FIELD_CONTRACT
                                                   if x["field"] == field), False)),
                "detail": reason,
            })
        maturity = {
            h: {d: dict(values) for d, values in directions.items()}
            for h, directions in sorted(s["maturity"].items())
        }
        out_cohorts[c] = {
            "observations": obs,
            "distinct_symbols": len(s["distinct_symbols"]),
            "distinct_runs": runs,
            "observations_per_run": round(obs / runs, 2) if runs else 0.0,
            "first_observation": s["first_ts"].isoformat() if s["first_ts"] else None,
            "latest_observation": s["latest_ts"].isoformat() if s["latest_ts"] else None,
            "matured": s["matured"],
            "unmatured": s["unmatured"],
            "missing_field_observations": s["missing_fields_obs"],
            "invalid_values": s["invalid_values"],
            "point_in_time_violations": s["pit_violations"],
            "explicitly_tagged": s["explicit"],
            "legacy_inferred": s["legacy_inferred"],
            "modern_matured": s["modern_matured"],
            "modern_unmatured": s["modern_unmatured"],
            "field_missingness": field_rows,
            "modern_required_field_missingness": [
                {"field": spec["field"],
                 "missing_count": s["modern_field_missing"].get(spec["field"], 0),
                 "missing_pct": round(
                     100.0 * s["modern_field_missing"].get(spec["field"], 0) / explicit, 2)
                 if explicit else 0.0}
                for spec in FIELD_CONTRACT if spec["run54"]
            ],
            "matured_outcome_field_missingness": [
                {"field": field,
                 "missing_count": s["outcome_field_missing"].get(field, 0),
                 "missing_pct": round(
                     100.0 * s["outcome_field_missing"].get(field, 0)
                     / s["matured_outcome_records"], 2)
                 if s["matured_outcome_records"] else 0.0}
                for field in OUTCOME_CONTRACT
            ],
            "modern_maturity_by_horizon_direction": maturity,
            "direction_examples": dict(s["direction_examples"]),
        }
    # Cohort overlap within a scan_run_id (a symbol tagged as >1 cohort for the
    # same selection event). candidate∩control is the serious one (Part 2).
    overlap = {"candidate_control": 0, "candidate_near_miss": 0,
               "near_miss_control": 0, "any": 0}
    for _run, syms in run_symbol_cohorts.items():
        for _sym, cohorts in syms.items():
            if len(cohorts) < 2:
                continue
            overlap["any"] += 1
            if CANDIDATE in cohorts and CONTROL in cohorts:
                overlap["candidate_control"] += 1
            if CANDIDATE in cohorts and NEAR_MISS in cohorts:
                overlap["candidate_near_miss"] += 1
            if NEAR_MISS in cohorts and CONTROL in cohorts:
                overlap["near_miss_control"] += 1

    return {
        "schema": "hsf-cohort-audit-1.2",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "total_observations": sum(c["observations"] for c in out_cohorts.values()),
        "duplicate_observation_ids": duplicate_ids,
        "conflicting_duplicates": conflicting_duplicates,
        "cohort_overlap_within_run": overlap,
        "cohorts": out_cohorts,
        "run54_contract": {
            "observation_fields": list(FIELD_CONTRACT),
            "outcome_fields_after_maturity": list(OUTCOME_CONTRACT),
            "future_price": "derived as entry_price * (1 + raw_return); not persisted",
            "null_policy": "exclude rows missing fields required by the specific analysis; never zero-fill",
        },
    }


def _load_live(limit: int = 100000) -> tuple:
    """Load observations + full outcome records from the live store, read-only.
    Returns ([], {}) when the DB is unavailable (non-fatal)."""
    from db.engine import get_neon_conn, get_sqlite_conn
    from db.hsf_observations import _loads, load_recent_observations
    obs = load_recent_observations(limit=limit) or []
    outcomes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    conn = get_neon_conn()
    if conn is None:
        try:
            conn = get_sqlite_conn()
        except Exception:
            conn = None
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT observation_id, record FROM hsf_observation_outcomes")
            for r in cur.fetchall() or []:
                vals = list(r.values()) if isinstance(r, dict) else r
                outcomes[str(vals[0])].append(_loads(vals[1]))
            cur.close()
        except Exception:
            pass
    return obs, outcomes


def render_text(report: Dict[str, Any]) -> str:
    lines = ["HSF Research Cohort Integrity Audit",
             "-----------------------------------",
             f"Total observations: {report['total_observations']}  "
             f"duplicate_ids={report['duplicate_observation_ids']}  "
             f"conflicting_duplicates={report['conflicting_duplicates']}"]
    for c in _COHORTS:
        s = report["cohorts"].get(c, {})
        lines.append(
            f"{c}: obs={s.get('observations', 0)} symbols={s.get('distinct_symbols', 0)} "
            f"runs={s.get('distinct_runs', 0)} obs/run={s.get('observations_per_run', 0)} "
            f"matured={s.get('matured', 0)} unmatured={s.get('unmatured', 0)} "
            f"missing_fields={s.get('missing_field_observations', 0)} "
            f"invalid={s.get('invalid_values', 0)} "
            f"pit_violations={s.get('point_in_time_violations', 0)} "
            f"legacy={s.get('legacy_inferred', 0)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only research cohort integrity audit")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()

    obs, outcomes = _load_live()
    report = audit_cohorts(obs, outcomes)
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cohort_audit.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
