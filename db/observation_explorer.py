"""Parameterized, bounded PostgreSQL queries for Admin Observation Explorer."""
from __future__ import annotations

import json
import time
from typing import Any, Mapping

from analytics.observation_explorer import MAX_EXPORT_ROWS, clamp_page
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS
from db.engine import get_neon_conn

COHORT_EXPR = "COALESCE(o.record->>'research_cohort', o.record->'market_context'->>'research_cohort')"
SCAN_EXPR = "COALESCE(o.record->'market_context'->>'scan_id', o.record->'research_metadata'->>'scan_id')"
SESSION_EXPR = "COALESCE(o.record->>'session', o.record->'research_metadata'->>'session')"
DESIGN_EXPR = "o.record->'market_context'->>'control_design'"
RANK_EXPR = "o.record->'research_metadata'->>'rank_at_observation'"
SCORE_EXPR = """CASE
    WHEN jsonb_typeof(o.record->'hsf_score') = 'number' THEN (o.record->>'hsf_score')::double precision
    WHEN jsonb_typeof(o.record->'models'->'hsf_score') = 'number'
      THEN (o.record->'models'->>'hsf_score')::double precision
    WHEN jsonb_typeof(o.record->'market_context'->'hsf_score') = 'number'
      THEN (o.record->'market_context'->>'hsf_score')::double precision
    ELSE NULL END"""
RETURN_EXPR = "COALESCE(NULLIF(oc.record->>'directional_return','')::double precision, NULLIF(oc.record->>'raw_return','')::double precision)"
MFE_EXPR = "NULLIF(oc.record->>'mfe','')::double precision"
MAE_EXPR = "NULLIF(oc.record->>'mae','')::double precision"


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def _execute(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    started = time.perf_counter()
    conn = get_neon_conn()
    if conn is None:
        return {"rows": [], "error": "database_unavailable", "duration_ms": None}
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = list(cur.fetchall() or [])
        cur.close()
        return {"rows": rows, "error": None, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"rows": [], "error": type(exc).__name__,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_where(filters: Mapping[str, Any], *, include_status: bool = True) -> tuple[str, tuple[Any, ...]]:
    """Build only static SQL fragments; every selected value stays in params."""
    clauses: list[str] = []
    params: list[Any] = []
    dataset = str(filters.get("dataset") or "scanner")
    if dataset == "stair_stepper":
        clauses.append("o.context = %s")
        params.append("day_trader:stair_stepper")
    else:
        clauses.append("o.context LIKE %s")
        params.append("scheduled:%")

    if filters.get("start"):
        clauses.append("o.timestamp >= %s")
        params.append(filters["start"])
    if filters.get("end"):
        clauses.append("o.timestamp < %s")
        params.append(filters["end"])

    cohorts = [str(value).upper() for value in filters.get("cohorts") or []]
    if cohorts:
        clauses.append(f"{COHORT_EXPR} = ANY(%s)")
        params.append(cohorts)
    symbols = [str(value).upper() for value in filters.get("symbols") or []]
    if symbols:
        clauses.append("o.symbol = ANY(%s)")
        params.append(symbols)
    if filters.get("signal"):
        clauses.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements(COALESCE(o.record->'scanners','[]'::jsonb)) sig "
            "WHERE sig->>'name' = %s)"
        )
        params.append(str(filters["signal"]))
    if filters.get("scan_id"):
        clauses.append(f"{SCAN_EXPR} = %s")
        params.append(str(filters["scan_id"]))
    if filters.get("session"):
        clauses.append(f"UPPER({SESSION_EXPR}) = %s")
        params.append(str(filters["session"]).upper())
    if filters.get("control_design"):
        clauses.append(f"{DESIGN_EXPR} = %s")
        params.append(str(filters["control_design"]))
    if filters.get("score_min") is not None:
        clauses.append(f"({SCORE_EXPR}) >= %s")
        params.append(float(filters["score_min"]))
    if filters.get("score_max") is not None:
        clauses.append(f"({SCORE_EXPR}) <= %s")
        params.append(float(filters["score_max"]))

    compatible_design = filters.get("compatible_control_design")
    if compatible_design and dataset == "scanner":
        clauses.append(f"({COHORT_EXPR} <> %s OR {DESIGN_EXPR} = %s)")
        params.extend((CONTROL, str(compatible_design)))

    if include_status and filters.get("status") not in (None, "", "All"):
        status = str(filters["status"]).upper()
        horizon_clause = " AND sx.horizon = %s" if filters.get("horizon") else ""
        exists = (
            "EXISTS (SELECT 1 FROM hsf_observation_outcomes sx "
            "WHERE sx.observation_id = o.observation_id"
            f"{horizon_clause} AND sx.record->>'data_status' = 'MATURED')"
        )
        usable = (
            "EXISTS (SELECT 1 FROM hsf_observation_outcomes sx "
            "WHERE sx.observation_id = o.observation_id"
            f"{horizon_clause} AND sx.record->>'data_status' = 'MATURED' "
            "AND (sx.record->>'directional_return' IS NOT NULL OR sx.record->>'raw_return' IS NOT NULL))"
        )
        excluded = (
            "EXISTS (SELECT 1 FROM hsf_observation_outcomes sx "
            "WHERE sx.observation_id = o.observation_id"
            f"{horizon_clause} AND UPPER(COALESCE(sx.record->>'data_status','')) "
            "IN ('EXCLUDED','UNAVAILABLE'))"
        )
        if status in {"PENDING", "MATURED", "USABLE", "EXCLUDED"}:
            selected = {
                "PENDING": f"NOT ({exists}) AND NOT ({excluded})",
                "MATURED": exists,
                "USABLE": usable,
                "EXCLUDED": excluded,
            }[status]
            clauses.append(selected)
            if filters.get("horizon"):
                occurrences = selected.count("%s")
                params.extend(str(filters["horizon"]) for _ in range(occurrences))
    return " AND ".join(clauses) if clauses else "TRUE", tuple(params)


def _base_cte(filters: Mapping[str, Any], *, include_status: bool = True) -> tuple[str, tuple[Any, ...]]:
    where, where_params = build_where(filters, include_status=include_status)
    horizon = filters.get("horizon")
    params: list[Any] = []
    if horizon:
        join = "LEFT JOIN hsf_observation_outcomes oc ON oc.observation_id = o.observation_id AND oc.horizon = %s"
        params.append(str(horizon))
    else:
        join = (
            "LEFT JOIN LATERAL (SELECT horizon, record FROM hsf_observation_outcomes x "
            "WHERE x.observation_id = o.observation_id ORDER BY horizon LIMIT 1) oc ON TRUE"
        )
    sql = f"""
        SELECT o.observation_id, o.symbol, o.timestamp, o.context, o.record,
               {COHORT_EXPR} AS cohort, {SCAN_EXPR} AS scan_id, {SESSION_EXPR} AS session,
               {DESIGN_EXPR} AS control_design, {RANK_EXPR} AS rank_at_observation,
               {SCORE_EXPR} AS hsf_score, oc.horizon, oc.record AS outcome_record
        FROM hsf_observations o
        {join}
        WHERE {where}
    """
    return sql, tuple(params) + where_params


def query_options(filters: Mapping[str, Any]) -> dict[str, Any]:
    base_filters = {key: filters.get(key) for key in (
        "dataset", "start", "end", "compatible_control_design",
    )}
    where, params = build_where(base_filters, include_status=False)
    sql = f"""
        WITH base AS (SELECT o.* FROM hsf_observations o WHERE {where})
        SELECT
          ARRAY(SELECT value FROM (
            SELECT DISTINCT sig->>'name' AS value FROM base o
            CROSS JOIN LATERAL jsonb_array_elements(COALESCE(o.record->'scanners','[]'::jsonb)) sig
            WHERE sig->>'name' IS NOT NULL ORDER BY value LIMIT 200
          ) listed) AS signals,
          ARRAY(SELECT DISTINCT {COHORT_EXPR} FROM base o
                WHERE {COHORT_EXPR} IS NOT NULL ORDER BY {COHORT_EXPR}) AS cohorts,
          ARRAY(SELECT DISTINCT {DESIGN_EXPR} FROM base o
                WHERE {DESIGN_EXPR} IS NOT NULL ORDER BY {DESIGN_EXPR}) AS control_designs,
          ARRAY(SELECT DISTINCT oc.horizon FROM hsf_observation_outcomes oc
                JOIN base o ON o.observation_id=oc.observation_id ORDER BY oc.horizon) AS horizons,
          ARRAY(SELECT DISTINCT UPPER({SESSION_EXPR}) FROM base o
                WHERE {SESSION_EXPR} IS NOT NULL ORDER BY UPPER({SESSION_EXPR})) AS sessions,
          (SELECT COUNT({SCORE_EXPR}) FROM base o) AS score_count
    """
    query = _execute(sql, params)
    row = query["rows"][0] if query["rows"] else {}
    keys = ("signals", "cohorts", "control_designs", "horizons", "sessions")
    result = {
        key: [str(value) for value in (_row_value(row, key, index) or []) if value]
        for index, key in enumerate(keys)
    }
    result["score_available"] = bool(int(_row_value(row, "score_count", 5) or 0))
    result["errors"] = {"options": query["error"]} if query["error"] else {}
    result["duration_ms"] = query["duration_ms"]
    return result


def query_summary(filters: Mapping[str, Any]) -> dict[str, Any]:
    base, params = _base_cte(filters)
    sql = f"""
        WITH base AS ({base})
        SELECT COUNT(*) AS observations,
               COUNT(*) FILTER (WHERE outcome_record->>'data_status' = 'MATURED') AS matured,
               COUNT(*) FILTER (WHERE outcome_record->>'data_status' = 'MATURED' AND
                   COALESCE(outcome_record->>'directional_return', outcome_record->>'raw_return') IS NOT NULL) AS usable,
               COUNT(*) FILTER (WHERE outcome_record IS NULL OR
                   UPPER(COALESCE(outcome_record->>'data_status','PENDING')) = 'PENDING') AS pending,
               COUNT(*) FILTER (WHERE UPPER(COALESCE(outcome_record->>'data_status',''))
                   IN ('EXCLUDED','UNAVAILABLE')) AS excluded,
               COUNT(*) FILTER (WHERE cohort = %s) AS candidate,
               COUNT(*) FILTER (WHERE cohort = %s) AS near_miss,
               COUNT(*) FILTER (WHERE cohort = %s) AS control,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY hsf_score) AS median_score,
               AVG(CASE WHEN outcome_record->>'data_status' = 'MATURED' THEN
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) END) AS mean_return,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY
                   CASE WHEN outcome_record->>'data_status' = 'MATURED' THEN
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) END) AS median_return,
               percentile_cont(0.25) WITHIN GROUP (ORDER BY
                   CASE WHEN outcome_record->>'data_status' = 'MATURED' THEN
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) END) AS p25_return,
               percentile_cont(0.75) WITHIN GROUP (ORDER BY
                   CASE WHEN outcome_record->>'data_status' = 'MATURED' THEN
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) END) AS p75_return,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY
                   CASE WHEN outcome_record->>'data_status' = 'MATURED'
                   THEN NULLIF(outcome_record->>'mfe','')::double precision END) AS median_mfe,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY
                   CASE WHEN outcome_record->>'data_status' = 'MATURED'
                   THEN NULLIF(outcome_record->>'mae','')::double precision END) AS median_mae,
               AVG(CASE WHEN outcome_record->>'data_status' = 'MATURED' AND
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) > 0 THEN 1.0
                   WHEN outcome_record->>'data_status' = 'MATURED' THEN 0.0 END) AS positive_rate,
               MIN(timestamp) AS oldest_observation, MAX(timestamp) AS latest_observation,
               MAX(NULLIF(outcome_record->>'evaluation_time','')::timestamptz) AS latest_matured
        FROM base
    """
    result = _execute(sql, params + (CANDIDATE, NEAR_MISS, CONTROL))
    row = result["rows"][0] if result["rows"] else {}
    keys = ("observations", "matured", "usable", "pending", "excluded",
            "candidate", "near_miss", "control",
            "median_score", "mean_return", "median_return", "p25_return", "p75_return",
            "median_mfe", "median_mae", "positive_rate",
            "oldest_observation", "latest_observation", "latest_matured")
    return {**{key: _row_value(row, key, index) for index, key in enumerate(keys)},
            "error": result["error"], "duration_ms": result["duration_ms"]}


def query_page(filters: Mapping[str, Any], *, page: int = 1, page_size: int = 50) -> dict[str, Any]:
    page, page_size = clamp_page(page, page_size)
    base, params = _base_cte(filters)
    sql = f"""
        WITH base AS ({base})
        SELECT *, COUNT(*) OVER() AS total_count
        FROM base ORDER BY timestamp DESC, observation_id DESC LIMIT %s OFFSET %s
    """
    result = _execute(sql, params + (page_size, (page - 1) * page_size))
    rows = []
    total = 0
    for row in result["rows"]:
        total = int(_row_value(row, "total_count", 13) or total)
        record = _loads(_row_value(row, "record", 4))
        record.setdefault("observation_id", _row_value(row, "observation_id", 0))
        record.setdefault("symbol", _row_value(row, "symbol", 1))
        record.setdefault("timestamp", _row_value(row, "timestamp", 2))
        record.setdefault("context", _row_value(row, "context", 3))
        if _row_value(row, "cohort", 5):
            record.setdefault("research_cohort", _row_value(row, "cohort", 5))
        market_context = record.setdefault("market_context", {})
        if isinstance(market_context, dict):
            market_context.setdefault("scan_id", _row_value(row, "scan_id", 6))
            market_context.setdefault("control_design", _row_value(row, "control_design", 8))
        outcome = _loads(_row_value(row, "outcome_record", 12))
        horizon = _row_value(row, "horizon", 11)
        if horizon and outcome:
            record.setdefault("outcomes", {})[str(horizon)] = outcome
        rows.append(record)
    return {"rows": rows, "total": total, "page": page, "page_size": page_size,
            "error": result["error"], "duration_ms": result["duration_ms"]}


def query_timeline(filters: Mapping[str, Any], *, group_by: str = "total") -> dict[str, Any]:
    where, params = build_where(filters)
    if group_by == "cohort":
        group_expr = COHORT_EXPR
    elif group_by == "status":
        group_expr = (
            "CASE WHEN EXISTS (SELECT 1 FROM hsf_observation_outcomes sx WHERE sx.observation_id=o.observation_id "
            "AND sx.record->>'data_status'='MATURED') THEN 'MATURED' ELSE 'PENDING' END"
        )
    elif group_by == "signal":
        sql = f"""
            SELECT date_trunc('hour', o.timestamp) AS bucket, sig->>'name' AS series, COUNT(*) AS count
            FROM hsf_observations o
            CROSS JOIN LATERAL jsonb_array_elements(COALESCE(o.record->'scanners','[]'::jsonb)) sig
            WHERE {where} GROUP BY bucket, series ORDER BY bucket, series
        """
        result = _execute(sql, params)
        return _timeline_result(result)
    else:
        group_expr = "'Total'"
    sql = f"""
        SELECT date_trunc('hour', o.timestamp) AS bucket, {group_expr} AS series, COUNT(*) AS count
        FROM hsf_observations o WHERE {where} GROUP BY bucket, series ORDER BY bucket, series
    """
    return _timeline_result(_execute(sql, params))


def _timeline_result(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = [{"bucket": _row_value(row, "bucket", 0), "series": _row_value(row, "series", 1),
             "count": int(_row_value(row, "count", 2) or 0)} for row in result["rows"]]
    return {"rows": rows, "error": result["error"], "duration_ms": result["duration_ms"]}


def query_cohort_comparison(filters: Mapping[str, Any]) -> dict[str, Any]:
    base_filters = dict(filters)
    base_filters["cohorts"] = []
    base, params = _base_cte(base_filters)
    sql = f"""
        WITH base AS ({base}), usable AS (
          SELECT *, COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                             NULLIF(outcome_record->>'raw_return','')::double precision) AS ret,
                    NULLIF(outcome_record->>'mfe','')::double precision AS mfe,
                    NULLIF(outcome_record->>'mae','')::double precision AS mae
          FROM base
        )
        SELECT cohort, COUNT(*) AS n,
               COUNT(*) FILTER (WHERE outcome_record->>'data_status'='MATURED') AS matured,
               COUNT(ret) AS usable,
               AVG(CASE WHEN ret > 0 THEN 1.0 WHEN ret IS NOT NULL THEN 0.0 END) AS positive_rate,
               AVG(ret) AS mean_return, percentile_cont(0.5) WITHIN GROUP (ORDER BY ret) AS median_return,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY mfe) AS median_mfe,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY mae) AS median_mae
        FROM usable WHERE cohort = ANY(%s) GROUP BY cohort ORDER BY cohort
    """
    result = _execute(sql, params + (list((CANDIDATE, NEAR_MISS, CONTROL)),))
    keys = ("cohort", "n", "matured", "usable", "positive_rate", "mean_return",
            "median_return", "median_mfe", "median_mae")
    rows = [{key: _row_value(row, key, index) for index, key in enumerate(keys)} for row in result["rows"]]
    return {"rows": rows, "error": result["error"], "duration_ms": result["duration_ms"]}


def query_integrity(filters: Mapping[str, Any]) -> dict[str, Any]:
    base_filters = {key: filters.get(key) for key in (
        "dataset", "start", "end", "scan_id", "session", "compatible_control_design",
    )}
    where, params = build_where(base_filters, include_status=False)
    sql = f"""
        WITH memberships AS (
          SELECT {SCAN_EXPR} AS scan_id, o.symbol, {COHORT_EXPR} AS cohort
          FROM hsf_observations o WHERE {where}
          GROUP BY scan_id, o.symbol, cohort
        ), overlaps AS (
          SELECT scan_id, symbol,
                 BOOL_OR(cohort=%s) AS candidate,
                 BOOL_OR(cohort=%s) AS near_miss,
                 BOOL_OR(cohort=%s) AS control
          FROM memberships WHERE scan_id IS NOT NULL GROUP BY scan_id, symbol
        )
        SELECT COUNT(*) FILTER (WHERE candidate AND near_miss) AS candidate_near_miss,
               COUNT(*) FILTER (WHERE candidate AND control) AS candidate_control,
               COUNT(*) FILTER (WHERE near_miss AND control) AS near_miss_control
        FROM overlaps
    """
    result = _execute(sql, params + (CANDIDATE, NEAR_MISS, CONTROL))
    row = result["rows"][0] if result["rows"] else {}
    counts = {key: int(_row_value(row, key, index) or 0) for index, key in enumerate(
        ("candidate_near_miss", "candidate_control", "near_miss_control")
    )}
    status = "UNKNOWN" if result["error"] else "PASS" if not any(counts.values()) else "OVERLAP_DETECTED"
    return {**counts, "status": status,
            "error": result["error"], "duration_ms": result["duration_ms"]}


def query_histogram(filters: Mapping[str, Any]) -> dict[str, Any]:
    base, params = _base_cte(filters)
    sql = f"""
        WITH base AS ({base}), returns AS (
          SELECT COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                          NULLIF(outcome_record->>'raw_return','')::double precision) AS value
          FROM base WHERE outcome_record->>'data_status'='MATURED'
        )
        SELECT FLOOR(value * 200.0) / 200.0 AS bucket, COUNT(*) AS count
        FROM returns WHERE value IS NOT NULL GROUP BY bucket ORDER BY bucket
    """
    result = _execute(sql, params)
    rows = [{"bucket": _row_value(row, "bucket", 0), "count": int(_row_value(row, "count", 1) or 0)}
            for row in result["rows"]]
    return {"rows": rows, "error": result["error"], "duration_ms": result["duration_ms"]}


def query_score_analysis(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Return score buckets and matured outcome summaries without row-level loading."""
    base, params = _base_cte(filters)
    sql = f"""
        WITH base AS ({base}), scored AS (
          SELECT hsf_score,
                 CASE WHEN hsf_score < 50 THEN '<50'
                      WHEN hsf_score < 60 THEN '50-59'
                      WHEN hsf_score < 70 THEN '60-69'
                      WHEN hsf_score < 80 THEN '70-79'
                      WHEN hsf_score < 90 THEN '80-89'
                      ELSE '90-100' END AS bucket,
                 CASE WHEN outcome_record->>'data_status'='MATURED' THEN
                   COALESCE(NULLIF(outcome_record->>'directional_return','')::double precision,
                            NULLIF(outcome_record->>'raw_return','')::double precision) END AS ret
          FROM base WHERE hsf_score IS NOT NULL
        )
        SELECT bucket, COUNT(*) AS observations, COUNT(ret) AS matured,
               AVG(ret) AS mean_return,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY ret) AS median_return,
               AVG(CASE WHEN ret > 0 THEN 1.0 WHEN ret IS NOT NULL THEN 0.0 END) AS positive_rate,
               MIN(hsf_score) AS score_min, MAX(hsf_score) AS score_max
        FROM scored GROUP BY bucket
        ORDER BY MIN(hsf_score)
    """
    result = _execute(sql, params)
    keys = ("bucket", "observations", "matured", "mean_return", "median_return",
            "positive_rate", "score_min", "score_max")
    rows = [{key: _row_value(row, key, index) for index, key in enumerate(keys)} for row in result["rows"]]
    return {"rows": rows, "error": result["error"], "duration_ms": result["duration_ms"]}


def query_detail(observation_id: str) -> dict[str, Any]:
    sql = """
        SELECT o.record, oc.horizon, oc.record AS outcome_record
        FROM hsf_observations o LEFT JOIN hsf_observation_outcomes oc
          ON oc.observation_id=o.observation_id
        WHERE o.observation_id=%s ORDER BY oc.horizon
    """
    result = _execute(sql, (str(observation_id),))
    if not result["rows"]:
        return {"observation": None, "error": result["error"], "duration_ms": result["duration_ms"]}
    observation = _loads(_row_value(result["rows"][0], "record", 0))
    observation.setdefault("observation_id", str(observation_id))
    for row in result["rows"]:
        horizon = _row_value(row, "horizon", 1)
        outcome = _loads(_row_value(row, "outcome_record", 2))
        if horizon and outcome:
            observation.setdefault("outcomes", {})[str(horizon)] = outcome
    return {"observation": observation, "error": result["error"], "duration_ms": result["duration_ms"]}


def query_export(filters: Mapping[str, Any], *, limit: int = MAX_EXPORT_ROWS) -> dict[str, Any]:
    safe_limit = max(1, min(MAX_EXPORT_ROWS, int(limit)))
    base, params = _base_cte(filters)
    sql = f"WITH base AS ({base}) SELECT * FROM base ORDER BY timestamp DESC, observation_id DESC LIMIT %s"
    result = _execute(sql, params + (safe_limit + 1,))
    rows = []
    for row in result["rows"][:safe_limit]:
        record = _loads(_row_value(row, "record", 4))
        outcome = _loads(_row_value(row, "outcome_record", 12))
        rows.append({
            "observation_id": _row_value(row, "observation_id", 0),
            "timestamp": _row_value(row, "timestamp", 2),
            "symbol": _row_value(row, "symbol", 1),
            "cohort": _row_value(row, "cohort", 5),
            "signals": ",".join(str(item.get("name")) for item in record.get("scanners") or [] if item.get("name")),
            "hsf_score": _row_value(row, "hsf_score", 10),
            "rank": _row_value(row, "rank_at_observation", 9),
            "scan_id": _row_value(row, "scan_id", 6),
            "control_design": _row_value(row, "control_design", 8),
            "horizon": _row_value(row, "horizon", 11),
            "forward_return": outcome.get("directional_return") if outcome.get("directional_return") is not None else outcome.get("raw_return"),
            "mfe": outcome.get("mfe"), "mae": outcome.get("mae"),
            "matured_at": outcome.get("evaluation_time"),
        })
    return {"rows": rows, "truncated": len(result["rows"]) > safe_limit,
            "error": result["error"], "duration_ms": result["duration_ms"]}
