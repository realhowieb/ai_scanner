"""Scan history and historical research for API clients (P1-67).

Pro features on the web (pricing: "Scan history & historical research"):
  * GET /v1/runs, /v1/runs/{id}: your own saved scans (web: Scanner > Scan
    History, db.runs.list_runs scoped to the user). db.runs.load_run_results
    does no ownership check, so get_run() checks it here first.
  * GET /v1/track-record, /v1/track-record/daily: saved scan picks vs SPY
    (web: ui.track_record, db.track_record), descriptive only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.store import DatabaseUnavailable

RANKINGS = {"breakout": "BreakoutScore", "prebreakout": "PreBreakoutProb"}
HORIZONS = (1, 3, 5, 10, 20)
MIN_SAMPLE_SIZE = 25   # ui.track_record: fewer matured picks reads "still building"
DISCLAIMER = ("Historical research: backtested on saved scan snapshots. These figures are descriptive and "
              "are not evidence that HSF signals work. Past performance is not indicative of future results.")


def _run_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": int(r["id"]), "name": r.get("name"), "label": r.get("label"),
            "row_count": r.get("row_count"), "duration_s": r.get("duration_sec"),
            "is_snapshot": bool(r.get("is_snapshot")), "created_at": r.get("created_at")}


def saved_runs(username: str, limit: int, include_snapshots: bool) -> List[Dict[str, Any]]:
    from db.runs import list_runs as _list_runs

    try:
        runs = _list_runs(limit=int(limit), include_snapshots=include_snapshots, username=username) or []
    except RuntimeError as e:
        raise DatabaseUnavailable("database unavailable") from e
    return [_run_row(r) for r in runs if str(r.get("username") or "").strip().lower() == username]


def _owned_run(username: str, run_id: int) -> Optional[Dict[str, Any]]:
    from api.store import _conn

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, label, username, row_count, duration_sec, is_snapshot, created_at "
                    "FROM runs WHERE id = %s AND lower(username) = %s", (int(run_id), username))
        row = cur.fetchone()
        cols = [d.name for d in cur.description]
        cur.close()
        conn.rollback()
    finally:
        conn.close()
    if row is None:
        return None
    return dict(row) if isinstance(row, dict) else dict(zip(cols, row))


def get_run(username: str, run_id: int, *, early_breakout: bool, max_results: int) -> Optional[Dict[str, Any]]:
    """One of your saved scans with its rows (None when missing or not yours)."""
    from api.scans import _scan_row
    from api.today import run_df
    from ui.entitlement_view import redact_prebreakout_rows
    from ui.market_scans import top_setups

    meta = _owned_run(username, run_id)
    if meta is None:
        return None
    df = run_df(int(run_id))
    opps = redact_prebreakout_rows(top_setups(df, n=100_000), allowed=early_breakout) if df is not None else []
    return {**_run_row(meta), "total": len(opps), "max_results": max_results, "limited": len(opps) > max_results,
            "setups": [_scan_row(o) for o in opps[:max_results]]}


# Track-record rows are computed once a day by the scheduler, so they're cached here.
TRACK_RECORD_TTL_S = 600
TRACK_RECORD_STALE_S = 6 * 3600


def track_record() -> Dict[str, Any]:
    from api.today import _cache, _cached

    out = _cached("track_record", _track_record, ttl_s=TRACK_RECORD_TTL_S, stale_s=TRACK_RECORD_STALE_S)
    if not out.get("summaries"):  # nothing (or a database blip): don't keep the empty answer
        _cache.clear_key("track_record")
    return out


def _track_record() -> Dict[str, Any]:
    from db.track_record import load_latest_track_records

    try:
        latest = {(int(r.get("horizon_days") or 0), r.get("ranking") or "breakout"): r
                  for r in load_latest_track_records() or []}
    except Exception:
        latest = {}
    rows: List[Dict[str, Any]] = []
    for horizon in HORIZONS:
        for ranking, label in RANKINGS.items():
            row = latest.get((horizon, ranking))
            if not row:
                continue
            row = dict(row)
            rows.append({"ranking": row.get("ranking") or ranking, "ranking_label": label,
                         "horizon_days": row.get("horizon_days", horizon),
                         "avg_excess_return": row.get("avg_return"), "median_excess_return": row.get("median_return"),
                         "win_rate": row.get("win_rate"), "sample_size": row.get("sample_size"),
                         "runs_used": row.get("runs_used"), "top_n": row.get("top_n"),
                         "benchmark": row.get("benchmark") or "SPY", "computed_at": row.get("computed_at"),
                         "sufficient": int(row.get("sample_size") or 0) >= MIN_SAMPLE_SIZE})
    sufficient = [r for r in rows if r.get("sufficient") and r.get("avg_excess_return") is not None]
    best = max(sufficient, key=lambda r: r.get("avg_excess_return") or -999) if sufficient else None
    summary = {"best_ranking": best.get("ranking_label") if best else None,
               "best_horizon_days": best.get("horizon_days") if best else None,
               "best_avg_excess_return": best.get("avg_excess_return") if best else None,
               "ready_horizons": len(sufficient), "total_horizons": len(rows),
               "read": (f"Best matured slice is {best.get('ranking_label')} at {best.get('horizon_days')}d."
                        if best else "Still building enough matured samples to call out a best slice.")}
    return {"disclaimer": DISCLAIMER, "min_sample_size": MIN_SAMPLE_SIZE, "summary": summary, "summaries": rows}


def track_record_daily(ranking: str, horizon: int, days: int) -> List[Dict[str, Any]]:
    from api.today import _cache, _cached

    key = ("track_record_daily", ranking, int(horizon), int(days))
    out = _cached(key, lambda: _track_record_daily(ranking, horizon, days),
                  ttl_s=TRACK_RECORD_TTL_S, stale_s=TRACK_RECORD_STALE_S)
    if not out:
        _cache.clear_key(key)
    return out


def _track_record_daily(ranking: str, horizon: int, days: int) -> List[Dict[str, Any]]:
    from db.track_record import load_daily_excess

    out = []
    for item in load_daily_excess(ranking, int(horizon), days=int(days)) or []:
        day, excess = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item.get("day"), item.get("avg_excess"))
        out.append({"day": day, "avg_excess_return": excess})
    return out
