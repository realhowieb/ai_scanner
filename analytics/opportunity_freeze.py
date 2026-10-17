"""Deliberate, user-independent freeze of HSF opportunities for calibration.

The scheduled cron pipeline owns the freeze, so signal-time HSF opportunities are
collected whether or not anyone opens the UI. They are frozen from the Scanner's
ranking of the latest saved full-market run (consolidate_scanner_results, the
same scores /v1/scans/latest shows), keyed on that run's created_at, so the
Track record buckets use the scores people actually saw. Idempotent per
(snapshot, ticker) via ON CONFLICT DO NOTHING.

Earlier rows came from the Market Brief's top-5 lists (build_opportunities),
which scored lower than the Scanner for the same name; readers drop those once a
snapshot has Scanner rows (db.signal_outcomes._SCANNER_SUPERSEDES_BRIEF), and
scripts/backfill_scanner_opportunities.py re-freezes past runs the same way.

Read-safe: returns 0 and never raises when data/DB is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Ranked Scanner setups frozen per run: what a Basic plan sees in the Scanner
# (config.TIERS_CONFIG basic max_results). Deeper than the Brief's top 5, so the
# lower score buckets get evidence too.
FREEZE_TOP_N = 25


def market_runs(limit: int = 60) -> List[Dict[str, Any]]:
    """Scheduled full-market runs, newest first (parsed created_at)."""
    from db.runs import list_runs
    from ui.market_scans import MARKET_USER
    from ui.market_scans import market_runs as _market_runs

    return _market_runs(list_runs(limit=limit, include_snapshots=True, username=MARKET_USER) or [])


def scanner_opportunities(run_id: int, top_n: int = FREEZE_TOP_N) -> List[Dict[str, Any]]:
    """The Scanner's top-n ranked HSF opportunities of one saved run."""
    from ui.market_scans import _run_df_uncached, top_setups

    return top_setups(_run_df_uncached(int(run_id)), n=top_n)


def source_scan_id(df: Any) -> Optional[str]:
    """The run's own scan id (SourceScanId = scan start time), when the run has
    exactly one. Lets research link each frozen row to its own scan record."""
    if df is None or getattr(df, "empty", True) or "SourceScanId" not in getattr(df, "columns", ()):
        return None
    ids = {str(v).strip() for v in df["SourceScanId"].dropna() if str(v).strip()}
    return ids.pop() if len(ids) == 1 else None


def _ticker_col(df: Any) -> Optional[str]:
    return next((c for c in ("Ticker", "Symbol") if c in df.columns), None)


def ai_confidence_by_ticker(df: Any, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """AI Confidence for the frozen tickers only, scored from the same saved scan
    row the Scanner ranked (no new data). Returns {ticker: {value, model_version}}
    for tickers the model could score; missing model or feature columns leave the
    ticker out, so nothing is filled in. Never touches the saved run."""
    col = _ticker_col(df) if df is not None and not getattr(df, "empty", True) else None
    want = {str(t).upper() for t in tickers if t}
    if col is None or not want:
        return {}
    from scan.ai_confidence import CONFIDENCE_COL, load_ai_confidence_bundle, score_ai_confidence

    sub = df[df[col].astype(str).str.strip().str.upper().isin(want)]
    if sub.empty:
        return {}
    scored = score_ai_confidence(sub.drop(columns=[CONFIDENCE_COL], errors="ignore").reset_index(drop=True))
    if CONFIDENCE_COL not in scored.columns:
        return {}
    _model, meta, _warn = load_ai_confidence_bundle()
    version = (meta or {}).get("model_version")
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in scored.iterrows():
        t = str(row.get(col) or "").strip().upper()
        try:
            v = float(row.get(CONFIDENCE_COL))
        except (TypeError, ValueError):
            continue
        if t and t not in out and v == v:
            out[t] = {"value": v, "model_version": str(version) if version else None}
    return out


def prebreakout_version(opp: Dict[str, Any]) -> Optional[str]:
    """The PreBreakout model version the scan's own inference recorded."""
    pb = (opp.get("models") or {}).get("prebreakout")
    if not isinstance(pb, dict) or pb.get("status") != "captured":
        return None
    art = pb.get("loaded_artifact")
    return str(art["version"]) if isinstance(art, dict) and art.get("version") else None


def freeze_run(run: Dict[str, Any], top_n: int = FREEZE_TOP_N, *, score_confidence: bool = True) -> int:
    """Freeze one saved run's Scanner opportunities at the run's created_at.

    Stamps each row with the run's scan id and the served PreBreakout version,
    and (live freezes only) the AI Confidence of that scan row. Backfills of old
    runs pass score_confidence=False: today's model must not be frozen onto a
    past observation."""
    from db.signal_outcomes import SCANNER_SIGNAL_TYPE, freeze_opportunities
    from ui.market_scans import _run_df_uncached, top_setups

    ts = run.get("created_at")
    if ts is None or run.get("id") is None:
        return 0
    df = _run_df_uncached(int(run["id"]))
    opps = top_setups(df, n=top_n)
    if not opps:
        return 0
    sid = source_scan_id(df)
    conf: Dict[str, Dict[str, Any]] = {}
    if score_confidence:
        try:
            conf = ai_confidence_by_ticker(df, [o.get("ticker") for o in opps])
        except Exception as e:  # noqa: BLE001 - research extra; the freeze still happens
            print(f"[opportunity_freeze] AI Confidence unavailable: {type(e).__name__}")
    for o in opps:
        o["source_scan_id"] = sid
        o["prebreakout_model_version"] = prebreakout_version(o)
        c = conf.get(str(o.get("ticker") or "").upper())
        o["ai_confidence"] = c["value"] if c else None
        o["ai_confidence_model_version"] = c["model_version"] if c else None
    return freeze_opportunities(ts, opps, signal_type=SCANNER_SIGNAL_TYPE)


def freeze_latest_opportunities() -> int:
    """Freeze the latest full-market run's Scanner opportunities.

    Returns the number of freeze attempts (existing rows are no-ops via ON
    CONFLICT). Best-effort; 0 on any failure.
    """
    try:
        runs = market_runs(limit=5)
        return freeze_run(runs[0]) if runs else 0
    except Exception:
        return 0

