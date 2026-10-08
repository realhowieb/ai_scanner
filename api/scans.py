"""GET /v1/scans/latest and GET /v1/stocks/{ticker} (P1-59 step 5).

Built from the same saved market scans and the same helpers as the Streamlit
Scanner and Stock Intelligence pages, so the app and the web show one answer.
Premium model output is redacted below Premium, as on the web.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional

from api.today import TTLCache, _cached, _iso, _num, market_runs, run_df, scan_freshness

SCAN_FIELDS = ("ticker", "score", "primary_setup", "status", "n_signals")
SCAN_NUMBERS = ("last", "chg_pct", "gap_pct", "rvol", "breakout_score", "prob")
CALIBRATION_TTL_S = 1800  # matured outcomes change once a day; same as the web
BAR_LIMIT = 120
# Stock pages get their own cache so a client paging through many tickers can't
# push the scan runs out of the shared one.
stock_cache = TTLCache(max_entries=128)


def json_safe(value: Any) -> Any:
    """NaN/inf -> None, datetimes -> ISO strings, recursively (strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
        return _iso(value)            # datetimes always with a timezone (UTC when stored without one)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            return None
    return value


def run_opportunities(run_id: int) -> List[Dict[str, Any]]:
    """Ranked HSF opportunities of one saved run (one per ticker), cached."""
    def load():
        from ui.market_scans import top_setups

        df = run_df(run_id)
        return top_setups(df, n=100_000) if df is not None else []

    return _cached(("opps", int(run_id)), load)


def max_results_for(tier: str) -> int:
    """Rows a plan sees in the Scanner (config.TIERS_CONFIG max_results)."""
    from config import TIERS_CONFIG

    cfg = TIERS_CONFIG.get(tier) or TIERS_CONFIG["basic"]
    return int(cfg.get("max_results") or 25)


def _scan_row(o: Dict[str, Any]) -> Dict[str, Any]:
    return {**{k: o.get(k) for k in SCAN_FIELDS},
            "signals": [str(s) for s in o.get("signals") or []],
            "fading": bool(o.get("fading")),
            **{k: _num(o.get(k)) for k in SCAN_NUMBERS}}


def latest_scan(entitlements: Dict[str, bool], tier: str, *, limit: int, offset: int,
                min_score: int = 0, signal: Optional[str] = None) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_rows

    cap = max_results_for(tier)
    runs = market_runs()
    if not runs:
        return {"scan_at": None, "total": 0, "max_results": cap, "limited": False, "setups": []}
    allowed = bool(entitlements.get("can_early_breakout"))
    opps = redact_prebreakout_rows(run_opportunities(int(runs[0]["id"])), allowed=allowed)
    if min_score:
        opps = [o for o in opps if int(o.get("score") or 0) >= int(min_score)]
    if signal:
        opps = [o for o in opps if signal in (o.get("signals") or [])]
    visible = opps[:cap]
    return {"scan_at": _iso(runs[0]["created_at"]), "total": len(opps), "max_results": cap,
            "limited": len(opps) > cap,
            "stale": scan_freshness(runs[0]["created_at"], dt.datetime.now(dt.timezone.utc))["stale"],
            "setups": [_scan_row(o) for o in visible[offset:offset + limit]]}


def _calibration_records() -> List[Dict[str, Any]]:
    def load():
        from analytics.hsf_calibration import build_calibration_dataset

        return build_calibration_dataset(days_back=180).get("records") or []

    return _cached("calibration", load, ttl_s=CALIBRATION_TTL_S)


def _ticker_rows(df: Any, ticker: str) -> List[Dict[str, Any]]:
    if df is None or "Ticker" not in getattr(df, "columns", []):
        return []
    sub = df[df["Ticker"].astype(str).str.strip().str.upper() == ticker]
    return sub.to_dict(orient="records")


def daily_bars(ticker: str, limit: int = BAR_LIMIT) -> Dict[str, Any]:
    """Daily OHLCV the scans already cached (no market-data call from the API)."""
    from db.prices import get_price_data_snapshot

    frames, _stale = get_price_data_snapshot([ticker], max_age_minutes=7 * 24 * 60)
    df = frames.get(ticker)
    if df is None or getattr(df, "empty", True):
        return {"bars": [], "as_of": None}
    cols = {str(c).lower().replace("adj close", "adj_close"): c for c in df.columns}
    pick = {k: cols.get(k) for k in ("open", "high", "low", "close", "volume")}
    if pick["close"] is None:
        return {"bars": [], "as_of": None}
    bars = []
    for idx, row in df.tail(limit).iterrows():
        day = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        bar = {"date": day, **{k: _num(row[c]) if c is not None else None for k, c in pick.items()}}
        if bar["close"] is not None:
            bars.append(bar)
    return {"bars": bars, "as_of": bars[-1]["date"] if bars else None}


def _stock_core(ticker: str) -> Dict[str, Any]:
    """The Stock Intelligence object for `ticker` before redaction (cached per ticker)."""
    def load():
        from db.signal_outcomes import fetch_ticker_opportunity_history
        from scheduler.morning_digest import _earnings_days_map
        from ui.stock_intelligence import build_stock_intelligence

        runs = market_runs()
        run = runs[0] if runs else None
        current_opp, rows = None, []
        if run is not None:
            current_opp = next((o for o in run_opportunities(int(run["id"])) if o["ticker"] == ticker), None)
            rows = _ticker_rows(run_df(int(run["id"])), ticker)
        try:
            history = list(fetch_ticker_opportunity_history(ticker) or [])
        except Exception:
            history = []
        try:
            calibration = _calibration_records()
        except Exception:
            calibration = None
        intel = build_stock_intelligence(
            ticker, current_opp=current_opp, current_row=rows[0] if rows and current_opp is None else None,
            history=history, regime=None, calibration_records=calibration,
            earnings_days=_earnings_days_map([ticker]).get(ticker))
        quote = {"last": None, "chg_pct": None}
        for r in rows:  # price even when the name doesn't qualify as a setup
            quote["last"] = quote["last"] if quote["last"] is not None else _num(r.get("Last"))
            quote["chg_pct"] = quote["chg_pct"] if quote["chg_pct"] is not None else _num(r.get("PctChange"))
        return {"intel": intel, "scan_at": _iso(run["created_at"]) if run else None,
                "in_latest_scan": bool(rows), "quote": quote}

    return stock_cache.get(ticker, load)


def stock_detail(ticker: str, entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_opportunity

    core = _stock_core(ticker)
    intel = redact_prebreakout_opportunity(core["intel"], allowed=bool(entitlements.get("can_early_breakout")))
    movement = intel.get("movement") or {}
    out = {
        "ticker": ticker,
        "scan_at": core["scan_at"],
        "in_latest_scan": core["in_latest_scan"],
        "has_setup": bool(intel.get("has_opportunity")),
        "from_history": bool(intel.get("from_history")),
        "price": _num(intel.get("price")) if intel.get("price") is not None else core["quote"]["last"],
        "change_pct": (_num(intel.get("change_pct")) if intel.get("change_pct") is not None
                       else core["quote"]["chg_pct"]),
        "hsf_score": intel.get("hsf_score"),
        "status": intel.get("status"),
        "primary_setup": intel.get("primary_setup"),
        "signals": [str(s) for s in intel.get("signals") or []],
        "score_components": intel.get("score_components"),
        "movement": movement.get("movement_state"),
        "score_change": movement.get("score_delta"),
        "reasons": [str(r) for r in intel.get("reasons") or []],
        "risks": [str(r) for r in intel.get("risks") or []],
        "watch_next": [str(r) for r in intel.get("watch_next") or []],
        "breakout_score": _num((intel.get("model") or {}).get("breakout_score")),
        "prob": _num((intel.get("model") or {}).get("prebreakout_prob")),
        "earnings_days": intel.get("earnings_days"),
        "history_summary": intel.get("history_summary"),
        "historical_context": intel.get("historical_context"),
        "outcome_cohort": intel.get("outcome_cohort"),
        "lifecycle": intel.get("lifecycle") or [],
    }
    # Historical research is Pro (can_track_record), as on the web's stock page (Run 85E).
    out["historical_locked"] = not bool(entitlements.get("can_track_record"))
    if out["historical_locked"]:
        out.update({"history_summary": None, "historical_context": None, "outcome_cohort": None})
    return json_safe(out)
