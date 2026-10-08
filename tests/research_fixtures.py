"""Shared builders for research dataset tests (opportunity rows + scan records)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

UTC = dt.timezone.utc


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def opp(oid: int, ticker: str, fired_at: str, *, score: Optional[float] = 72.0, version: Optional[str] = "1.0",
        setup: str = "breakout", status: str = "WATCH", signals=("breakout", "gapper"), chg_pct=3.1, gap_pct=2.0,
        prob=0.18, breakout_score=8.5, scored: bool = False, r1=None, r3=None, r5=None, mfe=None, mae=None,
        b1=None, b3=None, b5=None) -> Dict[str, Any]:
    row = {
        "id": oid, "ticker": ticker, "fired_at": ts(fired_at), "setup_score": breakout_score,
        "prebreakout_prob": prob,
        "indicators": {"n_signals": len(signals), "signals": list(signals), "chg_pct": chg_pct, "gap_pct": gap_pct,
                       "fading": False, "primary_setup": setup, "status": status},
        "raw_signal": {"hsf_score": score, "score_version": version, "primary_setup": setup, "status": status,
                       "score_components": {"signals_component": 30.0, "model_component": 25.0,
                                            "momentum_component": 17.0, "fading_penalty": 0.0}},
        "return_1d": r1, "return_3d": r3, "return_5d": r5, "mfe_5d": mfe, "mae_5d": mae,
        "outcome_computed_at": ts(fired_at) + dt.timedelta(days=8) if scored else None,
        "created_at": ts(fired_at),
    }
    if b1 is not None or b3 is not None or b5 is not None:
        row.update(benchmark_return_1d=b1, benchmark_return_3d=b3, benchmark_return_5d=b5)
    return row


def scan(symbol: str, scan_ts: str, *, written: Optional[str] = None, price=10.0, rvol=2.0, trend20=5.0,
         rank=3, context="scheduled:us_market", oid: Optional[str] = None, meta: bool = True) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "observation_id": oid or f"{symbol}-{scan_ts}-{context}", "symbol": symbol, "context": context,
        "schema_version": "hsf-obs-1.0", "scan_timestamp": ts(scan_ts).isoformat(),
        "timestamp": ts(scan_ts).replace(minute=0, second=0).isoformat(),
        "market": {"price": price, "volume": 1_000_000},
        "indicators": {"rvol": rvol, "atr_pct": 3.2, "gap_pct": 2.0, "chg_pct": 3.1},
        "scanners": [{"name": "breakout", "score": 8.5, "meta": {"is_breakout": True}}],
        "versions": {"prebreakout_model": "prebreakout-xgb-v16"},
        "market_context": {"scan_id": ts(scan_ts).isoformat()},
        "universe_version": "US_MARKET",
    }
    if meta:
        rec["research_metadata"] = {"row_features": {"trend_20d_pct": trend20, "trend_10d_pct": 2.0,
                                                     "rs_vs_spy": 1.1, "ema_cross": "golden"},
                                    "rank_at_observation": rank, "scan_id": ts(scan_ts).isoformat(),
                                    "universe_name": "US_MARKET", "scoring_version": "breakout-1"}
    return {"observation_id": rec["observation_id"], "symbol": symbol, "context": context,
            "timestamp": rec["timestamp"], "created_at": ts(written or scan_ts) + (
                dt.timedelta(0) if written else dt.timedelta(minutes=2)), "record": rec}
