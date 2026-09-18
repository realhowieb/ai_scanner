#!/usr/bin/env python3
"""Run 33 — Day Trader DT Score validation / calibration harness.

Direction-aware backtest of the Run 32 DT Score against historical intraday
bars: compute the Run 32 intelligence at each signal timestamp T using ONLY
bars <= T (no lookahead), then measure forward directional returns / MFE / MAE
using later bars, and aggregate by score bucket, direction, setup quality, and
conflicts.

Runs OFF the Streamlit path (a plain script) and modifies no production state.
It needs a historical INTRADAY (minute) bar source. The project currently ships
only daily-bar Alpaca support, so absent a supplied intraday loader (and creds)
this reports NO_INTRADAY_DATA honestly and performs NO calibration — it never
fabricates results to make the score look good.

Usage:
    python -m scripts.validate_day_trade_score [--out artifacts]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Minute offsets → intraday horizons (bar counts assume 1-minute bars).
HORIZONS = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
MFE_WINDOW = 60


def build_observation(*, timestamp: str, ticker: str, features: Dict[str, Any],
                      prices_after: Sequence[float]) -> Dict[str, Any]:
    """One validation observation. `features` are the Run 32 inputs computed from
    bars AT OR BEFORE T; `prices_after[0]` is the signal price and the rest are
    strictly-later bars used ONLY for outcomes."""
    from analytics.day_trade_intel import day_trade_intelligence
    from analytics.day_trade_validation import (
        directional_return,
        forward_returns,
        mfe_mae,
    )

    intel = day_trade_intelligence(features)
    obs: Dict[str, Any] = {
        "timestamp": timestamp, "ticker": ticker,
        "price_at_signal": prices_after[0] if prices_after else None,
        "direction": intel["direction"], "score": intel["score"],
        "setup_quality": intel["quality"], "conflicts": intel["conflicts"],
    }
    rets = forward_returns(prices_after, 0, HORIZONS)
    for h, r in rets.items():
        obs[f"return_{h}"] = r
        obs[f"directional_return_{h}"] = directional_return(intel["direction"], r)
    obs.update(mfe_mae(prices_after, 0, MFE_WINDOW, intel["direction"]))
    return obs


def build_report(observations: List[Dict[str, Any]], *, min_n: int = 10) -> Dict[str, Any]:
    """Aggregate observations into the validation report structure (pure)."""
    from analytics.day_trade_validation import bucket_report, score_distribution

    horizons = tuple(HORIZONS.keys())
    directional = [o for o in observations if str(o.get("direction")) in ("bullish", "bearish")]
    bull = [o for o in observations if o.get("direction") == "bullish"]
    bear = [o for o in observations if o.get("direction") == "bearish"]
    by_quality: Dict[str, List[Dict[str, Any]]] = {}
    for o in observations:
        by_quality.setdefault(str(o.get("setup_quality")), []).append(o)

    def _quality_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        d = [o for o in rows if str(o.get("direction")) in ("bullish", "bearish")]
        out = {"n": len(rows), "n_directional": len(d)}
        for h in horizons:
            vals = [o[f"directional_return_{h}"] for o in d if o.get(f"directional_return_{h}") is not None]
            out[f"hit_rate_{h}"] = (sum(1 for x in vals if x > 0) / len(vals)) if vals else None
            out[f"avg_dir_return_{h}"] = (sum(vals) / len(vals)) if vals else None
        return out

    return {
        "sample_size": len(observations),
        "directional_n": len(directional),
        "score_distribution": score_distribution([o.get("score") for o in observations]),
        "by_score_bucket": bucket_report(observations, horizons=horizons, min_n=min_n),
        "bullish": bucket_report(bull, horizons=horizons, min_n=min_n),
        "bearish": bucket_report(bear, horizons=horizons, min_n=min_n),
        "by_setup_quality": {q: _quality_summary(rows) for q, rows in by_quality.items()},
        "min_sample": min_n,
    }


def _load_intraday_observations() -> Optional[List[Dict[str, Any]]]:
    """Return historical observations if an intraday bar source is available, else
    None. The project ships only daily-bar Alpaca support, so this returns None
    unless a real minute-bar loader is wired in (kept as the honest boundary)."""
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate/calibrate Day Trader DT Score")
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--min-n", type=int, default=10)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = _load_intraday_observations()
    if not observations:
        report = {
            "status": "NO_INTRADAY_DATA",
            "reason": ("No historical intraday (minute) bar source is available in "
                       "this environment. The project ships daily-bar Alpaca support "
                       "only, and no intraday loader/credentials were supplied. Per "
                       "Run 33, calibration is NOT performed without evidence and the "
                       "Run 32 DT Score is left unchanged."),
            "calibration_changes": None,
            "run32_parameters": _run32_parameters(),
        }
    else:
        report = {"status": "OK", **build_report(observations, min_n=args.min_n),
                  "run32_parameters": _run32_parameters()}

    (out_dir / "day_trader_validation.json").write_text(json.dumps(report, indent=2, default=str))
    _write_markdown(out_dir / "day_trader_validation.md", report)
    print(f"[validate_day_trade_score] status={report['status']} -> {out_dir}")
    return 0


def _run32_parameters() -> Dict[str, Any]:
    """Snapshot the current (unchanged) Run 32 scoring parameters for the report."""
    from analytics import day_trade_intel as di
    return {
        "weights": di._WEIGHTS,
        "adx_floor": di._ADX_FLOOR, "adx_full": di._ADX_FULL,
        "rvol_ordinary": di._RVOL_ORDINARY, "rvol_strong": di._RVOL_STRONG,
        "vwap_full": di._VWAP_FULL, "momentum_full": di._MOM_FULL, "gap_full": di._GAP_FULL,
        "conflict_penalty": di._CONFLICT_PENALTY, "conflict_penalty_cap": di._CONFLICT_PENALTY_CAP,
        "strong_threshold": 65, "weak_threshold": 40, "min_directional": di._MIN_DIRECTIONAL,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Day Trader DT Score — Validation Report", ""]
    lines.append(f"**Status:** {report.get('status')}")
    if report.get("status") == "NO_INTRADAY_DATA":
        lines += ["", report["reason"], "",
                  "## Run 32 parameters (unchanged)", "```json",
                  json.dumps(report["run32_parameters"], indent=2), "```"]
    else:
        d = report.get("score_distribution", {})
        lines += ["", f"Sample: **{report.get('sample_size')}** "
                  f"(directional {report.get('directional_n')})", "",
                  "## Score distribution",
                  f"mean {d.get('mean')} · median {d.get('median')} · std {d.get('std')} · "
                  f"P10 {d.get('p10')} · P90 {d.get('p90')}", "",
                  "## By score bucket (direction-aware)",
                  "See day_trader_validation.json for full per-bucket / bull / bear / "
                  "quality tables."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
