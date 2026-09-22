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
        "diagnostic_inputs": {key: features.get(key) for key in (
            "chg_pct", "gap_pct", "rvol", "vs_vwap_pct", "adx", "supertrend_direction", "ewo")},
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


def minute_bars_to_observations(symbol: str, bars: Sequence[Dict[str, Any]],
                                *, sample_every: int = 15) -> List[Dict[str, Any]]:
    """Build no-lookahead validation observations from ascending 1-minute bars.

    At each sampled signal bar i, the Run 32 INTRADAY-derivable inputs are
    computed from bars[0..i] ONLY (session VWAP, intraday change); outcomes come
    from bars after i. Daily-derived indicators (ADX / SuperTrend / EWO / gap /
    RVOL baseline) are NOT reconstructed from minute bars here and are passed as
    None — the Run 32 engine renormalizes over available evidence. This is an
    honest, partial validation of the intraday signal subset; full-fidelity
    validation additionally needs the daily-indicator history.
    """
    obs: List[Dict[str, Any]] = []
    if len(bars) < 2:
        return obs
    closes = [b.get("c") for b in bars]
    day_open = bars[0].get("o")
    cum_pv = cum_v = 0.0
    for i, b in enumerate(bars):
        typ = ((b.get("h") or 0) + (b.get("l") or 0) + (b.get("c") or 0)) / 3.0
        vol = b.get("v") or 0
        cum_pv += typ * vol
        cum_v += vol
        if i == 0 or i % sample_every != 0:
            continue
        price = b.get("c")
        vwap = (cum_pv / cum_v) if cum_v else None
        feat = {
            "chg_pct": ((price - day_open) / day_open * 100) if (day_open and price is not None) else None,
            "vs_vwap_pct": ((price - vwap) / vwap * 100) if (vwap and price is not None) else None,
            "gap_pct": None, "adx": None, "rvol": None,
            "supertrend_direction": None, "ewo": None,
        }
        obs.append(build_observation(timestamp=str(b.get("t")), ticker=symbol,
                                     features=feat, prices_after=closes[i:]))
    return obs


def _load_intraday_observations() -> Optional[List[Dict[str, Any]]]:
    """Load historical observations from Alpaca 1-minute bars when configured.

    Reads DTV_SYMBOLS (comma-separated), DTV_START, DTV_END (ISO). Returns None
    when no symbols/creds are configured (the harness then reports NO_INTRADAY_DATA
    honestly). Never raises."""
    import os

    symbols = [s.strip().upper() for s in (os.getenv("DTV_SYMBOLS") or "").split(",") if s.strip()]
    start = os.getenv("DTV_START")
    if not symbols or not start:
        return None
    try:
        from data.price_alpaca import fetch_minute_bars
    except Exception:
        return None
    end = os.getenv("DTV_END")
    sample_every = int(os.getenv("DTV_SAMPLE_EVERY", "15"))
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            bars = fetch_minute_bars(sym, start, end)
        except Exception:
            bars = []
        if not bars:
            continue
        # Prefer FULL-feature reconstruction (daily ADX/SuperTrend/EWO + gap +
        # RVOL) when a daily frame is available; else the intraday-only subset.
        daily_df = _fetch_daily_frame(sym)
        if daily_df is not None:
            try:
                from analytics.day_trade_reconstruct import reconstruct_observations

                out.extend(reconstruct_observations(sym, daily_df, bars, sample_every=sample_every))
                continue
            except Exception:
                pass
        out.extend(minute_bars_to_observations(sym, bars, sample_every=sample_every))
    return out or None


def _fetch_daily_frame(symbol: str):
    """Daily OHLCV frame (title-cased) for the daily-indicator reconstruction, or
    None when unavailable. Reuses the existing daily Alpaca path (>= ~60 sessions
    so ADX/EWO have enough history)."""
    try:
        from data.price_alpaca import download_multi_alpaca

        frames = download_multi_alpaca([symbol], period="120d", interval="1d",
                                       prepost=False, timeout_s=20.0)
        df = frames.get(symbol.upper())
        if df is None:
            df = frames.get(symbol)
        if df is None or getattr(df, "empty", True):
            return None
        # Ensure title-cased OHLCV columns the indicators expect.
        rename = {c: str(c).title() for c in df.columns}
        return df.rename(columns=rename)
    except Exception:
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
        from scripts.diagnose_dt_tiers import build_diagnostic, render_markdown

        tier_report = build_diagnostic(observations, profile="rejected_v2")
        (out_dir / "dt_tier_diagnostic.json").write_text(json.dumps(tier_report, indent=2, allow_nan=False) + "\n")
        (out_dir / "dt_tier_diagnostic.md").write_text(render_markdown(tier_report))

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
