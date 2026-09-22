"""Run 33 — historical reconstruction of the FULL Run 32 Day Trader feature set.

Production Day Trader layers DAILY indicators (ADX 14, SuperTrend 13/2, EWO 5/35 —
see market_data.fetch_daily_range_metrics) onto intraday VWAP / RVOL / gap /
change. To validate DT Score honestly at a historical intraday timestamp T we
must reconstruct every input as it was KNOWN at T:

  * Daily indicators (ADX / SuperTrend / EWO): computed on the daily bars up to
    and including the PRIOR completed session (D-1) — during a live session on
    day D the latest *completed* daily bar is D-1, so this matches production and
    introduces no lookahead.
  * gap %      = (day-D open − D-1 close) / D-1 close.
  * RVOL       = cumulative intraday volume through T / average daily volume
    over the prior `rvol_lookback` sessions.
  * VWAP / vs-VWAP / intraday change: from the day-D minute bars up to T.

Outcomes (returns / MFE / MAE) use only bars AFTER T, same session. Pure: takes a
daily OHLCV frame + minute bars and returns observations; no network here.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def _et_date(ts: Any) -> Optional[_dt.date]:
    """Trading (ET) date from an RFC3339/ISO timestamp or datetime."""
    try:
        if isinstance(ts, _dt.datetime):
            dt = ts
        else:
            dt = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_ET).date()
    except Exception:
        return None


def _finite_latest_upto(series, cutoff_date) -> Optional[float]:
    """Last finite value of a daily series at or before cutoff_date."""
    try:
        s = series[[d.date() <= cutoff_date for d in series.index]].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None


def daily_indicator_series(daily_df):
    """Compute the daily ADX / SuperTrend(direction) / EWO series once."""
    from scan.indicators import adx, ewo, supertrend

    out = {}
    try:
        out["adx"] = adx(daily_df, 14)
    except Exception:
        out["adx"] = None
    try:
        st = supertrend(daily_df, 13, 2.0)
        out["supertrend_direction"] = st["direction"]
    except Exception:
        out["supertrend_direction"] = None
    try:
        out["ewo"] = ewo(daily_df, 5, 35)
    except Exception:
        out["ewo"] = None
    return out


def reconstruct_observations(
    symbol: str,
    daily_df,
    minute_bars: Sequence[Dict[str, Any]],
    *,
    sample_every: int = 15,
    rvol_lookback: int = 20,
) -> List[Dict[str, Any]]:
    """Build full-feature, no-lookahead observations for one symbol.

    `daily_df` is a title-cased OHLCV DataFrame (Open/High/Low/Close/Volume) with a
    DatetimeIndex covering enough history for the daily indicators (>= ~37 bars).
    `minute_bars` are ascending 1-minute bars for the validation day(s).
    """
    from analytics.day_trade_intel import day_trade_intelligence
    from analytics.day_trade_validation import (
        directional_return,
        forward_returns,
        mfe_mae,
    )
    from scripts.validate_day_trade_score import HORIZONS, MFE_WINDOW

    ind = daily_indicator_series(daily_df)

    # Group minute bars by ET trading date, preserving order.
    by_day: Dict[_dt.date, List[Dict[str, Any]]] = {}
    for b in minute_bars:
        d = _et_date(b.get("t"))
        if d is not None:
            by_day.setdefault(d, []).append(b)

    try:
        daily_dates = [ix.date() for ix in daily_df.index]
    except Exception:
        return []

    obs: List[Dict[str, Any]] = []
    for day, bars in by_day.items():
        prior = [d for d in daily_dates if d < day]
        if not prior:
            continue  # need a prior completed session
        dm1 = prior[-1]
        # Daily indicators as-of D-1 (no lookahead).
        adx_val = _finite_latest_upto(ind["adx"], dm1) if ind["adx"] is not None else None
        ewo_val = _finite_latest_upto(ind["ewo"], dm1) if ind["ewo"] is not None else None
        st_dir = None
        if ind["supertrend_direction"] is not None:
            try:
                sd = ind["supertrend_direction"]
                sd = sd[[d.date() <= dm1 for d in sd.index]].dropna()
                st_dir = str(sd.iloc[-1]) if len(sd) else None
            except Exception:
                st_dir = None
        try:
            dm1_close = float(daily_df.loc[[ix for ix in daily_df.index if ix.date() == dm1][-1], "Close"])
        except Exception:
            dm1_close = None
        try:
            vols = daily_df["Volume"][[ix.date() <= dm1 for ix in daily_df.index]].dropna()
            avg_vol = float(vols.iloc[-rvol_lookback:].mean()) if len(vols) else None
        except Exception:
            avg_vol = None

        day_open = bars[0].get("o")
        gap_pct = (((day_open - dm1_close) / dm1_close * 100)
                   if (day_open is not None and dm1_close) else None)

        closes = [b.get("c") for b in bars]
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
                "gap_pct": gap_pct,
                "adx": adx_val,
                "rvol": (cum_v / avg_vol) if avg_vol else None,
                "supertrend_direction": st_dir,
                "ewo": ewo_val,
            }
            intel = day_trade_intelligence(feat)
            o: Dict[str, Any] = {
                "timestamp": str(b.get("t")), "ticker": symbol, "price_at_signal": price,
                "direction": intel["direction"], "score": intel["score"],
                "setup_quality": intel["quality"], "conflicts": intel["conflicts"],
                "diagnostic_inputs": dict(feat),
            }
            rets = forward_returns(closes[i:], 0, HORIZONS)
            for h, r in rets.items():
                o[f"return_{h}"] = r
                o[f"directional_return_{h}"] = directional_return(intel["direction"], r)
            o.update(mfe_mae(closes[i:], 0, MFE_WINDOW, intel["direction"]))
            obs.append(o)
    return obs
