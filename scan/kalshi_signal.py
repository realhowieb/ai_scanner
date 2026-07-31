"""Kalshi BTC decision engine: a selective up/down/No-Trade read for 15-min contracts.

Pure, deterministic, headless-testable. Prediction markets reward selectivity, so
this is a *decision engine*, not a signal generator — it will say **No Trade**
rather than force a call. A directional read requires trend agreement (EMA / VWAP
/ MACD all pointing the same way); conviction then comes from EMA separation &
distance, RSI, MACD strength, volume, volatility, and optional higher-timeframe
agreement. `evaluate_ev` compares the read's win probability to Kalshi's implied
price so marginal (already-priced) trades get passed.

Educational only; win probability is a heuristic, not an outcome-trained forecast.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

MIN_BARS = 35
MIN_VOLUME_RATIO = 0.7      # reject setups below this RVOL (#5)
MIN_ATR_PCT = 0.05          # skip when volatility is dead-flat (#7)
CONF_SMALL, CONF_NORMAL, CONF_STRONG = 40, 70, 85  # conviction tiers (#1)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _vwap(df) -> Optional[float]:
    try:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        denom = float(vol.sum())
        return float((tp * vol).sum() / denom) if denom > 0 else None
    except Exception:
        return None


def _htf_direction(htf_df) -> Optional[str]:
    """'up'/'down' trend on a higher timeframe via EMA9 vs EMA21, or None."""
    try:
        if htf_df is None or pd is None:
            return None
        from scan.indicators import ema

        c = pd.to_numeric(htf_df["Close"], errors="coerce").dropna()
        if len(c) < 21:
            return None
        return "up" if float(ema(c, 9).iloc[-1]) >= float(ema(c, 21).iloc[-1]) else "down"
    except Exception:
        return None


def compute_kalshi_signal(df, *, sr_lookback: int = 30, htf_df=None) -> Optional[Dict[str, Any]]:
    """Selective BTC decision for Kalshi up/down contracts, or None if too little data.

    Returns a rich decision: recommendation tier (Strong Buy Up / Buy Up / No Trade
    / Buy Down / Strong Buy Down), direction, confidence, win probability, position
    size, the gates (trend/volume/volatility/HTF), per-factor contributions, and
    entry/exit geometry. `htf_df` (higher-timeframe bars) is optional confirmation.
    """
    if pd is None or df is None or len(df) < MIN_BARS:
        return None
    try:
        from scan.indicators import atr, ema, macd, rsi

        closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(closes) < MIN_BARS:
            return None
        price = float(closes.iloc[-1])
        if price <= 0:
            return None

        ema9_s = ema(closes, 9)
        ema21_s = ema(closes, 21)
        ema9, ema21 = float(ema9_s.iloc[-1]), float(ema21_s.iloc[-1])
        ema9_slope = float(ema9_s.iloc[-1] - ema9_s.iloc[-4]) if len(ema9_s) >= 4 else 0.0
        rsi14 = float(rsi(closes, 14).iloc[-1])
        _ml, _sig, hist = macd(closes)
        macd_hist = float(hist.iloc[-1])
        vwap = _vwap(df)
        try:
            atr14 = float(atr(df, 14).iloc[-1])
        except Exception:
            atr14 = price * 0.01
        if atr14 != atr14 or atr14 <= 0:
            atr14 = price * 0.01
        atr_pct = atr14 / price * 100.0

        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        avg_vol = float(vol.tail(20).mean() or 0.0)
        rvol = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

        lows = pd.to_numeric(df["Low"], errors="coerce").dropna()
        highs = pd.to_numeric(df["High"], errors="coerce").dropna()
        support = float(lows.tail(sr_lookback).min())
        resistance = float(highs.tail(sr_lookback).max())

        # --- #2 trend agreement (VWAP optional if unavailable) ---
        price_vs_vwap = None if vwap is None else (price >= vwap)
        up_ok = (ema9 >= ema21) and (macd_hist >= 0) and (price_vs_vwap is not False)
        down_ok = (ema9 < ema21) and (macd_hist < 0) and (price_vs_vwap is not True)
        direction = "up" if up_ok else "down" if down_ok else "none"

        # --- gates (#5 volume, #7 volatility) ---
        volume_ok = rvol >= MIN_VOLUME_RATIO
        volatility_ok = atr_pct >= MIN_ATR_PCT

        # --- #4 higher-timeframe confirmation ---
        htf_dir = _htf_direction(htf_df)
        htf_agree = None if (htf_dir is None or direction == "none") else (htf_dir == direction)

        # --- conviction: per-factor contributions (#3 distance/separation/slope) ---
        contributions: List[Tuple[str, float]] = []
        if direction != "none":
            sep_pct = abs(ema9 - ema21) / price * 100.0
            dist_pct = abs(price - ema9) / price * 100.0
            slope_ok = (ema9_slope >= 0) == (direction == "up")
            contributions = [
                ("EMA separation", _clip(sep_pct / 0.30, 0, 1) * 30.0),
                ("Price vs EMA9", _clip(dist_pct / 0.40, 0, 1) * 20.0),
                ("MACD strength", _clip(abs(macd_hist) / (price * 0.001), 0, 1) * 20.0),
                ("RSI", _clip(abs(rsi14 - 50.0) / 25.0, 0, 1) * 15.0),
                ("EMA slope", 10.0 if slope_ok else -8.0),
            ]
            if rvol >= 1.5:
                contributions.append(("Volume surge", 10.0))
            elif rvol >= MIN_VOLUME_RATIO:
                contributions.append(("Volume", _clip((rvol - 0.7) / 0.8, 0, 1) * 5.0))
            if htf_agree is True:
                contributions.append(("1h trend agrees", 10.0))
            elif htf_agree is False:
                contributions.append(("1h trend disagrees", -15.0))
            if atr_pct >= 0.20:
                contributions.append(("Volatility expanding", 5.0))

        confidence = round(_clip(sum(pts for _n, pts in contributions), 0, 100))

        # --- #1 No-Trade filter + conviction tiers ---
        gated = (direction == "none") or (not volume_ok) or (not volatility_ok)
        if gated or confidence < CONF_SMALL:
            recommendation, action, size, tradeable = "No Trade", "No Trade", "None", False
            direction_out = "none"
        else:
            direction_out = direction
            tradeable = True
            if confidence >= CONF_STRONG:
                tier, size = "Strong ", "Large"
            elif confidence >= CONF_NORMAL:
                tier, size = "", "Medium"
            else:
                tier, size = "", "Small"
            action = "Buy Up" if direction == "up" else "Buy Down"
            recommendation = f"{tier}{action}"

        # Heuristic win probability (NOT outcome-trained — see #9). Only meaningful
        # when tradeable; clamped so it never reads as a certainty.
        win_probability = int(round(_clip(50.0 + confidence * 0.35, 50, 85))) if tradeable else None

        # entry / exit geometry (only when tradeable)
        entry_zone = target = stop = None
        if tradeable and direction == "up":
            entry_zone = (round(price - 0.25 * atr14, 2), round(price, 2))
            target = round(min(price + atr14, resistance) if resistance > price else price + atr14, 2)
            stop = round(max(price - 0.75 * atr14, support) if support < price else price - 0.75 * atr14, 2)
        elif tradeable and direction == "down":
            entry_zone = (round(price, 2), round(price + 0.25 * atr14, 2))
            target = round(max(price - atr14, support) if support < price else price - atr14, 2)
            stop = round(min(price + 0.75 * atr14, resistance) if resistance > price else price + 0.75 * atr14, 2)

        # Why No Trade / which factors drove it
        gate_reasons = []
        if direction == "none":
            gate_reasons.append("Trend not aligned (EMA / VWAP / MACD disagree)")
        if not volume_ok:
            gate_reasons.append(f"Volume too thin ({rvol:.2f}× < {MIN_VOLUME_RATIO}×)")
        if not volatility_ok:
            gate_reasons.append(f"Volatility too low (ATR {atr_pct:.2f}%)")
        if tradeable and confidence < CONF_SMALL:
            gate_reasons.append("Confidence below trade threshold")

        top_factors = sorted(
            [(n, round(p, 1)) for n, p in contributions], key=lambda x: abs(x[1]), reverse=True
        )

        return {
            "recommendation": recommendation,     # 'Strong Buy Up' | 'Buy Up' | 'No Trade' | …
            "direction": direction_out,           # 'up' | 'down' | 'none'
            "action": action,                     # kept for back-compat
            "tradeable": tradeable,
            "position_size": size,                # None | Small | Medium | Large
            "confidence": int(confidence),
            "win_probability": win_probability,   # 50-85 heuristic, or None
            "probability": win_probability,       # back-compat alias
            "price": round(price, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "ema9_slope": round(ema9_slope, 2),
            "rsi": round(rsi14, 1),
            "macd_hist": round(macd_hist, 2),
            "vwap": round(vwap, 2) if vwap is not None else None,
            "rvol": round(rvol, 2),
            "volume_spike": bool(rvol >= 1.5),
            "atr": round(atr14, 2),
            "atr_pct": round(atr_pct, 2),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "trend_aligned": direction != "none",
            "volume_ok": bool(volume_ok),
            "volatility_ok": bool(volatility_ok),
            "htf_direction": htf_dir,
            "htf_agree": htf_agree,
            "entry_zone": entry_zone,
            "target": target,
            "stop": stop,
            "contributions": top_factors,
            "gate_reasons": gate_reasons,
        }
    except Exception:
        return None


def evaluate_ev(
    direction: str, win_prob_pct: Optional[float], yes_price_pct: Optional[float],
    *, edge_threshold_pts: float = 4.0,
) -> Optional[Dict[str, Any]]:
    """Expected value of the aligned side vs Kalshi's implied price (#8, #10).

    For a binary contract bought at price p that pays $1 on a win, EV per $1
    notional = win_prob − p. `yes_price_pct` is the market YES price (%). Buying
    'down' means buying NO at (1 − YES). Recommends BUY only when the edge clears
    `edge_threshold_pts` — i.e. our probability beats what the market already
    charges. Returns None when inputs are missing.
    """
    if direction not in ("up", "down") or win_prob_pct is None or yes_price_pct is None:
        return None
    wp = win_prob_pct / 100.0
    yes = _clip(yes_price_pct / 100.0, 0.0, 1.0)
    side = "YES" if direction == "up" else "NO"
    price = yes if direction == "up" else (1.0 - yes)
    edge = wp - price                                  # EV per $1 notional
    ev_return = (edge / price * 100.0) if price > 0 else 0.0
    recommend = (edge * 100.0) >= edge_threshold_pts
    return {
        "side": side,                                  # buy YES (up) or NO (down)
        "entry_price_pct": round(price * 100.0, 1),    # what you'd pay, in %
        "win_prob_pct": round(wp * 100.0, 1),
        "edge_pts": round(edge * 100.0, 1),            # win_prob − price, in points
        "ev_return_pct": round(ev_return, 1),          # EV per $ staked
        "recommend": bool(recommend),                  # BUY vs PASS
    }
