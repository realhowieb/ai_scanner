"""Kalshi BTC scanner signal: directional read on BTC for up/down contracts.

Pure, deterministic, headless-testable. Takes an OHLCV DataFrame (BTC bars) and
returns a single actionable read for Kalshi's up/down BTC event contracts:

  direction ("up"/"down"), confidence (0-100), a heuristic probability estimate,
  the underlying indicator readings (EMA 9/21, RSI, MACD, VWAP, RVOL, support/
  resistance), an entry zone, and an exit / cash-out target + stop.

No network here — the caller supplies the bars (data.crypto_btc). Educational
signal only; the probability is a heuristic, not a guarantee.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

MIN_BARS = 35


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _vwap(df) -> Optional[float]:
    """Rolling VWAP over the supplied window: sum(typical*vol)/sum(vol)."""
    try:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        denom = float(vol.sum())
        if denom <= 0:
            return None
        return float((tp * vol).sum() / denom)
    except Exception:
        return None


def compute_kalshi_signal(df, *, sr_lookback: int = 30) -> Optional[Dict[str, Any]]:
    """Directional BTC read for Kalshi up/down contracts, or None if too little data.

    Blends EMA 9/21 trend, RSI(14), MACD histogram, and price-vs-VWAP into a
    weighted directional score; a volume spike (RVOL) boosts confidence. Entry /
    exit zones come from ATR and the nearest support/resistance.
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

        ema9 = float(ema(closes, 9).iloc[-1])
        ema21 = float(ema(closes, 21).iloc[-1])
        rsi14 = float(rsi(closes, 14).iloc[-1])
        _macd_line, _sig, hist = macd(closes)
        macd_hist = float(hist.iloc[-1])
        vwap = _vwap(df)
        try:
            atr14 = float(atr(df, 14).iloc[-1])
        except Exception:
            atr14 = price * 0.01
        if atr14 != atr14 or atr14 <= 0:  # NaN/zero guard
            atr14 = price * 0.01

        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        avg_vol = float(vol.tail(20).mean() or 0.0)
        rvol = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

        lows = pd.to_numeric(df["Low"], errors="coerce").dropna()
        highs = pd.to_numeric(df["High"], errors="coerce").dropna()
        support = float(lows.tail(sr_lookback).min())
        resistance = float(highs.tail(sr_lookback).max())

        # --- weighted directional votes in [-1, +1] ---
        votes: List[Tuple[str, float, float, str]] = []  # (name, vote, weight, detail)

        ema_gap_pct = (ema9 - ema21) / price * 100.0
        votes.append((
            "EMA 9/21 trend", _clip(ema_gap_pct / 0.30, -1, 1), 2.0,
            f"EMA9 {ema9:,.0f} {'>' if ema9 >= ema21 else '<'} EMA21 {ema21:,.0f}",
        ))
        votes.append((
            "RSI(14)", _clip((rsi14 - 50.0) / 25.0, -1, 1), 1.0,
            f"RSI {rsi14:.0f} ({'bullish' if rsi14 >= 50 else 'bearish'})",
        ))
        votes.append((
            "MACD", _clip(macd_hist / (price * 0.001), -1, 1), 1.5,
            f"histogram {macd_hist:+.1f} ({'up' if macd_hist >= 0 else 'down'})",
        ))
        if vwap is not None:
            votes.append((
                "VWAP", _clip((price - vwap) / price / 0.002, -1, 1), 1.5,
                f"price {'above' if price >= vwap else 'below'} VWAP {vwap:,.0f}",
            ))

        weight_sum = sum(w for _n, _v, w, _d in votes)
        net = sum(v * w for _n, v, w, _d in votes)
        strength = abs(net) / weight_sum if weight_sum else 0.0
        direction = "up" if net >= 0 else "down"

        confidence = strength * 100.0
        vol_spike = rvol >= 1.5
        if vol_spike:
            confidence = min(99.0, confidence + 10.0)  # volume confirms the move
        confidence = round(_clip(confidence, 0, 99))

        # Heuristic probability the move continues — clamped so it never reads as
        # a certainty. NOT a calibrated forecast.
        probability = round(_clip(50.0 + confidence * 0.35, 50, 85))

        # Entry zone: slight pullback into the move. Target: ~1 ATR toward the
        # nearest S/R. Stop: ~0.75 ATR against.
        if direction == "up":
            entry_zone = (round(price - 0.25 * atr14, 2), round(price, 2))
            target = round(min(price + atr14, resistance) if resistance > price
                           else price + atr14, 2)
            stop = round(max(price - 0.75 * atr14, support) if support < price
                         else price - 0.75 * atr14, 2)
        else:
            entry_zone = (round(price, 2), round(price + 0.25 * atr14, 2))
            target = round(max(price - atr14, support) if support < price
                           else price - atr14, 2)
            stop = round(min(price + 0.75 * atr14, resistance) if resistance > price
                         else price + 0.75 * atr14, 2)

        reasons = [f"{name}: {detail}" for name, v, _w, detail in votes if abs(v) > 0.05]

        return {
            "direction": direction,                       # 'up' | 'down'
            "action": "Buy Up" if direction == "up" else "Buy Down",
            "confidence": int(confidence),                # 0-100
            "probability": int(probability),              # 50-85 heuristic
            "price": round(price, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "rsi": round(rsi14, 1),
            "macd_hist": round(macd_hist, 2),
            "vwap": round(vwap, 2) if vwap is not None else None,
            "rvol": round(rvol, 2),
            "volume_spike": bool(vol_spike),
            "atr": round(atr14, 2),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "entry_zone": entry_zone,                     # (low, high)
            "target": target,
            "stop": stop,
            "reasons": reasons,
        }
    except Exception:
        return None
