"""Settle immutable fired signals with forward market outcomes."""
from __future__ import annotations

from typing import Any, Dict, Optional


def _entry_position(bars, fire_date) -> Optional[int]:
    try:
        closes = bars["Close"].dropna()
        for pos, ts in enumerate(closes.index):
            d = ts.date() if hasattr(ts, "date") else ts
            if d >= fire_date:
                return pos
    except Exception:
        pass
    return None


def score_signal(bars, fired_at) -> Optional[Dict[str, Any]]:
    """Return 1/3/5D close returns and 5D MFE/MAE, or None if incomplete."""
    try:
        fire_date = fired_at.date() if hasattr(fired_at, "date") else fired_at
        closes = bars["Close"].dropna()
        entry_pos = _entry_position(bars, fire_date)
        if entry_pos is None or entry_pos + 5 >= len(closes):
            return None
        entry = float(closes.iloc[entry_pos])
        if entry <= 0:
            return None

        out: Dict[str, Any] = {}
        for horizon in (1, 3, 5):
            exit_ = float(closes.iloc[entry_pos + horizon])
            out[f"return_{horizon}d"] = round((exit_ - entry) / entry, 6)

        window = bars.iloc[entry_pos + 1 : entry_pos + 6]
        highs = window["High"] if "High" in window.columns else window["Close"]
        lows = window["Low"] if "Low" in window.columns else window["Close"]
        out["mfe_5d"] = round((float(highs.max()) - entry) / entry, 6)
        out["mae_5d"] = round((float(lows.min()) - entry) / entry, 6)
        return out
    except Exception:
        return None


def score_pending_signal_outcomes(max_signals: int = 1000) -> int:
    """Fill settled outcomes for frozen fired signals. Returns rows updated."""
    try:
        from db.signal_outcomes import list_pending_outcomes, save_outcome
    except Exception:
        return 0

    signals = list_pending_outcomes(limit=max_signals)
    if not signals:
        return 0

    symbols = sorted({str(s["ticker"]).upper() for s in signals if s.get("ticker")})
    if not symbols:
        return 0

    try:
        from data.price_alpaca import download_multi_alpaca

        bars_by_symbol = download_multi_alpaca(
            symbols,
            period="60d",
            interval="1d",
            prepost=False,
            timeout_s=25.0,
        )
    except Exception:
        return 0
    if not bars_by_symbol:
        return 0

    saved = 0
    for signal in signals:
        sym = str(signal.get("ticker") or "").upper()
        bars = bars_by_symbol.get(sym)
        if bars is None:
            bars = bars_by_symbol.get(sym.replace("-", "."))
        outcome = score_signal(bars, signal["fired_at"]) if bars is not None else None
        if outcome is None:
            outcome = {
                "return_1d": None,
                "return_3d": None,
                "return_5d": None,
                "mfe_5d": None,
                "mae_5d": None,
            }
        try:
            if save_outcome(signal_id=int(signal["id"]), **outcome):
                saved += 1
        except Exception:
            continue
    return saved
