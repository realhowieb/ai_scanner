"""Settle immutable fired signals with forward market outcomes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Benchmark for Outcome Intelligence: the same SPY the track record uses.
BENCHMARK = "SPY"


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
            sorted(set(symbols) | {BENCHMARK}),
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
        if outcome.get("return_1d") is not None:
            _save_benchmark_for(signal, bars_by_symbol.get(BENCHMARK))
    return saved


def benchmark_returns(bench_bars, fired_at) -> Optional[Dict[str, Optional[float]]]:
    """The benchmark's 1/3/5D returns over the signal's own window, scored by the
    SAME score_signal as the stock (first bar on/after the fire date, close to
    close). None when the benchmark window is incomplete."""
    if bench_bars is None:
        return None
    scored = score_signal(bench_bars, fired_at)
    if scored is None:
        return None
    return {k: scored.get(k) for k in ("return_1d", "return_3d", "return_5d")}


def _save_benchmark_for(signal: Dict[str, Any], bench_bars) -> bool:
    try:
        from db.signal_outcomes import save_benchmark

        rets = benchmark_returns(bench_bars, signal["fired_at"])
        if rets is None:
            return False  # left NULL; the backfill retries it
        return bool(save_benchmark(signal_id=int(signal["id"]), symbol=BENCHMARK, **rets))
    except Exception:
        return False


def backfill_benchmark_returns(max_rows: int = 2000) -> int:
    """Fill the benchmark for matured rows scored before it existed (bounded).

    Uses only bars inside each row's already-completed window, so it adds no
    information the outcome itself didn't already use. Returns rows filled."""
    try:
        from db.signal_outcomes import list_benchmark_backfill
    except Exception:
        return 0
    rows: List[Dict[str, Any]] = list_benchmark_backfill(limit=max_rows)
    if not rows:
        return 0
    try:
        from datetime import datetime, timezone

        oldest = min(r["fired_at"] for r in rows if r.get("fired_at") is not None)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        days = min(1500, (datetime.now(timezone.utc) - oldest).days + 20)
        from data.price_alpaca import download_multi_alpaca

        bars = (download_multi_alpaca([BENCHMARK], period=f"{days}d", interval="1d",
                                      prepost=False, timeout_s=25.0) or {}).get(BENCHMARK)
    except Exception:
        return 0
    if bars is None:
        return 0
    return sum(1 for r in rows if _save_benchmark_for(r, bars))
