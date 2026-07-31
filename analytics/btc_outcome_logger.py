"""15-min BTC outcome logger: collect features + settle ground-truth results.

Two steps, run together from the cron (best-effort):
  settle_due()  — for logged windows past their close, pull Kalshi's settled
                  result and record whether BTC finished above the strike.
  log_current() — snapshot the current KXBTC15M window: decision-engine features,
                  its prediction, and the Kalshi implied price (pending outcome).

Over time this builds the labeled dataset behind item #9 — training the model on
the actual 15-min above/below outcome instead of a heuristic. Never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _features(sig: Dict[str, Any], market: Dict[str, Any], spot: float) -> Dict[str, Any]:
    """Model-facing features from the engine read + the contract + clock."""
    price = float(sig.get("price") or spot or 0.0) or 1.0
    vwap = sig.get("vwap")
    strike = market.get("floor_strike")
    ct = market.get("close_time")
    hour = ct.hour if isinstance(ct, datetime) else None
    dow = ct.weekday() if isinstance(ct, datetime) else None
    return {
        "ema9_slope": sig.get("ema9_slope"),
        "ema_sep_pct": round((sig.get("ema9", 0) - sig.get("ema21", 0)) / price * 100, 4),
        "price_vs_ema9_pct": round((price - sig.get("ema9", price)) / price * 100, 4),
        "vwap_dist_pct": (round((price - vwap) / price * 100, 4) if vwap else None),
        "rsi": sig.get("rsi"),
        "macd_hist": sig.get("macd_hist"),
        "atr_pct": sig.get("atr_pct"),
        "volume_ratio": sig.get("rvol"),
        "htf_agree": sig.get("htf_agree"),
        "hour_utc": hour,
        "day_of_week": dow,
        "kalshi_yes_pct": market.get("up_prob_pct"),
        "spot_minus_strike": (round(spot - float(strike), 2) if strike else None),
        "confidence": sig.get("confidence"),
        "tradeable": sig.get("tradeable"),
    }


def log_current() -> bool:
    """Snapshot the current 15-min window as a pending outcome row."""
    try:
        from data.crypto_btc import fetch_btc_bars, latest_btc_price
        from data.kalshi_markets import fetch_btc_15min
        from db.btc_outcomes import log_window
        from scan.kalshi_signal import compute_kalshi_signal

        spot = latest_btc_price()
        df = fetch_btc_bars("5m")
        htf = fetch_btc_bars("1h")
        if df is None or spot is None:
            print("[btc_logger] no bars/spot; skip log")
            return False
        sig = compute_kalshi_signal(df, htf_df=htf)
        market = fetch_btc_15min(spot)
        if not sig or not market or not market.get("ticker"):
            print("[btc_logger] no signal/market; skip log")
            return False
        ok = log_window(
            market["ticker"],
            close_time=market.get("close_time"),
            strike=market.get("floor_strike"),
            spot=spot,
            pred_direction=sig.get("direction"),
            pred_confidence=sig.get("confidence"),
            pred_win_prob=sig.get("win_probability"),
            kalshi_yes_pct=market.get("up_prob_pct"),
            features=_features(sig, market, spot),
        )
        print(f"[btc_logger] logged window {market['ticker']} "
              f"pred={sig.get('recommendation')} yes={market.get('up_prob_pct')}%")
        return ok
    except Exception as e:
        print(f"[btc_logger] log_current failed: {type(e).__name__}: {e}")
        return False


def settle_due() -> int:
    """Settle any logged windows now past close using Kalshi's result. Returns count."""
    try:
        from data.kalshi_markets import fetch_market
        from db.btc_outcomes import pending_due, record_result
    except Exception:
        return 0
    settled = 0
    for row in pending_due():
        tk = row.get("window_ticker")
        try:
            m = fetch_market(tk)
        except Exception:
            m = None
        if not m or not m.get("settled") or m.get("result_up") is None:
            continue  # not finalized yet — retry next run
        if record_result(tk, bool(m["result_up"]), m.get("expiration_value")):
            settled += 1
            print(f"[btc_logger] settled {tk}: result_up={m['result_up']} "
                  f"@ {m.get('expiration_value')}")
    return settled


def run_btc_outcome_logger() -> Dict[str, Any]:
    """Settle due windows, then log the current one. Best-effort entrypoint."""
    now = datetime.now(timezone.utc)
    n_settled = settle_due()
    logged = log_current()
    print(f"[btc_logger] {now:%H:%M} settled={n_settled} logged={int(logged)}")
    return {"settled": n_settled, "logged": bool(logged)}


if __name__ == "__main__":
    run_btc_outcome_logger()
