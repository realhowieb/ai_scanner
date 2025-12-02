# --- Premium Breakout v2 Engine + utilities (merged) ---
from typing import List, Dict, Tuple, Optional

import time

import numpy as np
import pandas as pd
import streamlit as st

from market_data import get_latest_quotes


def safe_call(
    fn,
    *args,
    retries: int = 2,
    sleep_s: float = 0.8,
    label: str = "",
    **kwargs,
):
    """Retry wrapper to harden flaky providers (yfinance, etc.). Supports kwargs."""
    last_err = None
    for i in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            try:
                st.caption(
                    f"⚠️ {label or fn.__name__} failed (attempt {i+1}/{retries+1}): {e}"
                )
            except Exception:
                pass
            time.sleep(sleep_s)
    raise last_err






def _coerce_scan_output(out, tickers: List[str]) -> pd.DataFrame:
    """Coerce various real-scan return types into a DataFrame."""
    if out is None:
        return pd.DataFrame()
    if isinstance(out, pd.DataFrame):
        return out
    try:
        if isinstance(out, list):
            if len(out) == 0:
                return pd.DataFrame()
            if isinstance(out[0], dict):
                return pd.DataFrame(out)
            if isinstance(out[0], str):
                return pd.DataFrame({"Ticker": out})
        if isinstance(out, dict):
            if all(isinstance(v, (int, float)) for v in out.values()):
                return pd.DataFrame(
                    {"Ticker": list(out.keys()), "BreakoutScore": list(out.values())}
                )
            if all(isinstance(v, dict) for v in out.values()):
                rows = []
                for k, v in out.items():
                    r = {"Ticker": k}
                    r.update(v)
                    rows.append(r)
                return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def run_breakout_scan_v2(
    tickers: List[str],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    """Premium Breakout v2.1 Engine (Balanced mode).

    Uses the shared price-data module (which is now Alpaca-first with
    yfinance fallback) for daily OHLCV history, and `get_latest_quotes`
    for the most recent last/prev_close/volume where available.
    """
    if not tickers:
        return pd.DataFrame()

    # Fetch historical daily bars via ai_scanner.data.prices
    price_data: Dict[str, pd.DataFrame] = {}
    try:
        from ai_scanner.data.prices import fetch_price_data_parallel  # type: ignore
    except Exception:
        fetch_price_data_parallel = None  # type: ignore
    try:
        from ai_scanner.data.prices import fetch_price_data_batch  # type: ignore
    except Exception:
        fetch_price_data_batch = None  # type: ignore

    if fetch_price_data_parallel is not None:
        try:
            price_data, _skipped = fetch_price_data_parallel(tickers)
        except Exception:
            price_data = {}

    if not price_data and fetch_price_data_batch is not None:
        try:
            price_data, _skipped = fetch_price_data_batch(tickers)
        except Exception:
            price_data = {}

    if not price_data:
        if diagnostics:
            try:
                st.caption("📉 v2: price_data is EMPTY – no history returned for this universe.")
            except Exception:
                pass
        return pd.DataFrame()

    # Debug: how much data did we actually get?
    if diagnostics:
        try:
            st.caption(
                f"📊 v2: got price history for {len(price_data)} of {len(tickers)} symbols."
            )
        except Exception:
            pass

    # Fetch latest quotes from Alpaca to override last/prev/volume when possible
    try:
        live_quotes = get_latest_quotes(tickers)
    except Exception:
        live_quotes = {}

    rows: List[Dict] = []

    for sym in tickers:
        try:
            df_sym = price_data.get(sym)
            if df_sym is None or df_sym.empty:
                continue

            try:
                close = df_sym["Close"].dropna()
                high = df_sym["High"].dropna()
                vol = df_sym["Volume"].dropna()
            except Exception:
                continue

            if close.empty or high.empty or vol.empty:
                continue
            if len(close) < 5 or len(vol) < 5:
                continue

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
            last_vol = float(vol.iloc[-1])

            # If live Alpaca quotes are available, override last/prev/volume
            q = live_quotes.get(sym)
            if isinstance(q, dict):
                try:
                    if q.get("last") is not None:
                        last_close = float(q["last"])
                    if q.get("prev_close") is not None:
                        prev_close = float(q["prev_close"])
                    if q.get("volume") is not None:
                        last_vol = float(q["volume"])
                except Exception:
                    pass

            if last_close <= 0:
                continue

            if not (min_price <= last_close <= max_price):
                continue

            gap_pct = (
                ((last_close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
            )

            window_h = min(20, len(high))
            high20 = float(high.tail(window_h).max()) if window_h > 0 else last_close
            breakout_pos = (last_close / high20) if high20 > 0 else 0.0

            if len(close) >= 20:
                past20 = float(close.iloc[-20])
            else:
                past20 = float(close.iloc[0])
            trend20 = (
                ((last_close - past20) / past20) * 100 if past20 > 0 else 0.0
            )

            if len(close) >= 10:
                past10 = float(close.iloc[-10])
            else:
                past10 = float(close.iloc[0])
            trend10 = (
                ((last_close - past10) / past10) * 100 if past10 > 0 else 0.0
            )

            window_v = min(20, len(vol))
            avg_vol20 = (
                float(vol.tail(window_v).mean()) if window_v > 0 else float(vol.iloc[-1])
            )
            vol_rel20 = (last_vol / avg_vol20) if avg_vol20 > 0 else 1.0
            dollar_vol20 = avg_vol20 * last_close

            returns = close.pct_change().dropna()
            if not returns.empty:
                tail = returns.tail(min(20, len(returns)))
                vol20_pct = float(tail.std() * 100.0)
            else:
                vol20_pct = 0.0

            min_avg_vol = 200_000
            min_dollar_vol = 10_000_000
            if avg_vol20 < min_avg_vol:
                continue
            if dollar_vol20 < min_dollar_vol:
                continue
            if vol_rel20 < 0.8:
                continue

            if unusual_volume and vol_rel20 < 1.5:
                continue

            if gap_pct < min_gap:
                continue

            comp_gap = max(0.0, gap_pct) / 4.0
            comp_breakout = max(0.0, breakout_pos - 0.9) * 15.0
            comp_trend20 = max(0.0, trend20) / 6.0
            comp_trend10 = max(0.0, trend10) / 4.0
            comp_vol_rel = max(0.0, vol_rel20 - 1.0) * 3.0
            price_factor = np.clip(last_close / 20.0, 0.2, 1.5)
            dv_component = np.clip((np.log10(dollar_vol20 + 1) - 5.5), 0.0, 4.0)

            vol_penalty = np.clip((vol20_pct - 10.0) / 3.0, 0.0, 5.0)

            raw_score = (
                0.20 * comp_gap
                + 0.22 * comp_breakout
                + 0.18 * comp_trend20
                + 0.14 * comp_trend10
                + 0.14 * comp_vol_rel
                + 0.12 * dv_component
            )

            raw_score = raw_score * float(price_factor) - 0.15 * vol_penalty
            score = float(np.clip(raw_score * 10.0, 0.0, 100.0))

            pattern_tags = []
            if breakout_pos >= 0.98 and trend20 > 0 and gap_pct >= min_gap:
                pattern_tags.append("BreakoutHigh")
            if trend20 > 20 and vol_rel20 >= 1.3:
                pattern_tags.append("Momentum")
            if trend20 > 10 and vol20_pct <= 8:
                pattern_tags.append("SteadyClimb")
            if not pattern_tags and vol20_pct >= 20 and gap_pct >= 5:
                pattern_tags.append("HighVolRunner")
            if not pattern_tags:
                pattern_tags.append("Base/Neutral")
            pattern_tag = ",".join(pattern_tags)

            score_factors = []
            if comp_gap > 0:
                score_factors.append("Gap")
            if comp_breakout > 0:
                score_factors.append("NearHigh")
            if comp_trend20 > 0:
                score_factors.append("Trend20D")
            if comp_trend10 > 0:
                score_factors.append("Trend10D")
            if comp_vol_rel > 0:
                score_factors.append("Volume")
            if dv_component > 0:
                score_factors.append("Liquidity")
            if vol_penalty > 0:
                score_factors.append("VolPenalty")
            score_note = "+".join(score_factors) if score_factors else "Neutral mix"

            rows.append(
                {
                    "Ticker": sym,
                    "BreakoutScore": round(score, 2),
                    "Last": round(last_close, 2),
                    "Volume": int(last_vol),
                    "Gap%": round(gap_pct, 2),
                    "BreakoutPos20D": round(breakout_pos, 3),
                    "Trend20D%": round(trend20, 2),
                    "Trend10D%": round(trend10, 2),
                    "VolRel20": round(vol_rel20, 2),
                    "DollarVol20": round(dollar_vol20, 2),
                    "Volatility20D%": round(vol20_pct, 2),
                    "PatternTag": pattern_tag,
                    "ScoreNote": score_note,
                    "RS_Rank": np.nan,
                    "Premarket": premarket,
                    "AfterHours": afterhours,
                    "UnusualVol": unusual_volume and vol_rel20 >= 1.5,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    try:
        if "Trend20D%" in df.columns and len(df) > 1:
            df["RS_Rank"] = df["Trend20D%"].rank(pct=True) * 100.0
            df["RS_Rank"] = df["RS_Rank"].round(1)
    except Exception:
        pass

    df = df.sort_values("BreakoutScore", ascending=False).head(top_n).reset_index(
        drop=True
    )
    return df


def run_breakout_scan(
    tickers: List[str],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    """Public entry point for breakout scans.

    This wrapper intentionally delegates directly to the legacy
    `scan.breakout.run_breakout_scan` implementation so that we can
    rely on the previous, stable behaviour while the v2 engine and
    data plumbing are refined.
    """
    from . import breakout as legacy_breakout

    return legacy_breakout.run_breakout_scan(
        tickers,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )


@st.cache_data(ttl=600, show_spinner=False)
def cached_real_scan(
    tickers: Tuple[str, ...],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool,
) -> pd.DataFrame:
    """Cached wrapper around run_breakout_scan."""
    return run_breakout_scan(
        list(tickers),
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )