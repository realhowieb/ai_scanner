from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def _prep_symbol_df(df_sym: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sanity-check a single symbol's OHLCV DataFrame.

    - Strips/normalizes column names (e.g., 'close' -> 'Close').
    - Sorts by index.
    - Ensures we have at least Close and Volume columns.
    """
    if df_sym is None or not isinstance(df_sym, pd.DataFrame) or df_sym.empty:
        return pd.DataFrame()

    df = df_sym.copy()
    try:
        df = df.rename(columns=lambda c: str(c).strip().capitalize())
    except Exception:
        # If renaming fails, bail on this symbol.
        return pd.DataFrame()

    # Require at least Close + Volume to proceed.
    if "Close" not in df.columns or "Volume" not in df.columns:
        return pd.DataFrame()

    # Best-effort sort by index (date).
    try:
        df = df.sort_index()
    except Exception:
        pass

    return df


def _compute_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute core metrics for the latest bar of a symbol."""
    if df.empty or "Close" not in df.columns:
        return {}

    if len(df) < 2:
        # Not enough history to compute changes/gaps; treat as invalid.
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        close = float(last["Close"])
        prev_close = float(prev["Close"])
    except Exception:
        return {}

    if prev_close <= 0:
        pct_change = 0.0
        gap_pct = 0.0
    else:
        pct_change = (close / prev_close - 1.0) * 100.0
        if "Open" in df.columns:
            try:
                open_today = float(last["Open"])
                gap_pct = (open_today / prev_close - 1.0) * 100.0
            except Exception:
                gap_pct = 0.0
        else:
            gap_pct = 0.0

    # Volume & relative volume
    try:
        vol_today = float(last.get("Volume", float("nan")))
    except Exception:
        vol_today = float("nan")

    if "Volume" in df.columns and len(df) >= 21:
        try:
            vol_hist = df["Volume"].iloc[-21:-1]  # last 20 prior days
            vol_avg_20 = float(vol_hist.mean())
        except Exception:
            vol_avg_20 = float("nan")
    else:
        vol_avg_20 = float("nan")

    if vol_avg_20 and vol_avg_20 > 0:
        rel_vol = vol_today / vol_avg_20
    else:
        rel_vol = float("nan")

    # 20-day high for breakout context
    if "High" in df.columns and len(df) >= 20:
        try:
            high_20 = float(df["High"].iloc[-20:].max())
        except Exception:
            high_20 = close
    else:
        high_20 = close

    is_breakout = False
    try:
        if high_20 > 0 and close >= 0.999 * high_20:
            is_breakout = True
    except Exception:
        is_breakout = False

    return {
        "Close": close,
        "PrevClose": prev_close,
        "PctChange": pct_change,
        "GapPct": gap_pct,
        "Volume": vol_today,
        "VolAvg20": vol_avg_20,
        "RelVol": rel_vol,
        "High20": high_20,
        "IsBreakout": is_breakout,
    }


def _clip(x, lo: float, hi: float) -> float:
    """Clamp a value into [lo, hi]; NaN/invalid -> 0.0.

    Used to bound the percentage inputs to the breakout score so a single
    micro-cap with an extreme move (e.g. a 2,900% 20-day trend) can't blow the
    score up into the thousands and dominate the rankings.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(lo, min(hi, x))


def run_breakout_scan(
    price_data,
    spy_df,
    premarket,
    afterhours,
    unusual_volume,
    min_gap,
    min_price,
    max_price,
    top_n,
    diagnostics,
    progress_cb: Optional[Callable[[int, int, str], Any]] = None,
    heartbeat_every: int = 25,
    min_dollar_vol: float = 0.0,
):
    """Breakout scan with richer metrics.

    This implementation:
    - Normalizes each symbol's OHLCV data.
    - Computes basic breakout metrics (pct change, gap, 20-day high, rel volume).
    - Adds 10/20 day trend, breakout position vs 20-day high, 20-day volatility,
      and dollar volume.
    - Applies price filters, optional gap and unusual volume filters.
    - Uses SPY (if provided) to compute a simple relative-strength metric.
    - Ranks results by a composite breakout score and returns the top N rows.
    """
    # Optional: Streamlit diagnostics
    try:
        import streamlit as st  # type: ignore
    except Exception:
        st = None  # type: ignore[assignment]

    if not isinstance(price_data, dict) or not price_data:
        if st is not None and diagnostics:
            try:
                st.error("❌ run_breakout_scan: price_data is EMPTY – no OHLCV fetched.")
            except Exception:
                pass
        return pd.DataFrame()

    # Prepare SPY for relative strength, if available.
    spy_prepped = _prep_symbol_df(spy_df) if spy_df is not None else pd.DataFrame()
    spy_close = None
    if not spy_prepped.empty and "Close" in spy_prepped.columns:
        try:
            spy_close = float(spy_prepped["Close"].iloc[-1])
        except Exception:
            spy_close = None

    if st is not None and diagnostics:
        try:
            st.caption(
                f"📊 Breakout debug: incoming price_data has {len(price_data)} symbols. "
                f"Sample: {list(price_data.keys())[:10]}"
            )
        except Exception:
            pass

    attempted = 0
    added = 0
    skipped_price = 0
    skipped_liquidity = 0
    skipped_unusual = 0
    skipped_gap = 0
    skipped_invalid = 0

    # Accumulate per-symbol result rows
    rows: List[Dict[str, Any]] = []

    items = list(price_data.items())
    total = len(items)

    # Kick off progress immediately so UI can show 1/N even before work starts
    if progress_cb:
        try:
            progress_cb(0, total, "starting")
        except Exception:
            pass

    for i, (symbol, raw_df) in enumerate(items, start=1):
        attempted += 1

        # Optional heartbeat/progress callback (safe: never raises)
        if progress_cb and (i == 1 or i % heartbeat_every == 0 or i == total):
            try:
                progress_cb(i, total, str(symbol))
            except Exception:
                pass

        df = _prep_symbol_df(raw_df)
        if df.empty:
            skipped_invalid += 1
            continue

        metrics = _compute_basic_metrics(df)
        if not metrics:
            skipped_invalid += 1
            continue

        close = metrics["Close"]
        _ = metrics["PrevClose"]
        pct_change = metrics["PctChange"]
        gap_pct = metrics["GapPct"]
        vol_today = metrics["Volume"]
        vol_avg_20 = metrics["VolAvg20"]
        rel_vol = metrics["RelVol"]
        high_20 = metrics["High20"]
        is_breakout = metrics["IsBreakout"]

        # 1) Price filter
        if not (min_price <= close <= max_price):
            skipped_price += 1
            continue

        # 2) Gap filter: if min_gap > 0, require |gap| >= min_gap
        if min_gap is not None and min_gap > 0:
            if abs(gap_pct) < min_gap:
                skipped_gap += 1
                continue

        # 3) Unusual volume filter: require rel_vol >= 1.5 by default
        if unusual_volume:
            try:
                # NaN check + threshold
                if rel_vol != rel_vol or rel_vol < 1.5:
                    skipped_unusual += 1
                    continue
            except Exception:
                skipped_unusual += 1
                continue

        # --- Extended metrics ---

        # Trend over last 20 / 10 closes
        trend_20 = float("nan")
        trend_10 = float("nan")
        try:
            if len(df) >= 21:
                trend_20 = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1.0) * 100.0
            if len(df) >= 11:
                trend_10 = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-11]) - 1.0) * 100.0
        except Exception:
            trend_20 = float("nan")
            trend_10 = float("nan")

        # Position vs 20-day high (1.0 == at high)
        breakout_pos_20d = float("nan")
        try:
            if high_20 and high_20 > 0:
                breakout_pos_20d = close / high_20
        except Exception:
            breakout_pos_20d = float("nan")

        # Dollar volume using today's volume and last price, or 20-day avg
        dollar_vol_20 = float("nan")
        try:
            if vol_avg_20 and vol_avg_20 > 0:
                dollar_vol_20 = close * vol_avg_20
            elif vol_today and vol_today > 0:
                dollar_vol_20 = close * vol_today
        except Exception:
            dollar_vol_20 = float("nan")

        # 3b) Liquidity filter — exclude thin micro-caps (the pump/penny names
        # that otherwise dominate by score). Uses 20-day-avg dollar volume so a
        # stock that was quiet then spiked is correctly screened out.
        if min_dollar_vol and min_dollar_vol > 0:
            if not (dollar_vol_20 == dollar_vol_20) or dollar_vol_20 < min_dollar_vol:
                skipped_liquidity += 1
                continue

        # 20-day volatility (std dev of daily returns)
        vol_20d = float("nan")
        try:
            if len(df) >= 21:
                rets = df["Close"].pct_change().iloc[-20:]
                vol_20d = float(rets.std() * 100.0)
        except Exception:
            vol_20d = float("nan")

        # Relative strength vs SPY (simple ratio)
        rs = None
        try:
            if spy_close and spy_close > 0:
                rs = close / spy_close
        except Exception:
            rs = None

        # Pattern tag: basic labeling based on breakout / gap / volume
        pattern_tag = "Base/Neutral"
        try:
            if is_breakout and rel_vol and rel_vol >= 1.5 and gap_pct >= 0:
                pattern_tag = "BreakoutHigh"
            elif is_breakout and gap_pct >= 0:
                pattern_tag = "NearHigh"
            elif gap_pct <= -2.0:
                pattern_tag = "GapDown"
        except Exception:
            pattern_tag = "Base/Neutral"

        # Composite breakout score:
        #   - Favor strong short-term trend and % change
        #   - Reward being close to 20-day high
        #   - Reward higher relative volume and positive gap
        # Each percentage input is clipped to a sane band so extreme micro-cap
        # moves can't explode the score into the thousands. This keeps scores on
        # a bounded, intuitive range (roughly 0-100) where the threshold means
        # something, and stops pump/penny names from dominating purely on
        # magnitude.
        score = 0.0
        try:
            # Base on daily move (cap ±30%)
            score += _clip(pct_change, -30.0, 30.0)

            # Trend weighting (cap each trend at +50% before weighting)
            if trend_20 == trend_20 and trend_20 > 0:
                score += _clip(trend_20, 0.0, 50.0) * 0.4
            if trend_10 == trend_10 and trend_10 > 0:
                score += _clip(trend_10, 0.0, 50.0) * 0.6

            # Position vs high (closer to 1.0 is better)
            if breakout_pos_20d == breakout_pos_20d:
                score += (breakout_pos_20d - 0.9) * 50.0  # reward > 0.9

            # Gap contribution (cap ±30%)
            score += _clip(gap_pct, -30.0, 30.0) * 0.5

            # Volume contribution (cap rel_vol at 5x)
            if rel_vol and rel_vol == rel_vol:
                score += (_clip(rel_vol, 1.0, 5.0) - 1.0) * 10.0

            # Small bonus for confirmed breakout
            if is_breakout:
                score += 5.0
        except Exception:
            # Fall back to a bounded pct_change score on any error
            score = _clip(pct_change, -30.0, 30.0)

        rows.append(
            {
                "Ticker": symbol,
                "BreakoutScore": score,
                "Last": close,
                "Volume": vol_today,
                "GapPct": gap_pct,
                "BreakoutPos20D": breakout_pos_20d,
                "Trend20D%": trend_20,
                "Trend10D%": trend_10,
                "VolRel20": rel_vol,
                "DollarVol20": dollar_vol_20,
                "Volatility20D%": vol_20d,
                "PatternTag": pattern_tag,
                "RSvsSPY": rs,
                "IsBreakout": is_breakout,
            }
        )
        added += 1

    df_out = pd.DataFrame(rows)

    if st is not None and diagnostics:
        try:
            st.caption(
                f"🧪 Breakout summary: attempted={attempted}, added={added}, "
                f"skipped_price={skipped_price}, skipped_liquidity={skipped_liquidity}, "
                f"skipped_gap={skipped_gap}, "
                f"skipped_unusual={skipped_unusual}, skipped_invalid={skipped_invalid}"
            )
        except Exception:
            pass

    if df_out.empty:
        return df_out

    # Rank by composite breakout score and return top N
    try:
        df_out = df_out.sort_values("BreakoutScore", ascending=False).head(top_n)
    except Exception:
        # As a fallback, just limit the number of rows.
        df_out = df_out.head(top_n)

    # Final safe progress call so UI can reliably flip to complete
    if progress_cb:
        try:
            progress_cb(total, total, "done")
        except Exception:
            pass

    return df_out


def breakout_scanner(
    price_data,
    min_price: float = 5.0,
    max_price: float = 1000.0,
    *,
    include_ta: bool = False,
    spy_df=None,
    premarket: bool = False,
    afterhours: bool = False,
    unusual_volume: bool = False,
    min_gap: float = 0.0,
    top_n: int = 100,
    diagnostics: bool = False,
    progress_cb: Optional[Callable[[int, int, str], Any]] = None,
):
    """Backward-compatible wrapper around :func:`run_breakout_scan`.

    Older headless scan modules used ``breakout_scanner(price_data, min_price,
    max_price, include_ta=..., spy_df=...)``. Keep that path working while the
    app standardizes on ``run_breakout_scan``.
    """
    return run_breakout_scan(
        price_data=price_data,
        spy_df=spy_df,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
        progress_cb=progress_cb,
    )
