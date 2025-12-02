from __future__ import annotations

from typing import Dict, Any, List

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
):
    """Rebuilt breakout scan.

    This implementation:
    - Normalizes each symbol's OHLCV data.
    - Computes basic breakout metrics (pct change, gap, 20-day high, rel volume).
    - Applies price filters, optional gap and unusual volume filters.
    - Uses SPY (if provided) to compute a simple relative-strength metric.
    - Ranks results by a composite score and returns the top N rows.
    """
    # Optional: Streamlit diagnostics
    try:
        import streamlit as st  # type: ignore
    except Exception:
        st = None  # type: ignore[assignment]

    if not isinstance(price_data, dict) or not price_data:
        if st is not None:
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

    if st is not None:
        try:
            st.caption(
                f"📊 Breakout debug: incoming price_data has {len(price_data)} symbols. "
                f"Sample: {list(price_data.keys())[:10]}"
            )
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []

    attempted = 0
    added = 0
    skipped_price = 0
    skipped_unusual = 0
    skipped_gap = 0
    skipped_invalid = 0

    for symbol, raw_df in price_data.items():
        attempted += 1
        df = _prep_symbol_df(raw_df)
        if df.empty:
            skipped_invalid += 1
            continue

        metrics = _compute_basic_metrics(df)
        if not metrics:
            skipped_invalid += 1
            continue

        close = metrics["Close"]
        pct_change = metrics["PctChange"]
        gap_pct = metrics["GapPct"]
        vol_today = metrics["Volume"]
        rel_vol = metrics["RelVol"]
        high_20 = metrics["High20"]
        is_breakout = metrics["IsBreakout"]

        # 1) Price filter
        if not (min_price <= close <= max_price):
            skipped_price += 1
            continue

        # 2) Gap filter: if min_gap > 0, require gap >= min_gap
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

        # Relative strength vs SPY (simple ratio)
        rs = None
        try:
            if spy_close and spy_close > 0:
                rs = close / spy_close
        except Exception:
            rs = None

        # Composite breakout score:
        # - Favor strong % change
        # - Slightly favor bigger gaps
        # - Boost for unusual volume
        # - Boost if at/near 20-day high
        score = pct_change
        score += gap_pct * 0.5
        if rel_vol and rel_vol == rel_vol:  # not NaN
            score += (rel_vol - 1.0) * 10.0
        if is_breakout:
            score += 5.0

        rows.append(
            {
                "Ticker": symbol,
                "Last": close,
                "%Change": pct_change,
                "GapPct": gap_pct,
                "Volume": vol_today,
                "RelVol": rel_vol,
                "High20": high_20,
                "IsBreakout": is_breakout,
                "RSvsSPY": rs,
                "Score": score,
            }
        )
        added += 1

    df_out = pd.DataFrame(rows)

    if st is not None:
        try:
            st.caption(
                f"🧪 Breakout summary: attempted={attempted}, added={added}, "
                f"skipped_price={skipped_price}, skipped_gap={skipped_gap}, "
                f"skipped_unusual={skipped_unusual}, skipped_invalid={skipped_invalid}"
            )
        except Exception:
            pass

    if df_out.empty:
        return df_out

    # Rank by composite score and return top N
    try:
        df_out = df_out.sort_values("Score", ascending=False).head(top_n)
    except Exception:
        # As a fallback, just limit the number of rows.
        df_out = df_out.head(top_n)

    return df_out
