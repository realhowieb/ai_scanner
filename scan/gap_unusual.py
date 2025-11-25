# gap_unusual.py
from __future__ import annotations
from typing import Dict, Tuple, Iterable, Optional
import math
import pandas as pd


def _get_last(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    try:
        val = series.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _find_session_price(df: pd.DataFrame, session: str) -> Tuple[Optional[float], str]:
    """
    Try to infer a session price for premarket/postmarket/regular.
    Returns (price, source).
    - For 'regular': prefer today's Open, else last Close.
    - For 'premarket'/'postmarket': try common column names; if not found, fall back to Open (regular) or Close.
    """
    session = (session or "regular").lower()

    # Normalize columns for case-insensitive lookup
    colmap = {c.lower(): c for c in df.columns}

    def pick_first(candidates: Iterable[str]) -> Optional[str]:
        for name in candidates:
            key = name.lower()
            if key in colmap:
                return colmap[key]
        # try fuzzy contains for pre/post market names
        for key_lower, orig in colmap.items():
            if any(k in key_lower for k in candidates):
                return orig
        return None

    # Common fallbacks
    last_close = _get_last(df.get("Close", pd.Series(dtype=float)))
    today_open = _get_last(df.get("Open", pd.Series(dtype=float)))

    if session == "regular":
        if today_open is not None and pd.notna(today_open):
            return today_open, "Open"
        return last_close, "Close"

    # Heuristics for pre-/post-market columns from various sources
    if session == "premarket":
        cand = pick_first([
            "premarket", "pre_market", "pre-market", "premarketprice", "pre_market_price", "pre", "preprice",
        ])
        if cand:
            v = _get_last(df[cand])
            if v is not None:
                return v, cand
        # Fallback to today's Open if available, else last Close
        if today_open is not None:
            return today_open, "Open(fallback)"
        return last_close, "Close(fallback)"

    if session == "postmarket":
        cand = pick_first([
            "postmarket", "post_market", "post-market", "postmarketprice", "post_price", "afterhours", "after_hours",
        ])
        if cand:
            v = _get_last(df[cand])
            if v is not None:
                return v, cand
        # Fallback: use last Close (end of regular)
        return last_close, "Close(fallback)"

    # Unknown -> treat as regular
    if today_open is not None:
        return today_open, "Open(fallback)"
    return last_close, "Close(fallback)"


def scan_gappers(
    price_data: Dict[str, pd.DataFrame],
    min_price: float,
    max_price: float,
    session: str = "premarket",   # "premarket" | "postmarket" | "regular"
) -> pd.DataFrame:
    """Compute gap % and $ versus previous close for each symbol.

    Expects each DataFrame to have at least a 'Close' column; 'Open' improves accuracy.
    If session-specific columns exist (e.g., premarket/postmarket), they'll be used.
    """
    rows = []
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        # Ensure chronological order
        try:
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
        except Exception:
            pass

        # Latest trade price (regular close) for price filter
        latest_close = _get_last(df.get("Close", pd.Series(dtype=float)))
        if latest_close is None or math.isnan(latest_close):
            continue
        if not (min_price <= latest_close <= max_price):
            continue

        # Previous close for gap baseline
        prev_close = None
        try:
            if df.shape[0] >= 2 and "Close" in df.columns:
                prev_close = float(df["Close"].iloc[-2])
        except Exception:
            prev_close = None

        if prev_close is None or prev_close == 0 or math.isnan(prev_close):
            continue

        session_price, source = _find_session_price(df, session)
        if session_price is None or math.isnan(session_price):
            # If we cannot compute a session price, skip
            continue

        gap_d = session_price - prev_close
        gap_pct = (gap_d / prev_close) * 100.0

        rows.append({
            "Symbol": symbol,
            "Price": latest_close,
            "PrevClose": prev_close,
            "SessionPrice": session_price,
            "Gap $": round(gap_d, 4),
            "Gap %": round(gap_pct, 4),
            "Session": session,
            "Source": source,
        })

    if not rows:
        return pd.DataFrame(columns=["Symbol", "Price", "PrevClose", "SessionPrice", "Gap $", "Gap %", "Session", "Source"]).sort_values(by="Gap %", ascending=False)

    out = pd.DataFrame(rows)
    out = out.sort_values(by="Gap %", ascending=False, kind="stable").reset_index(drop=True)
    return out


def scan_unusual_volume(
    price_data: Dict[str, pd.DataFrame],
    lookback_days: int = 20,
    min_price: float = 0.0,
    max_price: float = 10_000.0,
) -> pd.DataFrame:
    """Compute latest volume vs average over `lookback_days` (excluding the last row).

    Returns a table with Volume, AvgVolume, VolRatio, and Price.
    """
    rows = []
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
        except Exception:
            pass

        close = _get_last(df.get("Close", pd.Series(dtype=float)))
        if close is None or math.isnan(close) or not (min_price <= close <= max_price):
            continue

        if "Volume" not in df.columns or df["Volume"].empty:
            continue

        latest_vol = None
        try:
            latest_vol = float(df["Volume"].iloc[-1])
        except Exception:
            latest_vol = None

        if latest_vol is None or math.isnan(latest_vol):
            continue

        # Average over prior `lookback_days` (exclude the last row)
        start_idx = max(0, len(df) - (lookback_days + 1))
        end_idx = max(0, len(df) - 1)
        hist = df["Volume"].iloc[start_idx:end_idx]
        avg_vol = float(hist.mean()) if len(hist) > 0 else float("nan")
        if math.isnan(avg_vol) or avg_vol <= 0:
            continue

        vol_ratio = latest_vol / avg_vol

        rows.append({
            "Symbol": symbol,
            "Price": close,
            "LatestVolume": int(latest_vol),
            "AvgVolume": int(avg_vol),
            "VolRatio": round(vol_ratio, 4),
        })

    if not rows:
        return pd.DataFrame(columns=["Symbol", "Price", "LatestVolume", "AvgVolume", "VolRatio"]).sort_values(by="VolRatio", ascending=False)

    out = pd.DataFrame(rows)
    out = out.sort_values(by="VolRatio", ascending=False, kind="stable").reset_index(drop=True)
    return out


def combine_gap_unusual(
    gappers: pd.DataFrame,
    uv: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join gappers and unusual volume on Symbol and provide a combined score.

    If either input is empty, returns an empty DataFrame with expected columns.
    """
    if gappers is None or uv is None or gappers.empty or uv.empty:
        return pd.DataFrame(columns=[
            "Symbol", "Price", "PrevClose", "SessionPrice", "Gap $", "Gap %", "Session", "Source",
            "LatestVolume", "AvgVolume", "VolRatio", "Score"
        ])

    left = gappers.copy()
    right = uv.copy()

    # Ensure required columns exist
    required_g = {"Symbol", "Gap %"}
    required_u = {"Symbol", "VolRatio"}
    if not required_g.issubset(left.columns) or not required_u.issubset(right.columns):
        return pd.DataFrame(columns=[
            "Symbol", "Price", "PrevClose", "SessionPrice", "Gap $", "Gap %", "Session", "Source",
            "LatestVolume", "AvgVolume", "VolRatio", "Score"
        ])

    merged = pd.merge(left, right, on=["Symbol", "Price"], how="inner") if "Price" in left.columns and "Price" in right.columns else pd.merge(left, right, on="Symbol", how="inner")

    # Simple combined score: abs(Gap %) * VolRatio
    try:
        merged["Score"] = (merged["Gap %"].abs() * merged["VolRatio"]).round(4)
    except Exception:
        merged["Score"] = pd.NA

    cols_order = [
        "Symbol", "Price", "PrevClose", "SessionPrice", "Gap $", "Gap %", "Session", "Source",
        "LatestVolume", "AvgVolume", "VolRatio", "Score"
    ]
    for c in cols_order:
        if c not in merged.columns:
            merged[c] = pd.NA

    merged = merged[cols_order].sort_values(by=["Score", "VolRatio", "Gap %"], ascending=[False, False, False], kind="stable").reset_index(drop=True)
    return merged

def gap_unusual_volume_scanner(
    price_data: Dict[str, pd.DataFrame],
    *,
    min_gap_pct: float = 4.0,
    vol_window: int = 20,
    min_vol_mult: float = 2.0,
    session: str = "premarket",
    min_price: float = 0.0,
    max_price: float = 10_000.0,
    combine: bool = True,
):
    """Backward-compatible wrapper used by older UI code.

    It computes gappers and unusual volume, applies the legacy thresholds,
    and returns either the combined table (default) or the pair `(gappers, uv)`
    when `combine=False`.
    """
    # Compute raw tables
    gappers = scan_gappers(
        price_data,
        min_price=min_price,
        max_price=max_price,
        session=session,
    )
    uv = scan_unusual_volume(
        price_data,
        lookback_days=vol_window,
        min_price=min_price,
        max_price=max_price,
    )

    # Apply thresholds if present
    if gappers is None or gappers.empty:
        gappers_f = pd.DataFrame(columns=gappers.columns if gappers is not None else [
            "Symbol", "Price", "PrevClose", "SessionPrice", "Gap $", "Gap %", "Session", "Source"
        ])
    else:
        gappers_f = gappers[(gappers["Gap %"].abs() >= float(min_gap_pct))]

    if uv is None or uv.empty:
        uv_f = pd.DataFrame(columns=uv.columns if uv is not None else [
            "Symbol", "Price", "LatestVolume", "AvgVolume", "VolRatio"
        ])
    else:
        uv_f = uv[(uv["VolRatio"] >= float(min_vol_mult))]

    if not combine:
        return gappers_f.reset_index(drop=True), uv_f.reset_index(drop=True)

    merged = combine_gap_unusual(gappers_f, uv_f)
    return merged.reset_index(drop=True)