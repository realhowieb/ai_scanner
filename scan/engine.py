


# --- Premium Breakout v2 Engine + utilities (merged) ---
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


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


def _override_last_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Override df['Last'] with live-ish last trade prices."""
    if yf is None or df is None or df.empty or "Ticker" not in df.columns:
        return df
    last_map = {}
    for t in df["Ticker"].astype(str).tolist():
        try:
            tk = yf.Ticker(t)
            price = None
            try:
                fi = getattr(tk, "fast_info", {}) or {}
                price = fi.get("last_price")
            except Exception:
                price = None
            if price is None:
                try:
                    info = tk.info or {}
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
                except Exception:
                    price = None
            if price is not None and np.isfinite(price):
                last_map[t] = float(price)
        except Exception:
            continue
    if last_map:
        out = df.copy()
        if "Last" not in out.columns:
            out["Last"] = np.nan
        out["Last"] = out["Ticker"].map(last_map).fillna(out["Last"])
        return out
    return df


def safe_yf_download(
    tickers: List[str],
    *,
    period: str = "1mo",
    interval: str = "1d",
    group_by: str = "ticker",
) -> pd.DataFrame:
    """Batch yfinance.download with retries and per-ticker fallback.

    Behaviour:
      - Uses a single batched yfinance.download call (fast path).
      - If the batch fails entirely, retries via safe_call and returns an empty
        DataFrame on total failure.
      - If the batch succeeds but some tickers are missing from the MultiIndex
        columns, it will retry those missing tickers one-by-one and attempt to
        merge them into the batch result.
    """
    if yf is None or not tickers:
        return pd.DataFrame()

    clean_tickers = [t for t in tickers if isinstance(t, str) and t]
    if not clean_tickers:
        return pd.DataFrame()

    tickers_str = " ".join(clean_tickers)

    def _download_batch():
        return yf.download(
            tickers=tickers_str,
            period=period,
            interval=interval,
            group_by=group_by,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    try:
        data = safe_call(_download_batch, label=f"yfinance batch ({len(clean_tickers)} symbols)")
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    if not isinstance(data.columns, pd.MultiIndex):
        return data

    present = {str(c[0]) for c in data.columns}
    missing = [t for t in clean_tickers if t not in present]

    if not missing:
        return data

    for sym in missing:
        try:
            def _download_single():
                return yf.download(
                    tickers=sym,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )

            single = safe_call(_download_single, label=f"yfinance retry {sym}")
            if single is None or single.empty:
                continue

            if not isinstance(single.index, type(data.index)):
                data = data.join(single, how="outer", rsuffix=f"_{sym}")
                continue

            for col in single.columns:
                data[(sym, col)] = single[col]
        except Exception as e:
            try:
                st.caption(f"⚠️ Retry for {sym} failed: {e}")
            except Exception:
                pass
            continue

    return data


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
    """Premium Breakout v2.1 Engine (Balanced mode)."""
    if yf is None or not tickers:
        rows = []
        for t in tickers:
            price = float(np.random.uniform(min_price, max_price))
            vol = int(np.random.randint(300_000, 20_000_000))
            gap = float(np.random.uniform(min_gap, min_gap + 10))
            score = float(np.random.uniform(0, 100))
            rows.append(
                {
                    "Ticker": t,
                    "BreakoutScore": round(score, 2),
                    "Last": round(price, 2),
                    "Volume": vol,
                    "Gap%": round(gap, 2),
                    "BreakoutPos20D": np.nan,
                    "Trend20D%": np.nan,
                    "Trend10D%": np.nan,
                    "VolRel20": np.nan,
                    "DollarVol20": np.nan,
                    "Volatility20D%": np.nan,
                    "RS_Rank": np.nan,
                    "PatternTag": "Stub",
                    "ScoreNote": "Stub/no yfinance.",
                    "Premarket": premarket,
                    "AfterHours": afterhours,
                    "UnusualVol": False,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df[df["Last"].between(min_price, max_price)]
        df = df[df["Gap%"] >= min_gap]
        df = df.sort_values("BreakoutScore", ascending=False).head(top_n).reset_index(
            drop=True
        )
        return df

    batch = safe_yf_download(tickers, period="1mo", interval="1d", group_by="ticker")
    if batch is None or batch.empty:
        return pd.DataFrame()

    rows: List[Dict] = []

    def _get_series(sym: str, field: str) -> Optional[pd.Series]:
        if batch is None or batch.empty:
            return None
        if isinstance(batch.columns, pd.MultiIndex):
            try:
                return batch[(sym, field)].dropna()
            except Exception:
                try:
                    return batch[(field, sym)].dropna()
                except Exception:
                    return None
        try:
            return batch[field].dropna()
        except Exception:
            return None

    for sym in tickers:
        try:
            close = _get_series(sym, "Close")
            high = _get_series(sym, "High")
            vol = _get_series(sym, "Volume")
            if close is None or high is None or vol is None:
                continue
            if len(close) < 5 or len(vol) < 5:
                continue

            close = close.dropna()
            high = high.dropna()
            vol = vol.dropna()
            if close.empty or high.empty or vol.empty:
                continue

            last_close = float(close.iloc[-1])
            if last_close <= 0:
                continue

            if not (min_price <= last_close <= max_price):
                continue

            prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
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
            last_vol = float(vol.iloc[-1])
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
    """Wrapper that prefers v2 engine but can fall back to any legacy engine."""
    try:
        return run_breakout_scan_v2(
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
    except Exception:
        # Fallback: try a legacy engine in scan.breakout if present
        try:
            from . import breakout as legacy_breakout

            if hasattr(legacy_breakout, "run_breakout_scan"):
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
        except Exception:
            pass
        raise


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