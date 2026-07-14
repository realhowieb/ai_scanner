"""Small helpers shared by result rendering UI."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st

# Don't surface a track record until it's statistically meaningful — a handful
# of snapshots over one week is noise and can misrepresent the signal in either
# direction. Require a real sample before showing anything to users.
TRACK_RECORD_MIN_SAMPLE = 150
TRACK_RECORD_MIN_RUNS = 8


@st.cache_data(ttl=600, show_spinner=False)
def _cached_track_record(horizon_days: int = 5):
    from db.track_record import load_latest_track_record

    return load_latest_track_record(horizon_days=horizon_days)


def render_track_record_badge() -> None:
    """Show the latest signal track record (forward-return performance).

    Cached 10 min: this renders on every rerun and each uncached read opens a
    fresh Neon connection.
    """
    try:
        tr = _cached_track_record(5)
    except Exception:
        tr = None
    if not tr or not tr.get("sample_size"):
        return
    if int(tr.get("sample_size") or 0) < TRACK_RECORD_MIN_SAMPLE or int(
        tr.get("runs_used") or 0
    ) < TRACK_RECORD_MIN_RUNS:
        return
    avg = tr.get("avg_return")
    win = tr.get("win_rate")
    n = tr.get("sample_size")
    h = tr.get("horizon_days", 5)
    bench = tr.get("benchmark") or "SPY"
    top_n = tr.get("top_n") or 5
    if avg is None or win is None:
        return
    # Hero number + per-horizon tiles: this is the trust asset, give it
    # visual weight (st.metric polarity arrows; no custom colors needed).
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric(
        f"vs {bench} · {h}d (top-{top_n})",
        f"{avg:+.1%}",
        delta=f"{win:.0%} beat the benchmark",
        delta_color="normal" if avg >= 0 else "inverse",
    )
    for col, horizon in ((hc2, 1), (hc3, 20)):
        try:
            other = _cached_track_record(horizon)
        except Exception:
            other = None
        if other and other.get("avg_return") is not None and int(other.get("sample_size") or 0) >= TRACK_RECORD_MIN_SAMPLE:
            col.metric(f"vs {bench} · {horizon}d", f"{other['avg_return']:+.1%}")
    st.caption(
        f"📈 Backtested on saved snapshots (n={n}) — past performance is not "
        "indicative of future results."
    )

# Columns that are model internals / duplicates — hidden from the results grid.
RESULTS_HIDDEN_COLUMNS = ("PreBreakoutProb", "BreakoutPos20D", "IsBreakout")


def results_column_config() -> dict:
    """Number formats for the fast-mode results grid.

    Rendered by the data grid itself (st.column_config), so fast mode keeps its
    speed while losing the raw 10-decimal floats.
    """
    cc = st.column_config
    return {
        "BreakoutScore": cc.NumberColumn("Score", format="%.1f"),
        "Last": cc.NumberColumn(format="%.2f"),
        "Volume": cc.NumberColumn(format="localized"),
        "GapPct": cc.NumberColumn("Gap %", format="%.2f%%"),
        "Trend10D%": cc.NumberColumn("Trend 10d", format="%.1f%%"),
        "Trend20D%": cc.NumberColumn("Trend 20d", format="%.1f%%"),
        "Volatility20D%": cc.NumberColumn("Vol 20d", format="%.1f%%"),
        "VolRel20": cc.NumberColumn("RVOL", format="%.2fx"),
        "DollarVol20": cc.NumberColumn("$Vol 20d", format="compact"),
        "RSvsSPY": cc.NumberColumn("RS vs SPY", format="%.2f"),
        "PreBreakoutProb%": cc.NumberColumn("PreBreakout", format="%.1f%%"),
        "AI Confidence": cc.NumberColumn(format="%.1f%%"),
    }


YF_DISABLED_KEY = "yf_disabled"
YF_DISABLED_REASON_KEY = "yf_disabled_reason"
YF_WARNED_KEY = "yf_disabled_warned"


def quiet_provider_loggers() -> None:
    """Reduce noisy provider logs without hiding app exceptions."""
    for name in ("yfinance", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.ERROR)


def is_yahoo_crumb_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    msg_l = msg.lower()
    if "invalid crumb" in msg_l:
        return True
    return "http error 401" in msg_l or ("401" in msg_l and "unauthorized" in msg_l)


def disable_yfinance_for_session(reason: str) -> None:
    st.session_state[YF_DISABLED_KEY] = True
    st.session_state[YF_DISABLED_REASON_KEY] = reason


def warn_yfinance_disabled_once() -> None:
    if st.session_state.get(YF_WARNED_KEY):
        return
    if not st.session_state.get(YF_DISABLED_KEY):
        return

    st.session_state[YF_WARNED_KEY] = True
    reason = st.session_state.get(YF_DISABLED_REASON_KEY) or "Yahoo Finance blocked the request (401)."
    st.caption(f"⚠️ Live quotes temporarily disabled this session: {reason}")


def get_results_df() -> Optional[pd.DataFrame]:
    """Return the current results DataFrame from session_state."""
    return st.session_state.get("results_df")


def ticker_column(df: pd.DataFrame) -> str | None:
    if "Ticker" in df.columns:
        return "Ticker"
    if "Symbol" in df.columns:
        return "Symbol"
    return None


def sync_selected_ticker_from_table(
    selection_obj: object,
    df: pd.DataFrame,
    picker_key: str,
    *,
    selected_key: str = "results_selected_ticker",
) -> None:
    """Sync Streamlit dataframe row selection into ticker picker state."""
    rows = getattr(getattr(selection_obj, "selection", None), "rows", None)
    if not rows:
        return

    try:
        idx = int(rows[0])
    except (TypeError, ValueError, IndexError):
        return

    if idx < 0 or idx >= len(df):
        return

    col = ticker_column(df)
    if not col:
        return

    ticker = str(df.iloc[idx][col]).strip().upper()
    if not ticker:
        return

    st.session_state[selected_key] = ticker
    st.session_state[picker_key] = ticker


def find_row_for_ticker(df: Optional[pd.DataFrame], ticker: str | None) -> Optional[pd.Series]:
    """Find the first row in df matching ticker."""
    if df is None or df.empty or not ticker:
        return None

    normalized_ticker = str(ticker).strip().upper()
    if not normalized_ticker:
        return None

    col = ticker_column(df)
    if not col:
        return None

    matches = df[col].astype(str).str.strip().str.upper() == normalized_ticker
    if matches.any():
        return df[matches].iloc[0]
    return None


def auto_details_ticker(df: pd.DataFrame) -> str | None:
    """Choose the best ticker to show to Basic users."""
    if df is None or df.empty or "Ticker" not in df.columns:
        return None

    if "BreakoutScore" in df.columns:
        scores = pd.to_numeric(df["BreakoutScore"], errors="coerce")
        if scores.notna().any():
            idx = int(scores.fillna(-1e18).idxmax())
            ticker = str(df.loc[idx, "Ticker"]).strip().upper()
            return ticker or None

    ticker = str(df.iloc[0]["Ticker"]).strip().upper()
    return ticker or None


def as_optional_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def row_to_jsonable_dict(row: pd.Series) -> dict[str, object]:
    return {
        key: (None if isinstance(value, float) and pd.isna(value) else value)
        for key, value in row.to_dict().items()
    }
