"""P2-26 — Day Trader "Stair-steppers" check (1-minute trend consistency).

On demand (a button, not every auto-refresh) it fetches recent 1-minute bars
for the symbols on the Day Trader page and lists the ones moving in a tight,
straight line: Linear Regression R² of close vs. time, the trend per hour and
the deepest pullback. Descriptive only; thresholds are the user's settings,
not claims, and nothing here touches HSF Score or ranking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

MAX_CHECK = 40              # symbols per check (keeps it to a few Alpaca pages)
RECENT_LOOKBACK_H = 4       # covers the live session incl. pre/post-market
FALLBACK_LOOKBACK_D = 4     # market closed: reach back to the last session


def _fetch(symbols: Tuple[str, ...], hours: float) -> Dict[str, list]:
    from data.price_alpaca import fetch_minute_bars_multi

    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return fetch_minute_bars_multi(list(symbols), start, max_pages=40)


def fetch_recent_minute_bars(symbols: Sequence[str]) -> Dict[str, list]:
    """1-minute bars for `symbols`: the last few hours, or the last session when
    the market is closed. Raises on provider errors (caller shows a message)."""
    syms = tuple(sorted({str(s).strip().upper() for s in symbols if str(s).strip()}))
    if not syms:
        return {}
    bars = _fetch(syms, RECENT_LOOKBACK_H)
    if not any(bars.values()):
        bars = _fetch(syms, FALLBACK_LOOKBACK_D * 24)
    return bars


if st is not None:
    _cached_bars = st.cache_data(ttl=120, show_spinner=False)(fetch_recent_minute_bars)
else:  # pragma: no cover
    _cached_bars = fetch_recent_minute_bars


def build_rows(bars_by_symbol: Dict[str, list], symbols: Sequence[str], window: int) -> List[Dict[str, Any]]:
    from analytics.stair_step import stair_step_metrics

    rows = []
    for sym in symbols:
        m = stair_step_metrics(bars_by_symbol.get(sym) or [], window=window)
        rows.append({"ticker": sym, **m})
    return rows


def _fmt_time(ts: Any) -> str:
    try:
        from zoneinfo import ZoneInfo

        return ts.astimezone(ZoneInfo("America/New_York")).strftime("%a %H:%M ET")
    except Exception:
        return "—"


def render_stair_steppers(symbols: Sequence[str]) -> None:
    """Collapsible Day Trader section. Never raises."""
    if st is None or not symbols:
        return
    try:
        from analytics.stair_step import DEFAULT_WINDOW, is_stair_stepper

        with st.expander("🪜 Stair-steppers: smooth 1-minute trends", expanded=False):
            st.caption(
                "Finds symbols moving in a tight, straight line on the 1-minute chart. "
                "**R²** is how closely the last bars follow a straight line (1.0 = perfectly "
                "straight). It describes the recent path; it is not a prediction.")
            c1, c2, c3 = st.columns(3)
            direction = c1.selectbox("Direction", ["up", "down", "either"], key="ss_direction")
            r2_min = c2.slider("Minimum R²", 0.50, 0.99, 0.80, 0.01, key="ss_r2")
            window_options = [10, 15, 20, 30, 45, 60]
            window = c3.selectbox("Bars fitted", window_options, index=window_options.index(DEFAULT_WINDOW),
                                  key="ss_window", help="Most recent 1-minute bars of the latest session.")
            c4, c5 = st.columns(2)
            max_pb = c4.number_input("Max pullback %", 0.1, 5.0, 1.0, 0.1, key="ss_pullback",
                                     help="Deepest dip against the trend within the window, from closes.")
            min_trend = c5.number_input("Min trend % per hour", 0.0, 20.0, 0.5, 0.1, key="ss_trend",
                                        help="Filters out straight but nearly flat lines.")
            checked = list(symbols)[:MAX_CHECK]
            label = f"Check {len(checked)} symbols"
            if len(symbols) > MAX_CHECK:
                st.caption(f"Checks the first {MAX_CHECK} of {len(symbols)} symbols on this page.")
            if st.button(label, key="ss_run"):
                st.session_state["ss_symbols"] = tuple(checked)
            if tuple(checked) != st.session_state.get("ss_symbols"):
                return
            try:
                with st.spinner("Loading 1-minute bars…"):
                    bars = _cached_bars(tuple(checked))
            except Exception:
                st.warning("Couldn't load 1-minute bars right now. Try again in a minute.")
                return
            rows = build_rows(bars, checked, int(window))
            hits = [r for r in rows if is_stair_stepper(
                r, r2_min=float(r2_min), direction=direction,
                max_pullback_pct=float(max_pb), min_trend_pct_per_hour=float(min_trend))]
            as_of = max((r["as_of"] for r in rows if r.get("as_of")), default=None)
            if as_of is not None:
                st.caption(f"Latest 1-minute bar: {_fmt_time(as_of)}.")
            if hits:
                import pandas as pd

                hits.sort(key=lambda r: r["r2"], reverse=True)
                st.dataframe(pd.DataFrame([{
                    "Ticker": r["ticker"], "R²": r["r2"],
                    "Trend %/hr": r["trend_pct_per_hour"],
                    "Max pullback %": r["max_pullback_pct"], "Bars": r["bars"],
                } for r in hits]), hide_index=True, width="stretch",
                    column_config={"R²": st.column_config.NumberColumn(format="%.3f")})
            else:
                st.info("No symbols match these settings right now.")
            thin = [r["ticker"] for r in rows if r["status"] != "ok"]
            if thin:
                st.caption(f"Not enough 1-minute data to judge ({len(thin)}): {', '.join(thin[:15])}"
                           + ("…" if len(thin) > 15 else "")
                           + ". Thinly traded names have gaps on the free data feed.")
    except Exception:
        st.caption("Stair-stepper check unavailable.")
