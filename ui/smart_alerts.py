"""Smart alert suggestions from current scan results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - pure helpers are tested without Streamlit
    st = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SmartAlertSuggestion:
    ticker: str
    alert_type: str
    label: str
    reason: str
    threshold: float | None = None
    direction: str | None = None


def build_smart_alert_suggestions(
    df: pd.DataFrame,
    *,
    max_suggestions: int = 5,
) -> list[SmartAlertSuggestion]:
    if df is None or df.empty:
        return []

    suggestions: list[SmartAlertSuggestion] = []
    seen: set[tuple[str, str, str | None]] = set()
    for row in _top_rows(df, limit=max(max_suggestions * 4, 12)):
        ticker = _ticker(row)
        if not ticker:
            continue
        for suggestion in _row_suggestions(row, ticker):
            key = (suggestion.ticker, suggestion.alert_type, suggestion.direction)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(suggestion)
            if len(suggestions) >= max_suggestions:
                return suggestions
    return suggestions


def render_smart_alert_suggestions(
    df: pd.DataFrame,
    *,
    key_prefix: str = "results",
) -> None:
    if st is None:
        return
    suggestions = build_smart_alert_suggestions(df)
    if not suggestions:
        return

    with st.expander("🔔 Smart alert suggestions", expanded=False):
        st.caption("One-click setup ideas from the current scan. Nothing is created until you confirm it on Alerts.")
        for idx, item in enumerate(suggestions):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{item.label}**  \n{item.reason}")
            if cols[1].button("Use", key=f"{key_prefix}_smart_alert_{idx}_{item.ticker}_{item.alert_type}"):
                _prefill_alert_form(item)
                try:
                    st.switch_page("pages/alerts.py")
                except (RuntimeError, ValueError):
                    st.caption("Open Alerts from the sidebar to finish creating it.")


def _top_rows(df: pd.DataFrame, *, limit: int) -> Iterable[dict[str, Any]]:
    frame = df.copy()
    sort_col = _first_existing(
        frame,
        ["AI Confidence", "PreBreakoutProb%", "PreBreakoutProb", "BreakoutScore"],
    )
    if sort_col:
        values = pd.to_numeric(frame[sort_col], errors="coerce")
        frame = frame.assign(_smart_sort=values.fillna(-1.0)).sort_values("_smart_sort", ascending=False)
    return frame.head(limit).to_dict(orient="records")


def _row_suggestions(row: dict[str, Any], ticker: str) -> list[SmartAlertSuggestion]:
    out: list[SmartAlertSuggestion] = []
    breakout_score = _num(row, "BreakoutScore", "Score")
    prebreakout_pct = _prob_pct(row, "PreBreakoutProb%", "PreBreakoutProb", "PreBreakout")
    rvol = _num(row, "RVOL", "VolRel20")
    ema_cross = str(row.get("EMA Cross") or row.get("ema_cross") or "").strip().lower()

    if breakout_score is not None and breakout_score >= 10:
        out.append(
            SmartAlertSuggestion(
                ticker=ticker,
                alert_type="breakout",
                label=f"{ticker}: breakout score alert",
                reason=f"Breakout score is elevated at {breakout_score:.1f}.",
                threshold=max(8.0, round(breakout_score * 0.9, 1)),
            )
        )

    if prebreakout_pct is not None and prebreakout_pct >= 60:
        out.append(
            SmartAlertSuggestion(
                ticker=ticker,
                alert_type="breakout",
                label=f"{ticker}: pre-breakout watch alert",
                reason=f"PreBreakout probability is {prebreakout_pct:.1f}%; watch for score confirmation.",
                threshold=8.0,
            )
        )

    if rvol is not None and rvol >= 1.5:
        out.append(
            SmartAlertSuggestion(
                ticker=ticker,
                alert_type="rvol",
                label=f"{ticker}: RVOL alert",
                reason=f"Relative volume is already {rvol:.2f}x average.",
                threshold=max(1.5, round(rvol, 1)),
            )
        )

    if ema_cross in {"golden", "golden cross"}:
        out.append(
            SmartAlertSuggestion(
                ticker=ticker,
                alert_type="ema_cross",
                label=f"{ticker}: Golden Cross alert",
                reason="EMA 9/21 is showing a bullish cross setup.",
                direction="bullish",
            )
        )
    elif ema_cross in {"death", "death cross"}:
        out.append(
            SmartAlertSuggestion(
                ticker=ticker,
                alert_type="ema_cross",
                label=f"{ticker}: Death Cross alert",
                reason="EMA 9/21 is showing a bearish cross setup.",
                direction="bearish",
            )
        )

    return out


def _prefill_alert_form(item: SmartAlertSuggestion) -> None:
    if item.alert_type == "breakout":
        st.session_state["alert_break_thr"] = float(item.threshold or 8.0)
        st.session_state["alert_break_wl"] = False
    elif item.alert_type == "rvol":
        st.session_state["alert_rvol_tk"] = item.ticker
        st.session_state["alert_rvol_thr"] = float(item.threshold or 2.0)
    elif item.alert_type == "ema_cross":
        st.session_state["alert_ema_tk"] = item.ticker
        st.session_state["alert_ema_dir"] = item.direction or "bullish"


def _ticker(row: dict[str, Any]) -> str:
    return str(row.get("Ticker") or row.get("Symbol") or row.get("ticker") or "").strip().upper()


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _num(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        try:
            value = row.get(name)
            if value is not None and not pd.isna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _prob_pct(row: dict[str, Any], *names: str) -> float | None:
    value = _num(row, *names)
    if value is None:
        return None
    return value * 100.0 if 0 <= value <= 1 else value
