from __future__ import annotations

from typing import Any, List

import streamlit as st

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


WATCHLIST_VALUE_COLUMNS = (
    "PreBreakoutProb%",
    "AI Confidence",
    "BreakoutScore",
    "GapPct",
    "Trend10D%",
)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _symbol_key(row: Any) -> str:
    for col in ("Ticker", "Symbol", "ticker", "symbol"):
        try:
            value = row.get(col)
        except AttributeError:
            value = None
        if str(value or "").strip():
            return str(value).strip().upper()
    return ""


def _rows_by_symbol(results_df: Any) -> dict[str, Any]:
    if results_df is None or not hasattr(results_df, "iterrows"):
        return {}
    rows: dict[str, Any] = {}
    try:
        iterator = results_df.iterrows()
    except (AttributeError, TypeError, ValueError):
        return {}
    for _, row in iterator:
        sym = _symbol_key(row)
        if sym and sym not in rows:
            rows[sym] = row
    return rows


def classify_watchlist_signal(row: Any) -> tuple[str, str]:
    """Return a compact signal label and reason for a watchlist row."""
    pre = _to_float(row.get("PreBreakoutProb%") if hasattr(row, "get") else None)
    ai_conf = _to_float(row.get("AI Confidence") if hasattr(row, "get") else None)
    score = _to_float(row.get("BreakoutScore") if hasattr(row, "get") else None)
    is_breakout = _to_bool(row.get("IsBreakout")) if hasattr(row, "get") else False

    if is_breakout or (ai_conf is not None and ai_conf >= 70):
        return "Active breakout", "AI confidence or breakout flag is high."
    if pre is not None and pre >= 60:
        return "Heating up", "Pre-breakout probability is elevated."
    if score is not None and score >= 20:
        return "Strong setup", "Breakout score is elevated."
    if pre is not None and pre <= 5 and ai_conf is not None and ai_conf <= 5:
        return "Cooling down", "Both model signals are quiet."
    return "Watching", "No strong signal in the latest scan yet."


def summarize_watchlist_intelligence(tickers: List[str], results_df: Any) -> list[dict[str, Any]]:
    """Match watchlist symbols against the latest scan results."""
    watch = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not watch:
        return []
    rows_by_symbol = _rows_by_symbol(results_df)
    if not rows_by_symbol:
        return []

    out: list[dict[str, Any]] = []
    for sym in watch:
        row = rows_by_symbol.get(sym)
        if row is None:
            out.append(
                {
                    "Ticker": sym,
                    "Signal": "Not in latest scan",
                    "Reason": "Run a scan including this ticker.",
                    "PreBreakout": None,
                    "AI Confidence": None,
                    "Score": None,
                }
            )
            continue
        signal, reason = classify_watchlist_signal(row)
        out.append(
            {
                "Ticker": sym,
                "Signal": signal,
                "Reason": reason,
                "PreBreakout": _to_float(row.get("PreBreakoutProb%")),
                "AI Confidence": _to_float(row.get("AI Confidence")),
                "Score": _to_float(row.get("BreakoutScore")),
            }
        )
    priority = {
        "Active breakout": 0,
        "Heating up": 1,
        "Strong setup": 2,
        "Watching": 3,
        "Cooling down": 4,
        "Not in latest scan": 5,
    }
    return sorted(out, key=lambda item: (priority.get(str(item["Signal"]), 9), str(item["Ticker"])))


def _metric(row: Any, name: str) -> float | None:
    return _to_float(row.get(name)) if hasattr(row, "get") else None


def _delta(current: Any, previous: Any, name: str) -> float | None:
    current_value = _metric(current, name)
    previous_value = _metric(previous, name)
    if current_value is None or previous_value is None:
        return None
    value = current_value - previous_value
    return 0.0 if abs(value) < 0.05 else value


def _format_delta(value: float | None, suffix: str = "") -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{suffix}"


def _delta_display(value: float | None) -> str:
    """Δ-cell text: '—' for no/zero change, signed otherwise (avoids '+0.0' noise)."""
    if value is None:
        return ""
    if abs(value) < 0.05:
        return "—"
    return f"{value:+.1f}"


def _change_summary(pre_delta: float | None, ai_delta: float | None, score_delta: float | None) -> str:
    present = [d for d in (pre_delta, ai_delta, score_delta) if d is not None]
    if not present:
        return "No comparable model metrics."
    # All present deltas are effectively zero → say so once, don't recite "0.0".
    if all(abs(d) < 0.05 for d in present):
        return "No change since the previous saved scan."
    parts = []
    if pre_delta is not None and abs(pre_delta) >= 0.05:
        parts.append(f"PreBreakout {_format_delta(pre_delta, ' pts')}")
    if ai_delta is not None and abs(ai_delta) >= 0.05:
        parts.append(f"AI {_format_delta(ai_delta, ' pts')}")
    if score_delta is not None and abs(score_delta) >= 0.05:
        parts.append(f"Score {_format_delta(score_delta)}")
    return "; ".join(parts)


def _movement_status(pre_delta: float | None, ai_delta: float | None, score_delta: float | None) -> str:
    positive = [v for v in (pre_delta, ai_delta) if v is not None and v >= 15]
    negative = [v for v in (pre_delta, ai_delta) if v is not None and v <= -15]
    if positive or (score_delta is not None and score_delta >= 5):
        return "Heating up"
    if negative or (score_delta is not None and score_delta <= -5):
        return "Cooling down"
    return "Stable"


def summarize_watchlist_movers(
    tickers: List[str],
    current_df: Any,
    previous_df: Any,
) -> list[dict[str, Any]]:
    """Compare watchlist symbols in the current scan versus a prior scan."""
    watch = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not watch:
        return []

    current = _rows_by_symbol(current_df)
    previous = _rows_by_symbol(previous_df)
    if not current and not previous:
        return []

    rows: list[dict[str, Any]] = []
    for sym in watch:
        cur = current.get(sym)
        prev = previous.get(sym)
        if cur is None and prev is None:
            continue
        if cur is not None and prev is None:
            rows.append(
                {
                    "Ticker": sym,
                    "Move": "New in latest scan",
                    "Change": "This watchlist ticker newly passed the scan filters.",
                    "Pre Δ": None,
                    "AI Δ": None,
                    "Score Δ": None,
                }
            )
            continue
        if cur is None and prev is not None:
            rows.append(
                {
                    "Ticker": sym,
                    "Move": "Dropped out",
                    "Change": "This ticker was present before but is not in the latest scan.",
                    "Pre Δ": None,
                    "AI Δ": None,
                    "Score Δ": None,
                }
            )
            continue

        pre_delta = _delta(cur, prev, "PreBreakoutProb%")
        ai_delta = _delta(cur, prev, "AI Confidence")
        score_delta = _delta(cur, prev, "BreakoutScore")
        rows.append(
            {
                "Ticker": sym,
                "Move": _movement_status(pre_delta, ai_delta, score_delta),
                "Change": _change_summary(pre_delta, ai_delta, score_delta),
                "Pre Δ": pre_delta,
                "AI Δ": ai_delta,
                "Score Δ": score_delta,
            }
        )

    priority = {"Heating up": 0, "New in latest scan": 1, "Stable": 2, "Cooling down": 3, "Dropped out": 4}
    return sorted(rows, key=lambda item: (priority.get(str(item["Move"]), 9), str(item["Ticker"])))


def filter_watchlist_movers(
    movers: list[dict[str, Any]],
    *,
    include_stable: bool = False,
) -> list[dict[str, Any]]:
    """Hide low-signal stable rows unless the user asks for the full audit."""
    if include_stable:
        return list(movers)
    return [row for row in movers if row.get("Move") != "Stable"]


def _signature_by_symbol(results_df: Any, tickers: List[str]) -> dict[str, tuple[float | None, ...]]:
    rows = _rows_by_symbol(results_df)
    out: dict[str, tuple[float | None, ...]] = {}
    for sym in tickers:
        row = rows.get(sym)
        if row is not None:
            out[sym] = tuple(_metric(row, col) for col in WATCHLIST_VALUE_COLUMNS)
    return out


def _load_previous_results_for_watchlist(tickers: List[str], current_df: Any) -> Any:
    """Best-effort load of the most recent different saved run for comparison."""
    watch = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not watch:
        return None
    current_signature = _signature_by_symbol(current_df, watch)
    username = st.session_state.get("username") or None
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df

        for run in list_runs(limit=8, include_snapshots=False, username=username):
            run_id = run.get("id") if isinstance(run, dict) else None
            if run_id is None:
                continue
            candidate = normalize_results_to_df(load_run_results(int(run_id)))
            if candidate is None:
                continue
            candidate_signature = _signature_by_symbol(candidate, watch)
            if not candidate_signature:
                continue
            if candidate_signature == current_signature:
                continue
            return candidate
    except (RuntimeError, TypeError, ValueError, OSError, ImportError, AttributeError):
        return None
    return None


def _dataframe(rows: list[dict[str, Any]]) -> Any:
    frame_mod = pd
    if frame_mod is None:
        import pandas as frame_mod  # type: ignore
    return frame_mod.DataFrame(rows)


def render_watchlist_intelligence(tickers: List[str]) -> None:
    df = st.session_state.get("results_df")
    rows = summarize_watchlist_intelligence(tickers, df)
    if not rows:
        return

    previous_df = _load_previous_results_for_watchlist(tickers, df)
    movers = summarize_watchlist_movers(tickers, df, previous_df)

    with st.expander("Watchlist intelligence 2.0", expanded=False):
        hot = sum(1 for row in rows if row["Signal"] in {"Active breakout", "Heating up", "Strong setup"})
        cold = sum(1 for row in rows if row["Signal"] == "Cooling down")
        heating = sum(1 for row in movers if row["Move"] in {"Heating up", "New in latest scan"})
        cooling = sum(1 for row in movers if row["Move"] in {"Cooling down", "Dropped out"})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actionable", hot)
        c2.metric("Cooling", cold)
        c3.metric("Heating since last", heating)
        c4.metric("Fading since last", cooling)

        st.markdown("##### Now")
        try:
            st.dataframe(
                _dataframe(rows),
                width="stretch",
                hide_index=True,
                key="watchlist_intelligence_now",
                column_config={
                    "PreBreakout": st.column_config.NumberColumn("PreBreakout", format="%.1f%%"),
                    "AI Confidence": st.column_config.NumberColumn(format="%.1f%%"),
                    "Score": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        except (RuntimeError, TypeError, ValueError, OSError, ImportError, AttributeError):
            st.write(rows)

        st.markdown("##### Since previous saved scan")
        include_stable = st.checkbox(
            "Show stable rows",
            value=False,
            key="watchlist_intelligence_show_stable",
        )
        visible_movers = filter_watchlist_movers(movers, include_stable=include_stable)
        if visible_movers:
            # Render Δ as text so a zero shows as '—' rather than a noisy '+0.0';
            # NumberColumn's printf format can't suppress the sign on zero.
            display_movers = []
            for row in visible_movers:
                out = dict(row)
                out["Pre Δ"] = _delta_display(row.get("Pre Δ"))
                out["AI Δ"] = _delta_display(row.get("AI Δ"))
                out["Score Δ"] = _delta_display(row.get("Score Δ"))
                display_movers.append(out)
            try:
                st.dataframe(
                    _dataframe(display_movers),
                    width="stretch",
                    hide_index=True,
                    key="watchlist_intelligence_movers",
                )
            except (RuntimeError, TypeError, ValueError, OSError, ImportError, AttributeError):
                st.write(visible_movers)
        elif movers:
            st.caption("No meaningful movers since the previous saved scan.")
        else:
            st.caption("Run and save another scan to see what is heating up or cooling off.")
