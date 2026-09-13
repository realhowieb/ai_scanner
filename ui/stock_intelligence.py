"""HSF Stock Intelligence — the canonical single-ticker deep dive.

Composes EXISTING HSF intelligence into one stock story: the canonical HSF
Opportunity Score + components, the deterministic why-it-ranked explanation and
risk flags, canonical movement, the opportunity lifecycle reconstructed from
frozen history, historical calibration context, model metrics, market regime,
and earnings. It NEVER re-implements the score, invents facts, or writes data.

build_stock_intelligence(...) is pure (all data injected) and testable;
render_stock_intelligence(...) fetches the read-only inputs and renders. Opening
the page is read-only — no freezes, snapshots, outcome writes, or Claude calls.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_STATUS_RANK = {"CAUTION": 1, "WATCH": 2, "STRONG": 3}
_SETUP_LABELS = {
    "golden_cross": "Golden Cross", "breakout": "Breakout", "prebreakout": "PreBreakout",
    "gapper": "Gapper", "gainer": "Momentum",
}


def _is_watched(ticker: str) -> bool:
    if st is None:
        return False
    active = {
        str(t).strip().upper()
        for t in (st.session_state.get("active_watchlist_tickers") or [])
        if str(t).strip()
    }
    return str(ticker or "").strip().upper() in active


def _watch_label(ticker: str) -> str:
    return "★ Watching" if _is_watched(ticker) else "☆ Watch"


def _rank(s: Optional[str]) -> int:
    return _STATUS_RANK.get(str(s or "").upper(), 0)


def build_current_opportunity(ticker: str, current_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Score the ticker's CURRENT state from a scanner row via the CANONICAL
    scorer (reusing the scanner adapter). None when there's no usable row."""
    if not current_row:
        return None
    from ui.opportunities import (
        _POSITIVE_SIGNALS,
        HSF_SCORE_VERSION,
        _primary_setup,
        _status,
        score_breakdown,
    )
    from ui.results_intelligence import _row_to_signal_fields

    f = _row_to_signal_fields(current_row)
    pos = [s for s in _POSITIVE_SIGNALS if s in f["signals"]]
    if len(pos) < 2 and f["breakout_score"] is None and f["prob"] is None:
        return None
    bd = score_breakdown(n_signals=len(pos), breakout_score=f["breakout_score"],
                         prob=f["prob"], chg_pct=f["chg_pct"], fading=f["fading"])
    return {
        "ticker": str(ticker).upper(), "score": bd["score"], "score_version": HSF_SCORE_VERSION,
        "score_components": {k: bd[k] for k in
                             ("signals_component", "model_component", "momentum_component", "fading_penalty")},
        "primary_setup": _primary_setup(pos, f["fading"]), "n_signals": len(pos),
        "signals": pos, "fading": f["fading"], "status": _status(bd["score"], f["fading"]),
        "breakout_score": f["breakout_score"], "prob": f["prob"],
        "chg_pct": f["chg_pct"], "gap_pct": f["gap_pct"], "last": f.get("last"), "rvol": f.get("rvol"),
    }


def reconstruct_lifecycle(history: Optional[List[Dict[str, Any]]], *, min_delta: int = 3) -> List[Dict[str, Any]]:
    """Deterministic lifecycle from frozen HSF observations (never fabricated).

    Each history item: {time, score, status, score_version, signals}. Sorted by
    time; the first is 'HSF history begins' (we never claim it was the true first
    qualifying moment). Version-incompatible steps are VERSION_CHANGED with no
    numeric delta. Reports signal additions/removals between steps.
    """
    rows = [h for h in (history or []) if h.get("score") is not None and h.get("time") is not None]
    rows = sorted(rows, key=lambda h: h["time"])
    events: List[Dict[str, Any]] = []
    prev = None
    for i, h in enumerate(rows):
        sigs = set(h.get("signals") or [])
        ev = {"time": h["time"], "score": h["score"], "status": h.get("status"),
              "score_version": h.get("score_version"), "signals": sorted(sigs),
              "first": i == 0, "movement": None, "transition": None,
              "signals_added": [], "signals_removed": []}
        if i == 0:
            ev["label"] = "HSF history begins"
        else:
            pv = prev.get("score_version")
            cv = h.get("score_version")
            if pv is not None and cv is not None and str(pv) != str(cv):
                ev["movement"] = "VERSION_CHANGED"
                ev["label"] = "Score version changed"
            else:
                delta = int(round(h["score"] - prev["score"]))
                ev["delta"] = delta
                ev["movement"] = ("RISING" if delta >= min_delta
                                  else "FALLING" if delta <= -min_delta else "UNCHANGED")
                if prev.get("status") and prev["status"] != h.get("status"):
                    ev["transition"] = (prev["status"], h.get("status"))
                psig = set(prev.get("signals") or [])
                ev["signals_added"] = sorted(sigs - psig)
                ev["signals_removed"] = sorted(psig - sigs)
                if ev["transition"]:
                    ev["label"] = f"{ev['transition'][0]} → {ev['transition'][1]}"
                elif ev["movement"] == "RISING":
                    ev["label"] = f"▲ RISING +{delta}"
                elif ev["movement"] == "FALLING":
                    ev["label"] = f"▼ FALLING {delta}"
                else:
                    ev["label"] = "UNCHANGED"
        events.append(ev)
        prev = h
    return events


def _watch_next(status: Optional[str], signals: List[str], movement: Optional[str],
                fading: bool, earnings_days: Optional[int]) -> List[str]:
    """Deterministic conditions to watch — no price/level/entry/exit language."""
    items: List[str] = []
    if "breakout" in signals:
        items.append("Whether the breakout status remains active")
    if "golden_cross" in signals:
        items.append("Whether the EMA 9/21 golden cross stays confirmed")
    if status == "WATCH":
        items.append("Whether the HSF Score strengthens into STRONG")
    if status == "STRONG":
        items.append("Whether the confirming signals persist")
    if fading or movement == "FALLING":
        items.append("Whether fading momentum continues to weaken the setup")
    if earnings_days is not None and 0 <= earnings_days <= 5:
        when = "today" if earnings_days == 0 else ("tomorrow" if earnings_days == 1 else f"in {earnings_days} days")
        items.append(f"Earnings {when} may change the setup")
    if not items:
        items.append("Whether new confirming signals appear")
    return items[:5]


def build_stock_intelligence(
    ticker: str,
    *,
    current_opp: Optional[Dict[str, Any]] = None,
    current_row: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    regime: Optional[str] = None,
    calibration_records: Optional[List[Dict[str, Any]]] = None,
    earnings_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Pure normalized Stock Intelligence object. All inputs injected (the
    renderer fetches them). Safe on missing/partial data — fields are None when
    unavailable, never fabricated.

    Current-state precedence: a pre-built `current_opp` (already canonically
    scored by Scanner/Brief) is used ONLY when its ticker matches this one — a
    hard state-isolation guard so a stale row from another ticker can never leak
    in. Otherwise a `current_row` is scored canonically; otherwise the latest
    frozen observation is used as a from_history fallback.
    """
    ticker = str(ticker or "").strip().upper()
    hist = list(history or [])
    current = None
    if current_opp and str(current_opp.get("ticker") or "").strip().upper() == ticker:
        current = dict(current_opp)  # copy so we never mutate the caller's object
    if current is None:
        current = build_current_opportunity(ticker, current_row)

    # If there's no live scanner row, fall back to the latest frozen observation
    # for the current HSF state (no live price, but real score/status/signals).
    if current is None and hist:
        latest = sorted(hist, key=lambda h: h["time"])[-1]
        if latest.get("score") is not None:
            current = {
                "ticker": ticker, "score": latest["score"], "status": latest.get("status"),
                "score_version": latest.get("score_version"), "signals": list(latest.get("signals") or []),
                "n_signals": len(latest.get("signals") or []), "primary_setup": None,
                "fading": "fading" in (latest.get("signals") or []),
                "breakout_score": None, "prob": None, "chg_pct": None, "last": None,
                "score_components": None, "from_history": True,
            }

    # Movement vs the prior frozen observation (canonical, version-safe).
    movement = None
    if current is not None:
        prev_rows = None
        earlier = [h for h in sorted(hist, key=lambda h: h["time"]) if h.get("score") is not None]
        if earlier:
            p = earlier[-1] if current.get("from_history") and len(earlier) >= 2 else earlier[-1]
            # exclude the latest if it's the same record we used as 'current'
            if current.get("from_history") and len(earlier) >= 2:
                p = earlier[-2]
            prev_rows = [{"ticker": ticker, "score": p["score"], "status": p.get("status"),
                          "score_version": p.get("score_version")}]
        from ui.opportunities import compare_opportunities
        movement = compare_opportunities([current], prev_rows)[0]

    # Explanation + risks (deterministic, canonical).
    reasons: List[str] = []
    risks: List[str] = []
    if current is not None:
        from ui.opportunities import build_opportunity_explanation
        ex = build_opportunity_explanation(
            current, earnings_today=[ticker] if earnings_days == 0 else [])
        reasons, risks = ex["reasons"], ex["risks"]
    if current and current.get("status") == "CAUTION" and "CAUTION status" not in risks:
        risks.append("CAUTION status")
    if earnings_days is not None and 1 <= earnings_days <= 5 and current is not None:
        risks.append(f"Earnings in {earnings_days} day(s)")

    # Lifecycle + history summary.
    lifecycle = reconstruct_lifecycle(hist)
    matured = [h for h in hist if h.get("matured")]
    positive = sum(1 for h in matured if h.get("mfe_5d") is not None and float(h["mfe_5d"]) >= 0.04)
    history_summary = {
        "observations": len(hist), "matured": len(matured), "positive": positive,
    } if hist else None

    # Historical calibration context for the current bucket.
    hist_ctx = None
    if current is not None and current.get("score") is not None and calibration_records is not None:
        try:
            from analytics.hsf_calibration import historical_context
            hist_ctx = historical_context(calibration_records, current["score"])
        except Exception:
            hist_ctx = None

    signals = current.get("signals") if current else []
    watch_next = _watch_next(
        current.get("status") if current else None, signals or [],
        movement.get("movement_state") if movement else None,
        bool(current.get("fading")) if current else False, earnings_days)

    return {
        "ticker": ticker,
        "has_opportunity": current is not None,
        "price": current.get("last") if current else None,
        "change_pct": current.get("chg_pct") if current else None,
        "hsf_score": current.get("score") if current else None,
        "score_version": current.get("score_version") if current else None,
        "score_components": current.get("score_components") if current else None,
        "status": current.get("status") if current else None,
        "movement": movement,
        "primary_setup": current.get("primary_setup") if current else None,
        "signals": signals or [],
        "n_signals": current.get("n_signals") if current else 0,
        "reasons": reasons,
        "risks": risks,
        "model": {"breakout_score": current.get("breakout_score") if current else None,
                  "prebreakout_prob": current.get("prob") if current else None},
        "market_regime": regime,
        "earnings_days": earnings_days,
        "historical_context": hist_ctx,
        "outcome_cohort": _outcome_cohort(current),
        "history_summary": history_summary,
        "lifecycle": lifecycle,
        "watch_next": watch_next,
        "from_history": bool(current.get("from_history")) if current else False,
    }


def _outcome_cohort(current: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Run 25 read-only HSF-state historical context for the current state's
    cohort (same status + score band), at H24. Descriptive persistence only —
    never a price probability. None when unavailable / insufficient history."""
    if not current or current.get("status") is None or current.get("score") is None:
        return None
    try:
        from db.opportunity_outcomes import get_similar_state_outcomes

        c = get_similar_state_outcomes(status=current.get("status"), score=current.get("score"))
    except Exception:
        return None
    return c if c.get("available") else None


# ------------------------------- render --------------------------------------
_ICON = {"STRONG": "🟢", "WATCH": "🟡", "CAUTION": "🟠"}


def _regime_from_session() -> Optional[str]:
    try:
        return st.session_state.get("_last_market_regime")
    except Exception:
        return None


def render_stock_intelligence(
    ticker: str,
    *,
    current_opp: Optional[Dict[str, Any]] = None,
    current_row: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    render_chart_for_ticker: Optional[Callable[[str], None]] = None,
) -> None:
    """Render the canonical Stock Intelligence view. Read-only; never raises."""
    if st is None:
        return
    ticker = str(ticker or "").strip().upper()
    if not ticker:
        st.info("Enter a ticker to see its HSF intelligence.")
        return

    # Read-only fetches (small history query + cached calibration + session regime).
    history = []
    try:
        from db.signal_outcomes import fetch_ticker_opportunity_history
        history = fetch_ticker_opportunity_history(ticker)
    except Exception:
        history = []
    calibration_records = None
    try:
        from ui.market_brief import _calibration_records_cached
        calibration_records = _calibration_records_cached()
    except Exception:
        calibration_records = None
    earnings_days = None
    try:
        from ui.ai_confidence_explain import _earnings_days_for
        earnings_days = _earnings_days_for(ticker)
    except Exception:
        earnings_days = None

    intel = build_stock_intelligence(
        ticker, current_opp=current_opp, current_row=current_row, history=history,
        regime=_regime_from_session(), calibration_records=calibration_records,
        earnings_days=earnings_days)

    _render_header(intel)
    _render_why_and_risks(intel)
    _render_lifecycle(intel)
    _render_signals_and_model(intel)
    _render_watch_next(intel)
    _render_historical(intel)
    _render_actions(intel, render_chart_for_ticker)


def _render_header(intel: Dict[str, Any]) -> None:
    st.markdown(f"## {intel['ticker']}")
    st.caption(_watch_label(intel["ticker"]))
    price = intel.get("price")
    chg = intel.get("change_pct")
    if price is not None or chg is not None:
        pline = f"${price:,.2f}" if price is not None else ""
        cline = f"  ·  {chg:+.2f}%" if chg is not None else ""
        st.markdown(f"**{pline}{cline}**")
    if not intel["has_opportunity"]:
        st.info("No active HSF opportunity for this ticker right now "
                "(fewer than 2 confirming signals and no model score). "
                "Its HSF history, if any, is shown below.")
    else:
        score, status = intel["hsf_score"], intel["status"]
        line = f"**HSF {score} · {status}**  ·  v{intel.get('score_version')}"
        mv = intel.get("movement") or {}
        state = mv.get("movement_state")
        from ui.opportunities import movement_badge
        badge = movement_badge(mv) if mv else "—"
        tags = []
        if state == "VERSION_CHANGED":
            tags.append("previous score used a different HSF version")
        elif badge and badge != "—":
            tags.append(f"{badge} · {state}")
        if mv.get("status_transition"):
            tags.append(f"{mv['status_transition'][0]} → {mv['status_transition'][1]}")
        st.markdown(line + ("  ·  " + "  ·  ".join(tags) if tags else ""))
        if intel.get("from_history"):
            st.caption("From the latest recorded HSF observation (no live scan row).")
        setup = intel.get("primary_setup")
        if setup:
            st.markdown(f"Primary setup: **{setup}** · {intel['n_signals']} confirming signals")
    if intel.get("market_regime"):
        st.caption(f"Market: {intel['market_regime']}")


def _render_why_and_risks(intel: Dict[str, Any]) -> None:
    if intel.get("reasons"):
        st.markdown("#### Why HSF cares")
        st.markdown("\n".join(f"- ✓ {r}" for r in intel["reasons"]))
    st.markdown("#### Risks / conflicts")
    if intel.get("risks"):
        st.markdown("\n".join(f"- ⚠ {r}" for r in intel["risks"]))
    else:
        st.caption("No major HSF risk flags detected from the current signal set. "
                   "(This does not imply the stock is safe.)")


def _render_lifecycle(intel: Dict[str, Any]) -> None:
    life = intel.get("lifecycle") or []
    if not life:
        return
    st.markdown("#### Opportunity lifecycle")
    if len(life) == 1:
        st.caption("Only one recorded HSF observation so far — the timeline "
                   "appears as history accumulates.")
    for ev in life:
        try:
            ts = ev["time"].strftime("%b %d %I:%M %p") if hasattr(ev["time"], "strftime") else str(ev["time"])
        except Exception:
            ts = str(ev["time"])
        label = ev.get("label", "")
        added = f" · +{', '.join(ev['signals_added'])}" if ev.get("signals_added") else ""
        st.markdown(f"- **{ts}** — {label} · HSF {ev['score']} · {ev.get('status') or ''}{added}")


def _render_signals_and_model(intel: Dict[str, Any]) -> None:
    sigs = intel.get("signals") or []
    if sigs:
        st.markdown("#### Current signals")
        st.markdown("\n".join(f"- ✓ {_SETUP_LABELS.get(s, s)}" for s in sigs))
    comps = intel.get("score_components")
    model = intel.get("model") or {}
    with st.expander("HSF Score breakdown & model metrics", expanded=False):
        if comps:
            st.markdown(
                f"Signals {comps.get('signals_component')}/48 · "
                f"Model {comps.get('model_component')}/38 · "
                f"Momentum {comps.get('momentum_component')}/14 · "
                f"Risk penalty −{comps.get('fading_penalty')}  →  **HSF {intel['hsf_score']}**")
        st.caption("HSF Score is an opportunity ranking (0–100), NOT a probability.")
        bits = []
        if model.get("breakout_score") is not None:
            bits.append(f"BreakoutScore {model['breakout_score']:g}")
        if model.get("prebreakout_prob") is not None:
            bits.append(f"PreBreakout model confidence {model['prebreakout_prob']:g}%")
        if bits:
            st.markdown("Model outputs (separate from HSF Score): " + " · ".join(bits))


def _render_watch_next(intel: Dict[str, Any]) -> None:
    wn = intel.get("watch_next") or []
    if not wn:
        return
    st.markdown("#### What to watch next")
    st.markdown("\n".join(f"- {w}" for w in wn))


def _render_historical(intel: Dict[str, Any]) -> None:
    ctx = intel.get("historical_context")
    summ = intel.get("history_summary")
    if not ctx and not summ and not intel.get("outcome_cohort"):
        return
    st.markdown("#### Historical context")
    if ctx and ctx.get("sufficient"):
        st.caption(f"HSF {ctx['bucket']} · positive-outcome rate {ctx['positive_rate']*100:.0f}% "
                   f"(reached +4% in 5D) · n={ctx['n']} · {ctx['confidence'].title()}")
    elif ctx and ctx.get("n"):
        st.caption(f"HSF {ctx['bucket']} · Still building history · n={ctx['n']}")
    else:
        st.caption("Still building history.")
    if summ and summ["observations"]:
        st.caption(f"This ticker: {summ['observations']} prior HSF observation(s), "
                   f"{summ['matured']} matured, {summ['positive']} positive "
                   "(observations, not trades).")
    coh = intel.get("outcome_cohort")
    if coh and coh.get("available") and coh.get("follow_through_rate") is not None:
        st.caption(
            f"Similar HSF states (status {coh['status']} · HSF score {coh['score_band']}, "
            f"{coh['horizon']}): {coh['comparable']} observations, "
            f"{coh['follow_through_rate']*100:.0f}% persisted or strengthened "
            f"· evidence strength: {str(coh.get('evidence_strength', 'EARLY')).title()} "
            "(historical HSF-state persistence, not a price probability).")


def _render_actions(intel: Dict[str, Any], render_chart_for_ticker) -> None:
    t = intel["ticker"]
    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    if a1.button("📈 Chart", key=f"si_chart_{t}"):
        st.session_state[f"si_show_chart_{t}"] = True
    if a2.button(_watch_label(t), key=f"si_watch_{t}"):
        try:
            from ui.market_brief import _add_to_watchlist
            _add_to_watchlist(t)
        except Exception:
            st.caption("Watchlist unavailable.")
    if a3.button("🔔 Alert", key=f"si_alert_{t}"):
        st.session_state["alert_price_tk"] = t
        if intel.get("price") is not None:
            try:
                st.session_state["alert_price_val"] = round(float(intel["price"]), 2)
            except (TypeError, ValueError):
                pass
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open Alerts from the sidebar — it's pre-filled.")
    if st.session_state.get(f"si_show_chart_{t}"):
        try:
            if render_chart_for_ticker:
                render_chart_for_ticker(t)
            else:
                from ui.charts import render_chart_for_ticker as _chart
                _chart(t, key=f"si_chartimg_{t}")
        except Exception:
            st.caption("Chart unavailable.")
