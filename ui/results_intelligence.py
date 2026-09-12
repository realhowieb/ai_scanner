"""Scanner / Results intelligence layer — turn raw scan output into a ranked,
consolidated HSF opportunity view.

Reuses the CANONICAL HSF logic from ui.opportunities (score_breakdown,
compare_opportunities, classify_market_regime, build_opportunity_explanation,
movement_badge) — this file never re-implements the score or business rules. It
only adapts the scanner DataFrame's real columns into the (n_signals,
breakout_score, prob, chg_pct, fading) inputs the canonical scorer expects, then
consolidates by ticker, classifies views, and enriches movement.

Signal presence is derived from REAL scanner fields with documented, conservative
thresholds (a definition of "signal present", not a fabricated metric value):

  breakout      IsBreakout is truthy
  golden_cross  EMACross == "Golden"
  prebreakout   PreBreakoutProb% >= PREBREAKOUT_SIGNAL_MIN
  gapper        |GapPct| >= GAP_SIGNAL_MIN
  gainer        PctChange >= GAINER_SIGNAL_MIN
  fading        PctChange <= FADING_SIGNAL_MAX

Model inputs: breakout_score = BreakoutScore, prob = PreBreakoutProb%,
momentum = PctChange (GapPct fallback). Missing fields contribute nothing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Documented signal-derivation thresholds (transparent, conservative).
PREBREAKOUT_SIGNAL_MIN = 50.0   # PreBreakoutProb% at/above this = model confirmation
GAP_SIGNAL_MIN = 2.0            # |gap %| at/above this = a gap
GAINER_SIGNAL_MIN = 2.0         # day % at/above this = positive momentum
FADING_SIGNAL_MAX = -2.0        # day % at/below this = fading/reversal


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    try:
        return bool(v) and v == v  # not NaN
    except Exception:
        return False


def _col(row: Dict[str, Any], *names) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _row_to_signal_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the canonical scorer inputs from one scanner result row."""
    gap = _num(_col(row, "GapPct", "Gap %"))
    chg = _num(_col(row, "PctChange", "Chg %", "chg_pct"))
    prob = _num(_col(row, "PreBreakoutProb%", "PreBreakoutProb"))
    bscore = _num(_col(row, "BreakoutScore"))
    ema = _col(row, "EMACross")
    signals = set()
    if _truthy(_col(row, "IsBreakout")):
        signals.add("breakout")
    if isinstance(ema, str) and ema.strip().lower() == "golden":
        signals.add("golden_cross")
    if prob is not None and prob >= PREBREAKOUT_SIGNAL_MIN:
        signals.add("prebreakout")
    if gap is not None and abs(gap) >= GAP_SIGNAL_MIN:
        signals.add("gapper")
    if chg is not None and chg >= GAINER_SIGNAL_MIN:
        signals.add("gainer")
    fading = chg is not None and chg <= FADING_SIGNAL_MAX
    return {
        "signals": signals, "fading": fading,
        "breakout_score": bscore, "prob": prob,
        "chg_pct": chg if chg is not None else gap,
        "gap_pct": gap,
        "last": _num(_col(row, "Last", "last", "Price")),
        "rvol": _num(_col(row, "VolRel20")),
    }


def _ticker_of(row: Dict[str, Any]) -> str:
    return str(_col(row, "Ticker", "Symbol", "ticker") or "").strip().upper()


def consolidate_scanner_results(
    rows: List[Dict[str, Any]], *, top_n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Consolidate scanner rows into ranked HSF opportunities (one per ticker).

    Merges duplicate tickers (union of signals, best/ freshest values), scores
    each via the CANONICAL score_breakdown, keeps only names that qualify
    (>=2 confirming signals OR a model score — same rule as build_opportunities),
    and sorts deterministically. Non-qualifying rows are simply not returned
    here (they remain in ALL RESULTS). Safe on empty/malformed input.
    """
    from ui.opportunities import (
        HSF_SCORE_VERSION,
        _primary_setup,
        _status,
        score_breakdown,
    )

    _POSITIVE = ["golden_cross", "breakout", "prebreakout", "gapper", "gainer"]
    acc: Dict[str, Dict[str, Any]] = {}
    for raw in (rows or []):
        if not isinstance(raw, dict):
            continue
        t = _ticker_of(raw)
        if not t:
            continue
        f = _row_to_signal_fields(raw)
        cur = acc.get(t)
        if cur is None:
            acc[t] = {"ticker": t, **f}
        else:
            # Merge duplicates: union signals; prefer non-None / stronger values.
            cur["signals"] |= f["signals"]
            cur["fading"] = cur["fading"] or f["fading"]
            for k in ("breakout_score", "prob"):
                if f[k] is not None and (cur[k] is None or f[k] > cur[k]):
                    cur[k] = f[k]
            for k in ("chg_pct", "gap_pct", "last", "rvol"):
                if cur.get(k) is None and f.get(k) is not None:
                    cur[k] = f[k]

    opps = []
    for s in acc.values():
        pos = [x for x in _POSITIVE if x in s["signals"]]
        if len(pos) < 2 and s["breakout_score"] is None and s["prob"] is None:
            continue  # doesn't qualify as an HSF opportunity
        bd = score_breakdown(n_signals=len(pos), breakout_score=s["breakout_score"],
                             prob=s["prob"], chg_pct=s["chg_pct"], fading=s["fading"])
        score = bd["score"]
        opps.append({
            "ticker": s["ticker"], "score": score, "score_version": HSF_SCORE_VERSION,
            "score_components": {k: bd[k] for k in
                                 ("signals_component", "model_component",
                                  "momentum_component", "fading_penalty")},
            "primary_setup": _primary_setup(pos, s["fading"]),
            "n_signals": len(pos), "signals": pos, "fading": s["fading"],
            "status": _status(score, s["fading"]),
            "breakout_score": s["breakout_score"], "prob": s["prob"],
            "chg_pct": s["chg_pct"], "gap_pct": s["gap_pct"],
            "last": s.get("last"), "rvol": s.get("rvol"),
        })

    # Deterministic ranking: score, status strength, signal count, model, ticker.
    _rank = {"STRONG": 3, "WATCH": 2, "CAUTION": 1}
    opps.sort(key=lambda o: (
        -o["score"], -_rank.get(o["status"], 0), -o["n_signals"],
        -((o["breakout_score"] or 0) + (o["prob"] or 0)), o["ticker"],
    ))
    return opps[:top_n] if top_n else opps


def enrich_movement(
    opps: List[Dict[str, Any]],
    previous_rows: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Version-safe movement enrichment vs the previous opportunity snapshot.

    Thin wrapper over the CANONICAL compare_opportunities, which already returns
    NO_BASELINE (no previous snapshot), NEW, VERSION_CHANGED (incompatible score
    version — no ▲/▼), and RISING/FALLING/UNCHANGED. Kept as a named entry point
    for the scanner; the movement definition lives in one place.
    """
    from ui.opportunities import compare_opportunities

    return compare_opportunities(opps, previous_rows)


def classify_views(compared: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Views over the SAME consolidated opportunities (no rescan)."""
    top = list(compared)  # already deterministically ranked
    new = [c for c in compared if c["movement_state"] == "NEW"]
    fading = [c for c in compared
              if c.get("fading")
              or c["movement_state"] == "FALLING"
              or (c.get("status_transition") and _rank(c["status_transition"][1]) < _rank(c["status_transition"][0]))]
    fading_tickers = {c["ticker"] for c in fading}
    developing = [c for c in compared
                  if c["ticker"] not in fading_tickers
                  and (c.get("status") == "WATCH" or c["movement_state"] == "RISING"
                       or (c.get("status_transition") and _rank(c["status_transition"][1]) > _rank(c["status_transition"][0])))]
    developing.sort(key=lambda c: (
        0 if c["movement_state"] == "RISING" else 1,
        -(c.get("score_delta") or 0), -c["score"], -c["n_signals"], c["ticker"],
    ))
    return {"top": top, "developing": developing, "new": new, "fading": fading}


def _rank(status: Optional[str]) -> int:
    return {"STRONG": 3, "WATCH": 2, "CAUTION": 1}.get(str(status or "").upper(), 0)


def summarize_results(
    compared: List[Dict[str, Any]],
    *,
    total_matches: int,
    has_previous: bool,
    regime: Optional[str] = None,
) -> Dict[str, Any]:
    """Counts for the results summary header. NEW/RISING omitted when there is
    no valid previous snapshot to compare against."""
    out = {
        "total_matches": int(total_matches),
        "opportunities": len(compared),
        "strong": sum(1 for c in compared if c["status"] == "STRONG"),
        "watch": sum(1 for c in compared if c["status"] == "WATCH"),
        "caution": sum(1 for c in compared if c["status"] == "CAUTION"),
        "regime": regime,
        "has_previous": bool(has_previous),
    }
    if has_previous:
        out["new"] = sum(1 for c in compared if c["movement_state"] == "NEW")
        out["rising"] = sum(1 for c in compared if c["movement_state"] == "RISING")
        out["falling"] = sum(1 for c in compared if c["movement_state"] == "FALLING")
    return out


# ----------------------------- Streamlit render ------------------------------
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_STATUS_ICON = {"STRONG": "🟢", "WATCH": "🟡", "CAUTION": "🟠"}
_VIEWS = ["Top ranked", "Developing", "New", "Fading", "All results"]


def _movement_cell(c: Dict[str, Any]) -> str:
    from ui.opportunities import movement_badge
    if c.get("movement_state") == "VERSION_CHANGED":
        return "ver"
    return movement_badge(c)


# Columns that materially change scanner intelligence (HSF score, status,
# primary setup, signal count, movement, view classification). Presentation-only
# fields (e.g. price/Last, Volume, Spark) are deliberately excluded so the cache
# isn't invalidated by changes that can't alter the intelligence.
_FINGERPRINT_COLS = (
    "Ticker", "Symbol", "IsBreakout", "EMACross",
    "PreBreakoutProb%", "PreBreakoutProb", "BreakoutScore", "PctChange", "GapPct",
)


def _df_signature(df: Any) -> str:
    """Content fingerprint over the intelligence-relevant columns only.

    Order-independent (consolidation/ranking are order-independent), NaN/None/
    missing-column/empty safe, and cheap (vectorized hash over a small subset —
    never re-scores or touches the DB/network). Changes iff a field that can
    alter the intelligence changes; row reordering does not.
    """
    try:
        if df is None or getattr(df, "empty", True):
            return "empty"
        cols = [c for c in _FINGERPRINT_COLS if c in getattr(df, "columns", [])]
        if not cols:
            return f"na:{len(df)}:{len(getattr(df, 'columns', []))}"
        import hashlib

        from pandas.util import hash_pandas_object

        # Per-row uint64 hashes (NaN-safe), then combine order-independently by
        # sorting the row hashes before digesting — so the same set of rows in
        # any order yields the same fingerprint.
        row_hashes = hash_pandas_object(df[cols], index=False)
        ordered = sorted(int(x) for x in row_hashes.to_numpy())
        digest = hashlib.sha1(repr(ordered).encode()).hexdigest()[:16]
        return f"{len(df)}:{len(cols)}:{digest}"
    except Exception:
        return "na"


def render_scanner_intelligence(
    df: Any,
    *,
    key_prefix: str = "results",
    render_chart_for_ticker: Optional[Callable[[str], None]] = None,
) -> None:
    """Intelligence-first layer above the raw results table. Never raises; the
    raw ALL RESULTS table below is unaffected if anything here fails."""
    if st is None or df is None or getattr(df, "empty", True):
        return
    try:
        rows = df.to_dict("records")
    except Exception:
        return
    total = len(rows)

    # Consolidation + movement are cached per result-set so switching views or
    # any rerun does NOT recompute or re-hit the DB. READ-ONLY: the scanner
    # never writes snapshots or freezes opportunities — Market Brief owns that,
    # so opening Results adds no DB writes and no brief rebuild.
    sig = _df_signature(df)
    cache = st.session_state.get(f"{key_prefix}_intel_cache")
    if cache and cache.get("sig") == sig:
        compared, has_prev = cache["compared"], cache["has_prev"]
    else:
        opps = consolidate_scanner_results(rows)
        previous_rows = None
        try:
            import datetime as dt

            from db.opportunity_snapshots import load_previous_opportunity_snapshot
            # Same-context only: the scanner never writes snapshots (read-only,
            # Run 20B), so it has no 'scanner' baseline and correctly reports
            # NO_BASELINE instead of misleading movement against Market Brief
            # opportunities (a different universe). Movement lights up here only
            # if/when the scanner persists its own comparable history.
            prev = load_previous_opportunity_snapshot(
                dt.datetime.now(dt.timezone.utc), context="scanner")
            previous_rows = prev.get("opportunities") if prev else None
        except Exception:
            previous_rows = None
        compared = enrich_movement(opps, previous_rows)
        has_prev = bool(previous_rows)
        st.session_state[f"{key_prefix}_intel_cache"] = {
            "sig": sig, "compared": compared, "has_prev": has_prev}

    # Regime is read from whatever Market Brief last computed (cheap, cached in
    # session); we never trigger a brief rebuild just to label the scanner.
    regime = st.session_state.get("_last_market_regime")
    summary = summarize_results(compared, total_matches=total,
                                has_previous=has_prev, regime=regime)
    views = classify_views(compared)

    # --- Summary header (compact) ---
    st.markdown("### 🎯 HSF Opportunities")
    bits = [f"**{summary['opportunities']}** opportunities of {total} matches"]
    if summary["strong"]:
        bits.append(f"🟢 {summary['strong']} STRONG")
    if summary["watch"]:
        bits.append(f"🟡 {summary['watch']} WATCH")
    if has_prev and summary.get("rising"):
        bits.append(f"▲ {summary['rising']} rising")
    if has_prev and summary.get("new"):
        bits.append(f"NEW {summary['new']}")
    st.markdown(" &nbsp;·&nbsp; ".join(bits))
    if summary.get("regime"):
        st.caption(f"Market regime: {summary['regime']}")

    if not compared:
        st.caption("No names met HSF opportunity qualification (2+ confirming "
                   "signals or a model score). See the full results table below.")
        return

    # --- Intelligence views (over the SAME results — no rescan) ---
    view = st.radio("View", _VIEWS, horizontal=True, key=f"{key_prefix}_intel_view",
                    label_visibility="collapsed")
    if view == "All results":
        st.caption("Full scanner output is in the results table below.")
        return
    keymap = {"Top ranked": "top", "Developing": "developing", "New": "new", "Fading": "fading"}
    items = views.get(keymap[view], [])
    if view == "New" and not has_prev:
        st.caption("No previous scan to compare against yet — NEW appears once a "
                   "prior snapshot exists.")
        return
    if not items:
        st.caption(f"No {view.lower()} opportunities in this scan.")
        return

    # Ticker search within the current results (no rescan).
    q = st.text_input("Search ticker", key=f"{key_prefix}_intel_search",
                      placeholder="Filter this view by ticker…", label_visibility="collapsed").strip().upper()
    if q:
        filtered = [c for c in items if q in c["ticker"]]
        if not filtered:
            st.caption(f"{q} is not in the {view.lower()} view.")
            return
        items = filtered

    rows_out = []
    for i, c in enumerate(items[:25], 1):
        tr = c.get("status_transition")
        status = f"{tr[0]} → {tr[1]}" if tr else f"{_STATUS_ICON.get(c['status'], '')} {c['status']}"
        rows_out.append({
            "#": i, "Ticker": c["ticker"], "HSF": c["score"], "Δ": _movement_cell(c),
            "Setup": c["primary_setup"], "Signals": c["n_signals"], "Status": status,
        })
    try:
        cc = st.column_config
        st.dataframe(rows_out, hide_index=True, width="stretch", column_config={
            "#": cc.NumberColumn(width="small"),
            "HSF": cc.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "Δ": cc.TextColumn(width="small"), "Signals": cc.NumberColumn(width="small"),
        })
    except Exception:
        st.dataframe(rows_out, hide_index=True, width="stretch")

    tickers = [c["ticker"] for c in items]
    pick = st.selectbox("Inspect", tickers, key=f"{key_prefix}_intel_pick",
                        label_visibility="collapsed")
    c = next((x for x in items if x["ticker"] == pick), None)
    if c:
        _render_result_detail(c, key_prefix=key_prefix, render_chart_for_ticker=render_chart_for_ticker)


def _render_result_detail(
    c: Dict[str, Any], *, key_prefix: str,
    render_chart_for_ticker: Optional[Callable[[str], None]],
) -> None:
    from ui.opportunities import build_opportunity_explanation, movement_badge

    sub = []
    if c.get("movement_state") == "VERSION_CHANGED":
        sub.append("score version changed · movement unavailable")
    else:
        b = movement_badge(c)
        if b and b != "—":
            sub.append(b)
    tr = c.get("status_transition")
    if tr:
        sub.append(f"{tr[0]} → {tr[1]}")
    price = f" · ${c['last']:,.2f}" if c.get("last") is not None else ""
    chg = f" ({c['chg_pct']:+.1f}%)" if c.get("chg_pct") is not None else ""
    st.markdown(f"**{c['ticker']} — HSF {c['score']}/100 · {c['status']}**{price}{chg}"
                + ("  ·  " + "  ·  ".join(sub) if sub else ""))

    ex = build_opportunity_explanation(c, earnings_today=[])
    if ex["reasons"]:
        st.markdown("**Why this ranked**")
        st.markdown("\n".join(f"- ✓ {r}" for r in ex["reasons"]))
    if ex["risks"]:
        st.markdown("**Risk**")
        st.markdown("\n".join(f"- ⚠ {r}" for r in ex["risks"]))

    # Score component transparency (canonical breakdown).
    comps = c.get("score_components") or {}
    if comps:
        st.caption(
            f"Signals {comps.get('signals_component')}/48 · "
            f"Model {comps.get('model_component')}/38 · "
            f"Momentum {comps.get('momentum_component')}/14 · "
            f"Fading {comps.get('fading_penalty')}"
        )

    # Historical context (reuses Run 18B calibration; 'still building' until n>=10).
    try:
        from analytics.hsf_calibration import historical_context
        from ui.market_brief import _calibration_records_cached
        ctx = historical_context(_calibration_records_cached(), c.get("score"))
        if ctx and ctx.get("sufficient"):
            st.caption(f"📊 Historical context · {ctx['bucket']}: "
                       f"{ctx['positive_rate']*100:.0f}% positive outcome · n={ctx['n']} · {ctx['confidence'].title()}")
        elif ctx and ctx.get("n"):
            st.caption(f"📊 Historical context · Still building history · n={ctx['n']}")
    except Exception:
        pass

    # Actions — reuse existing chart / watchlist / alert (no duplicate systems).
    a0, a1, a2, a3 = st.columns(4)
    if a0.button("🔬 Full intel", key=f"{key_prefix}_intel_open_{c['ticker']}"):
        st.session_state["hsf_stock_ticker"] = c["ticker"]
        st.session_state["hsf_stock_opp"] = c
        try:
            st.switch_page("pages/stock.py")
        except Exception:
            st.caption("Open 'Stock Intel' from the sidebar.")
    if a1.button("📈 Chart", key=f"{key_prefix}_intel_chart_{c['ticker']}"):
        st.session_state[f"{key_prefix}_intel_show_chart"] = c["ticker"]
    if a2.button("👁 Watch", key=f"{key_prefix}_intel_watch_{c['ticker']}"):
        try:
            from ui.market_brief import _add_to_watchlist
            _add_to_watchlist(c["ticker"])
        except Exception:
            st.caption("Watchlist unavailable.")
    if a3.button("🔔 Alert", key=f"{key_prefix}_intel_alert_{c['ticker']}"):
        st.session_state["alert_price_tk"] = c["ticker"]
        if c.get("last") is not None:
            try:
                st.session_state["alert_price_val"] = round(float(c["last"]), 2)
            except (TypeError, ValueError):
                pass
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open Alerts from the sidebar — it's pre-filled.")
    if st.session_state.get(f"{key_prefix}_intel_show_chart") == c["ticker"] and render_chart_for_ticker:
        try:
            render_chart_for_ticker(c["ticker"])
        except Exception:
            st.caption("Chart unavailable.")
