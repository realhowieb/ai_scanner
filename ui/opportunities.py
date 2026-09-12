"""HSF Opportunity Score — rank the brief's signals into Top Opportunities.

Pure, testable logic (no Streamlit): aggregates the market brief's existing
signal lists per ticker, computes a 0-100 confluence-weighted score from ONLY
data that exists, and builds a deterministic plain-English explanation. Never
fabricates RVOL, EMA distance, resistance, earnings, or confidence numbers.

HSF OPPORTUNITY SCORE (0-100) — confluence-weighted, real-data-only
-------------------------------------------------------------------
The app's thesis is *confluence* (a name showing up across independent
scanners), so breadth of confirming signals is the heaviest component. Each
component contributes only when its underlying data is present; a missing input
scores 0 and never invents a value.

  signals_component (0-48): breadth of independent confirming signals.
      Each of {breakout, golden_cross, prebreakout, gapper, gainer} = 12 pts,
      capped at 48 (so ~4 confirming signals maxes it).
  model_component  (0-38): the strongest available model score, whichever
      exists — normalized BreakoutScore (score/60 * 38, capped) OR the
      PreBreakout model probability (prob/100 * 38). Not additive: we take the
      max so two model views don't double-count.
  momentum_component (0-14): positive intraday move only —
      clamp(chg_pct, 0, 6) / 6 * 14. Negative moves add nothing here.
  fading_penalty (-20): applied when the name also shows a 'loser'/fading tag
      (gap fade / reversal risk), so a confirmed-but-fading name is demoted.

  raw   = signals + model + momentum - fading_penalty
  score = round(clamp(raw, 0, 100))

Status tiers: STRONG (>=75 and not fading) · CAUTION (fading, or <50) · WATCH.

Example numbers in the product spec are illustrative — actual scores depend on
each session's real signal data.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

# Component weights (documented above). Tune here; keep the docstring in sync.
_SIGNAL_POINTS = 12
_SIGNAL_CAP = 48
_MODEL_CAP = 38
_BREAKOUT_SCORE_FULL = 60.0  # BreakoutScore value treated as "full marks"
_MOMENTUM_CAP = 14
_MOMENTUM_FULL_PCT = 6.0
_FADING_PENALTY = 20

# Positive confirming signals (order = display priority for "primary setup").
_POSITIVE_SIGNALS = ["golden_cross", "breakout", "prebreakout", "gapper", "gainer"]
_SETUP_LABELS = {
    "golden_cross": "Golden Cross",
    "breakout": "Breakout",
    "prebreakout": "PreBreakout",
    "gapper": "Gapper",
    "gainer": "Momentum",
    "fading": "Gap Fading",
}

_FLAG_RE = re.compile(r"[⚠🚩].*$")


def _default_base_ticker(raw: Any) -> str:
    """Strip earnings/warning flag suffixes and normalize to a bare symbol."""
    if raw is None:
        return ""
    t = _FLAG_RE.sub("", str(raw)).strip().upper()
    return t.split()[0] if t else ""


def build_opportunities(
    data: Dict[str, Any],
    *,
    base_ticker: Optional[Callable[[Any], str]] = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Aggregate the brief's signal lists into ranked opportunities.

    Returns a list of dicts (highest score first), each carrying the score, the
    primary setup, confirming-signal count, status, and the real underlying
    fields used (for the deterministic explanation). Safe on empty/partial data.
    """
    norm = base_ticker or _default_base_ticker
    if not isinstance(data, dict):
        return []

    acc: Dict[str, Dict[str, Any]] = {}

    def slot(raw: Any) -> Optional[Dict[str, Any]]:
        t = norm(raw)
        if not t:
            return None
        return acc.setdefault(
            t,
            {"ticker": t, "signals": set(), "breakout_score": None,
             "prob": None, "chg_pct": None, "gap_pct": None},
        )

    for g in (data.get("gappers") or []):
        s = slot(g.get("ticker"))
        if s is None:
            continue
        s["signals"].add("gapper")
        if g.get("gap_pct") is not None:
            s["gap_pct"] = _to_float(g.get("gap_pct"))
        if g.get("chg_pct") is not None:
            s["chg_pct"] = _to_float(g.get("chg_pct"))

    for t in (data.get("golden") or []):
        s = slot(t)
        if s is not None:
            s["signals"].add("golden_cross")

    for pair in (data.get("top_setups") or []):
        try:
            t, score = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("breakout")
            s["breakout_score"] = _to_float(score)

    for p in (data.get("picks") or []):
        s = slot(p.get("symbol"))
        if s is not None:
            s["signals"].add("prebreakout")
            s["prob"] = _to_float(p.get("prob"))

    for pair in (data.get("gainers") or []):
        try:
            t, chg = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("gainer")
            if s["chg_pct"] is None:
                s["chg_pct"] = _to_float(chg)

    for pair in (data.get("losers") or []):
        try:
            t, chg = pair
        except (TypeError, ValueError):
            continue
        s = slot(t)
        if s is not None:
            s["signals"].add("fading")
            if s["chg_pct"] is None:
                s["chg_pct"] = _to_float(chg)

    opportunities = []
    for s in acc.values():
        pos = [sig for sig in _POSITIVE_SIGNALS if sig in s["signals"]]
        fading = "fading" in s["signals"]
        # A pure single-list name with no confluence and no model score isn't an
        # "opportunity" — require either confluence (2+) or a model score.
        if len(pos) < 2 and s["breakout_score"] is None and s["prob"] is None:
            continue
        score = build_opportunity_score(
            n_signals=len(pos),
            breakout_score=s["breakout_score"],
            prob=s["prob"],
            chg_pct=s["chg_pct"],
            fading=fading,
        )
        opportunities.append({
            "ticker": s["ticker"],
            "score": score,
            "primary_setup": _primary_setup(pos, fading),
            "n_signals": len(pos),
            "signals": pos,
            "fading": fading,
            "status": _status(score, fading),
            "breakout_score": s["breakout_score"],
            "prob": s["prob"],
            "chg_pct": s["chg_pct"],
            "gap_pct": s["gap_pct"],
        })

    opportunities.sort(key=lambda o: o["score"], reverse=True)
    return opportunities[: max(0, int(top_n))]


def build_opportunity_score(
    *,
    n_signals: int,
    breakout_score: Optional[float],
    prob: Optional[float],
    chg_pct: Optional[float],
    fading: bool,
) -> int:
    """Compute the 0-100 HSF Opportunity Score. See module docstring for the
    formula. Missing inputs contribute 0 (never a fabricated value)."""
    signals_component = min(int(n_signals) * _SIGNAL_POINTS, _SIGNAL_CAP)

    model_candidates = [0.0]
    if breakout_score is not None:
        model_candidates.append(min(float(breakout_score) / _BREAKOUT_SCORE_FULL, 1.0) * _MODEL_CAP)
    if prob is not None:
        model_candidates.append(min(max(float(prob), 0.0) / 100.0, 1.0) * _MODEL_CAP)
    model_component = max(model_candidates)

    momentum_component = 0.0
    if chg_pct is not None and chg_pct > 0:
        momentum_component = min(float(chg_pct), _MOMENTUM_FULL_PCT) / _MOMENTUM_FULL_PCT * _MOMENTUM_CAP

    raw = signals_component + model_component + momentum_component
    if fading:
        raw -= _FADING_PENALTY
    return int(round(max(0.0, min(100.0, raw))))


def _primary_setup(pos_signals: List[str], fading: bool) -> str:
    for sig in _POSITIVE_SIGNALS:
        if sig in pos_signals:
            return _SETUP_LABELS[sig]
    return _SETUP_LABELS["fading"] if fading else "Signal"


def _status(score: int, fading: bool) -> str:
    if fading:
        return "CAUTION"
    if score >= 75:
        return "STRONG"
    if score < 50:
        return "CAUTION"
    return "WATCH"


def build_opportunity_explanation(
    opp: Dict[str, Any],
    *,
    earnings_today: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Deterministic 'why it ranked' + risk flags from real fields only.

    Returns {"reasons": [...], "risks": [...]}. No invented metrics: every line
    is backed by a value present on the opportunity or the earnings list.
    """
    reasons: List[str] = []
    risks: List[str] = []
    signals = opp.get("signals") or []

    if "golden_cross" in signals:
        reasons.append("Fresh EMA 9/21 golden cross")
    if "breakout" in signals:
        bs = opp.get("breakout_score")
        reasons.append(f"Breakout setup (score {bs:g})" if bs is not None else "Breakout setup")
    if "prebreakout" in signals:
        prob = opp.get("prob")
        reasons.append(f"PreBreakout model confidence {prob:g}%" if prob is not None else "PreBreakout model pick")
    if "gapper" in signals:
        gap = opp.get("gap_pct")
        reasons.append(f"Gapping {gap:+.1f}%" if gap is not None else "Gapping today")
    if "gainer" in signals:
        chg = opp.get("chg_pct")
        reasons.append(f"Positive momentum {chg:+.1f}%" if chg is not None else "Positive momentum")

    n = int(opp.get("n_signals") or 0)
    if n >= 2:
        reasons.append(f"Confluence — appears across {n} independent signals")

    if opp.get("fading"):
        chg = opp.get("chg_pct")
        risks.append(f"Fading / reversal risk ({chg:+.1f}%)" if chg is not None else "Fading / reversal risk")
    et = {str(x).upper() for x in (earnings_today or [])}
    if opp.get("ticker", "").upper() in et:
        risks.append("Earnings reporting today")

    return {"reasons": reasons, "risks": risks}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
